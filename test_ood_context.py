# test on eval tasks with out-of-distribution context
# use a fixed policy to collect context for different test tasks

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torchkit.pytorch_utils import set_gpu_mode
import utils.config_utils as config_utl
from utils import helpers as utl, offline_utils as off_utl
from offline_rl_config import args_gridworld_block, args_cheetah_vel, args_ant_dir, args_point_robot_v1,\
    args_hopper_param, args_walker_param
import numpy as np
import random

from models.encoder import RNNEncoder, MLPEncoder, SelfAttnEncoder
from algorithms.dqn import DQN
from algorithms.sac import SAC
from models.generative import CVAE
from environments.make_env import make_env
from torchkit import pytorch_utils as ptu
from torchkit.networks import FlattenMlp
from data_management.storage_policy import MultiTaskPolicyStorage
from utils import evaluation as utl_eval
from utils.tb_logger import TBLogger
from models.policy import TanhGaussianPolicy
from offline_learner import OfflineMetaLearner

import matplotlib.pyplot as plt
#import matplotlib.colors as mcolors
from sklearn import manifold


class OfflineContrastive(OfflineMetaLearner):
    # algorithm class of offline meta-rl with relabelling

    def __init__(self, args, train_dataset, train_goals, eval_dataset, eval_goals):
        """
        Seeds everything.
        Initialises: logger, environments, policy (+storage +optimiser).
        """

        self.args = args

        # make sure everything has the same seed
        utl.seed(self.args.seed)

        # initialize tensorboard logger
        #if self.args.log_tensorboard:
        #    self.tb_logger = TBLogger(self.args)

        self.args, _ = off_utl.expand_args(self.args, include_act_space=True)
        if self.args.act_space.__class__.__name__ == "Discrete":
            self.args.policy = 'dqn'
        else:
            self.args.policy = 'sac'

        # load augmented buffer to self.storage
        #self.load_buffer(train_dataset, train_goals)
        if self.args.pearl_deterministic_encoder:
            self.args.augmented_obs_dim = self.args.obs_dim + self.args.task_embedding_size
        else:
            self.args.augmented_obs_dim = self.args.obs_dim + self.args.task_embedding_size * 2
        self.goals = train_goals
        self.eval_goals = eval_goals
        # context set, to extract task encoding
        self.context_dataset = train_dataset
        self.eval_context_dataset = eval_dataset


        # initialize policy
        self.initialize_policy()

        # initialize task encoder
        self.encoder = MLPEncoder(
                hidden_size=self.args.aggregator_hidden_size,
                num_hidden_layers=2,
                task_embedding_size=self.args.task_embedding_size,
                action_size=self.args.act_space.n if self.args.act_space.__class__.__name__ == "Discrete" else self.args.action_dim,
                state_size=self.args.obs_dim,
                reward_size=1,
                term_size=0, # encode (s,a,r,s') only
                normalize=self.args.normalize_z
        	).to(ptu.device)
        #self.encoder.load_state_dict(torch.load(self.args.encoder_model_path, map_location=ptu.device))

        # context encoder: convert (batch, N, dim) to (batch, dim)
        self.context_encoder = SelfAttnEncoder(input_dim=self.args.task_embedding_size,
            num_output_mlp=self.args.context_encoder_output_layers, task_gt_dim=self.goals[0].shape[0],
            ).to(ptu.device)
        #self.context_encoder_optimizer = torch.optim.Adam(self.context_encoder.parameters(), lr=self.args.encoder_lr)


        # create environment for evaluation
        self.env = make_env(args.env_name,
                            args.max_rollouts_per_task,
                            seed=args.seed,
                            n_tasks=self.args.num_eval_tasks)
        # fix the possible eval goals to be the testing set's goals
        self.env.set_all_goals(eval_goals)

        # create env for eval on training tasks
        self.env_train = make_env(args.env_name,
                            args.max_rollouts_per_task,
                            seed=args.seed,
                            n_tasks=self.args.num_train_tasks)
        self.env_train.set_all_goals(train_goals)


    def evaluate(self, trainset=False, ood=True):
        num_episodes = self.args.max_rollouts_per_task
        num_steps_per_episode = self.env.unwrapped._max_episode_steps
        num_tasks = self.args.num_train_tasks if trainset else self.args.num_eval_tasks
        obs_size = self.env.unwrapped.observation_space.shape[0]

        returns_per_episode = np.zeros((num_tasks, num_episodes))
        success_rate = np.zeros(num_tasks)

        rewards = np.zeros((num_tasks, self.args.trajectory_len))
        reward_preds = np.zeros((num_tasks, self.args.trajectory_len))
        observations = np.zeros((num_tasks, self.args.trajectory_len + 1, obs_size))
        if self.args.policy == 'sac':
            log_probs = np.zeros((num_tasks, self.args.trajectory_len))

        eval_env = self.env_train if trainset else self.env
        for task in eval_env.unwrapped.get_all_task_idx():
            obs = ptu.from_numpy(eval_env.reset(task))
            obs = obs.reshape(-1, obs.shape[-1])
            step = 0

            if ood:
                obs_context, actions_context, rewards_context, next_obs_context, terms_context = self.sample_context_batch_ood([task], trainset=trainset)
            else:
                obs_context, actions_context, rewards_context, next_obs_context, terms_context = self.sample_context_batch([task], trainset=trainset)
            #task_desc = self.encoder.context_encoding(obs=obs_context, actions=actions_context, 
            #   rewards=rewards_context, next_obs=next_obs_context, terms=terms_context)
            n_timesteps, batch_size, _ = obs_context.shape
            task_desc = self.encoder(
                    obs=obs_context.reshape(n_timesteps*batch_size, -1), 
                    action=actions_context.reshape(n_timesteps*batch_size, -1), 
                    reward=rewards_context.reshape(n_timesteps*batch_size, -1), 
                    next_obs=next_obs_context.reshape(n_timesteps*batch_size, -1),
                ).view(n_timesteps, batch_size, -1).transpose(0,1)
            #print(task_desc.shape)
            task_desc = self.context_encoder(task_desc)
            #print(task_desc.shape)

            observations[task, step, :] = ptu.get_numpy(obs[0, :obs_size])

            for episode_idx in range(num_episodes):
                running_reward = 0.
                for step_idx in range(num_steps_per_episode):
                    # add distribution parameters to observation - policy is conditioned on posterior
                    augmented_obs = torch.cat((obs, task_desc), dim=-1)
                    if self.args.policy == 'dqn':
                        action, value = self.agent.act(obs=augmented_obs, deterministic=True)
                    else:
                        action, _, _, log_prob = self.agent.act(obs=augmented_obs,
                                                                deterministic=self.args.eval_deterministic,
                                                                return_log_prob=True)

                    # observe reward and next obs
                    next_obs, reward, done, info = utl.env_step(eval_env, action.squeeze(dim=0))
                    running_reward += reward.item()
                    # done_rollout = False if ptu.get_numpy(done[0][0]) == 0. else True
                    # update encoding
                    #task_sample, task_mean, task_logvar, hidden_state = self.update_encoding(obs=next_obs,
                    #                                                                         action=action,
                    #                                                                         reward=reward,
                    #                                                                         done=done,
                    #                                                                         hidden_state=hidden_state)
                    rewards[task, step] = reward.item()
                    #reward_preds[task, step] = ptu.get_numpy(
                    #    self.vae.reward_decoder(task_sample, next_obs, obs, action)[0, 0])

                    observations[task, step + 1, :] = ptu.get_numpy(next_obs[0, :obs_size])
                    if self.args.policy != 'dqn':
                        log_probs[task, step] = ptu.get_numpy(log_prob[0])

                    if "is_goal_state" in dir(eval_env.unwrapped) and eval_env.unwrapped.is_goal_state():
                        success_rate[task] = 1.
                    # set: obs <- next_obs
                    obs = next_obs.clone()
                    step += 1

                returns_per_episode[task, episode_idx] = running_reward

        # reward_preds is 0 here
        if self.args.policy == 'dqn':
            return returns_per_episode, success_rate, observations, rewards, reward_preds
        else:
            return returns_per_episode, success_rate, log_probs, observations, rewards, reward_preds

    def load_behavior_policy(self, path, hidden):
        q1_network = FlattenMlp(input_size=self.args.obs_dim + self.args.action_dim,
                                output_size=1,
                                hidden_sizes=[hidden, hidden])
        q2_network = FlattenMlp(input_size=self.args.obs_dim + self.args.action_dim,
                                output_size=1,
                                hidden_sizes=[hidden,hidden])
        policy = TanhGaussianPolicy(obs_dim=self.args.obs_dim,
                                    action_dim=self.args.action_dim,
                                    hidden_sizes=[hidden,hidden])
        self.context_agent = SAC(
            policy,
            q1_network,
            q2_network,

            actor_lr=self.args.actor_lr,
            critic_lr=self.args.critic_lr,
            gamma=self.args.gamma,
            tau=self.args.soft_target_tau,

            entropy_alpha=self.args.entropy_alpha,
            automatic_entropy_tuning=self.args.automatic_entropy_tuning,
            alpha_lr=self.args.alpha_lr
        ).to(ptu.device)

        self.context_agent.load_state_dict(torch.load(path, map_location=ptu.device))

    # use context_agent to colloct context for each task
    def sample_context_batch_ood(self, tasks, trainset=True):
        test_env = make_env(self.args.env_name,
            self.args.max_rollouts_per_task,
            seed=self.args.seed,
            n_tasks=1)

        if trainset:
            goals = self.goals[tasks]
        else:
            goals = self.eval_goals[tasks]
        #print(goals)

        context = []
        for i, g in enumerate(goals):
            obs_c, act_c, rew_c, next_obs_c, term_c = [],[],[],[],[]

            for rollout in range(self.args.num_context_trajs):
                test_env.set_goal(g)
                obs = ptu.from_numpy(test_env.reset())
                obs = obs.reshape(-1, obs.shape[-1])
                done_rollout = False

                while not done_rollout:
                    if self.args.policy == 'dqn':
                        action, _ = self.context_agent.act(obs=obs)   # DQN
                    else:
                        action, _, _, _ = self.context_agent.act(obs=obs)   # SAC
                    # observe reward and next obs
                    next_obs, reward, done, info = utl.env_step(test_env, action.squeeze(dim=0))
                    done_rollout = False if ptu.get_numpy(done[0][0]) == 0. else True

                    # add data to policy buffer - (s+, a, r, s'+, term)
                    term = test_env.unwrapped.is_goal_state() if "is_goal_state" in dir(test_env.unwrapped) else False
                    rew_to_buffer = ptu.get_numpy(reward.squeeze(dim=0))
                    
                    obs_c.append(ptu.get_numpy(obs.squeeze(dim=0)))
                    act_c.append(ptu.get_numpy(action.squeeze(dim=0)))
                    next_obs_c.append(ptu.get_numpy(next_obs.squeeze(dim=0)))
                    rew_c.append(rew_to_buffer)
                    term_c.append(np.array([term], dtype=float))

                    # set: obs <- next_obs
                    obs = next_obs.clone()

            obs_c = ptu.FloatTensor(np.stack(obs_c))
            act_c = ptu.FloatTensor(np.stack(act_c))
            rew_c = ptu.FloatTensor(np.stack(rew_c))
            next_obs_c = ptu.FloatTensor(np.stack(next_obs_c))
            term_c = ptu.FloatTensor(np.stack(term_c))
            #print(obs_c.shape, act_c.shape, rew_c.shape, next_obs_c.shape, term_c.shape)

            context_i = [obs_c, act_c, rew_c, next_obs_c, term_c]
            context.append(context_i)

        ret = [torch.stack([context[i][j] for i in range(len(tasks))], dim=0).transpose(0,1) for j in range(5)]
        #print(ret[0].shape)
        #sys.exit(0)
        return ret


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--env-type', default='gridworld')
    # parser.add_argument('--env-type', default='point_robot_sparse')
    # parser.add_argument('--env-type', default='cheetah_vel')
    parser.add_argument('--env-type', default='gridworld_block')
    args, rest_args = parser.parse_known_args()
    env = args.env_type

    # --- GridWorld ---
    if env == 'gridworld_block':
        args = args_gridworld_block.get_args(rest_args)
    elif env == 'cheetah_vel':
        args = args_cheetah_vel.get_args(rest_args)
    elif env == 'point_robot':
        args = args_point_robot.get_args(rest_args)
    elif env == 'ant_dir':
        args = args_ant_dir.get_args(rest_args)
    elif env == 'point_robot_v1':
        args = args_point_robot_v1.get_args(rest_args)
    elif env == 'hopper_param':
        args = args_hopper_param.get_args(rest_args)
    elif env == 'walker_param':
        args = args_walker_param.get_args(rest_args)
    else:
        raise NotImplementedError

    #behavior_policy_list = []
    with open('ood_test_config/{}.txt'.format(env), 'r') as f:
        behavior_policy_list = f.read().splitlines()

    #print(behavior_policy_list)

    set_gpu_mode(torch.cuda.is_available() and args.use_gpu)

    args, _ = off_utl.expand_args(args) # add env information to args
    #print(args)

    dataset, goals = off_utl.load_dataset(data_dir=args.data_dir, args=args, arr_type='numpy')
    assert args.num_train_tasks + args.num_eval_tasks == len(goals)
    train_dataset, train_goals = dataset[0:args.num_train_tasks], goals[0:args.num_train_tasks]
    eval_dataset, eval_goals = dataset[args.num_train_tasks:], goals[args.num_train_tasks:]

    # load context collection policy


    learner = OfflineContrastive(args, train_dataset, train_goals, eval_dataset, eval_goals)
    #learner.train()
    #learner.load_models(num_iter=900, load_dir="logs/HalfCheetahVel-v0/offpolicy_contrastive_generative_64__0__02_06_11_12_46/models")
    #learner.load_models(num_iter=900, load_dir="logs/HalfCheetahVel-v0/offpolicy_contrastive_no_generative__0__02_06_19_12_37/models")
    #learner.load_models(num_iter=900, load_dir="logs/HalfCheetahVel-v0/offline_supervised__0__01_06_09_59_17/models")
    #learner.load_models(num_iter=900, load_dir="logs/PointRobot-v0/offpolicy_contrastive_generative_lr0.0003__0__07_10_20_53_48/models")
    #learner.load_models(num_iter=900, load_dir="logs/PointRobot-v1/offpolicy_contrastive_randomize__0__08_11_14_53_54/models")
    #learner.load_models(num_iter=900, load_dir="logs/PointRobot-v1/offline_supervised__0__18_10_16_37_37/models")
    #learner.load_models(num_iter=900, load_dir="logs/AntDir-v0/offpolicy_contrastive_no_generative__0__30_12_16_25_36/models")
    #learner.load_models(num_iter=900, load_dir="logs/HopperRandParams-v0/offpolicy_contrastive_no_generative__0__30_12_20_57_41/models")
    #learner.load_models(num_iter=900, load_dir="logs/HopperRandParams-v0/offline_supervised__0__30_12_21_01_36/models")
    

    path_pre = 'logs/HalfCheetahVel-v0/'
    path_2_pre = 'offpolicy_contrastive_separate'
    load_iter = 900
    behavior_policy_hidden = 128
    IID_ret, OOD_ret = [], []

    for p in os.listdir(path_pre):
        if p.startswith(path_2_pre):
            load_dir = os.path.join(path_pre, p, 'models')
            learner.load_models(num_iter=load_iter, load_dir=load_dir)
            print('model loaded:',load_dir)

            returns, success_rate, log_probs, observations, rewards, reward_preds = learner.evaluate(ood=False)
            print("in-distribution, return: ", np.mean(returns))
            IID_ret.append(np.mean(returns))

            ood_ret = []
            for l in behavior_policy_list:
                learner.load_behavior_policy(path=l, hidden=behavior_policy_hidden) # a random agent trained on another task to collect context
                #print('context collection policy loaded')

                returns, success_rate, log_probs, observations, rewards, reward_preds = learner.evaluate(ood=True)
                #print("ood returns", returns)
                print('behavior policy: {}, return: {} std:{}'.format(l, np.mean(returns), np.std(returns)))
                ood_ret.append(np.mean(returns))
            print('mean ood return:', np.mean(ood_ret))
            OOD_ret.append(np.mean(ood_ret))
            print()
    print('IID: mean {}, std {}'.format(np.mean(IID_ret), np.std(IID_ret)))
    print('OOD: mean {}, std {}'.format(np.mean(OOD_ret), np.std(OOD_ret)))

if __name__ == '__main__':
    main()
