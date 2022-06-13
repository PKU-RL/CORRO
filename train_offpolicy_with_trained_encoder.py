# train offpolicy rl with context-aggregator, after the pretraining of contrastive task encoder

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torchkit.pytorch_utils import set_gpu_mode
import utils.config_utils as config_utl
from utils import helpers as utl, offline_utils as off_utl
from offline_rl_config import args_gridworld_block, args_cheetah_vel, args_ant_dir, \
args_point_robot_v1, args_hopper_param, args_walker_param
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
        if self.args.log_tensorboard:
            self.tb_logger = TBLogger(self.args)

        self.args, _ = off_utl.expand_args(self.args, include_act_space=True)
        if self.args.act_space.__class__.__name__ == "Discrete":
            self.args.policy = 'dqn'
        else:
            self.args.policy = 'sac'

        # load augmented buffer to self.storage
        self.load_buffer(train_dataset, train_goals)
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
        self.encoder.load_state_dict(torch.load(self.args.encoder_model_path, map_location=ptu.device))

        # context encoder: convert (batch, N, dim) to (batch, dim)
        self.context_encoder = SelfAttnEncoder(input_dim=self.args.task_embedding_size,
            num_output_mlp=self.args.context_encoder_output_layers, task_gt_dim=self.goals[0].shape[0],
            ).to(ptu.device)
        self.context_encoder_optimizer = torch.optim.Adam(self.context_encoder.parameters(), lr=self.args.encoder_lr)


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

        #if self.args.env_name == 'GridNavi-v2' or self.args.env_name == 'GridBlock-v2':
        #    self.env.unwrapped.goals = [tuple(goal.astype(int)) for goal in self.goals]

        '''
        if self.args.relabel_type == 'gt':
            # create an env for reward/transition relabelling
            self.relabel_env = make_env(args.env_name,
                            args.max_rollouts_per_task,
                            seed=args.seed,
                            n_tasks=1)
        elif self.args.relabel_type == 'generative':
            self.generative_model = CVAE(
            	hidden_size=args.cvae_hidden_size,
                num_hidden_layers=args.cvae_num_hidden_layers,
            	z_dim=self.args.task_embedding_size,
                action_size=self.args.act_space.n if self.args.act_space.__class__.__name__ == "Discrete" else self.args.action_dim,
                state_size=self.args.obs_dim,
                reward_size=1).to(ptu.device)
            self.generative_model.load_state_dict(torch.load(self.args.generative_model_path, 
                map_location=ptu.device))
            self.generative_model.train(False)
            print('generative model loaded from {}'.format(self.args.generative_model_path))
        else: 
            raise NotImplementedError
        '''

        #self._preprocess_positive_samples()

        #print(self.evaluate())
        #self.vis_sample_embeddings('test.png')
        #sys.exit(0)

    def update(self, tasks):
        rl_losses_agg = {}
        if self.args.log_train_time:
	        time_cost = {'data_sampling':0, 'negatives_sampling':0, 'update_encoder':0, 'update_rl':0}

        for update in range(self.args.rl_updates_per_iter):
            if self.args.log_train_time:
                _t_cost = time.time()
            
            # sample rl batch, context batch and update agent
            # sample random RL batch
            obs, actions, rewards, next_obs, terms = self.sample_rl_batch(tasks, self.args.rl_batch_size) # [task, batch, dim]
            # sample corresponding context batch
            obs_context, actions_context, rewards_context, next_obs_context, terms_context = self.sample_context_batch(tasks) # [ts'=ts*num_context_traj, task, dim]

            n_timesteps, batch_size, _ = obs_context.shape
            with torch.no_grad():
                encodings = self.encoder(
                        obs=obs_context.reshape(n_timesteps*batch_size, -1), 
                        action=actions_context.reshape(n_timesteps*batch_size, -1), 
                        reward=rewards_context.reshape(n_timesteps*batch_size, -1), 
                        next_obs=next_obs_context.reshape(n_timesteps*batch_size, -1),
                    ).view(n_timesteps, batch_size, -1).transpose(0,1)
            

            # additional task loss for debug
            if self.args.use_additional_task_info:
                encoding, task_pred = self.context_encoder.forward_full(encodings)
                tasks_gt = self.goals[tasks]
                tasks_gt = ptu.FloatTensor(tasks_gt)
                task_pred_loss = nn.MSELoss()(task_pred, tasks_gt)
                self.context_encoder_optimizer.zero_grad()
                task_pred_loss.backward()
                self.context_encoder_optimizer.step()
                task_encoding = encoding.detach().unsqueeze(1)
            else:
                encoding = self.context_encoder(encodings)
                task_encoding = encoding.unsqueeze(1)
                self.context_encoder_optimizer.zero_grad()

            t, _, d = task_encoding.size()
            task_encoding = task_encoding.expand(t, self.args.rl_batch_size, d) # [task, batch(repeat), dim]

            obs = torch.cat((obs, task_encoding), dim=-1)
            next_obs = torch.cat((next_obs, task_encoding), dim=-1) # [task, batch, obs_dim+z_dim]

            # flatten out task dimension
            t, b, _ = obs.size()
            obs = obs.view(t * b, -1)
            actions = actions.view(t * b, -1)
            rewards = rewards.view(t * b, -1)
            next_obs = next_obs.view(t * b, -1)
            terms = terms.view(t * b, -1)
            #print('forward: q learning')
            # RL update (Q learning)
            #rl_losses = self.agent.update(obs, actions, rewards, next_obs, terms, action_space=self.env.action_space)
            if self.args.policy == 'dqn':
                rl_losses = self.agent.update(obs, actions, rewards, next_obs, terms)
                if not self.args.use_additional_task_info:
                    self.context_encoder_optimizer.step()
            elif self.args.policy == 'sac':
                rl_losses = self.agent.update_critic(obs, actions, rewards, next_obs, terms, action_space=self.env.action_space)
                if not self.args.use_additional_task_info:
                    self.context_encoder_optimizer.step()
                obs = obs.detach()
                next_obs = next_obs.detach()
                actor_losses = self.agent.update_actor(obs, actions, rewards, next_obs, terms, action_space=self.env.action_space)
                rl_losses.update(actor_losses)
            else:
                raise NotImplementedError

            '''
            if self.args.log_train_time:
                _t_now = time.time()
                time_cost['update_rl'] += (_t_now-_t_cost)
                _t_cost = _t_now
            '''
            if self.args.use_additional_task_info:
                rl_losses['task_pred_loss'] = task_pred_loss.item()


            for k, v in rl_losses.items():
                if update == 0:  # first iterate - create list
                    rl_losses_agg[k] = [v]
                else:  # append values
                    rl_losses_agg[k].append(v)
        # take mean
        for k in rl_losses_agg:
            rl_losses_agg[k] = np.mean(rl_losses_agg[k])
        self._n_rl_update_steps_total += self.args.rl_updates_per_iter

        if self.args.log_train_time:
            print(time_cost)

        return rl_losses_agg


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


    set_gpu_mode(torch.cuda.is_available() and args.use_gpu)

    args, _ = off_utl.expand_args(args) # add env information to args
    #print(args)

    dataset, goals = off_utl.load_dataset(data_dir=args.data_dir, args=args, arr_type='numpy')
    assert args.num_train_tasks + args.num_eval_tasks == len(goals)
    train_dataset, train_goals = dataset[0:args.num_train_tasks], goals[0:args.num_train_tasks]
    eval_dataset, eval_goals = dataset[args.num_train_tasks:], goals[args.num_train_tasks:]

    learner = OfflineContrastive(args, train_dataset, train_goals, eval_dataset, eval_goals)
    learner.train()


if __name__ == '__main__':
    main()
