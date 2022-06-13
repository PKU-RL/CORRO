# ablation: train contrastive transition representation without generative model

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torchkit.pytorch_utils import set_gpu_mode
import utils.config_utils as config_utl
from utils import helpers as utl, offline_utils as off_utl
from offline_rl_config import args_gridworld_block, args_cheetah_vel, args_ant_dir, args_hopper_param, args_walker_param, args_point_robot_v1
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
        '''
        if args.encoder_type == 'rnn':
            self.encoder = RNNEncoder(
                layers_before_gru=self.args.layers_before_aggregator,
                hidden_size=self.args.aggregator_hidden_size,
                layers_after_gru=self.args.layers_after_aggregator,
                task_embedding_size=self.args.task_embedding_size,
                action_size=self.args.act_space.n, # fixed a bug?
                action_embed_size=self.args.action_embedding_size,
                state_size=self.args.obs_dim,
                state_embed_size=self.args.state_embedding_size,
                reward_size=1,
                reward_embed_size=self.args.reward_embedding_size,
            ).to(ptu.device)
        elif args.encoder_type == 'mlp':
        '''
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


        #else:
        #    raise NotImplementedError
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.args.encoder_lr)

        # context encoder: convert (batch, N, dim) to (batch, dim)
        self.context_encoder = SelfAttnEncoder(input_dim=self.args.task_embedding_size,
            num_output_mlp=self.args.context_encoder_output_layers).to(ptu.device)
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

        self._preprocess_positive_samples()

        #print(self.evaluate())
        #self.vis_sample_embeddings('test.png')
        #sys.exit(0)

    # process the training dataset for fast positives sampling
    def _preprocess_positive_samples(self):
        ds = [np.concatenate(itm, axis=2) for itm in self.context_dataset]
        ds = np.stack(ds) # (n_task, ts, episode, 11)
        shape = ds.shape
        ds = ds.reshape(shape[0], -1, shape[-1]) # (n_task, n_sample, dim)
        #print(ds.shape)
        self.train_samples_dataset = ds

    # random sample positive samples (query, key) with size (batchsize, [s,a,r,s',t])
    # return [s,a,r,s',t], [s,a,r,s',t]
    # a very fast version!
    def sample_positive_pairs(self, batch_size, trainset=True):
        tasks_index = np.random.randint(0, self.train_samples_dataset.shape[0], size=(batch_size))
        query_index = np.random.randint(0, self.train_samples_dataset.shape[1], size=(batch_size))
        key_index = np.random.randint(0, self.train_samples_dataset.shape[1], size=(batch_size))
        
        query = self.train_samples_dataset[tasks_index, query_index]
        key = self.train_samples_dataset[tasks_index, key_index] # (batchsize, 11)
        
        sizes = [self.args.obs_dim, self.args.act_space.n if self.args.act_space.__class__.__name__ == "Discrete" else self.args.action_dim, 
            1, self.args.obs_dim, 1]

        query = ptu.FloatTensor(query)
        key = ptu.FloatTensor(key)
        query = torch.split(query, sizes, dim=1)
        key = torch.split(key, sizes, dim=1)
        assert len(query)==5 and len(key)==5

        return query, key

    # random sample (r, s'|s,a) over task distribution for negative samples
    # state, action (batchsize, dim)
    # return rewards, next_obs (batchsize, num_negatives, dim)
    def create_negatives(self, state, action, num_negatives):
        batch_size = state.shape[0]
        tasks_index = np.random.randint(0, self.train_samples_dataset.shape[0], size=(batch_size*num_negatives))
        ts_index = np.random.randint(0, self.train_samples_dataset.shape[1], size=(batch_size*num_negatives))
        sizes = [self.args.obs_dim, self.args.act_space.n if self.args.act_space.__class__.__name__ == "Discrete" else self.args.action_dim, 
            1, self.args.obs_dim, 1]
        samples = self.train_samples_dataset[tasks_index, ts_index]
        samples = ptu.FloatTensor(samples)
        samples = torch.split(samples, sizes, dim=1)
        assert len(samples)==5
        return samples

    # InfoNCE
    # q, k (b, dim); neg (b, N, dim)
    def contrastive_loss(self, q, k, neg):
        N = neg.shape[1]
        b = q.shape[0]
        l_pos = torch.bmm(q.view(b, 1, -1), k.view(b, -1, 1)) # (b,1,1)
        l_neg = torch.bmm(q.view(b, 1, -1), neg.transpose(1,2)) # (b,1,N)
        logits = torch.cat([l_pos.view(b, 1), l_neg.view(b, N)], dim=1)
        
        labels = torch.zeros(b, dtype=torch.long)
        labels = labels.to(ptu.device)
        cross_entropy_loss = nn.CrossEntropyLoss()
        loss = cross_entropy_loss(logits/self.args.infonce_temp, labels)
        #print(logits, labels, loss)
        return loss

    def update(self, tasks):
        rl_losses_agg = {}
        if self.args.log_train_time:
	        time_cost = {'data_sampling':0, 'negatives_sampling':0, 'update_encoder':0, 'update_rl':0}

        for update in range(self.args.rl_updates_per_iter):
            if self.args.log_train_time:
                _t_cost = time.time()
            #print('data sampling')
            
            # sample key, query, negative samples and train encoder
            # (batchsize, dim)
            queries, keys = self.sample_positive_pairs(self.args.contrastive_batch_size)
            obs_q, actions_q, rewards_q, next_obs_q, terms_q = queries
            obs_k, actions_k, rewards_k, next_obs_k, terms_k = keys

            if self.args.log_train_time:
                _t_now = time.time()
                time_cost['data_sampling'] += (_t_now-_t_cost)
                _t_cost = _t_now

            # (batchsize, N, dim)
            obs_neg, actions_neg, rewards_neg, next_obs_neg, terms_neg = self.create_negatives(obs_q, actions_q, self.args.n_negative_per_positive)
            
            if self.args.log_train_time:
                _t_now = time.time()
                time_cost['negatives_sampling'] += (_t_now-_t_cost)
                _t_cost = _t_now
            
            b_dot_N = self.args.contrastive_batch_size * self.args.n_negative_per_positive
            q_z = self.encoder.forward(obs_q, actions_q, rewards_q, next_obs_q)
            k_z = self.encoder.forward(obs_k, actions_k, rewards_k, next_obs_k)
            neg_z = self.encoder.forward(obs_neg.reshape(b_dot_N, -1), actions_neg.reshape(b_dot_N, -1), 
                rewards_neg.reshape(b_dot_N, -1), next_obs_neg.reshape(b_dot_N, -1)).view(
                self.args.contrastive_batch_size, self.args.n_negative_per_positive, -1)
            contrastive_loss = self.contrastive_loss(q_z, k_z, neg_z)


            self.encoder_optimizer.zero_grad()
            contrastive_loss.backward()
            self.encoder_optimizer.step()

            if self.args.log_train_time:
                _t_now = time.time()
                time_cost['update_encoder'] += (_t_now-_t_cost)
                _t_cost = _t_now

            
            rl_losses = {'contrastive_loss':contrastive_loss.item()}
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


    def log(self, iteration, train_stats):
        #super().log(iteration, train_stats)
        if self.args.save_model and (iteration % self.args.save_interval == 0):
            save_path = os.path.join(self.tb_logger.full_output_folder, 'models')
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            torch.save(self.encoder.state_dict(), os.path.join(save_path, "encoder{0}.pt".format(iteration)))

        if iteration % self.args.log_interval == 0:
            if self.args.log_tensorboard:
                for k in train_stats.keys():
                    self.tb_logger.writer.add_scalar('rl_losses/'+k, train_stats[k], 
                        self._n_rl_update_steps_total)
            print("Iteration -- {}, Elapsed time {:5d}[s]"
                .format(iteration, int(time.time() - self._start_time)), train_stats)

        # visualize embeddings
        if self.args.log_tensorboard and (iteration % self.args.log_vis_interval == 0):
            save_path = os.path.join(self.tb_logger.full_output_folder, 'vis_z')
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            self.vis_sample_embeddings(os.path.join(save_path, "train_fig{0}.png".format(iteration)), trainset=True)
            self.vis_sample_embeddings(os.path.join(save_path, "test_fig{0}.png".format(iteration)), trainset=False)


    # visualize the encodings of (s,a,r,s')
    # distinguish different tasks' critical samples and unimportant samples with different colors
    # use tsne
    def vis_sample_embeddings(self, save_path, trainset=True):
        self.training_mode(False)
        goals = self.goals if trainset else self.eval_goals
        x, y = [], []
        obs_context, actions_context, rewards_context, next_obs_context, _ = self.sample_context_batch(
            tasks=[i for i in range(len(goals))], trainset=trainset)
        #print(obs_context.shape)
        n_timesteps, n_tasks, _ = obs_context.shape
        encodings = self.encoder(
                obs_context.reshape(n_timesteps*n_tasks, -1),
                actions_context.reshape(n_timesteps*n_tasks, -1),
                rewards_context.reshape(n_timesteps*n_tasks, -1),
                next_obs_context.reshape(n_timesteps*n_tasks, -1)
            )
        encodings = encodings.reshape(n_timesteps, n_tasks, -1).cpu().detach().numpy()
        obs_context, actions_context, rewards_context, next_obs_context = \
            obs_context.cpu().detach().numpy(), actions_context.cpu().detach().numpy(), \
            rewards_context.cpu().detach().numpy(), next_obs_context.cpu().detach().numpy()

        if self.args.env_name == 'GridBlock-v2':
            test_env = make_env(self.args.env_name,
                self.args.max_rollouts_per_task,
                seed=self.args.seed,
                n_tasks=1)
            for i, t in enumerate(goals):
                test_env.set_goal(t.astype(np.int))
                #print("task", t)
                for j in range(obs_context.shape[0]):
                    is_critical = test_env.is_sample_contain_task(
                            obs_context[j,i],
                            np.argmax(actions_context[j,i]),
                            rewards_context[j,i,0],
                            next_obs_context[j,i],
                        )
                    x.append(encodings[j,i])
                    if is_critical:
                        y.append(i+1) # task-specific: 1~20
                    else:
                        y.append(0) # task-irrelevant: 0
        else:
            for i, t in enumerate(goals):
                for j in range(obs_context.shape[0]):
                    x.append(encodings[j,i])
                    y.append(i)
        
        
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        X_tsne = tsne.fit_transform(np.asarray(x))

        x_min, x_max = np.min(X_tsne, 0), np.max(X_tsne, 0)
        data = (X_tsne - x_min) / (x_max - x_min)

        if self.args.env_name == 'GridBlock-v2':
            colors = plt.cm.rainbow(np.linspace(0,1,len(goals)+1))
        else:
            colors = plt.cm.rainbow(np.linspace(0,1,len(goals)))
        #print(colors)

        plt.cla()
        fig = plt.figure()
        ax = plt.subplot(111)
        for i in range(data.shape[0]):
            plt.text(data[i, 0], data[i, 1], str(y[i]),
                    color=colors[y[i]], #plt.cm.Set1(y[i] / 21),
                    fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        plt.savefig(save_path)



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
    elif env == 'ant_dir':
        args = args_ant_dir.get_args(rest_args)
    elif env == 'hopper_param':
        args = args_hopper_param.get_args(rest_args)
    elif env == 'walker_param':
        args = args_walker_param.get_args(rest_args)
    elif env == 'point_robot_v1':
        args = args_point_robot_v1.get_args(rest_args)
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
