# generative model of transition and reward
# modeling P(s',r|s,a)

import os
import sys
import time
import argparse
import torch
import torch.nn.functional as F
from torchkit.pytorch_utils import set_gpu_mode
import utils.config_utils as config_utl
from utils import helpers as utl, offline_utils as off_utl
from relabel_model_config import args_gridworld_block, args_cheetah_vel, args_ant_dir, args_point_robot_v1, args_hopper_param, args_walker_param
import numpy as np

#from models.encoder import RNNEncoder, MLPEncoder
#from algorithms.dqn import DQN
#from algorithms.sac import SAC
from environments.make_env import make_env
from torchkit import pytorch_utils as ptu
from torchkit.networks import FlattenMlp
#from data_management.storage_policy import MultiTaskPolicyStorage
from utils import evaluation as utl_eval
from utils.tb_logger import TBLogger
#from models.policy import TanhGaussianPolicy
#from offline_learner import OfflineMetaLearner
from models.generative import CVAE
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, dataset, goals):
        self.dataset = dataset
        self.goals = goals
        self.size = len(goals) * dataset[0][0].shape[0] * dataset[0][0].shape[1]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        i_task = index % len(self.goals)
        t = index // len(self.goals)
        i_timestep = t % self.dataset[0][0].shape[0]
        i_episode = t // self.dataset[0][0].shape[0]
        obs, action, reward, next_obs = self.dataset[i_task][0][i_timestep, i_episode], self.dataset[i_task][1][i_timestep, i_episode],\
            self.dataset[i_task][2][i_timestep, i_episode], self.dataset[i_task][3][i_timestep, i_episode]
        #print(i_task, i_episode, i_timestep)
        return torch.from_numpy(obs), torch.from_numpy(action), torch.from_numpy(reward), torch.from_numpy(next_obs)


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

    utl.seed(args.seed)

    # initialize tensorboard logger
    if args.log_tensorboard:
        tb_logger = TBLogger(args)

    args, env = off_utl.expand_args(args) # add env information to args
    #print(args)
    print('loading dataset')
    dataset, goals = off_utl.load_dataset(data_dir=args.data_dir, args=args, arr_type='numpy')
    dataset = MyDataset(dataset, goals)
    print('dataset loaded')
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    model = CVAE(hidden_size=args.hidden_size,
                 num_hidden_layers=args.num_hidden_layers,
                 z_dim=args.z_dim,
                 action_size=env.action_space.n if env.action_space.__class__.__name__ == "Discrete" else args.action_dim,
                 state_size=args.obs_dim,
                 reward_size=1).to(ptu.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    model.train()

    iters_total = 0
    for epoch in range(args.n_epochs):
        n_iter = 0
        for obs, action, reward, next_obs in dataloader:
            obs = obs.to(ptu.device)
            action = action.to(ptu.device)
            reward = reward.to(ptu.device)
            next_obs = next_obs.to(ptu.device)
            mean, logvar, z_sample = model.forward_encoder(obs, action, reward, next_obs)
            next_obs_pred, reward_pred = model.forward_decoder(obs, action, z=z_sample)

            kl_loss = model.compute_kl_divergence(mean, logvar).mean()
            obs_recon_loss = F.mse_loss(next_obs_pred, next_obs)
            reward_recon_loss = F.mse_loss(reward_pred, reward)
            #print(kl_loss, obs_recon_loss, reward_recon_loss)
            total_loss = args.beta * kl_loss + obs_recon_loss + reward_recon_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            n_iter+=1
            iters_total += 1
            if (n_iter % args.log_interval) == 0:
                print('epoch {}, step {}/{}, kl loss: {}, state recon loss: {}, reward recon loss: {}, total loss: {}'.format(
                        epoch, n_iter, len(dataset) // args.batch_size, kl_loss, obs_recon_loss, reward_recon_loss,
                        total_loss))
                tb_logger.writer.add_scalar('kl_loss', kl_loss.item(), iters_total)
                tb_logger.writer.add_scalar('state_recon_loss', obs_recon_loss.item(), iters_total)
                tb_logger.writer.add_scalar('reward_recon_loss', reward_recon_loss.item(), iters_total)
                tb_logger.writer.add_scalar('total_loss', total_loss.item(), iters_total)


        if (epoch+1) % args.save_interval == 0:
            save_path = os.path.join(tb_logger.full_output_folder, 'models')
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            torch.save(model.state_dict(), os.path.join(save_path, "model{0}.pt".format(epoch+1)))



if __name__ == '__main__':
    main()
