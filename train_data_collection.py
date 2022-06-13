
import os
import argparse
import torch
import numpy as np
from learner import Learner
from torchkit.pytorch_utils import set_gpu_mode
from data_collection_config import args_cheetah_vel, args_gridworld_block, \
args_ant_dir, args_point_robot_v1, args_hopper_param, args_walker_param


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--env-type', default='gridworld')
    # parser.add_argument('--env-type', default='point_robot_sparse')
    # parser.add_argument('--env-type', default='cheetah_vel')
    parser.add_argument('--env-type', default='ant_semicircle_sparse')
    args, rest_args = parser.parse_known_args()
    env = args.env_type

    # --- GridWorld ---
    if env == 'gridworld':
        args = args_gridworld.get_args(rest_args)
    elif env == 'gridworld_block':
        args = args_gridworld_block.get_args(rest_args)
    # --- PointRobot ---
    elif env == 'point_robot':
        args = args_point_robot.get_args(rest_args)
    elif env == 'point_robot_v1':
        args = args_point_robot_v1.get_args(rest_args)
    # --- Mujoco ---
    elif env == 'cheetah_vel':
        args = args_cheetah_vel.get_args(rest_args)
    elif env == 'ant_semicircle':
        args = args_ant_semicircle.get_args(rest_args)
    elif env == 'ant_dir':
        args = args_ant_dir.get_args(rest_args)
    elif env == 'hopper_param':
        args = args_hopper_param.get_args(rest_args)
    elif env == 'walker_param':
        args = args_walker_param.get_args(rest_args)
    else:
        raise NotImplementedError

    set_gpu_mode(torch.cuda.is_available())

    if hasattr(args, 'save_buffer') and args.save_buffer:
        os.makedirs(args.main_save_dir, exist_ok=True)

    learner = Learner(args)

    learner.train()


if __name__ == '__main__':
    main()

