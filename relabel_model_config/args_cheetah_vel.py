import argparse
import torch
from utils.cli import boolean_argument


def get_args(rest_args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default='HalfCheetahVel-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-rollouts-per-task', default=1) # should be 1, not BAMDP
    parser.add_argument('--num-trajs-per-task', type=int, default=None) # None: use all the saved trajs per task
  
    parser.add_argument('--n-epochs', type=int, default=100, help='')
    parser.add_argument('--batch-size', type=int, default=128, help='')
    parser.add_argument('--learning-rate', type=float, default=0.0003)
    parser.add_argument('--model-type', type=str, default='CVAE', help='')
    parser.add_argument('--beta', type=float, default=0.2, help='KL divergence weight for CVAE')
    parser.add_argument('--z-dim', type=int, default=5, help='dimensionality of latent space')
    parser.add_argument('--hidden-size', type=int, default=64, help='')
    parser.add_argument('--num-hidden-layers', type=int, default=1, help='')


    # logging configs and others
    parser.add_argument('--log-interval', type=int, default=200, help='log every # iteration')
    parser.add_argument('--save-interval', type=int, default=1, help='save models interval, every # epoch')

    parser.add_argument('--main-data-dir', default='./batch_data')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--results-log-dir', default=None, help='directory to save agent logs (default: ./logs)')
    parser.add_argument('--output-file-prefix', default='generative')
    parser.add_argument('--log-tensorboard', type=int, default=True)
    parser.add_argument('--save-model', type=int, default=True)
    parser.add_argument('--use-gpu', default=True)
    parser.add_argument('--num-workers', type=int, default=4, help='')

    args = parser.parse_args(rest_args)

    return args
