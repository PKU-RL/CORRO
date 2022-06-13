import argparse
import torch
from utils.cli import boolean_argument


def get_args(rest_args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default='GridBlock-v2')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-rollouts-per-task', default=1) # should be 1, not BAMDP
    parser.add_argument('--num-trajs-per-task', type=int, default=None) # None: use all the saved trajs per task
    #parser.add_argument('--hindsight-relabelling', type=int, default=True)
    parser.add_argument('--num-train-tasks', default=20) 
    parser.add_argument('--num-eval-tasks', default=20) # split the saved tasks buffers into training and testing tasks

    # global training configs
    parser.add_argument('--meta-batch', type=int, default=16,
                        help='number of tasks to average the gradient across')
    parser.add_argument('--num-iters', default=1000)
    #parser.add_argument('--tasks-batch-size', default=8) # moved to --meta-batch
    
    # RL configs
    parser.add_argument('--rl-updates-per-iter', type=int, default=250, help='number of RL steps per iteration')
    parser.add_argument('--rl-batch-size', type=int, default=256, help='number of transitions in RL batch (per task)')
    parser.add_argument('--dqn-layers', nargs='+', default=[64, 64])
    parser.add_argument('--policy-lr', type=float, default=0.0003, help='learning rate for policy (default: 3e-4)')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--soft-target-tau', type=float, default=0.005,
                        help='soft target network update (default: 5e-3)')
    parser.add_argument('--eval-deterministic', default=True)


    # general encoder configs
    parser.add_argument('--num-context-trajs', type=int, default=4, help='number of trajs provided \
        for task encoding. context-batch-size=4*traj_len=80')
    parser.add_argument('--encoder-lr', type=float, default=0.0003, help='learning rate for encoder (default: 3e-4)')
    #parser.add_argument('--kl-weight', type=float, default=.05, help='weight for the KL term')

    # offline Pearl encoder configs
    parser.add_argument('--pearl-deterministic-encoder', type=int, default=True, help='if true, use deterministic\
        encoder; otherwise, use gaussian encoder, regularize with KL loss')
    parser.add_argument('--encoder-type', type=str, default='rnn', help='choose: rnn, mlp')
    parser.add_argument('--task-embedding-size', type=int, default=5, help='dimensionality of latent space')
    parser.add_argument('--aggregator-hidden-size', type=int, default=64, help='for both rnn and mlp')
    parser.add_argument('--layers-before-aggregator', nargs='+', type=int, default=[])
    parser.add_argument('--layers-after-aggregator', nargs='+', type=int, default=[])
    parser.add_argument('--action-embedding-size', type=int, default=5)
    parser.add_argument('--state-embedding-size', type=int, default=5)
    parser.add_argument('--reward-embedding-size', type=int, default=5)

    # relabelling configs
    # gt: relabel with gt reward/transition. separate: learn a model for each task. generative: learn a generative model
    parser.add_argument('--relabel-type', type=str, default='generative', help='choose: gt, separate, generative')
    parser.add_argument('--cvae-hidden-size', type=int, default=64)
    parser.add_argument('--cvae-num-hidden-layers', type=int, default=1)
    parser.add_argument('--generative-model-path', type=str, default=
        'logs/GridBlock-v2/generative__0__14_05_16_50_14/models/model100.pt')
    parser.add_argument('--aggregate-encoder-type', type=str, default='selfattn', help='choose: selfattn, mean')

    parser.add_argument('--contrastive-batch-size', type=int, default=64)
    parser.add_argument('--n-negative-per-positive', type=int, default=16)
    parser.add_argument('--normalize-z', type=int, default=True, help='encoding normalization')
    parser.add_argument('--infonce-temp', type=float, default=0.1, help='temperature param')
    


    # debug configs
    parser.add_argument('--use-additional-task-info', type=int, default=False, help='')
    #parser.add_argument('--use-sample-encoder', type=int, default=True, help='')
    parser.add_argument('--context-encoder-output-layers', type=int, default=0)

    '''
    # task encoder configs: for prediction-based method
    parser.add_argument('--vae-batch-num-rollouts-per-task', default=8, help='')
    parser.add_argument('--vae-lr', type=float, default=0.0003, help='learning rate for VAE (default: 3e-4)')
    parser.add_argument('--kl-weight', type=float, default=.05, help='weight for the KL term')
    parser.add_argument('--vae-batch-num-elbo-terms', default=None,
                        help='for how many timesteps to compute the ELBO; None uses all')

    # - decoder: rewards
    parser.add_argument('--decode-reward', default=True, help='use reward decoder')
    parser.add_argument('--input-prev-state', default=False, help='use prev state for rew pred')
    parser.add_argument('--input-action', default=False, help='use prev action for rew pred')
    parser.add_argument('--reward-decoder-layers', nargs='+', type=int, default=[32, 32])
    parser.add_argument('--rew-pred-type', type=str, default='deterministic', help='choose from: gaussian, deterministic, bernoulli')
    parser.add_argument('--multihead-for-reward', default=False, help='one head per reward pred (i.e. per state)')
    parser.add_argument('--rew-loss-coeff', type=float, default=1.0)
    # - decoder: state transitions
    parser.add_argument('--decode-state', type=int, default=True)
    parser.add_argument('--state-loss-coeff', type=float, default=1.0)
    parser.add_argument('--state-decoder-layers', nargs='+', type=int, default=[32, 32])
    parser.add_argument('--state-pred-type', type=str, default='deterministic', help='choose from: gaussian, deterministic')
    '''

    # logging configs and others
    parser.add_argument('--log-interval', type=int, default=10,
                        help='log interval, one log per n iterations')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='save models interval, every # iterations')
    #parser.add_argument('--eval-interval', default=20)

    parser.add_argument('--main-data-dir', default='./batch_data')
    parser.add_argument('--data-dir', default='data')
    #parser.add_argument('--save-dir-prefix', default='relabel')
    parser.add_argument('--results-log-dir', default=None, help='directory to save agent logs (default: ./logs)')
    parser.add_argument('--output-file-prefix', default='offline')
    parser.add_argument('--log-tensorboard', type=int, default=True)
    parser.add_argument('--save-model', type=int, default=True)
    #parser.add_argument('--save-dir', default='./trained_vae')
    parser.add_argument('--log-train-time', type=int, default=False, help='log training time cost')
    parser.add_argument('--use-gpu', default=True)

    args = parser.parse_args(rest_args)

    return args
