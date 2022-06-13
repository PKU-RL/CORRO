
import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import helpers as utl
from torchkit import pytorch_utils as ptu
from torchkit.networks import Mlp, FlattenMlp

# conditional VAE: vae of (s',r) with encoder/decoder conditioned on (s,a)
class CVAE(nn.Module):
    def __init__(self,
                 # network size
                 hidden_size=64,
                 num_hidden_layers=1,
                 z_dim=5,
                 # actions, states, rewards
                 action_size=5,
                 state_size=2,
                 reward_size=1
                 ):
        super(CVAE, self).__init__()

        self.action_size=action_size
        self.state_size=state_size
        self.reward_size=reward_size
        self.z_dim = z_dim
        self.encoder = FlattenMlp(input_size=state_size*2+action_size+reward_size,
                                    output_size=z_dim*2,
                                    hidden_sizes=[hidden_size for i in range(num_hidden_layers)])
        self.decoder = FlattenMlp(input_size=z_dim+state_size+action_size,
                                    output_size=state_size+reward_size,
                                    hidden_sizes=[hidden_size for i in range(num_hidden_layers)])

    def compute_kl_divergence(self, mean, logvar):
        return (- 0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).sum(dim=1))

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def forward_encoder(self, obs, action, reward, next_obs):
        z = self.encoder(obs, action, reward, next_obs)
        mean, logvar = torch.split(z, self.z_dim, dim=1)
        z_sample = self.reparameterize(mean, logvar)
        return mean, logvar, z_sample

    def forward_decoder(self, obs, action, z=None):
        # sampling
        if z==None:
            z = torch.randn_like(ptu.zeros((obs.shape[0], self.z_dim), requires_grad=False))
        out = self.decoder(obs, action, z)
        s_, r = out[:,0:self.state_size], out[:, self.state_size:]
        return s_, r


# single task model
class Predictor(nn.Module):
    def __init__(self,
                 # network size
                 hidden_size=64,
                 num_hidden_layers=2,
                 #z_dim=5,
                 # actions, states, rewards
                 action_size=5,
                 state_size=2,
                 reward_size=1
                 ):
        super(Predictor, self).__init__()

        self.action_size=action_size
        self.state_size=state_size
        self.reward_size=reward_size
        #self.z_dim = z_dim
        self.mlp = FlattenMlp(input_size=state_size+action_size,
                                    output_size=state_size+reward_size,
                                    hidden_sizes=[hidden_size for i in range(num_hidden_layers)])

    def forward(self, obs, action):
        out = self.mlp(obs, action)
        s_, r = out[:,0:self.state_size], out[:, self.state_size:]
        return s_, r