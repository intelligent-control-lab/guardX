import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

EPS = 1e-8

def alpha_kl(mu0, log_std0, mu1, log_std1, alpha):
    """
    torch symbol for mean KL divergence between two batches of diagonal gaussian distributions,
    where distributions are specified by means and log stds.
    (https://en.wikipedia.org/wiki/Kullback-Leibler_divergence#Multivariate_normal_distributions)
    """
    std0, std1 = torch.exp(log_std0), torch.exp(log_std1)
    factor = (std0**(1-alpha))*(std1**alpha) / torch.sqrt(alpha*(std1**2)+(1-alpha)*(std0**2))
    exp = alpha*(alpha-1)*(mu0-mu1)**2 / (2.0*(alpha*(std1**2)+(1-alpha)*(std0**2)))
    pre_sum = (factor*torch.exp(exp)-1) / (alpha*(alpha-1))
    all_kls = torch.sum(pre_sum, axis=1)
    return torch.mean(all_kls)

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def batch_discount_cumsum(x, discount):
    """the batch discounted cumulative sums of vectors, using magic from rllab

    Args:
        x (torch.tensor): vector x, shape = (B,length)
        discount (float): the discount factor

    Returns:
        torch.tensor: the batch discounted cumulative sums of vectors, shape = (B,length)
    """
    return np.asarray([discount_cumsum(x_row, discount) for x_row in x.cpu().numpy()])

class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)
    
    def _d_kl(self, obs, old_mu, old_log_std, device):
        raise NotImplementedError


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution
    
    def _d_kl(self, obs, old_mu, old_log_std, device, alpha):
        # kl divergence computation 
        mu = self.mu_net(obs.to(device))
        log_std = self.log_std 
        
        d_kl = alpha_kl(old_mu.to(device), old_log_std.to(device), mu, log_std, alpha) # debug test to see if P old in the front helps
        return d_kl


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.



class MLPActorCritic(nn.Module):


    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation).to(self.device)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation).to(self.device)

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation).to(self.device)

    def step(self, obs):
        with torch.no_grad():
            obs = obs.to(self.device)
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a, v, logp_a, pi.mean, torch.log(pi.stddev)

    def act(self, obs):
        return self.step(obs)[0]