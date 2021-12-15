"""This modules creates a discrete policy network.
A neural network can be used as policy method in different RL algorithms.
It accepts an observation of the environment and predicts an action.
"""
import akro
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from garage.torch.modules import MLPModule
from conditionalMAIL.DiscreteMLPPolicy import DiscreteMLPPolicy
from conditionalMAIL.models import FTNet, ModularNet

class AdaptivePolicy(DiscreteMLPPolicy):
    def __init__(self, env_spec, num_tasks, name, continuous=False, var=1.0, **kwargs):
        super().__init__(env_spec, name, **kwargs)
        self.S, self.A, self.P = env_spec.observation_space.flat_dim, env_spec.action_space.flat_dim, num_tasks
        self.cur_p = None
        self.use_zero_partner_vec = False

        self.continuous = continuous
        self.var = var

        self.net = None

    def process_observations(self, observations):
        # turn each observation from S into [P, S]
        # input: (batch, |S|)
        # output: (batch, |P+S|)

        batch = observations.size()[:-1]
        prod_batch = np.prod(batch)
        observations = observations.reshape(prod_batch, self.S)

        P_pad = torch.zeros( prod_batch, self.P )
        if not self.use_zero_partner_vec:
            P_pad[:, self.cur_p] = 1
        obs = torch.cat( (P_pad, observations), dim=-1) # [batch, P+S]
        ret = obs.reshape( *batch, self.P+self.S)

        return ret

    def set_task(self, task_idx):
        print("task: ", task_idx, flush=True)
        self.cur_p = task_idx

    def setup_for_testing(self, train_p_only=False):
        self.use_zero_partner_vec = True

    # pylint: disable=arguments-differ
    def forward(self, observations):
        batch = observations.size()[:-1]
        pad_obs = self.process_observations(observations)

        if self.continuous:
            mu = self.net(pad_obs)
            return Normal(loc=mu, scale=mu*0+self.var), dict()

        logits = self.net(pad_obs)
        return Categorical(logits=logits), dict()


class MTPolicy(AdaptivePolicy):
    def __init__(self, env_spec, num_tasks, name='MTPolicy', continuous=False, var=1.0, **kwargs):
        super().__init__(env_spec, num_tasks, name, continuous, var, **kwargs)
        self.net = MLPModule(input_dim=self.P + self.S, output_dim=self.A, **kwargs)


class LatentPolicy(AdaptivePolicy):
    def __init__(self, env_spec, num_tasks, name='LatentPolicy', continuous=False, var=1.0, **kwargs):
        super().__init__(env_spec, num_tasks, name, continuous, var, **kwargs)
        self.Pnet = MLPModule(input_dim=self.P, output_dim=self.P, **kwargs)
        self.net = MLPModule(input_dim=self.P + self.S, output_dim=self.A, **kwargs)

    def process_observations(self, observations):
        ret = super(LatentPolicy, self).process_observations(observations)
        retP = ret[:, :self.P]
        retS = ret[:, self.P:]
        retP = F.softmax( self.Pnet(retP), dim=-1 )
        ret = torch.cat( (retP, retS), dim=-1 )
        return ret

    def setup_for_testing(self, train_p_only=False):
        if train_p_only:
            for s in self.net.parameters():
                s.requires_grad = False

        self.use_zero_partner_vec = True


class TTPolicy(AdaptivePolicy):
    def __init__(self, env_spec, num_tasks, rank, kernel, name='TTPolicy', continuous=False, var=1.0, **kwargs):
        super().__init__(env_spec, num_tasks, name, continuous, var, **kwargs)
        self.net = FTNet(dimensions=[self.P,self.S], rank=rank, latents=64, kernel=kernel, output_dim=self.A)

    def setup_for_testing(self, train_p_only=False):
        P_net, S_net = self.net.list_of_nets
        if train_p_only:
            for s in S_net.parameters():
                s.requires_grad = False

        self.use_zero_partner_vec = True


class ModularPolicy(AdaptivePolicy):
    def __init__(self, env_spec, num_tasks, name='Modular', continuous=False, var=1.0, **kwargs):
        super().__init__(env_spec, num_tasks, name, continuous, var, **kwargs)
        self.net = ModularNet(input_dim=self.S, output_dim=self.A, latent_dim=64, num_partners=self.P, continuous=continuous)

    def setup_for_testing(self, train_p_only=False):
        nets = self.net.partner_nets

        def reset(m):
            if isinstance(m, torch.nn.Linear):
                m.reset_parameters()
        nets.apply(reset)

        if train_p_only:
            for s in self.net.main_net.parameters():
                s.requires_grad = False

    def forward_full(self, observations):
        batch = observations.size()[:-1]
        partner_idx_vec = torch.ones( *batch, dtype=np.long ).to(observations.device) * self.cur_p
        logits, wass_dist = self.net(observations, partner_idx_vec=partner_idx_vec)
        return logits, wass_dist

    # pylint: disable=arguments-differ
    def forward(self, observations):
        if self.continuous:
            mu, _ = self.forward_full(observations)
            return Normal(loc=mu, scale=mu*0+self.var), dict()

        logits, _ = self.forward_full(observations)
        return Categorical(logits=logits), dict()

    def marginal_reg(self, observations):
        _, wass_dist = self.forward_full(observations)
        return wass_dist
