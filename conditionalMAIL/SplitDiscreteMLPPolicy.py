"""This modules creates a discrete policy network.
A neural network can be used as policy method in different RL algorithms.
It accepts an observation of the environment and predicts an action.
"""
import akro
import numpy as np
import torch
from torch.distributions import Categorical

from garage.torch.modules import MLPModule
from garage.torch.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from garage.torch import global_device
from garage.torch.policies.policy import Policy
from conditionalMAIL.utils import np_to_torch

from conditionalMAIL.DiscreteMLPPolicy import DiscreteMLPPolicy

class SplitDiscreteMLPPolicy(DiscreteMLPPolicy):
    """Implements a discrete policy network.
    The policy network selects action based on the state of the environment.
    It uses a PyTorch neural network module to fit the function of pi(s).
    """


    def __init__(self, env_spec, name='SplitPolicy', **kwargs):
        """Initialize class with multiple attributes.
        Args:
            env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
            name (str): Policy name.
            kwargs : Additional keyword arguments passed to the MLPModule.
        """
        super().__init__(env_spec, name, **kwargs)

        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self.net = torch.nn.ModuleList([
            MLPModule(input_dim=self._obs_dim // 2,
                                 output_dim=6,
                                 **kwargs),
            MLPModule(input_dim=self._obs_dim // 2,
                                 output_dim=6,
                                 **kwargs)
        ])

        assert( 2*(self._obs_dim // 2) == self._obs_dim )
        assert( 6*6 == self._action_dim )

        self.agent_idx = None

    # pylint: disable=arguments-differ
    def forward(self, observations):
        """Compute actions from the observations.
        Args:
            observations (torch.Tensor): Batch of observations on default
                torch device.
        Returns:
            torch.Tensor: Batch of actions.
        """
        if not isinstance(observations, torch.Tensor):
            observations = np_to_torch(observations)

        if self.agent_idx is not None:
            return self.split_forward(split_observations=observations, agent_idx=self.agent_idx)

        obs = torch.chunk(observations, 2, dim=-1)
        logits = [self.net[i](obs[i]) for i in [0,1]] # (batch, 6) and (batch, 6)
        # watch out for list comprehension ordering
        # currently it is (i0, j0), (i0, j1)...(i1,j0)...
        logits = torch.stack([logits[0][:,i] + logits[1][:,j] for i in range(6) for j in range(6)], dim=-1)
        return Categorical(logits=logits), dict()

    def split_forward(self, split_observations, agent_idx):
        if not isinstance(split_observations, torch.Tensor):
            split_observations = np_to_torch(split_observations)

        with torch.no_grad():
            logits = self.net[agent_idx](split_observations)
        # print("split_forward", logits, agent_idx)
        return Categorical(logits=logits), dict()

    def split_get_action(self, split_observations, agent_idx):
        # print("split_get_action", agent_idx)
        dist, _ = self.split_forward(split_observations, agent_idx)
        action = dist.sample().cpu().numpy()
        return action

    def set_onesided(self, agent_idx):  # set agent_idx=None to undo onesidedness
        self.agent_idx = agent_idx
