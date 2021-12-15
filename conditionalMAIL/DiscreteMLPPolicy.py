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



class DiscreteMLPPolicy(DeterministicMLPPolicy):
    """Implements a discrete policy network.
    The policy network selects action based on the state of the environment.
    It uses a PyTorch neural network module to fit the function of pi(s).
    """


    def __init__(self, env_spec, name='DiscreteMLPPolicy', **kwargs):
        """Initialize class with multiple attributes.
        Args:
            env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
            name (str): Policy name.
            kwargs : Additional keyword arguments passed to the MLPModule.
        """
        super().__init__(env_spec, name, **kwargs)

    # pylint: disable=arguments-differ
    def forward(self, observations):
        """Compute actions from the observations.
        Args:
            observations (torch.Tensor): Batch of observations on default
                torch device.
        Returns:
            torch.Tensor: Batch of actions.
        """
        logits = self._module(observations)
        return Categorical(logits=logits), dict()

    def get_actions(self, observations):
        """Get actions given observations.
        Args:
            observations (np.ndarray): Observations from the environment.
        Returns:
            tuple:
                * np.ndarray: Predicted actions.
                * dict:
                    * list[float]: Mean of the distribution
                    * list[float]: Log of standard deviation of the
                        distribution
        """
        if not isinstance(observations[0], np.ndarray) and not isinstance(
                observations[0], torch.Tensor):
            observations = self._env_spec.observation_space.flatten_n(
                observations)
        # frequently users like to pass lists of torch tensors or lists of
        # numpy arrays. This handles those conversions.
        if isinstance(observations, list):
            if isinstance(observations[0], np.ndarray):
                observations = np.stack(observations)
            elif isinstance(observations[0], torch.Tensor):
                observations = torch.stack(observations)

        if isinstance(self._env_spec.observation_space, akro.Image) and \
                len(observations.shape) < \
                len(self._env_spec.observation_space.shape):
            observations = self._env_spec.observation_space.unflatten_n(
                observations)
        with torch.no_grad():
            if not isinstance(observations, torch.Tensor):
                observations = torch.as_tensor(observations).float().to(
                    global_device())
            dist, info = self.forward(observations)
            return dist.sample().cpu().numpy(), {
                k: v.detach().cpu().numpy()
                for (k, v) in info.items()
            }

    def get_action(self, observation):
        """Get a single action given an observation.
        Args:
            observation (np.ndarray): Observation from the environment.
        Returns:
            tuple:
                * np.ndarray: Predicted action.
                * dict:
                    * list[float]: Mean of the distribution
                    * list[float]: Log of standard deviation of the
                        distribution
        """
        with torch.no_grad():
            dist, _ = self(torch.Tensor(observation).unsqueeze(0))
            return dist.sample().cpu().numpy()[0], dict()

    def set_task(self, task_idx):
        pass
