"""Implementation of Behavioral Cloning in PyTorch."""
import itertools
import copy

from dowel import tabular, logger
import numpy as np
import torch

from conditionalMAIL.MTBCTest import MTBCTest

class MTBCParticleTest(MTBCTest):
    def _compute_expert_loss(self, observations, actions):
        assert(self._loss == 'log_prob')
        batch = observations.size(0)
        learner_output = self.learner(observations)
        action_dist, _ = learner_output

        ret = action_dist.log_prob(actions)
        ret = ret.view(batch, -1)[:,:2].sum(dim=1)
        ret = -torch.mean(ret)
        return ret

    def _compute_partner_loss(self, observations, actions):
        assert(self._loss == 'log_prob')
        batch = observations.size(0)
        learner_output = self.learner(observations)
        action_dist, _ = learner_output

        ret = action_dist.log_prob(actions)
        ret = ret.view(batch, -1)[:,2:].sum(dim=1)
        ret = -torch.mean(ret)
        return ret
