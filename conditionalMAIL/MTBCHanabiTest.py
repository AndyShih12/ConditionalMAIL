"""Implementation of Behavioral Cloning in PyTorch."""
import itertools
import copy

from dowel import tabular, logger
import numpy as np
import torch

from garage import make_optimizer
from conditionalMAIL.utils import np_to_torch
from conditionalMAIL.MTBCTest import MTBCTest

class MTBCHanabiTest(MTBCTest):
    def _train_once(self, runner, epoch):
        """Obtain samplers and train for one epoch.

        Args:
            runner (LocalRunner): LocalRunner to which may be used to obtain
                samples.
            epoch (int): The current epoch.

        Returns:
            List[float]: Losses.

        """

        evaluate_batch = self._obtain_samples(runner, epoch, sample_type="evaluate")
        finetune_batch = self._obtain_samples(runner, epoch, sample_type="finetune")
        indices = np.random.permutation(len(finetune_batch.actions))
        print("finetune batch size: ", len(finetune_batch.actions))

        minibatches = np.array_split(indices, self._minibatches_per_epoch)
        losses = []

        for i, minibatch in enumerate(minibatches):
            evaluate_observations = np_to_torch(evaluate_batch.observations)
            evaluate_actions = np_to_torch(evaluate_batch.actions)
            evaluate_turns = np_to_torch(evaluate_batch.env_infos['turn'])
            evaluate_legal_moves = np_to_torch(evaluate_batch.env_infos['legal_moves'])

            expert_loss = self._compute_expert_loss(evaluate_observations, evaluate_actions, evaluate_turns, evaluate_legal_moves)
            losses.append(expert_loss.item())
            with tabular.prefix(self._name + '/'):
                tabular.record('MeanLoss', expert_loss.item())
                tabular.record('Batch', i)
            logger.log(tabular)

            growing_batches = np.concatenate( minibatches[:i+1] )
            finetune_observations = np_to_torch(finetune_batch.observations[ growing_batches ])
            finetune_actions = np_to_torch(finetune_batch.actions[ growing_batches ])
            finetune_turns = np_to_torch(finetune_batch.env_infos['turn'][ growing_batches ])
            finetune_legal_moves = np_to_torch(finetune_batch.env_infos['legal_moves'][ growing_batches ])

            print(len(finetune_observations))
            for _ in range(1):
                self._optimizer.zero_grad()
                partner_loss = self._compute_partner_loss(finetune_observations, finetune_actions, finetune_turns, finetune_legal_moves)
                partner_loss.backward()
                self._optimizer.step()

        with tabular.prefix(self._name + '/'):
            tabular.record('Batch', int(1e9)) # some large number to make the last log sink to bottom

        return losses

    def _compute_loss(self, observations, actions, turns_mask, legal_moves):
        observations = observations[turns_mask]
        actions = actions[turns_mask]
        legal_moves = legal_moves[turns_mask]

        actions = actions % legal_moves

        return super()._compute_loss(observations, actions)

    def _compute_expert_loss(self, observations, actions, turns, legal_moves):
        turns_mask = turns.long() != 0
        return self._compute_loss(observations, actions, turns_mask, legal_moves)

    def _compute_partner_loss(self, observations, actions, turns, legal_moves):
        turns_mask = turns.long() == 0
        return self._compute_loss(observations, actions, turns_mask, legal_moves)
