"""Model-Agnostic Meta-Learning (MAML) algorithm implementation for RL."""
import collections
import copy
import itertools

from dowel import tabular
import numpy as np
import torch

from garage import _Default, make_optimizer, TimeStepBatch
from garage import log_multitask_performance
from garage import TrajectoryBatch
from garage.misc import tensor_utils
from garage.sampler import OnPolicyVectorizedSampler
from garage.torch import update_module_params
from garage.torch.optimizers import ConjugateGradientOptimizer
from garage.torch.optimizers import DifferentiableSGD
from conditionalMAIL.utils import np_to_torch


class MAMLBC:
    """Model-Agnostic Meta-Learning (MAML).

    Args:
        inner_algo (garage.torch.algos.VPG): The inner algorithm used for
            computing loss.
        env (garage.envs.GarageEnv): A gym environment.
        policy (garage.torch.policies.Policy): Policy.
        meta_optimizer (Union[torch.optim.Optimizer, tuple]):
            Type of optimizer.
            This can be an optimizer type such as `torch.optim.Adam` or a tuple
            of type and dictionary, where dictionary contains arguments to
            initialize the optimizer e.g. `(torch.optim.Adam, {'lr' : 1e-3})`.
        meta_batch_size (int): Number of tasks sampled per batch.
        inner_lr (float): Adaptation learning rate.
        outer_lr (float): Meta policy learning rate.
        num_grad_updates (int): Number of adaptation gradient steps.
        meta_evaluator (garage.experiment.MetaEvaluator): A meta evaluator for
            meta-testing. If None, don't do meta-testing.
        evaluate_every_n_epochs (int): Do meta-testing every this epochs.

    """

    def __init__(self,
                 env,
            policy, # learner's policy, not expert policy
                 batch_size,
            sources=None,
            source_type="trajectory", # "policy" or "trajectory"
                 meta_optimizer=torch.optim.Adam,
                 meta_batch_size=3,
                 inner_lr=0.1,
                 outer_lr=1e-3,
            loss='log_prob',
            minibatches_per_epoch=16,
                 num_grad_updates=1,
                 meta_evaluator=None,
                 evaluate_every_n_epochs=1):
        self._sources = sources
        self._sources = [itertools.cycle(iter(source)) for source in self._sources]
        self._source = self._sources[0]
        self.source_type = source_type
        self._policy = policy
        self._batch_size = batch_size

        self._meta_evaluator = meta_evaluator
        self._env = env
        self._num_grad_updates = num_grad_updates
        self._meta_batch_size = meta_batch_size
        self._inner_optimizer = DifferentiableSGD(self._policy, lr=inner_lr) # should this be inner_algo.policy?
        self._meta_optimizer = make_optimizer(meta_optimizer,
                                              module=policy,
                                              lr=_Default(outer_lr),
                                              eps=_Default(1e-5))
        self._evaluate_every_n_epochs = evaluate_every_n_epochs

        if loss not in ('log_prob', 'mse'):
            raise ValueError('Loss should be either "log_prob" or "mse".')
        self._loss = loss
        self._minibatches_per_epoch = minibatches_per_epoch

    def train(self, runner):
        """Obtain samples and start training for each epoch.

        Args:
            runner (LocalRunner): LocalRunner is passed to give algorithm
                the access to runner.step_epochs(), which provides services
                such as snapshotting and sampler control.
        """
        for _ in runner.step_epochs():
            all_samples, all_params = self._obtain_samples(runner)
            losses = self.train_once(runner, all_samples, all_params)
            runner.step_itr += 1

    def train_once(self, runner, all_samples, all_params):
        """Train the algorithm once.

        Args:
            runner (garage.experiment.LocalRunner): The experiment runner.
            all_samples (list[list[MAMLTrajectoryBatch]]): A two
                dimensional list of MAMLTrajectoryBatch of size
                [meta_batch_size * (num_grad_updates + 1)]
            all_params (list[dict]): A list of named parameter dictionaries.
                Each dictionary contains key value pair of names (str) and
                parameters (torch.Tensor).

        Returns:
            float: Average return.

        """
        itr = runner.step_itr

        meta_objective = self._compute_meta_loss(all_samples, all_params)

        self._meta_optimizer.zero_grad()
        meta_objective.backward()

        self._meta_optimize(all_samples, all_params)

        # Log
        loss_after = self._compute_meta_loss(all_samples,
                                             all_params,
                                             set_grad=False)

        with torch.no_grad():
            loss_before, loss_after = meta_objective.item(), loss_after.item()
            with tabular.prefix(self._policy.name + '/'):
                tabular.record('Iteration', itr)
                tabular.record('LossBefore', loss_before)
                tabular.record('LossAfter', loss_after)
                tabular.record('dLoss', loss_before - loss_after)

        if self._meta_evaluator and itr % self._evaluate_every_n_epochs == 0:
            self._meta_evaluator.evaluate(self)

        return loss_after

    def _obtain_samples(self, runner):
        """Obtain samples for each task before and after the fast-adaptation.

        Args:
            runner (LocalRunner): A local runner instance to obtain samples.

        Returns:
            tuple: Tuple of (all_samples, all_params).
                all_samples (list[MAMLTrajectoryBatch]): A list of size
                    [meta_batch_size * (num_grad_updates + 1)]
                all_params (list[dict]): A list of named parameter
                    dictionaries.

        """
        tasks = self._env.sample_tasks(self._meta_batch_size)
        all_samples = [[] for _ in range(len(tasks))]
        all_params = []
        theta = dict(self._policy.named_parameters())

        for i, task in enumerate(tasks):
            self._set_task(runner, task)

            for j in range(self._num_grad_updates + 1):

                batches = []
                while (sum(len(batch.actions)
                        for batch in batches) < self._batch_size):
                    batches.append(next(self._source))

                batch_samples = TimeStepBatch.concatenate(*batches)
                all_samples[i].append(batch_samples)

                # The last iteration does only sampling but no adapting
                if j < self._num_grad_updates:
                    # A grad need to be kept for the next grad update
                    # Except for the last grad update
                    require_grad = j < self._num_grad_updates - 1
                    self._adapt(batch_samples, set_grad=require_grad)

            all_params.append(dict(self._policy.named_parameters()))
            # Restore to pre-updated policy
            update_module_params(self._policy, theta)

        return all_samples, all_params

    def _adapt(self, batch_samples, set_grad=True):
        """Performs one MAML inner step to update the policy.

        Args:
            batch_samples (MAMLTrajectoryBatch): Samples data for one
                task and one gradient step.
            set_grad (bool): if False, update policy parameters in-place.
                Else, allow taking gradient of functions of updated parameters
                with respect to pre-updated parameters.

        """
        # pylint: disable=protected-access
        batch_observations, batch_actions = np_to_torch(batch_samples.observations[1:]), np_to_torch(batch_samples.actions[1:])
        loss = self._compute_loss(batch_observations, batch_actions)

        # Update policy parameters with one SGD step
        self._inner_optimizer.zero_grad()
        loss.backward(create_graph=set_grad)

        with torch.set_grad_enabled(set_grad):
            self._inner_optimizer.step()

    def _meta_optimize(self, all_samples, all_params):
        self._meta_optimizer.step(lambda: self._compute_meta_loss(
            all_samples, all_params, set_grad=False))

    def _compute_meta_loss(self, all_samples, all_params, set_grad=True):
        """Compute loss to meta-optimize.

        Args:
            all_samples (list[list[MAMLTrajectoryBatch]]): A two
                dimensional list of MAMLTrajectoryBatch of size
                [meta_batch_size * (num_grad_updates + 1)]
            all_params (list[dict]): A list of named parameter dictionaries.
                Each dictionary contains key value pair of names (str) and
                parameters (torch.Tensor).
            set_grad (bool): Whether to enable gradient calculation or not.

        Returns:
            torch.Tensor: Calculated mean value of loss.

        """
        theta = dict(self._policy.named_parameters())
        #old_theta = dict(self._old_policy.named_parameters())

        losses = []
        for task_samples, task_params in zip(all_samples, all_params):
            for i in range(self._num_grad_updates):
                require_grad = i < self._num_grad_updates - 1 or set_grad
                self._adapt(task_samples[i], set_grad=require_grad)

            #update_module_params(self._old_policy, task_params)
            with torch.set_grad_enabled(set_grad):
                # pylint: disable=protected-access
                last_update = task_samples[-1]
                last_update_observations, last_update_actions = np_to_torch(last_update.observations[1:]), np_to_torch(last_update.actions[1:])
                loss = self._compute_loss(last_update_observations, last_update_actions)
            losses.append(loss)

            update_module_params(self._policy, theta)
            #update_module_params(self._old_policy, old_theta)

        return torch.stack(losses).mean()

    def _compute_loss(self, observations, actions):
        """Compute loss of inner algo

        Args:
            observations (torch.Tensor): Observations used to select actions.
                Has shape :math:`(B, O^*)`, where :math:`B` is the batch
                dimension and :math:`O^*` are the observation dimensions.
            actions (torch.Tensor): The actions of the expert.
                Has shape :math:`(B, A^*)`, where :math:`B` is the batch
                dimension and :math:`A^*` are the action dimensions.

        Returns:
            torch.Tensor: The loss through which gradient can be propagated
                back to the learner. Depends on self._loss.

        """
        assert(self._loss == 'log_prob')
        batch = observations.size(0)
        learner_output = self._policy(observations)
        action_dist, _ = learner_output

        ret = action_dist.log_prob(actions)
        ret = ret.view(batch, -1).sum(dim=1)
        ret = -torch.mean(ret)
        return ret

    def _set_task(self, runner, task):
        # pylint: disable=protected-access, no-self-use
        if self.source_type == "policy":
            for env in runner._sampler._vec_env.envs:
                env.set_task(task)

        self._source = self._sources[int(task)]

    @property
    def policy(self):
        """Current policy of the inner algorithm.

        Returns:
            garage.torch.policies.Policy: Current policy of the inner
                algorithm.

        """
        return self._policy
