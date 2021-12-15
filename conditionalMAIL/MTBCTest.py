"""Implementation of Behavioral Cloning in PyTorch."""
import itertools
import copy

from dowel import tabular, logger
import numpy as np
import torch

from garage import _Default, log_performance, make_optimizer, TimeStepBatch, \
    TrajectoryBatch
from garage.np import obtain_evaluation_samples
from garage.np.algos.rl_algorithm import RLAlgorithm
from garage.np.policies import Policy
from garage.sampler import RaySampler
from garage.sampler import OnPolicyVectorizedSampler
from conditionalMAIL.utils import np_to_torch

class MTBCTest(RLAlgorithm):
    """Behavioral Cloning.

    Based on Model-Free Imitation Learning with Policy Optimization:
        https://arxiv.org/abs/1605.08478

    Args:
        env_spec (garage.envs.EnvSpec): Specification of environment.
        learner (garage.torch.Policy): Policy to train.
        batch_size (int): Size of optimization batch.
        source (garage.Policy or Generator[garage.TimeStepBatch]): Expert to
            clone. If a policy is passed, will set `.policy` to source and use
            the runner to sample from the policy.
        max_path_length (int or None): Required if a policy is passed as
            source.
        policy_optimizer (torch.optim.Optimizer): Optimizer to be used to
            optimize the policy.
        policy_lr (float): Learning rate of the policy optimizer.
        loss (str): Which loss function to use. Must be either 'log_prob' or
            'mse'. If set to 'log_prob' (the default), `learner` must be a
            `garage.torch.StochasticPolicy`.
        minibatches_per_epoch (int): Number of minibatches per epoch.
        name (str): Name to use for logging.

    Raises:
        ValueError: If `source` is a `garage.Policy` and `max_path_length` is
            not passed or `learner` is not a `garage.torch.StochasticPolicy`
            and loss is 'log_prob'.

    """

    # pylint: disable=too-few-public-methods

    def __init__(
            self,
            env,
            learner,
            *,
            batch_size,
            finetune_sources=None,
            evaluate_sources=None,
            source_type="policy", # "policy" or "trajectory"
            max_path_length=None,
            policy_optimizer=torch.optim.Adam,
            policy_lr=_Default(1e-3),
            loss='log_prob',
            minibatches_per_epoch=16,
            update_per_minibatch=1,
            name='MTBCTest',

            marginal_reg=None,
            policy_valuefn_task_switching=True,
    ):
        assert(source_type == "trajectory") # self._sources doesn't handle finetune/evaluate sources
        self._finetune_sources = finetune_sources
        self._evaluate_sources = evaluate_sources
        self.source_type = source_type
        self.learner = learner
        self.orig_learner = learner
        self.policy_optimizer = policy_optimizer
        self.policy_lr = policy_lr
        self._optimizer = make_optimizer(self.policy_optimizer,
                                         module=self.learner,
                                         lr=self.policy_lr)
        if loss not in ('log_prob', 'mse'):
            raise ValueError('Loss should be either "log_prob" or "mse".')
        self._loss = loss
        self._minibatches_per_epoch = minibatches_per_epoch
        self._update_per_minibatch = update_per_minibatch
        self._eval_env = None
        self._batch_size = batch_size
        self._name = name

        # Public fields for sampling.
        self.env = env
        self.max_path_length = max_path_length
        self.sampler_cls = None

        if self.source_type == "policy":
            if max_path_length is None:
                raise ValueError('max_path_length must be passed if the '
                                 'source is a policy')
            self.sampler_cls = OnPolicyVectorizedSampler #RaySampler
            self._finetune_sources = finetune_sources
            self._evaluate_sources = evaluate_sources
        else:
            self._finetune_sources = [itertools.cycle(iter(source)) for source in self._finetune_sources]
            self._evaluate_sources = [itertools.cycle(iter(source)) for source in self._evaluate_sources]

        self.task = 0
        self.marginal_reg = marginal_reg
        self.policy_valuefn_task_switching = policy_valuefn_task_switching

    def train(self, runner):
        """Obtain samplers and start actual training for each epoch.

        Args:
            runner (LocalRunner): LocalRunner is passed to give algorithm
                the access to runner.step_epochs(), which provides services
                such as snapshotting and sampler control.

        """
        # if not self._eval_env:
        #     self._eval_env = runner.get_env_copy()
        for epoch in runner.step_epochs():
            task = self.env.sample_tasks(1)[0]
            self.set_task(runner, task)

            # if self._eval_env is not None: # if using this, make sure eval_env is using same task as env
            #     log_performance(epoch,
            #                     obtain_evaluation_samples(
            #                         self.learner, self._eval_env),
            #                     discount=1.0)

            losses = self._train_once(runner, epoch)
            with tabular.prefix(self._name + '/'):
                tabular.record('MeanLoss', np.mean(losses))
                tabular.record('StdLoss', np.std(losses))
                tabular.record('Epoch', epoch)

    def set_task(self, runner, task):
        # task is a string
        self.task = int(task)
        if self.source_type == "policy":
            for env in runner._sampler._vec_env.envs:
                env.set_task(task)

        if self.policy_valuefn_task_switching:
            self.learner.set_task(task_idx=self.task)

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

            expert_loss = self._compute_expert_loss(evaluate_observations, evaluate_actions)
            losses.append(expert_loss.item())
            with tabular.prefix(self._name + '/'):
                tabular.record('MeanLoss', expert_loss.item())
                tabular.record('Batch', i)
            logger.log(tabular)

            growing_batches = np.concatenate( minibatches[:i+1] )
            finetune_observations = np_to_torch(finetune_batch.observations[ growing_batches ])
            finetune_actions = np_to_torch(finetune_batch.actions[ growing_batches ])

            print(len(finetune_observations))
            for _ in range(self._update_per_minibatch):
                self._optimizer.zero_grad()
                partner_loss = self._compute_partner_loss(finetune_observations, finetune_actions)
                partner_loss.backward()
                self._optimizer.step()

        with tabular.prefix(self._name + '/'):
            tabular.record('Batch', int(1e9)) # some large number to make the last log sink to bottom

        return losses

    def _obtain_samples(self, runner, epoch, sample_type):
        """Obtain samples from self._source.

        Args:
            runner (LocalRunner): LocalRunner to which may be used to obtain
                samples.
            epoch (int): The current epoch.

        Returns:
            TimeStepBatch: Batch of samples.

        """
        if sample_type == "finetune":
            self._sources = self._finetune_sources
        elif sample_type == "evaluate":
            self._sources = self._evaluate_sources
        else:
            raise Exception("not implemented sample type %s" % sample_type)

        if self.source_type == "policy":
            batch = TrajectoryBatch.from_trajectory_list(
                self.env.spec, runner.obtain_samples(epoch))
            log_performance(epoch, batch, 1.0, prefix='Expert')
            return batch
        else:
            batches = []
            while (sum(len(batch.actions)
                       for batch in batches) < self._batch_size):
                batches.append(next(self._sources[self.task]))
            return TimeStepBatch.concatenate(*batches)

    def _compute_loss(self, observations, actions):
        assert(self._loss == 'log_prob')
        batch = observations.size(0)
        learner_output = self.learner(observations)
        action_dist, _ = learner_output

        ret = action_dist.log_prob(actions)
        ret = ret.view(batch, -1).sum(dim=1)
        ret = -torch.mean(ret)
        return ret

    def _compute_expert_loss(self, observations, actions):
        return self._compute_loss(observations, actions)

    def _compute_partner_loss(self, observations, actions):
        return self._compute_loss(observations, actions)
