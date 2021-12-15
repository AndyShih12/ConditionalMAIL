import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import torch
import sys

class MABEnv(gym.Env):
    """
    Two player game.
    There are N (say 4) possible arms to pull.
        0 1 2 3
    Some of the arms are forbidden (determined by the state)
    Goal is to pull same arm k times in a row.
    """

    def __init__(self, n, a, mask, other_agents, enlarge_a=1):
        """
            n: number of states
            a: number of arms
            Mask is a boolean array of size n*a, denoting which arms are good at which settings
            other_agents is an array of size [P, n, a], denoting the probability of p-th partner's taking action a at state s

            state space is n
            action space is a
        """
        super(MABEnv, self).__init__()

        self.K = 100
        self.KA = enlarge_a

        self.n = n * self.K
        self.a = a * self.KA  # Enlarge A
        self.mask = mask
        self.other_agents = other_agents
        self.solo = (self.other_agents == None)

        self.action_space = spaces.Discrete(self.a)
        self.observation_space = spaces.Box(np.array([0]), np.array([self.n]), dtype=np.float32)
        self.reset()

        self.cur_task = 0

        print("states %u actions %u" % (self.n, self.a))

    def _eval_partner(self, partner, state):
        if isinstance(partner, (list,np.ndarray) ):
            return partner[state]
        else:
            state = torch.tensor([state]).float()
            dist, _ = partner.forward(state)
            return dist.probs.detach().numpy()

    def step(self, action):
        s = self.state[0] // self.K

        if self.solo:
            correct = self.mask[s][action]
        else:
            partner = self.other_agents[self.cur_task]
            partner_prob = self._eval_partner(partner=partner, state=self.state[0])
            partner_action = np.random.choice(self.a, size=1, p=partner_prob)

            match = (action == partner_action)
            correct = match and self.mask[s][action]

        done = True
        reward = 1 if correct else 0

        return [self.state, reward, done, {}]

    def reset(self):
        self.state = np.array([np.random.randint(self.n)])
        return self.state

    def render(self):
        print(self.state)


    def sample_tasks(self, n):
        """Sample a list of `num_tasks` tasks.
        Args:
            n (str): Number of tasks to sample.
        Returns:
            list[str]: A list of tasks.
        """
        print('sampling %u tasks' % n)
        task_ints = np.random.randint(0, self.num_tasks(), size=n)
        return [str(i) for i in task_ints]

    @property
    def all_task_names(self):
        """list[str]: Return a list of dummy task names."""
        return [str(i) for i in range(self.num_tasks())]

    def num_tasks(self):
        return len(self.other_agents)

    def set_task(self, task):
        """Reset with a task.
        Args:
            task (str): A task id.
        """
        task = int(task)
        self.cur_task = task
