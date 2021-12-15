import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import torch
import sys

class ParticleEnvN(gym.Env):
    """
    N player game.
    Continuous actions (dx, dy)

    N landmarks, work together to have one agent at each landmark

    must get from point a to point b
    """

    def __init__(self, other_agents, ego_agent_idx=0):
        """
            2D grid from [0,0] to [10,10]
        """
        super(ParticleEnv, self).__init__()

        self.other_agents = other_agents
        self.solo = (self.other_agents == None)

        self.action_space = spaces.Box(np.array([0, 0, 0, 0]), np.array([1, 1, 1, 1]), dtype=np.float32)     # dx1, dy1, dx2, dy2
        self.observation_space = spaces.Box(np.array([0, 0]), np.array([10, 10]), dtype=np.float32)
        self.reset()

        self.cur_task = 0
        self.agent_idx = ego_agent_idx
        self.horizon = 20

    def step(self, action):
        if self.solo:
            joint_action = action
        else:
            partner = self.other_agents[self.cur_task]
            other_action = partner.get_action(self.state)[0]
            if self.agent_idx == 0:
                joint_action = np.concatenate((action[:2], other_action[2:]), axis=-1)
            else:
                joint_action = np.concatenate((other_action[:2], action[2:]), axis=-1)
        joint_action = joint_action[:2] + joint_action[2:]

        self.state = self.state + joint_action
        self.t = self.t + 1

        done = (self.t == self.horizon)
        reward = 20 - np.linalg.norm( np.array([[9.,9.]]) - self.state )

        return [self.state, reward, done, {}]

    def reset(self):
        self.state = np.array([1., 1.])
        self.t = 0
        return self.state

    def render(self):
        pass
        #print(self.state)


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
