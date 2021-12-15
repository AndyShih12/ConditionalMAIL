from garage.envs import GarageEnv
from garage.sampler import OnPolicyVectorizedSampler
from garage import TimeStepBatch
import numpy as np


import gym
from gym import error, spaces, utils
from hanabi_learning_environment.rl_env import HanabiEnv
from hanabi_learning_environment import pyhanabi

class HanabiEnvWrapper(HanabiEnv, gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, config, other_agent, ego_agent_idx=0):
        self.config = config
        super(HanabiEnvWrapper, self).__init__(config=self.config)

        observation_shape = super().vectorized_observation_shape()
        self.observation_space = spaces.Box(low=np.zeros(observation_shape[0]), high=np.ones(observation_shape[0]))
        self.action_space = spaces.Discrete(self.game.max_moves())

        self.agent_idx = ego_agent_idx
        self.other_agent = other_agent

    def set_other_agent(self, other_agent):
        self.other_agent = other_agent

    def reset(self):
        obs = super().reset()
        obs = obs['player_observations'][obs['current_player']]['vectorized']
        return obs

    def step(self, action):
        turn = self.state.cur_player()
        if turn != self.agent_idx and self.other_agent is not None:
            action, _ = self.other_agent.get_action(observation=self.obs)

        # action is a integer from 0 to self.action_space
        # we map it to one of the legal moves
        # the legal move array may be too small in some cases, so just modulo action by the array length
        legal_moves = self.state.legal_moves()
        move = legal_moves[action % len(legal_moves)].to_dict()

        obs, reward, done, info = super().step(move)
        self.obs = obs['player_observations'][obs['current_player']]['vectorized']
        info['turn'] = turn
        info['legal_moves'] = len(legal_moves)

        return self.obs, reward, done, info

class HanabiEnvMultiWrapper(HanabiEnvWrapper):
    metadata = {'render.modes': ['human']}
    def __init__(self, config, other_agents):

        self.cur_task = 0
        self.other_agents = other_agents
        self.solo = (self.other_agents == None)

        other_agent = None if self.solo else self.other_agents[0]
        super(HanabiEnvMultiWrapper, self).__init__(config=config, other_agent=other_agent)

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
        self.set_other_agent(self.other_agents[task])


def getHanabiEnv(other_agents):
    config = {
        "colors":                   1,
        "ranks":                    5,
        "players":                  4,
        "hand_size":                2,
        "max_information_tokens":   3,
        "max_life_tokens":          3,
        "observation_type":         pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value
    }
    env = HanabiEnvMultiWrapper(config=config, other_agents=other_agents)
    return GarageEnv(env)


def collect_expert_trajectories(policy, env, fullbatch=100000):
    class DummyAlgo():
        def __init__(self, policy, max_path_length):
            self.policy = policy
            self.max_path_length = max_path_length

    algo = DummyAlgo(policy=policy, max_path_length=1000)
    sampler = OnPolicyVectorizedSampler(algo, env)

    trajs = []
    sampler.start_worker()

    batch = fullbatch
    for i in range(100):
        batch_size = max(1, batch//100)
        path = sampler.obtain_samples(itr=0, batch_size=batch_size)
        for p in path:
            p["terminals"] = p["dones"]
            p["next_observations"] = p["observations"]

            legal_moves = p["env_infos"]["legal_moves"]
            p["actions"] = p["actions"] % p["env_infos"]["legal_moves"]

        traj = TimeStepBatch.from_time_step_list(env_spec=env.spec, ts_samples=path)
        #print(traj)
        trajs.append(traj)
    sampler.shutdown_worker()

    return trajs.__iter__()
