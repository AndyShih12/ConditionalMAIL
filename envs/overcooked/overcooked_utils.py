import gym
import tqdm
import numpy as np
from overcooked_ai_py.utils import mean_and_std_err
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.planning.planners import MediumLevelPlanner, NO_COUNTERS_PARAMS

from garage.sampler import OnPolicyVectorizedSampler
from garage import TimeStepBatch

import akro
from garage.envs.env_spec import EnvSpec

DEFAULT_ENV_PARAMS = {
    "horizon": 400
}

MAX_HORIZON = 1e10

def collect_expert_trajectories(policy, env, fullbatch=100000, one_sided_obs=False):
    class DummyAlgo():
        def __init__(self, policy, max_path_length):
            self.policy = policy
            self.max_path_length = max_path_length

    algo = DummyAlgo(policy=policy, max_path_length=500)
    sampler = OnPolicyVectorizedSampler(algo, env)

    trajs = []
    sampler.start_worker()

    batch = fullbatch
    for i in range(10):
        batch_size = max(1, batch//10)
        path = sampler.obtain_samples(itr=0, batch_size=batch_size)
        for p in path:
            p["terminals"] = p["dones"]
            p["next_observations"] = p["observations"]
        traj = TimeStepBatch.from_time_step_list(env_spec=env.spec, ts_samples=path)
        if one_sided_obs:
            high = traj.env_spec.observation_space.high
            high = high[:high.shape[0]//2]
            env_spec = EnvSpec(action_space=traj.env_spec.action_space, observation_space=akro.Box(high * 0, high, dtype=np.float32))
            traj = TimeStepBatch(env_spec=env_spec,
                                observations=traj.observations[:,:traj.observations.shape[1]//2],
                                actions=traj.actions,
                                rewards=traj.rewards,
                                next_observations=traj.next_observations[:,:traj.next_observations.shape[1]//2],
                                terminals=traj.terminals,
                                env_infos=traj.env_infos,
                                agent_infos=traj.agent_infos)
        print(traj)
        trajs.append(traj)
    sampler.shutdown_worker()

    return trajs.__iter__()


class OvercookedEnv(object):
    """An environment wrapper for the OvercookedGridworld Markov Decision Process.

    The environment keeps track of the current state of the agent, updates
    it as the agent takes actions, and provides rewards to the agent.
    """

    def __init__(self, mdp, start_state_fn=None, horizon=MAX_HORIZON, debug=False):
        """
        mdp (OvercookedGridworld or function): either an instance of the MDP or a function that returns MDP instances
        start_state_fn (OvercookedState): function that returns start state for the MDP, called at each environment reset
        horizon (float): number of steps before the environment returns done=True
        """
        if isinstance(mdp, OvercookedGridworld):
            self.mdp_generator_fn = lambda: mdp
        elif callable(mdp) and isinstance(mdp(), OvercookedGridworld):
            self.mdp_generator_fn = mdp
        else:
            raise ValueError("Mdp should be either OvercookedGridworld instance or a generating function")

        self.horizon = horizon
        self.start_state_fn = start_state_fn
        self.reset()
        if self.horizon >= MAX_HORIZON and self.state.order_list is None and debug:
            print("Environment has (near-)infinite horizon and no terminal states")

    def __repr__(self):
        """Standard way to view the state of an environment programatically
        is just to print the Env object"""
        return self.mdp.state_string(self.state)

    def print_state_transition(self, a_t, r_t, info):
        print("Timestep: {}\nJoint action taken: {} \t Reward: {} + shape * {} \n{}\n".format(
            self.t, tuple(Action.ACTION_TO_CHAR[a] for a in a_t), r_t, info["shaped_r"], self)
        )

    @property
    def env_params(self):
        return {
            "start_state_fn": self.start_state_fn,
            "horizon": self.horizon
        }

    def display_states(self, *states):
        old_state = self.state
        for s in states:
            self.state = s
            print(self)
        self.state = old_state

    @staticmethod
    def print_state(mdp, s):
        e = OvercookedEnv(mdp, s)
        print(e)

    def copy(self):
        return OvercookedEnv(
            mdp=self.mdp.copy(),
            start_state_fn=self.start_state_fn,
            horizon=self.horizon
        )

    def step(self, joint_action):
        """Performs a joint action, updating the environment state
        and providing a reward.

        On being done, stats about the episode are added to info:
            ep_sparse_r: the environment sparse reward, given only at soup delivery
            ep_shaped_r: the component of the reward that is due to reward shaped (excluding sparse rewards)
            ep_length: length of rollout
        """
        assert not self.is_done()
        next_state, sparse_reward, reward_shaping = self.mdp.get_state_transition(self.state, joint_action)
        self.cumulative_sparse_rewards += sparse_reward
        self.cumulative_shaped_rewards += reward_shaping
        self.state = next_state
        self.t += 1
        done = self.is_done()
        info = {'shaped_r': reward_shaping}
        if done:
            info['episode'] = {
                'ep_sparse_r': self.cumulative_sparse_rewards,
                'ep_shaped_r': self.cumulative_shaped_rewards,
                'ep_length': self.t
            }
        return (next_state, sparse_reward + reward_shaping, done, info)

    def reset(self):
        """Resets the environment. Does NOT reset the agent."""
        # print("RESET", flush=True)
        self.mdp = self.mdp_generator_fn()
        if self.start_state_fn is None:
            self.state = self.mdp.get_standard_start_state()
        else:
            self.state = self.start_state_fn()
        self.cumulative_sparse_rewards = 0
        self.cumulative_shaped_rewards = 0
        self.t = 0

    def is_done(self):
        """Whether the episode is over."""
        return self.t >= self.horizon or self.mdp.is_terminal(self.state)

    def execute_plan(self, start_state, joint_action_plan, display=False):
        """Executes action_plan (a list of joint actions) from a start 
        state in the mdp and returns the resulting state."""
        # print("EXECUTE PLAN", flush=True)
        self.state = start_state
        done = False
        if display: print("Starting state\n{}".format(self))
        for joint_action in joint_action_plan:
            self.step(joint_action)
            done = self.is_done()
            if display: print(self)
            if done: break
        successor_state = self.state
        self.reset()
        return successor_state, done

    def run_agents(self, agent_pair, include_final_state=False, display=False, display_until=np.Inf):
        """
        Trajectory returned will a list of state-action pairs (s_t, joint_a_t, r_t, done_t).
        """
        assert self.cumulative_sparse_rewards == self.cumulative_shaped_rewards == 0, \
            "Did not reset environment before running agents"
        trajectory = []
        done = False

        if display: print(self)
        while not done:
            s_t = self.state
            a_t = agent_pair.joint_action(s_t)

            # Break if either agent is out of actions
            if any([a is None for a in a_t]):
                break

            s_tp1, r_t, done, info = self.step(a_t)
            trajectory.append((s_t, a_t, r_t, done))

            if display and self.t < display_until:
                self.print_state_transition(a_t, r_t, info)

        assert len(trajectory) == self.t, "{} vs {}".format(len(trajectory), self.t)

        # Add final state
        if include_final_state:
            trajectory.append((s_tp1, (None, None), 0, True))

        return np.array(trajectory), self.t, self.cumulative_sparse_rewards, self.cumulative_shaped_rewards

    def get_rollouts(self, agent_pair, num_games, display=False, final_state=False, agent_idx=0, reward_shaping=0.0, display_until=np.Inf, info=True):
        """
        Simulate `num_games` number rollouts with the current agent_pair and returns processed 
        trajectories.

        Only returns the trajectories for one of the agents (the actions _that_ agent took), 
        namely the one indicated by `agent_idx`.

        Returning excessive information to be able to convert trajectories to any required format 
        (baselines, stable_baselines, etc)

        NOTE: standard trajectories format used throughout the codebase
        """
        trajectories = {
            # With shape (n_timesteps, game_len), where game_len might vary across games:
            "ep_observations": [],
            "ep_actions": [],
            "ep_rewards": [], # Individual dense (= sparse + shaped * rew_shaping) reward values
            "ep_dones": [], # Individual done values

            # With shape (n_episodes, ):
            "ep_returns": [], # Sum of dense and sparse rewards across each episode
            "ep_returns_sparse": [], # Sum of sparse rewards across each episode
            "ep_lengths": [], # Lengths of each episode
            "mdp_params": [], # Custom MDP params to for each episode
            "env_params": [] # Custom Env params for each episode
        }

        for _ in tqdm.trange(num_games):
            agent_pair.set_mdp(self.mdp)

            trajectory, time_taken, tot_rews_sparse, tot_rews_shaped = self.run_agents(agent_pair, display=display, include_final_state=final_state, display_until=display_until)
            obs, actions, rews, dones = trajectory.T[0], trajectory.T[1], trajectory.T[2], trajectory.T[3]
            trajectories["ep_observations"].append(obs)
            trajectories["ep_actions"].append(actions)
            trajectories["ep_rewards"].append(rews)
            trajectories["ep_dones"].append(dones)
            trajectories["ep_returns"].append(tot_rews_sparse + tot_rews_shaped * reward_shaping)
            trajectories["ep_returns_sparse"].append(tot_rews_sparse)
            trajectories["ep_lengths"].append(time_taken)
            trajectories["mdp_params"].append(self.mdp.mdp_params)
            trajectories["env_params"].append(self.env_params)

            self.reset()
            agent_pair.reset()

        mu, se = mean_and_std_err(trajectories["ep_returns"])
        if info: print("Avg reward {:.2f} (std: {:.2f}, se: {:.2f}) over {} games of avg length {}".format(
            mu, np.std(trajectories["ep_returns"]), se, num_games, np.mean(trajectories["ep_lengths"]))
        )

        # Converting to numpy arrays
        trajectories = {k: np.array(v) for k, v in trajectories.items()}
        return trajectories

def joint_action_decode(action_int, lA=6):
    # only supports 2-player
    return action_int // lA, action_int % lA

def joint_action_encode(action_tup, lA=6):
    # only supports 2-player
    return action_tup[0] * lA + action_tup[1]

class OvercookedJoint(gym.Env):
    """
    Wrapper for the Env class above that is SOMEWHAT compatible with the standard gym API.

    NOTE: Observations returned are in a dictionary format with various information that is
    necessary to be able to handle the multi-agent nature of the environment. There are probably
    better ways to handle this, but we found this to work with minor modifications to OpenAI Baselines.

    NOTE: The index of the main agent in the mdp is randomized at each reset of the environment, and 
    is kept track of by the self.agent_idx attribute. This means that it is necessary to pass on this 
    information in the output to know for which agent index featurizations should be made for other agents.

    For example, say one is training A0 paired with A1, and A1 takes a custom state featurization.
    Then in the runner.py loop in OpenAI Baselines, we will get the lossless encodings of the state,
    and the true Overcooked state. When we encode the true state to feed to A1, we also need to know
    what agent index it has in the environment (as encodings will be index dependent).
    """

    def custom_init(self, base_env, featurize_fn, baselines=False):
        """
        base_env: OvercookedEnv
        featurize_fn: what function is used to featurize states returned in the 'both_agent_obs' field
        """
        if baselines:
            # NOTE: To prevent the randomness of choosing agent indexes
            # from leaking when using subprocess-vec-env in baselines (which 
            # seeding does not) reach, we set the same seed internally to all
            # environments. The effect is negligible, as all other randomness
            # is controlled by the actual run seeds
            np.random.seed(0)
        self.base_env = base_env
        self.mdp = base_env.mdp
        self.featurize_fn = featurize_fn
        self.observation_space = self._setup_observation_space()

        self.lA = len(Action.ALL_ACTIONS)
        self.action_space  = gym.spaces.Discrete( self.lA * self.lA )
        self.reset()

    def _setup_observation_space(self):
        dummy_state = self.mdp.get_standard_start_state()
        obs_shape = self.featurize_fn(dummy_state)[0].shape
        high = np.ones(obs_shape) * max(self.mdp.soup_cooking_time, self.mdp.num_items_for_soup, 5)

        high = np.repeat(high, 2)

        return gym.spaces.Box(high * 0, high, dtype=np.float32)

    def step(self, action):
        """
        action: 
            (agent with index self.agent_idx action, other agent action)
            is a tuple with the joint action of the primary and secondary agents in index format
            encoded as an int

        returns:
            observation: formatted to be standard input for self.agent_idx's policy
        """
        #action = np.argmax(action)
        #print("action", action, flush=True)
        action = joint_action_decode(action_int=action)
        #print(action)
        agent_action, other_agent_action = [Action.INDEX_TO_ACTION[a] for a in action]

        joint_action = (agent_action, other_agent_action)

        # print("last state")
        # print(self.base_env.mdp.state_string(self.base_env.state))
        next_state, reward, done, info = self.base_env.step(joint_action)
        # print(joint_action, self.base_env.t)
        # print(self.base_env.mdp.state_string(next_state), flush=True)
        ob_p0, ob_p1 = self.featurize_fn(next_state)
        both_agents_ob = (ob_p0, ob_p1)

        #print(ob_p0.shape, ob_p1.shape)
        obs = np.concatenate(both_agents_ob)
        #print(joint_action, reward)
        return obs, reward, done, {}#info

    def reset(self):
        """
        When training on individual maps, we want to randomize which agent is assigned to which
        starting location, in order to make sure that the agents are trained to be able to 
        complete the task starting at either of the hardcoded positions.

        NOTE: a nicer way to do this would be to just randomize starting positions, and not
        have to deal with randomizing indices.
        """
        self.base_env.reset()
        ob_p0, ob_p1 = self.featurize_fn(self.base_env.state)
        both_agents_ob = (ob_p0, ob_p1)

        #print(ob_p0.shape, ob_p1.shape)
        obs = np.concatenate(both_agents_ob)

        return obs

    def render(self, mode='human', close=False):
        pass

class OvercookedSingle(gym.Env):
    def custom_init(self, base_env, featurize_fn, other_agent, baselines=False, ego_agent_idx=0):
        """
        base_env: OvercookedEnv
        featurize_fn: what function is used to featurize states returned in the 'both_agent_obs' field
        """
        if baselines:
            # NOTE: To prevent the randomness of choosing agent indexes
            # from leaking when using subprocess-vec-env in baselines (which 
            # seeding does not) reach, we set the same seed internally to all
            # environments. The effect is negligible, as all other randomness
            # is controlled by the actual run seeds
            np.random.seed(0)
        self.base_env = base_env
        self.mdp = base_env.mdp
        self.featurize_fn = featurize_fn
        self.observation_space = self._setup_observation_space()

        self.lA = len(Action.ALL_ACTIONS)
        self.action_space  = gym.spaces.Discrete( self.lA )
        self.other_agent = other_agent
        self.agent_idx = ego_agent_idx
        self.reset()

    def set_other_agent(self, other_agent):
        self.other_agent = other_agent

    def _setup_observation_space(self):
        dummy_state = self.mdp.get_standard_start_state()
        obs_shape = self.featurize_fn(dummy_state)[0].shape
        high = np.ones(obs_shape) * max(self.mdp.soup_cooking_time, self.mdp.num_items_for_soup, 5)

        return gym.spaces.Box(high * 0, high, dtype=np.float32)

    def step(self, action):
        """
        action: 
            (agent with index self.agent_idx action, other agent action)
            is a tuple with the joint action of the primary and secondary agents in index format
            encoded as an int

        returns:
            observation: formatted to be standard input for self.agent_idx's policy
        """
        #other_action = self.other_agent.split_get_action(split_observations=self.other_agent_obs, agent_idx=1-self.agent_idx)
        other_action, _ = self.other_agent.get_actions(observations=self.other_agent_obs[np.newaxis,...])
        other_action = other_action[0]
        agent_action, other_agent_action = Action.INDEX_TO_ACTION[action], Action.INDEX_TO_ACTION[other_action]

        if self.agent_idx == 0:
            joint_action = (agent_action, other_agent_action)
        else:
            joint_action = (other_agent_action, agent_action)

        next_state, reward, done, info = self.base_env.step(joint_action)
        #print(self.base_env.mdp.state_string(next_state))
        ob_p0, ob_p1 = self.featurize_fn(next_state)
        if self.agent_idx == 0:
            obs, self.other_agent_obs = ob_p0, ob_p1
        else:
            obs, self.other_agent_obs = ob_p1, ob_p0

        return obs, reward, done, {}#info

    def reset(self):
        """
        When training on individual maps, we want to randomize which agent is assigned to which
        starting location, in order to make sure that the agents are trained to be able to 
        complete the task starting at either of the hardcoded positions.

        NOTE: a nicer way to do this would be to just randomize starting positions, and not
        have to deal with randomizing indices.
        """
        self.base_env.reset()
        ob_p0, ob_p1 = self.featurize_fn(self.base_env.state)
        if self.agent_idx == 0:
            obs, self.other_agent_obs = ob_p0, ob_p1
        else:
            obs, self.other_agent_obs = ob_p1, ob_p0

        return obs

    def render(self, mode='human', close=False):
        pass

class OvercookedMultiEnvWrapper(OvercookedSingle):
    def custom_init(self, base_env, featurize_fn, other_agents, baselines=False, ego_agent_idx=0):
        """
        base_env: OvercookedEnv
        featurize_fn: what function is used to featurize states returned in the 'both_agent_obs' field
        """
        self.cur_task = 0
        self.other_agents = other_agents

        super().custom_init(base_env, featurize_fn, other_agents[0], baselines, ego_agent_idx)

        #self.action_space  = gym.spaces.Discrete( 6*6 ) #!

    def _setup_observation_space(self):
        dummy_state = self.mdp.get_standard_start_state()
        obs_shape = self.featurize_fn(dummy_state)[0].shape
        high = np.ones(obs_shape) * max(self.mdp.soup_cooking_time, self.mdp.num_items_for_soup, 5)

        return gym.spaces.Box(high * 0, high, dtype=np.float32)

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


def getOvercookedEnv(joint_or_single, rew_shape, layout_name):
    assert(joint_or_single == "joint" or joint_or_single == "single")

    rew_shaping_params = {
        "PLACEMENT_IN_POT_REW": 3,
        "DISH_PICKUP_REWARD": 3,
        "SOUP_PICKUP_REWARD": 5,
        "DISH_DISP_DISTANCE_REW": 0,
        "POT_DISTANCE_REW": 0,
        "SOUP_DISTANCE_REW": 0,
    }
    rew_shaping_params = rew_shaping_params if rew_shape else None

    mdp = OvercookedGridworld.from_layout_name(layout_name=layout_name, rew_shaping_params=rew_shaping_params)
    mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, NO_COUNTERS_PARAMS, force_compute=False)

    env = OvercookedEnv(mdp, **DEFAULT_ENV_PARAMS)

    if joint_or_single == "joint":
        gym_env = OvercookedJoint() #gym.make("OvercookedJoint-v0")
        gym_env.custom_init(env, featurize_fn=lambda x: mdp.featurize_state(x, mlp))
    elif joint_or_single == "single":
        gym_env = OvercookedSingle() #gym.make("OvercookedSingle-v0")
        gym_env.custom_init(env, featurize_fn=lambda x: mdp.featurize_state(x, mlp), other_agent = None)

    return gym_env

def getOvercookedMultiEnv(other_agents, rew_shape, layout_name):
    rew_shaping_params = {
        "PLACEMENT_IN_POT_REW": 0,#3,
        "DISH_PICKUP_REWARD": 0,#3,
        "SOUP_PICKUP_REWARD": 0,#5,
        "DISH_DISP_DISTANCE_REW": 0,
        "POT_DISTANCE_REW": 0,
        "SOUP_DISTANCE_REW": 0,
    }
    rew_shaping_params = rew_shaping_params if rew_shape else None

    mdp = OvercookedGridworld.from_layout_name(layout_name=layout_name, rew_shaping_params=rew_shaping_params)
    mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, NO_COUNTERS_PARAMS, force_compute=False)

    env = OvercookedEnv(mdp, **DEFAULT_ENV_PARAMS)

    gym_env = OvercookedMultiEnvWrapper()
    gym_env.custom_init(env, featurize_fn=lambda x: mdp.featurize_state(x, mlp), other_agents=other_agents)


    return gym_env
