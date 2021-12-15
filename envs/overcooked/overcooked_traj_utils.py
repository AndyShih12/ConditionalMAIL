import gym
import argparse
import numpy as np
from human_aware_rl.human.process_dataframes import save_npz_file, get_trajs_from_data
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.planning.planners import MediumLevelPlanner, NO_COUNTERS_PARAMS
from envs.overcooked.overcooked_utils import joint_action_encode

from garage import TimeStepBatch

def get_expert_trajs(mode, layouts=None):
    assert(mode == "train" or mode == "test")

    DEFAULT_DATA_PARAMS = {
        "train_mdps": ["unident_s", "simple", "random1", "random0", "random3"],
        "ordered_trajs": True,
        "human_ai_trajs": False,
        "data_path": "./envs/overcooked/human_aware_rl/human_aware_rl/data/human/anonymized/clean_%s_trials.pkl" % mode
    }
    if layouts is not None:
        DEFAULT_DATA_PARAMS["train_mdps"] = layouts

    expert_trajs = get_trajs_from_data(**DEFAULT_DATA_PARAMS)

    num_trajs = len(expert_trajs["ep_observations"])
    expert_trajs["ep_actions"] = [[a[0] for a in expert_trajs["ep_actions"][i]] for i in range(num_trajs)]

    for i in range(0, num_trajs):
        expert_trajs['ep_observations'][i] = np.array( expert_trajs['ep_observations'][i] )
        expert_trajs['ep_actions'][i] = np.array( expert_trajs['ep_actions'][i] )
        expert_trajs['ep_dones'][i] = np.array( expert_trajs['ep_dones'][i] )

    return expert_trajs

def get_expert_joint_trajs(mode, layouts=None, one_sided_obs=False):
    assert(mode == "train" or mode == "test")
    # i-th episode is paried with (i+1)-th episode
    expert_single_trajs = get_expert_trajs(mode=mode, layouts=layouts)

    expert_joint_trajs = {
        # With shape (n_timesteps, game_len), where game_len might vary across games:
        "ep_observations": [],
        "ep_actions": [], # encoded as one action
        "ep_rewards": [], # One (same) reward value
        "ep_dones": [], # One (same) done values,

        # With shape (n_episodes, ):
        "ep_returns": [], # Sum of rewards across each episode
        "ep_lengths": [], # Lengths of each episode
        "mdp_params": [],
        "env_params": []
    }

    num_trajs = len(expert_single_trajs["ep_observations"])

    for i in range(0, num_trajs, 2):
        joint_obs = np.array([ np.concatenate( joint_obs ) for joint_obs in zip(expert_single_trajs["ep_observations"][i], expert_single_trajs["ep_observations"][i+1]) ])
        joint_actions = np.array([ joint_action_encode( joint_action ) for joint_action in zip(expert_single_trajs["ep_actions"][i], expert_single_trajs["ep_actions"][i+1]) ])
        expert_joint_trajs["ep_observations"].append( joint_obs if not one_sided_obs else expert_single_trajs["ep_observations"][i] )
        expert_joint_trajs["ep_actions"].append( joint_actions )
        expert_joint_trajs["ep_rewards"].append( expert_single_trajs["ep_rewards"][i] )
        expert_joint_trajs["ep_dones"].append( expert_single_trajs["ep_dones"][i] )

        expert_joint_trajs["ep_returns"].append( expert_single_trajs["ep_returns"][i] )
        expert_joint_trajs["ep_lengths"].append( expert_single_trajs["ep_lengths"][i] )
        expert_joint_trajs["mdp_params"].append( expert_single_trajs["mdp_params"][i] )
        expert_joint_trajs["env_params"].append( expert_single_trajs["env_params"][i] )

    return expert_joint_trajs

def get_single_timestepbatches(env_spec, mode, layout_name):
    assert(mode == "train" or mode == "test")

    expert_trajs = get_expert_trajs(mode=mode, layouts=[layout_name])
    print(expert_trajs.keys())
    print(expert_trajs["ep_lengths"])
    print(len(expert_trajs["ep_observations"]))
    # dict_keys(['ep_observations', 'ep_actions', 'ep_rewards', 'ep_dones', 'ep_returns', 'ep_lengths', mdp_params', 'env_params'])

    return trajs_to_timestepbatches(env_spec, expert_trajs)

def get_joint_timestepbatches(env_spec, mode, layout_name, one_sided_obs=False):
    assert(mode == "train" or mode == "test")

    expert_trajs = get_expert_joint_trajs(mode=mode, layouts=[layout_name], one_sided_obs=one_sided_obs)
    print(expert_trajs.keys())
    print(expert_trajs["ep_lengths"])
    print(len(expert_trajs["ep_observations"]))
    # dict_keys(['ep_observations', 'ep_actions', 'ep_rewards', 'ep_dones', 'ep_returns', 'ep_lengths', mdp_params', 'env_params'])

    return trajs_to_timestepbatches(env_spec, expert_trajs)

def trajs_to_timestepbatches(env_spec, expert_trajs):
    def get_timestepbatch(idx):
        print(expert_trajs['ep_actions'][idx])
        return TimeStepBatch(env_spec=env_spec,
                                observations=expert_trajs['ep_observations'][idx],
                                actions=expert_trajs['ep_actions'][idx],
                                rewards=expert_trajs['ep_rewards'][idx],
                                next_observations=np.concatenate( (expert_trajs['ep_observations'][idx][1:], expert_trajs['ep_observations'][idx][-1:]) ),
                                terminals=expert_trajs['ep_dones'][idx],
                                env_infos={},
                                agent_infos={})

    timestepbatches = [get_timestepbatch(idx) for idx in range(len(expert_trajs["ep_observations"]))]
    return timestepbatches

def split_timestepbatch(timestepbatch, num_splits):
    # split a single timestepbatch into num_splits, with splits even sized and indices chosen randomly
    # returns a list of timestepbatches
    n = len(timestepbatch.actions)
    indices = np.random.permutation(n)
    batches = np.array_split(indices, num_splits)

    split_timestepbatches = [TimeStepBatch(env_spec=timestepbatch.env_spec,
                    observations=timestepbatch.observations[batch],
                    actions=timestepbatch.actions[batch],
                    rewards=timestepbatch.rewards[batch],
                    next_observations=timestepbatch.next_observations[batch],
                    terminals=timestepbatch.terminals[batch],
                    env_infos=timestepbatch.env_infos,
                    agent_infos=timestepbatch.agent_infos) for batch in batches]

    return split_timestepbatches
