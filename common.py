import types
import numpy as np
import torch as th
from garage.torch.algos import PPO
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.torch.policies import GaussianMLPPolicy
from garage.experiment.snapshotter import Snapshotter

from conditionalMAIL.DiscreteMLPPolicy import DiscreteMLPPolicy
from conditionalMAIL.MAMLBC import MAMLBC
from conditionalMAIL.MTBC import MTBC
from conditionalMAIL.AdaptivePolicy import MTPolicy, LatentPolicy, TTPolicy, ModularPolicy

def get_dirname(env_name, args, base_dir, algo=None):
    if algo is None:
        algo = args.algo

    extra_infos = {
        "ppo": "",
        "maml": "",
        "tt": "_rank%u_%s" % (args.tt_rank, args.tt_kernel),
        "mt": "",
        "modular": "_reg%.2f" % (args.marginal_reg),
        "latent": "",
        "bc": "",
        "bc_single": "",
    }
    snapshot_dir = base_dir + '%s_%s_%u%s' % (env_name, algo, args.run, extra_infos[algo])
    return snapshot_dir

def get_algo_fn(algo, env, trajs, args, continuous=False):
    if algo == "ppo":
        PolicyClass = GaussianMLPPolicy if continuous else DiscreteMLPPolicy

        policy = PolicyClass(env.spec,
                    hidden_sizes=[64, 64],
                    hidden_nonlinearity=th.tanh,
                    output_nonlinearity=None)

        value_function = GaussianMLPValueFunction(env_spec=env.spec,
                    hidden_sizes=[64, 64],
                    hidden_nonlinearity=th.tanh,
                    output_nonlinearity=None)

        algo_fn = PPO(env_spec=env.spec,
                    policy=policy,
                    value_function=value_function,
                    discount=1.00,
                    policy_ent_coeff=0.0,
                    entropy_method="regularized",
                    center_adv=False)

    elif algo == "maml":
        PolicyClass = GaussianMLPPolicy if continuous else DiscreteMLPPolicy

        policy = PolicyClass(env.spec,
                    hidden_sizes=[64, 64],
                    hidden_nonlinearity=th.tanh,
                    output_nonlinearity=None)

        policy.set_task = types.MethodType( lambda self, task_idx: None, policy )

        algo_fn = MAMLBC(env=env,
                    policy=policy,
                    batch_size=500,
                    sources=trajs,
                    source_type="trajectory", # "policy" or "trajectory"
                    inner_lr=0.1,
                    outer_lr=1e-3,
                    meta_batch_size=3,)

    elif algo == "tt":
        policy = TTPolicy(env.spec,
                    num_tasks=env.num_tasks(),
                    rank=args.tt_rank,
                    kernel=args.tt_kernel,
                    continuous=continuous,
                    hidden_sizes=[64, 64],
                    hidden_nonlinearity=th.tanh,
                    output_nonlinearity=None)

        algo_fn = MTBC(env=env,
                    learner=policy,
                    batch_size=1000,
                    sources=trajs,
                    source_type="trajectory", # "policy" or "trajectory"
                    max_path_length=500,
                    policy_lr=1e-3)

    elif algo == "mt":
        policy = MTPolicy(env_spec=env.spec, num_tasks=env.num_tasks(), continuous=continuous,
                    hidden_sizes=[64, 64],
                    hidden_nonlinearity=th.tanh,
                    output_nonlinearity=None)

        algo_fn = MTBC(env=env,
                    learner=policy,
                    batch_size=1000,
                    sources=trajs,
                    source_type="trajectory", # "policy" or "trajectory"
                    max_path_length=500,
                    policy_lr=1e-3)

    elif algo == "modular":
        policy = ModularPolicy(env.spec, num_tasks=env.num_tasks(), continuous=continuous,
                    hidden_sizes=[64, 64],
                    hidden_nonlinearity=th.tanh,
                    output_nonlinearity=None)

        algo_fn = MTBC(env=env,
                    learner=policy,
                    batch_size=1000,
                    sources=trajs,
                    source_type="trajectory", # "policy" or "trajectory"
                    max_path_length=500,
                    policy_lr=1e-3,
                    marginal_reg=args.marginal_reg,
                    )

    elif algo == "latent":
        policy = LatentPolicy(env.spec, num_tasks=env.num_tasks(), continuous=continuous,
                    hidden_sizes=[64, 64],
                    hidden_nonlinearity=th.tanh,
                    output_nonlinearity=None)

        algo_fn = MTBC(env=env,
                    learner=policy,
                    batch_size=1000,
                    sources=trajs,
                    source_type="trajectory", # "policy" or "trajectory"
                    max_path_length=500,
                    policy_lr=1e-3,
                    )
    elif algo == "bc" or algo == "bc_single":
        PolicyClass = GaussianMLPPolicy if continuous else DiscreteMLPPolicy

        policy = PolicyClass(env.spec,
                    hidden_sizes=[64, 64],
                    hidden_nonlinearity=th.tanh,
                    output_nonlinearity=None)

        algo_fn = MTBC(env=env,
                    learner=policy,
                    batch_size=1000,
                    sources=trajs,
                    source_type="trajectory", # "policy" or "trajectory"
                    max_path_length=500,
                    policy_lr=1e-3)
    else:
        raise Exception("Algo %s not supported" % algo)

    return algo_fn, policy

def evaluate(policy, env):
    num_episodes = 20
    rewards = []
    for game in range(num_episodes):
        #print("ep %u" % game)
        obs = env.reset()
        done = False
        reward = 0

        while not done:
            action, _ = policy.get_action(obs)
            obs, newreward, done, _ = env.step(action)
            reward += newreward

        rewards.append(reward)

    env.close()

    avg_rew = np.mean(rewards)
    std_rew = np.std(rewards)
    print(f"Average Reward: {avg_rew}")
    print(f"Standard Deviation: {std_rew}")

    return avg_rew, std_rew


def load_policy(load_dir, algo):

    loaded_policy = Snapshotter().load(load_dir)
    print("Loaded policy from %s" % load_dir)

    if algo == "maml":      policy = loaded_policy['algo']._policy
    else:                   policy = loaded_policy['algo'].learner

    return policy
