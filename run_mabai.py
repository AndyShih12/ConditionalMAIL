import numpy as np
import argparse
import copy

from garage import wrap_experiment
from garage.experiment import LocalRunner
from garage.experiment.deterministic import set_seed
from garage.experiment.experiment import ExperimentContext
from garage.experiment.snapshotter import Snapshotter

from conditionalMAIL.MTBCTest import MTBCTest
from envs.mab.mab_utils import getMabEnv, collect_expert_trajectories
from common import get_dirname, get_algo_fn, evaluate

ENV_NAME = 'mabai'
BASE_DIR = './output/%s/' % ENV_NAME

def get_other_policies(mode, idx=None):
    if mode == "train":     rg = range(0,16)
    elif mode == "test":    rg = range(16,20)

    load_dirs = []
    for i in rg:
        args = copy.deepcopy(ARGS)
        args.algo, args.run = "ppo", i
        load_dir = get_dirname(env_name=ENV_NAME, args=args, base_dir=BASE_DIR)
        load_dirs.append(load_dir)

    if idx is not None:
        load_dirs = [load_dirs[idx]]

    other_agents = []

    for load_dir in load_dirs:
        loaded_policy = Snapshotter().load(load_dir)
        policy = loaded_policy['algo'].policy
        print("loaded other agent from %s" % load_dir)
        print(policy is not None)
        other_agents.append(policy)

    return other_agents


@wrap_experiment(log_dir=BASE_DIR, archive_launch_repo=False)
def ppo(ctxt, snapshot_dir, seed):
    ctxt = ExperimentContext(snapshot_dir=snapshot_dir, snapshot_mode="last", snapshot_gap=10)
    set_seed(seed)

    env = getMabEnv(other_agents=None)

    algo_fn, policy = get_algo_fn(ARGS.algo, env, None, ARGS)

    runner = LocalRunner(ctxt)
    runner.setup(algo_fn, env)
    runner.train(n_epochs=TRAIN_EPOCHS*2, batch_size=TRAIN_BATCH//10)

    env = getMabEnv(other_agents=[policy])

    runner = LocalRunner(ctxt)
    runner.setup(algo_fn, env)
    runner.train(n_epochs=TRAIN_EPOCHS*2, batch_size=TRAIN_BATCH//10)

@wrap_experiment(log_dir=BASE_DIR, archive_launch_repo=False)
def mtbc(ctxt, snapshot_dir, seed):
    ctxt = ExperimentContext(snapshot_dir=snapshot_dir, snapshot_mode="last", snapshot_gap=10)
    set_seed(seed)

    other_agents = get_other_policies(mode=ARGS.mode)
    env = getMabEnv(other_agents=other_agents)

    trajs = [collect_expert_trajectories(policy=other_agent, env=env, fullbatch=10000) for other_agent in other_agents]

    algo_fn, policy = get_algo_fn(ARGS.algo, env, trajs, ARGS)

    runner = LocalRunner(ctxt)
    runner.setup(algo_fn, env)
    runner.train(n_epochs=TRAIN_EPOCHS * env.num_tasks(), batch_size=TRAIN_BATCH)


@wrap_experiment(log_dir=BASE_DIR, archive_launch_repo=False)
def mtbc_test(ctxt, snapshot_dir, load_dir, seed):
    ctxt = ExperimentContext(snapshot_dir=snapshot_dir, snapshot_mode="none", snapshot_gap=10)
    set_seed(seed)

    other_agents = get_other_policies(mode=ARGS.mode)

    before_avg_rews, before_std_rews = [], []
    after_avg_rews, after_std_rews = [], []

    for idx, other_agent in enumerate(other_agents):
        env = getMabEnv(other_agents=[other_agent])
        trajs = [collect_expert_trajectories(policy=other_agent, env=env, fullbatch=10000)]

        loaded_policy = Snapshotter().load(load_dir)
        print("Loaded policy from %s" % load_dir)

        if ARGS.algo == "maml":      policy = loaded_policy['algo']._policy
        else:                        policy = loaded_policy['algo'].learner

        if ARGS.algo in ['tt', 'modular', 'latent']: policy.setup_for_testing(train_p_only=ARGS.adapt_partial) 

        algo_fn = MTBCTest(env=env,
                learner=policy,
                batch_size=TEST_BATCH,
                finetune_sources=trajs,
                evaluate_sources=trajs,
                source_type="trajectory",
                max_path_length=500,
                policy_lr=ARGS.lr,
                minibatches_per_epoch=16,
                )

        eval_env = env
        avg_rew, std_rew = evaluate(policy, eval_env)
        before_avg_rews.append(avg_rew)
        before_std_rews.append(std_rew)

        runner = LocalRunner(ctxt)
        runner.setup(algo_fn, env)
        runner.train(n_epochs=TEST_EPOCHS, batch_size=TEST_BATCH)

        eval_env = env
        avg_rew, std_rew = evaluate(policy, eval_env)
        after_avg_rews.append(avg_rew)
        after_std_rews.append(std_rew)

    print('#BeforeRew: ', np.mean(before_avg_rews), np.mean(before_std_rews))
    print('#AfterRew: ', np.mean(after_avg_rews), np.mean(after_std_rews))

def main():
    snapshot_dir = get_dirname(env_name=ENV_NAME, args=ARGS, base_dir=BASE_DIR)
    print(snapshot_dir)

    if ARGS.algo == 'ppo':
        ppo(snapshot_dir=snapshot_dir, seed=ARGS.run)
    elif ARGS.mode == "train":
        mtbc(snapshot_dir=snapshot_dir, seed=ARGS.run)
    elif ARGS.mode == "test":
        test_snapshot_dir = snapshot_dir + "_test"
        mtbc_test(snapshot_dir=test_snapshot_dir, load_dir=snapshot_dir, seed=ARGS.run)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run',               type=int, default=0,     help="Run ID. In case you want to run replicates")
    parser.add_argument('--algo',              type=str, choices=['ppo', 'maml', 'tt', 'mt', 'modular', 'latent'], required=True, help="which part of the algo to run")
    parser.add_argument('--mode',              type=str, choices=['train', 'test'], required=True, help="train or test")

    # algo-specific args
    parser.add_argument('--tt_rank',           type=int, default=4,     help="Tensortrain rank")
    parser.add_argument('--tt_kernel',         type=str, choices=['rbf', 'lin'], default='lin', help="RBF or linear kernel for Tensortrain.")
    parser.add_argument('--marginal_reg',      type=float, default=0.5,     help="Marginal reg")
    parser.add_argument('--adapt_partial',     action='store_true',      help="For TT/Modular: only tweak parts of the policy")

    parser.add_argument('--test_batch',        type=int, default=1000,     help="")
    parser.add_argument('--lr',                type=float, default=1e-3,     help="")

    ARGS = parser.parse_args()
    print(ARGS)

    TRAIN_EPOCHS = 10
    TRAIN_BATCH = 4000

    TEST_EPOCHS = 1
    TEST_BATCH = ARGS.test_batch

    main()
