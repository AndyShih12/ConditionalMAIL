import numpy as np
import argparse

from garage import wrap_experiment
from garage.envs import GarageEnv
from garage.experiment import LocalRunner
from garage.experiment.deterministic import set_seed
from garage.experiment.experiment import ExperimentContext
from garage.experiment.snapshotter import Snapshotter

from conditionalMAIL.MTBCTest import MTBCTest
from envs.overcooked.overcooked_utils import getOvercookedMultiEnv
from envs.overcooked.overcooked_traj_utils import get_single_timestepbatches, split_timestepbatch
from common import get_dirname, get_algo_fn, evaluate, load_policy

ENV_NAME = 'overcooked'
BASE_DIR = './output/%s/' % ENV_NAME

def get_trajs(env_spec, mode, layout_name):
    tsb_per_partner = get_single_timestepbatches(env_spec=env_spec, mode=mode, layout_name=layout_name) # [timestepbatch] of size num_partner
    print(len(tsb_per_partner))

    tsbs_per_partner = [split_timestepbatch(tsb, num_splits=20) for tsb in tsb_per_partner] # [[timestepbatch]] of size (num_partner, num_split)
    trajs = [tsbs.__iter__() for tsbs in tsbs_per_partner] # List of Generator obj

    p0_trajs, p1_trajs = trajs[::2], trajs[1::2]

    return p0_trajs, p1_trajs, trajs

@wrap_experiment(log_dir=BASE_DIR, archive_launch_repo=False)
def bc(ctxt, snapshot_dir, seed, traj_idx):
    ctxt = ExperimentContext(snapshot_dir=snapshot_dir, snapshot_mode="last", snapshot_gap=10)
    set_seed(seed)

    env = GarageEnv(getOvercookedMultiEnv(other_agents=[None, None], rew_shape=ARGS.rew_shape, layout_name=ARGS.layout))

    _, _, trajs = get_trajs(env_spec=env.spec, mode=ARGS.mode, layout_name=ARGS.layout) # List of Generator obj
    trajs = trajs[2*traj_idx:2*(traj_idx+1)]
    algo_fn, policy = get_algo_fn(ARGS.algo, env, trajs, ARGS)

    runner = LocalRunner(ctxt)
    runner.setup(algo_fn, env)
    runner.train(n_epochs=TRAIN_EPOCHS * env.num_tasks(), batch_size=TRAIN_BATCH)

@wrap_experiment(log_dir=BASE_DIR, archive_launch_repo=False)
def bc_single(ctxt, snapshot_dir, seed, traj_idx):
    ctxt = ExperimentContext(snapshot_dir=snapshot_dir, snapshot_mode="last", snapshot_gap=10)
    set_seed(seed)

    env = GarageEnv(getOvercookedMultiEnv(other_agents=[None], rew_shape=ARGS.rew_shape, layout_name=ARGS.layout))

    _, _, trajs = get_trajs(env_spec=env.spec, mode=ARGS.mode, layout_name=ARGS.layout) # List of Generator obj
    trajs = trajs[traj_idx:traj_idx+1]
    algo_fn, policy = get_algo_fn(ARGS.algo, env, trajs, ARGS)

    runner = LocalRunner(ctxt)
    runner.setup(algo_fn, env)
    runner.train(n_epochs=TRAIN_EPOCHS * env.num_tasks(), batch_size=TRAIN_BATCH)

@wrap_experiment(log_dir=BASE_DIR, archive_launch_repo=False)
def mtbc(ctxt, snapshot_dir, seed):
    ctxt = ExperimentContext(snapshot_dir=snapshot_dir, snapshot_mode="last", snapshot_gap=10)
    set_seed(seed)

    other_agents_placeholder = [None for _ in range(2 * TRAJ_CNT[ARGS.mode][ARGS.layout])]
    env = GarageEnv(getOvercookedMultiEnv(other_agents=other_agents_placeholder, rew_shape=ARGS.rew_shape, layout_name=ARGS.layout))

    _, _, trajs = get_trajs(env_spec=env.spec, mode=ARGS.mode, layout_name=ARGS.layout) # List of Generator obj
    algo_fn, policy = get_algo_fn(ARGS.algo, env, trajs, ARGS)

    runner = LocalRunner(ctxt)
    runner.setup(algo_fn, env)
    runner.train(n_epochs=TRAIN_EPOCHS * env.num_tasks(), batch_size=TRAIN_BATCH)

@wrap_experiment(log_dir=BASE_DIR, archive_launch_repo=False)
def mtbc_test(ctxt, snapshot_dir, load_dir, seed):
    ctxt = ExperimentContext(snapshot_dir=snapshot_dir, snapshot_mode="none", snapshot_gap=10)
    set_seed(seed)

    other_agents_placeholder = [None for _ in range(TRAJ_CNT[ARGS.mode][ARGS.layout])]
    env_placeholder = GarageEnv(getOvercookedMultiEnv(other_agents=[None], rew_shape=ARGS.rew_shape, layout_name=ARGS.layout))
    p0_trajs, p1_trajs, _ = get_trajs(env_spec=env_placeholder.spec, mode=ARGS.mode, layout_name=ARGS.layout) # List of Generator obj

    before_avg_rews, before_std_rews = [], []
    after_avg_rews, after_std_rews = [], []

    for idx, other_agent in enumerate(other_agents_placeholder):
        env = GarageEnv(getOvercookedMultiEnv(other_agents=[other_agent], rew_shape=ARGS.rew_shape, layout_name=ARGS.layout))

        loaded_policy = Snapshotter().load(load_dir)
        print("Loaded policy from %s" % load_dir)

        if ARGS.algo == "maml":      policy = loaded_policy['algo']._policy
        else:                        policy = loaded_policy['algo'].learner

        if ARGS.algo in ['tt', 'modular', 'latent']: policy.setup_for_testing(train_p_only=ARGS.adapt_partial)

        algo_fn = MTBCTest(env=env,
                learner=policy,
                batch_size=TEST_BATCH,
                finetune_sources=[p0_trajs[idx]],
                evaluate_sources=[p1_trajs[idx]],
                source_type="trajectory",
                max_path_length=500,
                policy_lr=ARGS.lr,
                minibatches_per_epoch=10,
                update_per_minibatch=ARGS.update_per,
                )

        bc_snapshot_dir = get_dirname(env_name=ENV_NAME + '_%s' % ARGS.layout, args=ARGS, base_dir=BASE_DIR, algo='bc_single')
        other_agent_policy = load_policy( "%s_%u_test" % (bc_snapshot_dir, 2*idx+1), algo='bc_single' )
        eval_env = GarageEnv(getOvercookedMultiEnv(other_agents=[other_agent_policy], rew_shape=ARGS.rew_shape, layout_name=ARGS.layout))
        avg_rew, std_rew = evaluate(policy, eval_env)
        before_avg_rews.append(avg_rew)
        before_std_rews.append(std_rew)

        runner = LocalRunner(ctxt)
        runner.setup(algo_fn, env)
        runner.train(n_epochs=TEST_EPOCHS, batch_size=TEST_BATCH)

        bc_snapshot_dir = get_dirname(env_name=ENV_NAME + '_%s' % ARGS.layout, args=ARGS, base_dir=BASE_DIR, algo='bc_single')
        other_agent_policy = load_policy( "%s_%u_test" % (bc_snapshot_dir, 2*idx+1), algo='bc_single' )
        eval_env = GarageEnv(getOvercookedMultiEnv(other_agents=[other_agent_policy], rew_shape=ARGS.rew_shape, layout_name=ARGS.layout))
        avg_rew, std_rew = evaluate(policy, eval_env)
        after_avg_rews.append(avg_rew)
        after_std_rews.append(std_rew)

    print('#BeforeRew: ', np.mean(before_avg_rews), np.mean(before_std_rews))
    print('#AfterRew: ', np.mean(after_avg_rews), np.mean(after_std_rews))

def main():
    snapshot_dir = get_dirname(env_name=ENV_NAME + '_%s' % ARGS.layout, args=ARGS, base_dir=BASE_DIR, algo=ARGS.algo)
    print(snapshot_dir)

    if ARGS.algo == 'bc_single':
        for traj_idx in range(2*TRAJ_CNT[ARGS.mode][ARGS.layout]):
            cur_snapshot_dir = "%s_%u_%s" % (snapshot_dir, traj_idx, ARGS.mode)
            bc_single(snapshot_dir=cur_snapshot_dir, seed=ARGS.run, traj_idx=traj_idx)

        for traj_idx in range(TRAJ_CNT[ARGS.mode][ARGS.layout]):
            policy0 = load_policy( "%s_%u_%s" % (snapshot_dir, 2*traj_idx, ARGS.mode), algo=ARGS.algo )
            policy1 = load_policy( "%s_%u_%s" % (snapshot_dir, 2*traj_idx + 1, ARGS.mode), algo=ARGS.algo )
            env = GarageEnv(getOvercookedMultiEnv(other_agents=[policy1], rew_shape=ARGS.rew_shape, layout_name=ARGS.layout))
            evaluate(policy0, env)
        return

    if ARGS.algo == 'bc':
        for traj_idx in range(TRAJ_CNT[ARGS.mode][ARGS.layout]):
            cur_snapshot_dir = "%s_%u" % (snapshot_dir, traj_idx)
            if ARGS.mode == "train":
                bc(snapshot_dir=cur_snapshot_dir, seed=ARGS.run, traj_idx=traj_idx)
                evaluate(load_dir=cur_snapshot_dir)
            elif ARGS.mode == "test":
                policy = load_policy(cur_snapshot_dir, algo=ARGS.algo)
                env = GarageEnv(getOvercookedMultiEnv(other_agents=[policy], rew_shape=ARGS.rew_shape, layout_name=ARGS.layout))
                evaluate(policy, env)
        return

    if ARGS.mode == "train":
        mtbc(snapshot_dir=snapshot_dir, seed=ARGS.run)
    elif ARGS.mode == "test":
        test_snapshot_dir = snapshot_dir + "_test"
        mtbc_test(snapshot_dir=test_snapshot_dir, load_dir=snapshot_dir, seed=ARGS.run)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run',               type=int, default=0,     help="Run ID. In case you want to run replicates")
    parser.add_argument('--layout',            type=str, choices=['simple', 'unident_s', 'random1', 'random0', 'random3'], required=True, help="layout name")
    parser.add_argument('--no_rew_shape',      action='store_false', default=True, dest="rew_shape", help="reward shaping params (for easier training)")

    parser.add_argument('--algo',              type=str, choices=['maml', 'tt', 'mt', 'modular', 'latent', 'bc', 'bc_single'], required=True, help="which part of the algo to run")
    parser.add_argument('--mode',              type=str, choices=['train', 'test'], required=True, help="train or test")

    # algo-specific args
    parser.add_argument('--tt_rank',           type=int, default=4,     help="Tensortrain rank")
    parser.add_argument('--tt_kernel',         type=str, choices=['rbf', 'lin'], default='lin', help="RBF or linear kernel for Tensortrain.")
    parser.add_argument('--marginal_reg',      type=float, default=0.5,     help="Marginal reg")
    parser.add_argument('--adapt_partial',     action='store_true',      help="For TT/Modular: only tweak parts of the policy")

    parser.add_argument('--test_batch',        type=int, default=1000,     help="")
    parser.add_argument('--lr',                type=float, default=1e-3,     help="")
    parser.add_argument('--update_per',        type=int, default=3,     help="")

    ARGS = parser.parse_args()
    print(ARGS)

    TRAIN_EPOCHS = 30
    TRAIN_BATCH = 4000

    TEST_EPOCHS = 1
    TEST_BATCH = ARGS.test_batch

    TRAJ_CNT = {
        "train": { # Joint count, which is half of single counts.
            'simple': 8,
            'unident_s': 9,
            'random1': 8,
            'random0': 6,
            'random3': 8,
        },
        "test": { # Joint count, which is half of single counts.
            'simple': 8,
            'unident_s': 8,
            'random1': 8,
            'random0': 6,
            'random3': 7,
        }
    }

    main()
