from garage.envs import GarageEnv
from garage.sampler import OnPolicyVectorizedSampler
from garage import TimeStepBatch
import numpy as np

from envs.mab.mab_env import MABEnv

def getMabEnv(other_agents, enlarge_a=1):
    N = 10
    mask = np.zeros((N,N))
    mask[ 0:4, 0:4 ] = 1
    mask[ 4:8, 4:8 ] = 1
    mask[ 8:10, 8:10 ] = 1

    mask = np.repeat(mask, enlarge_a, axis=-1) # Enlarge A

    env = MABEnv(n=N,
                 a=N,
                 mask=mask,
                 other_agents=other_agents,
                 enlarge_a=enlarge_a
                 )
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
        traj = TimeStepBatch.from_time_step_list(env_spec=env.spec, ts_samples=path)
        trajs.append(traj)
    sampler.shutdown_worker()

    return trajs.__iter__()