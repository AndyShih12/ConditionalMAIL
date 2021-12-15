#!/bin/bash

if [ "$1" == "hananbi" ]; then
    git clone https://github.com/rlworkgroup/garage.git
    cd garage/
    git checkout v2020.06.0
    sed -i 's/cloudpickle.loads(cloudpickle.dumps(self.env))//g' src/garage/sampler/on_policy_vectorized_sampler.py 
    sed -i 's/for _ in range(n_envs)/self.env/g' src/garage/sampler/on_policy_vectorized_sampler.py 
    pip install -e .
    cd ../
elif [ "$1" == "overcooked" ]; then
    cd envs/overcooked/
    git clone -b neurips2019 --recursive https://github.com/HumanCompatibleAI/human_aware_rl.git
    cd human_aware_rl/
    pip install -e .
    cd overcooked_ai/
    pip install -e .
    cd ../../../../
    pip install numpy==1.19.0
else
    echo "Argument must be 'hanabi' or 'overcooked'"
fi