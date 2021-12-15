## Installation

```
conda create -n ConditionalMAIL python=3.7
conda activate ConditionalMAIL
pip install -e .
```

## Commands

#### MAB
```
for r from 0 to 19 do:
    python run_mabai.py --run=${r} --algo=ppo --mode=train
python run_mabai.py --run=0 --algo=tt --mode=train
python run_mabai.py --run=0 --algo=tt --mode=test
```

#### PARTICLE
```
for r from 0 to 19 do:
    python run_particle.py --run=${r} --algo=ppo --mode=train
python run_particle.py --run=0 --algo=tt --mode=train
python run_particle.py --run=0 --algo=tt --mode=test
```

#### HANABI
Setup installation by running `./install.sh hanabi`
```
for r from 0 to 19 do:
    python run_hanabi.py --run=${r} --algo=ppo --mode=train
python run_hanabi.py --run=0 --algo=tt --mode=train
python run_hanabi.py --run=0 --algo=tt --mode=test
```

#### OVERCOOKED
Setup installation by running `./install.sh overcooked`
```
python run_overcooked.py --run=0  --layout=single --algo=bc_single --mode=test 
python run_overcooked.py --run=0  --layout=single --algo=tt --mode=train
python run_overcooked.py --run=0 --layout=single --algo=tt --mode=test
```