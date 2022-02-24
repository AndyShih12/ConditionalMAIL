This repository contains code for the paper:

[Conditional Imitation Learning for Multi-Agent Games](https://arxiv.org/abs/2201.01448)

```
"Conditional Imitation Learning for Multi-Agent Games"
Andy Shih, Stefano Ermon, Dorsa Sadigh
17th ACM/IEEE International Conference on Human-Robot Interaction (HRI 2022)

@inproceedings{ShihEShri22,
  author    = {Andy Shih and Stefano Ermon and Dorsa Sadigh},
  title     = {Conditional Imitation Learning for Multi-Agent Games},
  booktitle = {17th ACM/IEEE International Conference on Human-Robot Interaction (HRI)},
  month     = {march},
  year      = {2022},
  keywords  = {conference}
}
```

## Installation

```
conda create -n ConditionalMAIL python=3.7
conda activate ConditionalMAIL
pip install -e .
```

## Commands

#### MAB
```
for r in {0..19}
do
    python run_mabai.py --run=${r} --algo=ppo --mode=train
done
python run_mabai.py --run=0 --algo=tt --mode=train
python run_mabai.py --run=0 --algo=tt --mode=test
```

#### PARTICLE
```
for r in {0..19}
do
    python run_particle.py --run=${r} --algo=ppo --mode=train
done
python run_particle.py --run=0 --algo=tt --mode=train
python run_particle.py --run=0 --algo=tt --mode=test
```

#### HANABI
Setup installation by running `bash install.sh hanabi`
```
for r in {0..19}
do
    python run_hanabi.py --run=${r} --algo=ppo --mode=train
done
python run_hanabi.py --run=0 --algo=tt --mode=train
python run_hanabi.py --run=0 --algo=tt --mode=test
```

#### OVERCOOKED
Setup installation by running `bash install.sh overcooked`
```
python run_overcooked.py --run=0 --layout=simple --algo=bc_single --mode=test
python run_overcooked.py --run=0 --layout=simple --algo=tt --mode=train
python run_overcooked.py --run=0 --layout=simple --algo=tt --mode=test
```
