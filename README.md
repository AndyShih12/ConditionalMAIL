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