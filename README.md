# COMP4900 Project: Exploring Multi-Agent Coordination Under Physical Degradation in VMAS Football 

Hasan Fakih (101168868)

This repository contains modifications to [BenchMARL](https://github.com/facebookresearch/BenchMARL) and [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator) to study how multi-agent reinforcement learning policies adapt to dynamic agent capability changes through an injury system.

## Overview

I extended the VMAS Football environment with an **injury feature** that randomly impairs agents at episode reset, reducing their movement speed. This creates a more realistic and challenging training scenario where agents must:

1. **Adapt individually** when injured (reduced speed)
2. **Coordinate as a team** to compensate for impaired teammates
3. **Learn robust policies** that generalize across different injury configurations

I also added **separate actor/critic learning rates** to BenchMARL for improved training stability.

## Repository Structure

```
COMP4900-Project/
├── BenchMARL/
│   ├── base_experiment.yaml    # Experiment config with separate actor/critic lr and parameters I used
│   ├── experiment.py           # Modified to support per-component learning rates
│   ├── football.yaml           # Task config with injury parameters
│   └── football.py             # TaskConfig dataclass with injury fields
│
└── VMAS/
    ├── football.py                    # Modified football scenario with injuries
    └── vmas_football_modified.diff    # Git diff vs original VMAS
```

## Features Added

### 1. Injury System (VMAS)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `injury_probability` | float | 0.0 | Probability (0-1) that an agent gets injured at reset |
| `injury_speed_penalty` | float | 0.5 | Speed multiplier for injured agents (1.0 = no penalty) |
| `injury_obs_mode` | str | "none" | Observation mode: `"none"`, `"self"`, or `"all"` |

**Observation Modes:**
- `"none"`: No injury info in observations
- `"self"`: Each agent sees its own injury status (+1 obs dimension)
- `"all"`: Each agent sees all agents' injury statuses (+N obs dimensions)

### 2. Separate Learning Rates (BenchMARL)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `actor_lr` | float | null | Learning rate for actor/policy networks |
| `critic_lr` | float | null | Learning rate for critic/value networks |
| `alpha_lr` | float | null | Learning rate for entropy coefficient (SAC) |

If set to `null`, falls back to the global `lr` parameter.

## Installation

### Prerequisites

- **Python 3.10** (tested version)

```bash
# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install PyTorch
pip install torch torchvision

# Install TorchRL
pip install torchrl
```

### Install Modified VMAS

```bash
# Clone original VMAS
git clone https://github.com/proroklab/VectorizedMultiAgentSimulator.git
cd VectorizedMultiAgentSimulator

# Apply my modifications
cp /path/to/COMP4900-Project/VMAS/football.py vmas/scenarios/football.py

# Install in editable mode
pip install -e .
cd ..
```

### Install Modified BenchMARL

```bash
# Clone original BenchMARL
git clone https://github.com/facebookresearch/BenchMARL.git
cd BenchMARL

# Apply my modifications
cp /path/to/COMP4900-Project/BenchMARL/experiment.py benchmarl/experiment/experiment.py
cp /path/to/COMP4900-Project/BenchMARL/base_experiment.yaml benchmarl/conf/experiment/base_experiment.yaml
cp /path/to/COMP4900-Project/BenchMARL/football.yaml benchmarl/conf/task/vmas/football.yaml
cp /path/to/COMP4900-Project/BenchMARL/football.py benchmarl/environments/vmas/football.py

# Install in editable mode
pip install -e .
```

## Reproducing Experiments

### Basic Training Run

```bash
python benchmarl/run.py algorithm=mappo task=vmas/football
```

### Training with Injuries

```bash
python benchmarl/run.py algorithm=mappo task=vmas/football \
  task.injury_probability=0.3 \
  task.injury_speed_penalty=0.5 \
  task.injury_obs_mode=self
```

### Separate Actor/Critic Learning Rates

```bash
python benchmarl/run.py algorithm=mappo task=vmas/football \
  experiment.actor_lr=3e-4 \
  experiment.critic_lr=1e-3
```

### Benchmark: Injury Ablation Study

Run experiments across different injury configurations:

```bash
python benchmarl/run.py -m \
  algorithm=mappo \
  task=vmas/football \
  task.injury_probability=0.0,0.2,0.5,1.0 \
  task.injury_obs_mode=none,self,all \
  seed=0,1,2
```

### Full Experiment (As Run for Paper)

```bash
python benchmarl/run.py -m \
  algorithm=mappo \
  task=vmas/football \
  task.n_blue_agents=3 \
  task.n_red_agents=3 \
  task.ai_red_agents=True \
  task.ai_strength=0.5 \
  task.injury_probability=0.0,0.3,0.5,1.0 \
  task.injury_speed_penalty=0.5 \
  task.injury_obs_mode=none,self,all \
  experiment.actor_lr=3e-4 \
  experiment.critic_lr=6e-4 \
  experiment.max_n_frames=5_000_000 \
  experiment.gamma=0.999 \
  seed=0,1,2
```

### Evaluate a Trained Checkpoint

```bash
python benchmarl/evaluate.py /path/to/checkpoint.pt
```

### Resume Training from Checkpoint

```bash
python benchmarl/resume.py /path/to/checkpoint.pt
```

## Key Configuration Files

### `football.yaml` (Task Config)

```yaml
# Agents
n_blue_agents: 3
n_red_agents: 3
ai_red_agents: True  # Red team uses heuristic AI

# AI opponent difficulty
ai_strength: 0.5
ai_decision_strength: 0.5
ai_precision_strength: 0.5

# Injuries
injury_probability: 0.3
injury_speed_penalty: 0.5
injury_obs_mode: "self"

# Rewards
dense_reward: True
pos_shaping_factor_ball_goal: 10.0
pos_shaping_factor_agent_ball: 0.1
scoring_reward: 100.0
```

### `base_experiment.yaml` (Training Config)

```yaml
# Devices
train_device: "cuda"
sampling_device: "cuda"

# Learning rates
lr: 3e-4
actor_lr: 3e-4    # Separate actor lr
critic_lr: 6e-4   # Separate critic lr

# Training
gamma: 0.999
max_n_frames: 10_000_000
on_policy_collected_frames_per_batch: 60_000
on_policy_n_minibatch_iters: 10
on_policy_minibatch_size: 4096

# Logging
loggers: [wandb]
project_name: "benchmarl"
```

## Acknowledgments

- [BenchMARL](https://github.com/facebookresearch/BenchMARL) - Meta AI
- [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator) - Prorok Lab
- [TorchRL](https://github.com/pytorch/rl) - PyTorch

## License

This project extends MIT-licensed software. See original repositories for license details.
