# RL Cloud Resource Allocation (SAC + Streamlit Dashboard)

A reinforcement learning agent (Soft Actor-Critic) that learns to dynamically allocate cloud workers in a simulated queueing environment. The agent balances throughput, cost, SLA compliance, and latency while coping with realistic autoscaling delays.

## Features

- **Simulated cloud environment** with Poisson job arrivals, queueing, SLA tracking, and autoscaling delays
- **Workload scenarios**: steady, burst, and spike traffic patterns
- **SAC training** with parallel envs, VecNormalize, TensorBoard, checkpoints, and periodic evaluation
- **Baseline policies**: fixed workers, threshold autoscaling, random
- **Streamlit dashboard** for training metrics, policy comparison, and live simulation
- **Unit tests** and Docker support

## Installation

```bash
git clone https://github.com/backpropBoi/rl-cloud-resource-allocator.git
cd rl-cloud-resource-allocator
pip install -e ".[dev]"
```

Or with requirements only:

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick start

### 1. Train the agent

```bash
python -m src.train
```

Options:

```bash
python -m src.train --timesteps 500000 --n-envs 4 --workload burst --seed 42
python -m src.train --config config/default.yaml
```

Training logs go to `./logs/training_log.csv`. TensorBoard logs go to `./logs/tensorboard/`.

```bash
tensorboard --logdir logs/tensorboard
```

### 2. Evaluate against baselines

```bash
python -m src.evaluate --model ./models/sac_cloud_final.zip --episodes 10 --workload steady
python -m src.evaluate --no-model   # baselines only
```

Results are saved to `./logs/eval_results_<workload>.json` and `.csv`.

### 3. Launch the dashboard

```bash
streamlit run app/streamlit_dashboard.py
```

### 4. Run tests

```bash
pytest tests/ -v
```

### 5. Docker

```bash
docker build -t rl-cloud-allocator .
docker run -p 8501:8501 rl-cloud-allocator
```

## Configuration

Edit `config/default.yaml` to tune environment, reward weights, and training hyperparameters:

| Section | Key settings |
|---------|-------------|
| `env` | workers, arrival rate, episode length, scale delays, workload |
| `reward` | throughput, cost, SLA, latency, overflow, scaling weights |
| `training` | timesteps, learning rate, n_envs, eval/checkpoint frequency |

## Project structure

```
config/default.yaml      # Hyperparameters and paths
src/
  envs/cloud_env.py      # Gymnasium environment
  baselines.py           # Rule-based policies
  train.py               # SAC training script
  evaluate.py            # Evaluation vs baselines
  callbacks.py           # Training CSV logging
  config.py              # YAML loader
app/streamlit_dashboard.py
tests/test_cloud_env.py
```

## Environment details

- **Action**: continuous value in `[0, 1]` → desired worker count
- **Observation**: queue length, demand, workers, throughput, queue delta, utilization, pending scale, time
- **Reward**: throughput − resource cost − SLA penalties − latency − overflow − scaling cost
- **Autoscaling**: scale-up delay and scale-down cooldown mimic real cloud behavior

## License

MIT
