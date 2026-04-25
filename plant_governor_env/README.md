---
title: PlantGuard-Meta Environment
emoji: 🏭
colorFrom: green
colorTo: red
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - rl-environment
  - predictive-maintenance
---

# 🏭 PlantGuard-Meta: Industrial Plant Management RL Environment

An OpenEnv-compliant reinforcement learning environment for training LLMs to manage industrial plant operations. The agent must prevent cascade failures while managing a $10,000 maintenance budget and spare parts inventory across a 720-step shift (30 days × 24 hours).

**Theme:** #3.1 Professional Tasks — Industrial Plant Management

## Problem Statement

Industrial plants face cascading failures that cost millions. Current LLM agents lack the ability to make sequential, budget-constrained maintenance decisions under uncertainty. PlantGuard-Meta teaches LLMs to:

- **Read sensor data** and reason about failure risk
- **Manage budgets** with cost/benefit trade-offs
- **Order spare parts** proactively for future repairs
- **Dispatch repairs** at the right moment to prevent cascades
- **Reason explicitly** about temperature, torque, wear, and downtime

## Dataset: AI4I 2020 Predictive Maintenance

We use the [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) (10,000 rows) with real industrial sensor data.

### Key Statistics

| Metric | Value |
|--------|-------|
| Total samples | 10,000 |
| Failure rate | 3.39% (339 failures) |
| Sensors | 5 (air temp, process temp, RPM, torque, tool wear) |
| Product types | L (60%), M (30%), H (10%) |

### Failure Type Breakdown

| Type | Count | Rate | Description |
|------|-------|------|-------------|
| HDF | 115 | 1.15% | Heat Dissipation Failure |
| OSF | 98 | 0.98% | Overstrain Failure |
| PWF | 95 | 0.95% | Power Failure |
| TWF | 46 | 0.46% | Tool Wear Failure |
| RNF | 19 | 0.19% | Random Failure |

### What Predicts Failure?

| Sensor | Correlation | Failure Mean vs Normal |
|--------|-------------|----------------------|
| **Torque** | +0.191 (strongest) | 50.2 Nm vs 39.6 Nm (+26.7%) |
| **Tool Wear** | +0.105 | 143.8 min vs 106.7 min (+34.7%) |
| **Air Temperature** | +0.083 | 300.9 K vs 300.0 K |
| **Rotational Speed** | −0.044 | 1496 RPM vs 1540 RPM |
| **Process Temperature** | +0.036 | 310.3 K vs 310.0 K |

**Key insight:** High torque + high tool wear is the strongest failure predictor. Low-quality products (Type L) fail at 3.92% vs 2.09% for high-quality (Type H). Failures can cluster — median gap between failures is only 11.5 rows.

## Environment Design

### Observation Space (Memory-Constrained)

The agent receives **only the current step's data** — no past failures, no action history, no historical sensor readings. This forces the agent to develop robust reasoning from a single snapshot.

```python
{
    "air_temp": 300.1,          # Kelvin (295–304)
    "process_temp": 310.0,      # Kelvin (305–313)
    "rotational_speed": 1503.0, # RPM (1168–2886)
    "torque": 40.1,             # Nm (3.8–76.6)
    "tool_wear": 108.0,         # minutes (0–253)
    "shift_hour": 14,           # hour within shift (0–23)
    "remaining_budget": 9800.0  # dollars (0–10,000)
}
```

### Action Space (5 Tools)

| Tool | Cost | Effect |
|------|------|--------|
| `run_diagnostic` | $200 | Assess machine health |
| `adjust_load` | $0 | Reduce production intensity (0.1–0.9) |
| `dispatch_repair` | $1,500 ($0 with spare) | Fix machine, prevent cascade |
| `order_spare_part` | $100 | Stock one spare (max 1) |
| `do_nothing` | $0 | Monitor and wait |

### Reward Function (Multi-Component)

| Component | Value | Condition |
|-----------|-------|-----------|
| **Outcome** | +1,000 | Full 720-step shift completed |
| **Outcome** | −5,000 | Cascade failure occurred |
| **Survival** | +1 | Per step survived (always) |
| **Redundancy** | −5 | Same action repeated without state change |
| **Reasoning** | 0–350 | Keyword scoring on reasoning trace |
| **Spare Used** | +500 | Spare ordered AND used in repair (episode end) |
| **Spare Wasted** | −100 | Spare ordered but never used (episode end) |

**Reasoning Scoring:** The agent's free-text reasoning earns bonus reward for mentioning domain-relevant concepts:

- "cascade" → +150 | "trade-off" → +100 | "spare" → +80
- "downtime" → +60 | "temperature" → +50 | "budget" → +50
- Length > 30 words → +50 | **Cap: 350**

## Quick Start

### Connect to a Running Server

```python
import asyncio
from plant_governor_env import PlantGovernorClient, PlantAction

async def play():
    async with PlantGovernorClient(base_url="http://localhost:8000") as client:
        result = await client.reset(seed=42)
        print(f"Budget: ${result.observation.remaining_budget}")

        action = PlantAction(
            tool="run_diagnostic",
            reasoning="Checking temperature levels for cascade risk"
        )
        result = await client.step(action)
        print(f"Reward: {result.reward}, Budget: ${result.observation.remaining_budget}")

asyncio.run(play())
```

### Sync Usage

```python
from plant_governor_env import PlantGovernorClient, PlantAction

sync_client = PlantGovernorClient(base_url="http://localhost:8000").sync()
with sync_client:
    result = sync_client.reset()
    result = sync_client.step(PlantAction(tool="do_nothing", reasoning="Observing"))
```

## Running Locally

```bash
# Install dependencies
pip install -e .

# Start the server
uvicorn plant_governor_env.server.app:app --host 0.0.0.0 --port 8000

# Or use uv
uv run --project . server
```

## Building the Docker Image

```bash
docker build -t plantguard-meta:latest -f server/Dockerfile .
```

## Deploying to Hugging Face Spaces

```bash
openenv push
```

## Baseline Results

| Agent | Steps | Cascade | Total Reward | Avg/Step |
|-------|-------|---------|-------------|----------|
| Random | 30 | No | +15.0 | +0.5 |
| Qwen 2.5 7B (Ollama, zero-shot) | 30 | No | +1,680.0 | +56.0 |

The untrained Qwen 2.5 earns **112× more reward** than random, mostly through reasoning bonuses. However, it uses `adjust_load` 96% of the time and never orders spares or runs diagnostics strategically — demonstrating clear room for RL improvement.

## Training & Evidence Artifacts (Judging)

This repo includes a **re-runnable, end-to-end online rollout logger** against the deployed Hugging Face Space. It writes an episode-level CSV and produces the required plots.

### 1) Collect episode logs from the deployed Space

```bash
python -m plant_governor_env.train_online_ppo \
  --base-url "https://narendra78-plant-governor-env.hf.space" \
  --episodes 100 \
  --max-steps 720 \
  --out episode_log.csv
```

Optionally, if you have Ollama running locally:

```bash
python -m plant_governor_env.train_online_ppo \
  --base-url "https://narendra78-plant-governor-env.hf.space" \
  --episodes 50 \
  --max-steps 200 \
  --ollama-model "qwen2.5:7b" \
  --out episode_log.csv
```

### 2) Generate plots from `episode_log.csv`

```bash
python -m plant_governor_env.plot_episode_log --csv episode_log.csv
```

This writes:
- `reward_curve.png`
- `success_rate.png`
- `baseline_comparison.png` (only if your CSV includes a `policy` column)
- `reasoning_improvement.png` (only if your CSV includes a `reasoning_score` column)

## Project Structure

```
plant_governor_env/
├── __init__.py            # Package exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project config
├── models.py              # PlantAction, PlantObservation, PlantState
├── client.py              # PlantGovernorClient (EnvClient subclass)
└── server/
    ├── __init__.py
    ├── plant_governor_env_environment.py  # Core environment logic
    ├── app.py             # FastAPI application
    ├── requirements.txt   # Dependencies
    ├── ai4i2020.csv       # Dataset
    └── Dockerfile         # Container image
```

## References

- [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
- [OpenEnv Framework](https://github.com/meta-pytorch/OpenEnv)
- [TRL (Transformer Reinforcement Learning)](https://huggingface.co/docs/trl)
