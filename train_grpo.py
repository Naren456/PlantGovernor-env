#!/usr/bin/env python3
"""
PlantGuard-Meta — GRPO Training Script
========================================
Trains a Qwen 2.5 0.5B model with GRPO (Group Relative Policy Optimization)
to act as an industrial plant manager in the PlantGuard-Meta environment.

Hardware: RTX 4050 6GB VRAM (or Colab free tier)
Stack:    Unsloth + TRL + 4-bit QLoRA

Usage (local):
    pip install unsloth trl datasets
    python train_grpo.py

Usage (Colab):
    !pip install unsloth trl datasets
    # Then run cells from this script

The script:
  1. Builds a training dataset from the AI4I 2020 sensor readings
  2. Formats prompts asking the model to choose an action + reasoning
  3. Uses a multi-component reward function matching the environment
  4. Trains with GRPO for improved action selection and reasoning
"""

import json
import os
import re
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset

# ─── Unsloth fast loading ───────────────────────────────────────────────────
from unsloth import FastLanguageModel

# ─── TRL GRPO ───────────────────────────────────────────────────────────────
from trl import GRPOConfig, GRPOTrainer


# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = "unsloth/Qwen2.5-0.5B-Instruct"  # Small enough for 6GB VRAM
MAX_SEQ_LENGTH = 1024
LORA_RANK = 16       # LoRA rank (keep small for 6GB)
LORA_ALPHA = 32
OUTPUT_DIR = "./plantguard-grpo-output"
NUM_TRAIN_SAMPLES = 500   # Number of training prompts
NUM_GENERATIONS = 4        # Completions per prompt for GRPO
TRAINING_STEPS = 150       # Total training steps
BATCH_SIZE = 1             # Per-device batch size (small for 6GB)
GRAD_ACCUM = 4             # Effective batch = 1 * 4 = 4
LEARNING_RATE = 5e-6
MAX_NEW_TOKENS = 300       # Max tokens for model generation

# Reward constants (matching environment)
REASONING_KEYWORDS = {
    "temperature": 50,
    "budget": 50,
    "cascade": 150,
    "trade-off": 100,
    "spare": 80,
    "downtime": 60,
}
REASONING_LENGTH_BONUS = 50
MAX_REASONING_SCORE = 350


# =============================================================================
# 1. Load Dataset & Build Training Prompts
# =============================================================================

def load_sensor_data() -> pd.DataFrame:
    """Load the AI4I 2020 dataset from local or parent directories."""
    search_paths = [
        Path("ai4i2020.csv"),
        Path("plant_governor_env/server/ai4i2020.csv"),
        Path("../ai4i2020.csv"),
    ]
    for p in search_paths:
        if p.exists():
            df = pd.read_csv(p, encoding="utf-8-sig")
            df.columns = df.columns.str.strip()
            return df
    raise FileNotFoundError("ai4i2020.csv not found in any expected location")


SYSTEM_PROMPT = """You are PlantGuard-Meta, an AI plant operations manager.

You manage an industrial plant for a 720-hour shift. Your goal: survive without cascade failure while managing a $10,000 budget.

## Sensors (each step)
- air_temp: Air temperature (K), normal ~300
- process_temp: Process temperature (K), normal ~310
- rotational_speed: RPM, normal ~1500
- torque: Nm, normal ~40
- tool_wear: wear in minutes (0-253), higher = more risk
- shift_hour: current hour (0-23)
- remaining_budget: dollars left

## Tools
1. "run_diagnostic" - $200, check machine health
2. "adjust_load" - $0, reduce production (set load_reduction 0.1-0.9)
3. "dispatch_repair" - $1500 ($0 with spare), fixes machine
4. "order_spare_part" - $100, stock spare (max 1)
5. "do_nothing" - $0, monitor

## Key Rules
- Machine failure without dispatch_repair = CASCADE (game over, -5000)
- High torque (>50 Nm) + high tool wear (>150 min) = HIGH RISK
- Complete 720 steps = +1000 bonus
- Budget management is critical: repairs cost $1500, spares save money later
- Order spares BEFORE you need them
- Your reasoning is scored: mention temperature, budget, cascade, trade-off, spare, downtime

Respond with ONLY valid JSON:
{"tool": "<tool_name>", "reasoning": "<detailed_reasoning>", "load_reduction": null}"""


def build_prompt(row: pd.Series, step: int, budget: float,
                 spare: bool, last_action: str) -> str:
    """Build a user prompt from a sensor data row."""
    return f"""Step {step}/720 | Sensors:
- Air temp: {row['Air temperature [K]']:.1f} K
- Process temp: {row['Process temperature [K]']:.1f} K
- Rotational speed: {row['Rotational speed [rpm]']:.0f} RPM
- Torque: {row['Torque [Nm]']:.1f} Nm
- Tool wear: {row['Tool wear [min]']:.0f} min
- Shift hour: {step % 24}
- Budget: ${budget:.0f}
- Spare available: {'yes' if spare else 'no'}
- Last action: {last_action}

Choose your action. Respond with ONLY a JSON object."""


def create_training_dataset(df: pd.DataFrame, n_samples: int) -> Dataset:
    """
    Create training prompts from real sensor data.

    Strategy: over-sample high-risk scenarios (high torque, high wear,
    near-failure rows) so the model sees more interesting states.
    """
    prompts = []
    rng = random.Random(42)

    # Categorize rows by risk level
    high_risk = df[
        (df["Torque [Nm]"] > 50) | (df["Tool wear [min]"] > 150) |
        (df["Machine failure"] == 1)
    ].index.tolist()
    normal = df[
        ~df.index.isin(high_risk)
    ].index.tolist()

    for i in range(n_samples):
        # 60% high-risk, 40% normal for training diversity
        if rng.random() < 0.6 and high_risk:
            idx = rng.choice(high_risk)
        else:
            idx = rng.choice(normal)

        row = df.iloc[idx]
        step = rng.randint(0, 700)
        budget = rng.choice([10000, 9800, 9500, 9000, 8000, 7000, 5000, 3000])
        spare = rng.random() < 0.3
        last_action = rng.choice([
            "do_nothing", "adjust_load", "run_diagnostic",
            "order_spare_part", "dispatch_repair"
        ])

        user_prompt = build_prompt(row, step, budget, spare, last_action)

        # Format as chat messages for the model
        prompt_text = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

        prompts.append({
            "prompt": prompt_text,
            # Store context for reward computation
            "torque": float(row["Torque [Nm]"]),
            "tool_wear": float(row["Tool wear [min]"]),
            "air_temp": float(row["Air temperature [K]"]),
            "machine_failure": int(row["Machine failure"]),
            "budget": budget,
            "spare": spare,
            "step": step,
        })

    return Dataset.from_list(prompts)


# =============================================================================
# 2. Reward Function
# =============================================================================

def score_reasoning(reasoning: str) -> float:
    """Score reasoning text for domain-relevant keywords (0-350)."""
    text = reasoning.lower()
    score = 0
    for keyword, value in REASONING_KEYWORDS.items():
        if keyword in text:
            score += value
    if len(reasoning.split()) > 30:
        score += REASONING_LENGTH_BONUS
    return min(score, MAX_REASONING_SCORE)


def compute_reward(completion: str, torque: float, tool_wear: float,
                   machine_failure: int, budget: float, spare: bool,
                   **kwargs) -> float:
    """
    Multi-component reward function matching the environment.

    Components:
      1. Format reward: valid JSON gets bonus
      2. Action appropriateness: right tool for the situation
      3. Reasoning quality: keyword scoring
      4. Budget awareness: don't overspend
    """
    reward = 0.0

    # ── Parse the completion ───────────────────────────────────────────
    text = completion.strip()
    # Extract JSON from possible markdown wrapping
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        text = text[start:end]

    try:
        data = json.loads(text)
        tool = data.get("tool", "")
        reasoning = data.get("reasoning", "")
    except (json.JSONDecodeError, ValueError):
        # Invalid JSON: heavy penalty
        return -200.0

    # ── 1. Format reward (+50 for valid JSON with correct fields) ─────
    valid_tools = {"run_diagnostic", "adjust_load", "dispatch_repair",
                   "order_spare_part", "do_nothing"}
    if tool in valid_tools and isinstance(reasoning, str) and len(reasoning) > 0:
        reward += 50.0
    elif tool in valid_tools:
        reward += 20.0
    else:
        reward -= 100.0  # Invalid tool name
        return reward

    # ── 2. Action appropriateness ─────────────────────────────────────
    is_high_risk = torque > 50 or tool_wear > 150
    is_failure = machine_failure == 1
    is_low_budget = budget < 2000

    if is_failure:
        # Machine is failing — dispatch_repair is the BEST action
        if tool == "dispatch_repair":
            reward += 500.0  # Critical save
        elif tool == "run_diagnostic":
            reward += 50.0   # At least checking
        elif tool == "adjust_load":
            reward += 20.0   # Helps but won't prevent cascade
        elif tool == "do_nothing":
            reward -= 300.0  # Disaster — should have acted
        elif tool == "order_spare_part":
            reward -= 100.0  # Too late for spares

    elif is_high_risk:
        # High risk — proactive maintenance is good
        if tool == "dispatch_repair":
            reward += 200.0
        elif tool == "run_diagnostic":
            reward += 150.0
        elif tool == "order_spare_part" and not spare:
            reward += 100.0  # Good planning
        elif tool == "adjust_load":
            reward += 80.0
        elif tool == "do_nothing":
            reward -= 50.0   # Risky inaction

    else:
        # Normal conditions — conservative is fine
        if tool == "do_nothing":
            reward += 30.0
        elif tool == "adjust_load":
            reward += 20.0
        elif tool == "order_spare_part" and not spare:
            reward += 60.0   # Proactive spare ordering
        elif tool == "run_diagnostic":
            reward += 10.0
        elif tool == "dispatch_repair":
            reward -= 30.0   # Wasting money on unnecessary repair

    # Budget awareness
    cost_map = {"run_diagnostic": 200, "dispatch_repair": 1500,
                "order_spare_part": 100, "adjust_load": 0, "do_nothing": 0}
    cost = cost_map.get(tool, 0)
    if tool == "dispatch_repair" and spare:
        cost = 0  # Free with spare

    if is_low_budget and cost > budget:
        reward -= 100.0  # Can't afford this

    # ── 3. Reasoning quality (0-350) ──────────────────────────────────
    reasoning_score = score_reasoning(reasoning)
    # Scale reasoning to be meaningful but not dominant
    reward += reasoning_score * 0.5  # Max +175

    # ── 4. Reasoning length bonus ─────────────────────────────────────
    word_count = len(reasoning.split())
    if 20 < word_count < 80:
        reward += 20.0  # Good length
    elif word_count > 80:
        reward += 5.0   # Too verbose
    # Very short reasoning gets nothing extra

    return reward


def reward_function(prompts, completions, **kwargs):
    """
    Batch reward function for GRPO trainer.

    Args:
        prompts: List of prompt strings
        completions: List of completion strings (model outputs)
        **kwargs: Additional context from the dataset (torque, tool_wear, etc.)

    Returns:
        List of float rewards
    """
    rewards = []
    for i, completion in enumerate(completions):
        # Extract the completion text
        if isinstance(completion, list):
            # Handle chat format
            comp_text = completion[-1]["content"] if completion else ""
        elif isinstance(completion, dict):
            comp_text = completion.get("content", str(completion))
        else:
            comp_text = str(completion)

        r = compute_reward(
            completion=comp_text,
            torque=kwargs.get("torque", [40.0])[i] if isinstance(kwargs.get("torque"), list) else kwargs.get("torque", 40.0),
            tool_wear=kwargs.get("tool_wear", [100.0])[i] if isinstance(kwargs.get("tool_wear"), list) else kwargs.get("tool_wear", 100.0),
            machine_failure=kwargs.get("machine_failure", [0])[i] if isinstance(kwargs.get("machine_failure"), list) else kwargs.get("machine_failure", 0),
            budget=kwargs.get("budget", [10000])[i] if isinstance(kwargs.get("budget"), list) else kwargs.get("budget", 10000),
            spare=kwargs.get("spare", [False])[i] if isinstance(kwargs.get("spare"), list) else kwargs.get("spare", False),
        )
        rewards.append(float(r))

    return rewards


# =============================================================================
# 3. Main Training Loop
# =============================================================================

def main():
    print("=" * 70)
    print("  PlantGuard-Meta — GRPO Training with Unsloth")
    print("=" * 70)

    # ── Check GPU ──────────────────────────────────────────────────────
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"  GPU: {gpu} ({mem:.1f} GB)")
    else:
        print("  ⚠️  No GPU detected — training will be very slow!")

    # ── 1. Load model with Unsloth ─────────────────────────────────────
    print(f"\n📦 Loading model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect
        load_in_4bit=True,  # QLoRA for 6GB VRAM
    )

    # ── 2. Add LoRA adapters ───────────────────────────────────────────
    print(f"🔧 Adding LoRA adapters (rank={LORA_RANK})")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # ── 3. Build training dataset ──────────────────────────────────────
    print(f"\n📊 Building training dataset ({NUM_TRAIN_SAMPLES} prompts)")
    df = load_sensor_data()
    print(f"   Loaded {len(df)} rows from AI4I 2020 dataset")
    print(f"   Failure rate: {df['Machine failure'].mean()*100:.1f}%")

    dataset = create_training_dataset(df, NUM_TRAIN_SAMPLES)
    print(f"   Created {len(dataset)} training prompts")
    print(f"   Sample prompt (first 200 chars):")
    print(f"   {dataset[0]['prompt'][:200]}...")

    # ── 4. Configure GRPO trainer ──────────────────────────────────────
    print(f"\n⚙️  Configuring GRPO trainer")
    print(f"   Batch size: {BATCH_SIZE} × {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
    print(f"   Generations per prompt: {NUM_GENERATIONS}")
    print(f"   Training steps: {TRAINING_STEPS}")
    print(f"   Learning rate: {LEARNING_RATE}")

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        max_steps=TRAINING_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        fp16=not (torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False),
        max_completion_length=MAX_NEW_TOKENS,
        num_generations=NUM_GENERATIONS,
        # GRPO specific
        beta=0.04,  # KL penalty coefficient
        report_to="none",  # Set to "wandb" if you have wandb
        seed=42,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_function,
    )

    # ── 5. Train ───────────────────────────────────────────────────────
    print(f"\n🚀 Starting GRPO training...")
    print(f"   This will take ~{TRAINING_STEPS * 2}–{TRAINING_STEPS * 5} minutes on RTX 4050")
    print("-" * 70)

    trainer.train()

    # ── 6. Save model ─────────────────────────────────────────────────
    print(f"\n💾 Saving model to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Also save as merged 16-bit for inference
    merged_dir = f"{OUTPUT_DIR}-merged"
    print(f"💾 Saving merged model to {merged_dir}")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")

    print("\n" + "=" * 70)
    print("  ✅ Training complete!")
    print(f"  LoRA adapter: {OUTPUT_DIR}")
    print(f"  Merged model: {merged_dir}")
    print("=" * 70)

    # ── 7. Quick inference test ────────────────────────────────────────
    print("\n🧪 Quick inference test (post-training)...")
    FastLanguageModel.for_inference(model)

    test_prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nStep 100/720 | Sensors:\n- Air temp: 302.5 K\n- Process temp: 312.0 K\n- Rotational speed: 1400 RPM\n- Torque: 58.3 Nm\n- Tool wear: 180 min\n- Shift hour: 4\n- Budget: $8000\n- Spare available: yes\n- Last action: run_diagnostic\n\nChoose your action. Respond with ONLY a JSON object.<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            do_sample=True,
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\n  High-risk scenario response:")
    print(f"  {response[:300]}")

    # Score it
    r = compute_reward(response, torque=58.3, tool_wear=180,
                       machine_failure=0, budget=8000, spare=True)
    print(f"\n  Reward score: {r:+.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
