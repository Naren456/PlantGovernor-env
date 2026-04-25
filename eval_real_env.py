#!/usr/bin/env python3
"""
PlantGuard-Meta — Real Environment Evaluation
===============================================
Evaluates a trained model (or baseline) against the REAL deployed
OpenEnv environment on HuggingFace Spaces.

Shows before/after comparison: untrained vs GRPO-trained model.

Usage:
    python eval_real_env.py                          # Evaluate with Ollama baseline
    python eval_real_env.py --adapter ./plantguard-grpo-output  # Evaluate trained model
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

from plant_governor_env.client import PlantGovernorClient
from plant_governor_env.models import PlantAction

# ─── Config ──────────────────────────────────────────────────────────────────
HF_SPACE_URL = "https://narendra78-plant-governor-env.hf.space"
EVAL_SEEDS = [10, 28, 43, 63, 69]  # Seeds with varied failure patterns
MAX_STEPS = 50  # Per episode for evaluation


SYSTEM_PROMPT = """You are PlantGuard-Meta, an AI plant operations manager.
You manage an industrial plant for a 720-hour shift. Your goal: survive without cascade failure while managing a $10,000 budget.

## Tools
1. "run_diagnostic" - $200, check machine health
2. "adjust_load" - $0, reduce production (set load_reduction 0.1-0.9)
3. "dispatch_repair" - $1500 ($0 with spare), fixes machine
4. "order_spare_part" - $100, stock spare (max 1)
5. "do_nothing" - $0, monitor

## Key Rules
- High torque (>50 Nm) + high tool wear (>150 min) = HIGH RISK → dispatch_repair or order_spare_part
- Normal conditions → do_nothing or adjust_load
- Order spares BEFORE you need them
- Budget is limited: $10,000 total

Respond with ONLY valid JSON:
{"tool": "<tool_name>", "reasoning": "<detailed_reasoning>", "load_reduction": null}"""


def build_obs_prompt(obs, step, last_action):
    return f"""Step {step}/720 | Sensors:
- Air temp: {obs.air_temp:.1f} K
- Process temp: {obs.process_temp:.1f} K
- Rotational speed: {obs.rotational_speed:.0f} RPM
- Torque: {obs.torque:.1f} Nm
- Tool wear: {obs.tool_wear:.0f} min
- Shift hour: {obs.shift_hour}
- Budget: ${obs.remaining_budget:.0f}
- Last action: {last_action or 'none'}

Choose your action. Respond with ONLY a JSON object."""


def parse_response(text):
    """Parse model response into PlantAction."""
    text = text.strip()
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
        tool = data.get("tool", "do_nothing")
        reasoning = data.get("reasoning", "") or ""
        load_reduction = data.get("load_reduction")

        valid_tools = {"run_diagnostic", "adjust_load", "dispatch_repair",
                       "order_spare_part", "do_nothing"}
        if tool not in valid_tools:
            tool = "do_nothing"

        if tool == "adjust_load" and load_reduction is not None:
            load_reduction = max(0.1, min(0.9, float(load_reduction)))
        else:
            load_reduction = None

        return PlantAction(tool=tool, reasoning=reasoning, load_reduction=load_reduction)
    except Exception:
        return PlantAction(tool="do_nothing", reasoning="Parse error fallback")


async def run_episode_ollama(seed, model_name="qwen2.5:7b"):
    """Run one episode with Ollama model."""
    import ollama

    async with PlantGovernorClient(base_url=HF_SPACE_URL) as client:
        result = await client.reset(seed=seed)
        obs = result.observation
        total_reward = 0.0
        step = 0
        last_action = None
        action_counts = {}

        while not result.done and step < MAX_STEPS:
            prompt = build_obs_prompt(obs, step, last_action)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            try:
                response = ollama.chat(
                    model=model_name,
                    messages=messages,
                    options={"temperature": 0.7, "num_predict": 200},
                )
                action = parse_response(response["message"]["content"])
            except Exception as e:
                action = PlantAction(tool="do_nothing", reasoning=f"Error: {e}")

            result = await client.step(action)
            obs = result.observation
            reward = result.reward or 0.0
            total_reward += reward
            last_action = action.tool
            action_counts[action.tool] = action_counts.get(action.tool, 0) + 1
            step += 1

        state = await client.state()
        return {
            "seed": seed,
            "steps": state.step_count,
            "cascade": state.cascade_occurred,
            "complete": state.shift_complete,
            "total_reward": total_reward,
            "avg_reward": total_reward / max(step, 1),
            "budget_remaining": state.budget_remaining,
            "actions": action_counts,
        }


async def run_episode_trained(seed, model, tokenizer):
    """Run one episode with a trained Unsloth model."""
    import torch

    async with PlantGovernorClient(base_url=HF_SPACE_URL) as client:
        result = await client.reset(seed=seed)
        obs = result.observation
        total_reward = 0.0
        step = 0
        last_action = None
        action_counts = {}

        while not result.done and step < MAX_STEPS:
            prompt = build_obs_prompt(obs, step, last_action)
            full_prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

            inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                )
            response_text = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            action = parse_response(response_text)

            result = await client.step(action)
            obs = result.observation
            reward = result.reward or 0.0
            total_reward += reward
            last_action = action.tool
            action_counts[action.tool] = action_counts.get(action.tool, 0) + 1
            step += 1

        state = await client.state()
        return {
            "seed": seed,
            "steps": state.step_count,
            "cascade": state.cascade_occurred,
            "complete": state.shift_complete,
            "total_reward": total_reward,
            "avg_reward": total_reward / max(step, 1),
            "budget_remaining": state.budget_remaining,
            "actions": action_counts,
        }


async def run_episode_random(seed):
    """Run one episode with random actions."""
    import random

    async with PlantGovernorClient(base_url=HF_SPACE_URL) as client:
        result = await client.reset(seed=seed)
        total_reward = 0.0
        step = 0
        tools = ["run_diagnostic", "adjust_load", "dispatch_repair",
                 "order_spare_part", "do_nothing"]

        while not result.done and step < MAX_STEPS:
            tool = random.choice(tools)
            action = PlantAction(tool=tool, reasoning="random baseline action")
            result = await client.step(action)
            total_reward += result.reward or 0.0
            step += 1

        state = await client.state()
        return {
            "seed": seed,
            "steps": state.step_count,
            "cascade": state.cascade_occurred,
            "total_reward": total_reward,
            "avg_reward": total_reward / max(step, 1),
        }


def print_results_table(name, results):
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")
    print(f"  {'Seed':>6}  {'Steps':>5}  {'Cascade':>8}  {'Reward':>10}  {'Avg/Step':>8}")
    print(f"  {'─'*6}  {'─'*5}  {'─'*8}  {'─'*10}  {'─'*8}")
    for r in results:
        cascade = "🔴 YES" if r["cascade"] else "🟢 No"
        print(f"  {r['seed']:>6}  {r['steps']:>5}  {cascade:>8}  {r['total_reward']:>+10.1f}  {r['avg_reward']:>+8.2f}")

    avg_reward = sum(r["total_reward"] for r in results) / len(results)
    cascades = sum(1 for r in results if r["cascade"])
    print(f"\n  Summary: Avg reward = {avg_reward:+.1f} | Cascades = {cascades}/{len(results)}")


async def main():
    parser = argparse.ArgumentParser(description="Evaluate on real PlantGuard-Meta environment")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to trained LoRA adapter (e.g. ./plantguard-grpo-output)")
    parser.add_argument("--seeds", type=int, nargs="+", default=EVAL_SEEDS)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = parser.parse_args()

    global MAX_STEPS
    MAX_STEPS = args.max_steps

    print("=" * 60)
    print("  PlantGuard-Meta — Real Environment Evaluation")
    print(f"  Environment: {HF_SPACE_URL}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Max steps per episode: {MAX_STEPS}")
    print("=" * 60)

    # ── Random baseline ────────────────────────────────────────────
    print("\n🎲 Running random baseline...")
    random_results = []
    for seed in args.seeds:
        r = await run_episode_random(seed)
        random_results.append(r)
        print(f"  Seed {seed}: reward={r['total_reward']:+.1f}, cascade={r['cascade']}")
    print_results_table("RANDOM BASELINE", random_results)

    # ── Ollama baseline ────────────────────────────────────────────
    try:
        import ollama
        ollama.list()
        print("\n🤖 Running Ollama (qwen2.5:7b) baseline...")
        ollama_results = []
        for seed in args.seeds:
            r = await run_episode_ollama(seed)
            ollama_results.append(r)
            print(f"  Seed {seed}: reward={r['total_reward']:+.1f}, cascade={r['cascade']}, actions={r['actions']}")
        print_results_table("OLLAMA qwen2.5:7b (UNTRAINED)", ollama_results)
    except Exception as e:
        print(f"\n⚠️  Ollama not available: {e}")
        ollama_results = None

    # ── Trained model ──────────────────────────────────────────────
    if args.adapter and Path(args.adapter).exists():
        print(f"\n🎯 Running trained model from {args.adapter}...")
        import torch
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.adapter,
            max_seq_length=1024,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)

        trained_results = []
        for seed in args.seeds:
            r = await run_episode_trained(seed, model, tokenizer)
            trained_results.append(r)
            print(f"  Seed {seed}: reward={r['total_reward']:+.1f}, cascade={r['cascade']}, actions={r['actions']}")
        print_results_table("GRPO-TRAINED MODEL", trained_results)
    else:
        if args.adapter:
            print(f"\n⚠️  Adapter path not found: {args.adapter}")
        trained_results = None

    # ── Final comparison ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  COMPARISON SUMMARY")
    print("=" * 60)
    avg_random = sum(r["total_reward"] for r in random_results) / len(random_results)
    print(f"  Random baseline:    {avg_random:+.1f} avg reward")
    if ollama_results:
        avg_ollama = sum(r["total_reward"] for r in ollama_results) / len(ollama_results)
        print(f"  Ollama (untrained): {avg_ollama:+.1f} avg reward ({avg_ollama/max(abs(avg_random),0.1):+.1f}x vs random)")
    if trained_results:
        avg_trained = sum(r["total_reward"] for r in trained_results) / len(trained_results)
        print(f"  GRPO-trained:       {avg_trained:+.1f} avg reward ({avg_trained/max(abs(avg_random),0.1):+.1f}x vs random)")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
