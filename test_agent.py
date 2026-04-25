#!/usr/bin/env python3
"""
PlantGuard-Meta Local Test Script
==================================
Tests the environment end-to-end with a Qwen 2.5 agent via Ollama.

Usage:
    1. Start the server:   uvicorn plant_governor_env.server.app:app --port 8000
    2. Run this script:    python test_agent.py

The agent plays one full episode (up to 720 steps) making decisions based
on sensor readings and reasoning about cascade prevention, budget, and spares.
"""

import asyncio
import json
import sys
import time
from typing import Optional

import ollama

from plant_governor_env.client import PlantGovernorClient
from plant_governor_env.models import PlantAction

# ─── Configuration ──────────────────────────────────────────────────────────
OLLAMA_MODEL = "qwen2.5:7b"
SERVER_URL = "https://narendra78-plant-governor-env.hf.space"
MAX_STEPS = 30   # Cap for demo (full episode is 720)
PRINT_EVERY = 1  # Print every N steps


# ─── System Prompt ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are PlantGuard-Meta, an AI plant operations manager.

You manage an industrial plant for a 720-hour shift (30 days × 24 hours).
Your goal: survive the full shift without cascade failure while managing budget.

## Sensor Readings (provided each step)
- air_temp: Air temperature (K), normal ~300
- process_temp: Process temperature (K), normal ~310  
- rotational_speed: RPM, normal ~1500
- torque: Nm, normal ~40
- tool_wear: minutes of wear (0-253)
- shift_hour: current hour (0-23)
- remaining_budget: dollars left (starts at $10,000)

## Available Tools
1. "run_diagnostic" - $200, check machine health
2. "adjust_load" - $0, reduce production load (set load_reduction 0.1-0.9)
3. "dispatch_repair" - $1500 ($0 if spare available), fixes machine
4. "order_spare_part" - $100, stock one spare part (max 1)
5. "do_nothing" - $0, monitor and wait

## Rules
- If a machine fails and you didn't dispatch_repair, a CASCADE occurs (game over, -5000 penalty)
- Completing the full shift earns +1000
- Each step survived earns +1
- Repeating the same action without reason gets -5 penalty
- Your reasoning is scored: mention temperature, budget, cascade, trade-off, spare, downtime for bonus

## Response Format
You MUST respond with ONLY a valid JSON object, no other text:
{"tool": "<tool_name>", "reasoning": "<your reasoning>", "load_reduction": null}

Set load_reduction to a float (0.1-0.9) ONLY when using adjust_load, otherwise null.
"""


def build_user_prompt(obs_data: dict, step: int, last_action: Optional[str]) -> str:
    """Build the user prompt from observation data."""
    return f"""Step {step}/720 | Sensors:
- Air temp: {obs_data['air_temp']:.1f} K
- Process temp: {obs_data['process_temp']:.1f} K
- Rotational speed: {obs_data['rotational_speed']:.0f} RPM
- Torque: {obs_data['torque']:.1f} Nm
- Tool wear: {obs_data['tool_wear']:.0f} min
- Shift hour: {obs_data['shift_hour']}
- Budget remaining: ${obs_data['remaining_budget']:.0f}
- Last action: {last_action or 'none'}

Decide your next action. Respond with ONLY a JSON object."""


def parse_llm_response(response_text: str) -> PlantAction:
    """Parse LLM response into a PlantAction, with fallback."""
    text = response_text.strip()

    # Try to extract JSON from the response
    # Handle cases where LLM wraps in ```json ... ```
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    # Try to find JSON object in the text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        text = text[start:end]

    try:
        data = json.loads(text)
        tool = data.get("tool", "do_nothing")
        reasoning = data.get("reasoning", "")
        load_reduction = data.get("load_reduction")

        # Validate tool name
        valid_tools = ["run_diagnostic", "adjust_load", "dispatch_repair",
                       "order_spare_part", "do_nothing"]
        if tool not in valid_tools:
            tool = "do_nothing"
            reasoning = f"Invalid tool '{data.get('tool')}', defaulting to do_nothing. " + reasoning

        # Validate load_reduction
        if tool == "adjust_load" and load_reduction is not None:
            try:
                load_reduction = float(load_reduction)
                load_reduction = max(0.1, min(0.9, load_reduction))
            except (ValueError, TypeError):
                load_reduction = 0.5
        else:
            load_reduction = None

        return PlantAction(tool=tool, reasoning=reasoning, load_reduction=load_reduction)

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        # Fallback: do_nothing with the raw text as reasoning
        return PlantAction(
            tool="do_nothing",
            reasoning=f"Parse error ({e}), raw: {response_text[:100]}"
        )


def query_ollama(messages: list) -> str:
    """Query Ollama and return the response text."""
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        options={
            "temperature": 0.7,
            "num_predict": 300,
        },
    )
    return response["message"]["content"]


async def run_episode():
    """Run one full episode with the LLM agent."""
    print("=" * 70)
    print(f"  PlantGuard-Meta — Local Agent Test (Ollama: {OLLAMA_MODEL})")
    print("=" * 70)

    # ── Verify Ollama is running ────────────────────────────────────────
    try:
        ollama.list()
        print(f"✅ Ollama connected, using model: {OLLAMA_MODEL}")
    except Exception as e:
        print(f"❌ Ollama not available: {e}")
        print("   Start Ollama with: ollama serve")
        sys.exit(1)

    # ── Connect to environment server ──────────────────────────────────
    print(f"🔌 Connecting to environment at {SERVER_URL}...")

    async with PlantGovernorClient(base_url=SERVER_URL) as client:
        # Reset environment
        result = await client.reset(seed=10)
        obs = result.observation
        print(f"✅ Environment reset. Initial budget: ${obs.remaining_budget:.0f}")
        print("-" * 70)

        # ── Episode loop ───────────────────────────────────────────────
        total_reward = 0.0
        step = 0
        last_action = None
        action_counts = {}
        rewards_history = []

        start_time = time.time()

        while not result.done and step < MAX_STEPS:
            # Build observation dict for the prompt
            obs_data = {
                "air_temp": obs.air_temp,
                "process_temp": obs.process_temp,
                "rotational_speed": obs.rotational_speed,
                "torque": obs.torque,
                "tool_wear": obs.tool_wear,
                "shift_hour": obs.shift_hour,
                "remaining_budget": obs.remaining_budget,
            }

            # Query LLM
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(obs_data, step, last_action)},
            ]

            try:
                llm_response = query_ollama(messages)
                action = parse_llm_response(llm_response)
            except Exception as e:
                print(f"  ⚠️  LLM error at step {step}: {e}")
                action = PlantAction(tool="do_nothing", reasoning=f"LLM error: {e}")

            # Step the environment
            result = await client.step(action)
            obs = result.observation
            reward = result.reward or 0.0
            total_reward += reward

            # Track stats
            last_action = action.tool
            action_counts[action.tool] = action_counts.get(action.tool, 0) + 1
            rewards_history.append(reward)

            # Print step info
            if step % PRINT_EVERY == 0:
                status = "🔴 CASCADE!" if result.done and reward < -1000 else (
                    "🟢 COMPLETE!" if result.done else "⚡"
                )
                print(
                    f"  Step {step:3d} {status} | "
                    f"Tool: {action.tool:<18s} | "
                    f"Reward: {reward:+8.1f} | "
                    f"Budget: ${obs.remaining_budget:,.0f} | "
                    f"Temp: {obs.air_temp:.1f}K"
                )
                if len(action.reasoning) > 0:
                    # Show first 80 chars of reasoning
                    r = action.reasoning[:80] + ("..." if len(action.reasoning) > 80 else "")
                    print(f"         💭 {r}")

            step += 1

        elapsed = time.time() - start_time

        # ── Get final state ────────────────────────────────────────────
        state = await client.state()

        # ── Summary ────────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("  EPISODE SUMMARY")
        print("=" * 70)
        print(f"  Steps completed:   {state.step_count}")
        print(f"  Shift complete:    {'✅ Yes' if state.shift_complete else '❌ No'}")
        print(f"  Cascade occurred:  {'🔴 Yes' if state.cascade_occurred else '🟢 No'}")
        print(f"  Budget remaining:  ${state.budget_remaining:,.2f}")
        print(f"  Spare available:   {'🔧 Yes' if state.spare_available else '❌ No'}")
        print(f"  Total reward:      {total_reward:+,.1f}")
        print(f"  Avg reward/step:   {total_reward / max(step, 1):+.2f}")
        print(f"  Time elapsed:      {elapsed:.1f}s ({elapsed/max(step,1):.2f}s/step)")
        print()
        print("  Action Distribution:")
        for tool, count in sorted(action_counts.items(), key=lambda x: -x[1]):
            bar = "█" * (count * 40 // max(step, 1))
            print(f"    {tool:<20s} {count:3d} ({count*100//max(step,1):2d}%) {bar}")
        print()

        # Reward breakdown
        if rewards_history:
            print("  Reward Stats:")
            print(f"    Min:  {min(rewards_history):+.1f}")
            print(f"    Max:  {max(rewards_history):+.1f}")
            print(f"    Mean: {sum(rewards_history)/len(rewards_history):+.2f}")
        print("=" * 70)


# ─── Baseline comparison (random agent) ─────────────────────────────────────
async def run_random_baseline():
    """Run a random agent for comparison."""
    import random
    print("\n" + "=" * 70)
    print("  RANDOM BASELINE (for comparison)")
    print("=" * 70)

    async with PlantGovernorClient(base_url=SERVER_URL) as client:
        result = await client.reset(seed=10)  # Same seed for fair comparison
        total_reward = 0.0
        step = 0
        tools = ["run_diagnostic", "adjust_load", "dispatch_repair",
                 "order_spare_part", "do_nothing"]

        while not result.done and step < MAX_STEPS:
            tool = random.choice(tools)
            action = PlantAction(tool=tool, reasoning="random baseline")
            result = await client.step(action)
            total_reward += result.reward or 0.0
            step += 1

        state = await client.state()
        print(f"  Steps: {state.step_count} | "
              f"Complete: {state.shift_complete} | "
              f"Cascade: {state.cascade_occurred} | "
              f"Total reward: {total_reward:+,.1f}")
        print("=" * 70)


async def main():
    print("\n🏭 PlantGuard-Meta — Full Local Test\n")

    # Run random baseline first
    await run_random_baseline()

    # Run LLM agent
    await run_episode()


if __name__ == "__main__":
    asyncio.run(main())
