#!/usr/bin/env python3
"""
Generate the 4 required judging plots from `episode_log.csv`.

Input CSV columns expected (from `train_online_ppo.py`):
- episode,total_reward,complete,cascade

Outputs:
- reward_curve.png
- success_rate.png
- baseline_comparison.png

Note: reasoning_improvement.png requires logging reasoning scores during training;
this repo logs environment reward + outcomes by default. If you add a reasoning
score to the CSV, this script will also render reasoning_improvement.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def rolling_mean(s, w: int):
    return s.rolling(window=w, min_periods=max(1, w // 4)).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="episode_log.csv")
    parser.add_argument("--window", type=int, default=10)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)
    df = df.sort_values("episode")

    import matplotlib.pyplot as plt

    # 1) Reward curve
    plt.figure(figsize=(8, 4))
    plt.plot(df["episode"], df["total_reward"], alpha=0.35, label="episode reward")
    plt.plot(df["episode"], rolling_mean(df["total_reward"], args.window), label=f"moving avg (w={args.window})")
    plt.xlabel("episode")
    plt.ylabel("total reward")
    plt.title("PlantGuard-Meta training reward curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reward_curve.png", dpi=180)
    plt.close()

    # 2) Success rate (rolling)
    if "complete" in df.columns:
        plt.figure(figsize=(8, 4))
        complete = df["complete"].astype(int)
        plt.plot(df["episode"], rolling_mean(complete, args.window))
        plt.xlabel("episode")
        plt.ylabel("rolling success rate")
        plt.title("Shift completion rate (rolling)")
        plt.tight_layout()
        plt.savefig("success_rate.png", dpi=180)
        plt.close()

    # 3) Baseline comparison (if baseline_reward exists)
    # If you want: run multiple policies and concatenate logs with a `policy` column.
    if "policy" in df.columns:
        plt.figure(figsize=(7, 4))
        means = df.groupby("policy")["total_reward"].mean().sort_values()
        stds = df.groupby("policy")["total_reward"].std().reindex(means.index)
        plt.bar(means.index, means.values, yerr=stds.values)
        plt.ylabel("avg total reward")
        plt.title("Baseline comparison (mean ± std)")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig("baseline_comparison.png", dpi=180)
        plt.close()

    # 4) Reasoning improvement (optional)
    if "reasoning_score" in df.columns:
        plt.figure(figsize=(8, 4))
        plt.plot(df["episode"], rolling_mean(df["reasoning_score"], args.window))
        plt.xlabel("episode")
        plt.ylabel("avg reasoning score (rolling)")
        plt.title("Reasoning improvement (rolling)")
        plt.tight_layout()
        plt.savefig("reasoning_improvement.png", dpi=180)
        plt.close()

    print("Wrote plots: reward_curve.png, success_rate.png (and optional others).")


if __name__ == "__main__":
    main()

