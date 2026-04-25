# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
PlantGuard-Meta Environment Implementation.

Industrial plant management environment using the AI4I 2020 Predictive
Maintenance dataset. Each episode is a 720-step (30 days × 24 hours) shift
where the agent must prevent cascade failures while managing budget and
spare parts.

Key design decisions:
  - Dataset loaded once in __init__, episodes pick random 720-row slices.
  - _get_observation() returns ONLY current step data (no history).
  - Multi-component reward: outcome + survival + redundancy + reasoning + spare.
  - Failure occurs when `Machine failure == 1` before the last step.
"""

import os
import random
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import pandas as pd
from openenv.core.env_server.interfaces import Environment

try:
    from ..models import PlantAction, PlantObservation, PlantState
except ImportError:
    from models import PlantAction, PlantObservation, PlantState


# =============================================================================
# Constants
# =============================================================================

EPISODE_LENGTH = 720           # 30 days × 24 hours
INITIAL_BUDGET = 10_000.0      # Starting budget in dollars
MAX_REASONING_SCORE = 350      # Cap on reasoning bonus

# Action costs
COST_DIAGNOSTIC = 200.0
COST_REPAIR = 1500.0
COST_SPARE = 100.0

# Downtime mechanics
DOWNTIME_HOURS_NO_SPARE = 2
REWARD_DOWNTIME_PER_HOUR = -10.0

# Reward components
REWARD_SHIFT_COMPLETE = 1000.0
REWARD_CASCADE = -5000.0
REWARD_SURVIVAL_PER_STEP = 1.0
REWARD_REDUNDANCY_PENALTY = -5.0
REWARD_SPARE_USED = 500.0
REWARD_SPARE_WASTED = -100.0

# Reasoning keyword scores
REASONING_KEYWORDS = {
    "temperature": 50,
    "budget": 50,
    "cascade": 150,
    "trade-off": 100,
    "spare": 80,
    "downtime": 60,
}
REASONING_LENGTH_BONUS = 50    # Bonus for > 30 words


# =============================================================================
# Dataset Loading
# =============================================================================

def _load_dataset() -> pd.DataFrame:
    """
    Load the AI4I 2020 dataset. If the CSV is not found, generate synthetic
    data with similar distributions (normal distributions, ~3% failure rate).

    Returns:
        DataFrame with columns: Air temperature [K], Process temperature [K],
        Rotational speed [rpm], Torque [Nm], Tool wear [min], Machine failure.
    """
    # Search for CSV in common locations
    search_paths = [
        Path(__file__).resolve().parent.parent.parent / "ai4i2020.csv",
        Path(__file__).resolve().parent.parent / "ai4i2020.csv",
        Path(__file__).resolve().parent / "ai4i2020.csv",
        Path(os.getcwd()) / "ai4i2020.csv",
    ]

    for csv_path in search_paths:
        if csv_path.exists():
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            # Normalise column names (handle BOM and extra whitespace)
            df.columns = df.columns.str.strip()
            required = [
                "Air temperature [K]",
                "Process temperature [K]",
                "Rotational speed [rpm]",
                "Torque [Nm]",
                "Tool wear [min]",
                "Machine failure",
            ]
            if all(c in df.columns for c in required):
                return df[required].reset_index(drop=True)

    # ── Synthetic fallback ──────────────────────────────────────────────
    rng = np.random.default_rng(seed=42)
    n = 10_000

    air_temp = rng.normal(loc=300.0, scale=2.0, size=n).clip(295, 304)
    process_temp = air_temp + rng.normal(loc=10.0, scale=1.0, size=n)
    process_temp = process_temp.clip(305, 313)
    rot_speed = rng.normal(loc=1539.0, scale=300.0, size=n).clip(1168, 2886)
    torque = rng.normal(loc=40.0, scale=10.0, size=n).clip(3.8, 76.6)
    tool_wear = rng.integers(0, 254, size=n).astype(float)
    # ~3% failure rate
    failure = (rng.random(n) < 0.03).astype(int)

    return pd.DataFrame(
        {
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": rot_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
            "Machine failure": failure,
        }
    )


# =============================================================================
# Reasoning Scorer
# =============================================================================

def _score_reasoning(reasoning: str) -> int:
    """
    Score a reasoning trace based on domain-relevant keyword mentions
    and length.

    Scoring (capped at 350):
      - "temperature" → +50
      - "budget"      → +50
      - "cascade"     → +150
      - "trade-off"   → +100
      - "spare"       → +80
      - "downtime"    → +60
      - length > 30 words → +50
    """
    text = reasoning.lower()
    score = 0

    for keyword, value in REASONING_KEYWORDS.items():
        if keyword in text:
            score += value

    if len(reasoning.split()) > 30:
        score += REASONING_LENGTH_BONUS

    return min(score, MAX_REASONING_SCORE)


# =============================================================================
# Environment
# =============================================================================

class PlantGovernorEnvironment(Environment):
    """
    PlantGuard-Meta: Industrial plant management environment.

    The agent manages a plant for a 720-step shift (30 days × 24 hours).
    It can run diagnostics, adjust load, dispatch repairs, order spare parts,
    or do nothing. Failure occurs when the dataset's "Machine failure" column
    reads 1 at the current step (before the final step).

    Reward is multi-component:
      outcome   : +1000 (shift complete) / -5000 (cascade) / 0
      survival  : +1 per step (always)
      redundancy: -5 if same action repeated with no state change
      reasoning : 0-350 from _score_reasoning()
      spare     : +500 if spare ordered and used / -100 if wasted (episode end)

    Memory constraint: _get_observation() returns ONLY current step data.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Load dataset and initialise episode state."""
        super().__init__()
        self._data: pd.DataFrame = _load_dataset()
        self._data_len = len(self._data)

        # ── Episode state (set properly in reset()) ─────────────────────
        self._episode_id: str = ""
        self._step: int = 0
        self._start_idx: int = 0
        self._budget: float = INITIAL_BUDGET
        self._spare_available: bool = False
        self._spare_ordered: bool = False
        self._spare_used: bool = False
        self._cascade_occurred: bool = False
        self._shift_complete: bool = False
        self._done: bool = False
        self._last_action: Optional[str] = None
        self._last_state_changed: bool = False  # Track if last action caused a state change
        self._downtime_remaining: int = 0

    # ─────────────────────────────────────────────────────────────────────
    # Reset
    # ─────────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> PlantObservation:
        """
        Reset the environment for a new episode.

        Picks a random contiguous slice of 720 rows from the dataset.
        Resets budget to $10,000, clears spare flags, step counter, and
        cascade state.

        Args:
            seed: Optional random seed for reproducibility.
            episode_id: Optional custom episode identifier.

        Returns:
            Initial PlantObservation (current step sensor data + budget).
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Random starting index ensuring 720 rows fit
        max_start = self._data_len - EPISODE_LENGTH
        self._start_idx = random.randint(0, max_start)

        # Reset all episode state
        self._episode_id = episode_id or str(uuid4())
        self._step = 0
        self._budget = INITIAL_BUDGET
        self._spare_available = False
        self._spare_ordered = False
        self._spare_used = False
        self._cascade_occurred = False
        self._shift_complete = False
        self._done = False
        self._last_action = None
        self._last_state_changed = False
        self._downtime_remaining = 0

        return self._get_observation()

    # ─────────────────────────────────────────────────────────────────────
    # Step
    # ─────────────────────────────────────────────────────────────────────

    def step(
        self,
        action: PlantAction,  # type: ignore[override]
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> dict:
        """
        Execute one step in the environment.

        Order of operations:
          1. Deduct action cost (insufficient budget → do_nothing + penalty).
          2. Update spare inventory.
          3. Check failure (Machine failure == 1 and step < 719).
          4. Compute multi-component reward.
          5. Increment step counter, update tracking, set done flag.

        Args:
            action: PlantAction with tool, reasoning, and optional load_reduction.
            timeout_s: Ignored (for API compatibility).

        Returns:
            Dict compatible with OpenEnv server expectations:
              {"observation": PlantObservation, "reward": float, "done": bool, "info": dict}
        """
        if self._done:
            return {
                "observation": self._get_observation(),
                "reward": 0.0,
                "done": True,
                "info": {"already_done": True},
            }

        tool = action.tool
        reasoning = action.reasoning or ""
        info: dict[str, Any] = {}

        # ── 0. Downtime handling ───────────────────────────────────────
        # If a repair caused downtime, production is halted and the agent cannot
        # meaningfully act. We still advance time and apply a penalty.
        if self._downtime_remaining > 0:
            self._downtime_remaining -= 1
            reward = REWARD_SURVIVAL_PER_STEP + REWARD_DOWNTIME_PER_HOUR
            self._step += 1

            if self._step >= EPISODE_LENGTH:
                self._shift_complete = True
                self._done = True

            info.update(
                {
                    "downtime": True,
                    "downtime_remaining": self._downtime_remaining,
                    "effective_action": "downtime_wait",
                }
            )
            return {
                "observation": self._get_observation(),
                "reward": reward,
                "done": self._done,
                "info": info,
            }

        # ── 1. Deduct action cost ──────────────────────────────────────
        cost = self._get_action_cost(tool, spare_available=self._spare_available)
        effective_action = tool
        state_changed = False

        if cost > self._budget:
            # Insufficient budget: force do_nothing
            effective_action = "do_nothing"
            cost = 0.0

        self._budget -= cost

        # ── 2. Update spare inventory & track state changes ────────────
        if effective_action == "order_spare_part" and not self._spare_available:
            self._spare_available = True
            self._spare_ordered = True
            state_changed = True
        elif effective_action == "dispatch_repair":
            if self._spare_available:
                self._spare_available = False
                self._spare_used = True
                info["repair_cost_waived_by_spare"] = True
            state_changed = True  # Repair always counts as state change
            # If no spare was used, the repair causes downtime.
            if not info.get("repair_cost_waived_by_spare", False):
                self._downtime_remaining = DOWNTIME_HOURS_NO_SPARE
                info["downtime_triggered_hours"] = DOWNTIME_HOURS_NO_SPARE
        elif effective_action == "adjust_load":
            state_changed = True  # Load adjustment always counts as state change
        elif effective_action == "run_diagnostic":
            state_changed = True  # Diagnostic reveals new info
            # Reveal a diagnostic score (placeholder; can be made data-driven later).
            info["diagnostic_failure_risk_pct"] = int(np.random.randint(0, 101))

        # ── 3. Check failure ───────────────────────────────────────────
        row_idx = self._start_idx + self._step
        row = self._data.iloc[row_idx]
        machine_failure = int(row["Machine failure"])

        failure_this_step = False
        if machine_failure == 1 and self._step < (EPISODE_LENGTH - 1):
            # Repair action prevents cascade
            if effective_action != "dispatch_repair":
                self._cascade_occurred = True
                failure_this_step = True

        # ── 4. Compute multi-component reward ──────────────────────────
        reward = self._compute_reward(
            effective_action=effective_action,
            reasoning=reasoning,
            failure_this_step=failure_this_step,
            state_changed=state_changed,
        )

        # ── 5. Update tracking ─────────────────────────────────────────
        self._step += 1
        self._last_action = effective_action
        self._last_state_changed = state_changed

        # Check for episode completion
        if failure_this_step:
            self._done = True
        elif self._step >= EPISODE_LENGTH:
            self._shift_complete = True
            self._done = True
            # At episode end, add spare reward/penalty
            if self._spare_ordered:
                if self._spare_used:
                    reward += REWARD_SPARE_USED
                else:
                    reward += REWARD_SPARE_WASTED

        info.update(
            {
                "step": self._step,
                "effective_action": effective_action,
                "cascade_occurred": self._cascade_occurred,
                "shift_complete": self._shift_complete,
                "spare_available": self._spare_available,
                "budget_remaining": round(self._budget, 2),
                "downtime_remaining": self._downtime_remaining,
            }
        )

        return {
            "observation": self._get_observation(),
            "reward": float(reward),
            "done": bool(self._done),
            "info": info,
        }

    # ─────────────────────────────────────────────────────────────────────
    # State
    # ─────────────────────────────────────────────────────────────────────

    @property
    def state(self) -> PlantState:
        """
        Return full episode state.

        Returns:
            PlantState with episode tracking fields.
        """
        return PlantState(
            episode_id=self._episode_id,
            step_count=self._step,
            shift_complete=self._shift_complete,
            cascade_occurred=self._cascade_occurred,
            budget_remaining=round(self._budget, 2),
            spare_available=self._spare_available,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Private Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _get_observation(self) -> PlantObservation:
        """
        Build observation from ONLY the current step's data.

        Memory constraint: NO past failures, NO action history, NO
        historical sensor readings. Only the current row + budget.

        Returns:
            PlantObservation with current sensor data and budget.
        """
        row_idx = self._start_idx + min(self._step, EPISODE_LENGTH - 1)
        row = self._data.iloc[row_idx]

        return PlantObservation(
            air_temp=float(row["Air temperature [K]"]),
            process_temp=float(row["Process temperature [K]"]),
            rotational_speed=float(row["Rotational speed [rpm]"]),
            torque=float(row["Torque [Nm]"]),
            tool_wear=float(row["Tool wear [min]"]),
            shift_hour=int(self._step % 24),
            remaining_budget=round(self._budget, 2),
            metadata={
                "step": self._step,
                "cascade_occurred": self._cascade_occurred,
                "shift_complete": self._shift_complete,
                "spare_available": self._spare_available,
                "budget_remaining": round(self._budget, 2),
                "downtime_remaining": self._downtime_remaining,
            },
        )

    @staticmethod
    def _get_action_cost(tool: str, spare_available: bool) -> float:
        """Return the dollar cost of a given tool."""
        costs = {
            "run_diagnostic": COST_DIAGNOSTIC,
            "adjust_load": 0.0,
            "dispatch_repair": COST_REPAIR,
            "order_spare_part": COST_SPARE,
            "do_nothing": 0.0,
        }
        if tool == "dispatch_repair" and spare_available:
            return 0.0
        return costs.get(tool, 0.0)

    def _compute_reward(
        self,
        effective_action: str,
        reasoning: str,
        failure_this_step: bool,
        state_changed: bool,
    ) -> float:
        """
        Compute multi-component reward.

        Components:
          outcome   : +1000 (shift complete) / -5000 (cascade) / 0
          survival  : +1 (always added each step)
          redundancy: -5 if same action as previous AND no state change
          reasoning : 0–350 from keyword scoring
          spare     : handled at episode end in step()

        Args:
            effective_action: The action actually taken (may differ from requested).
            reasoning: Free-text reasoning trace.
            failure_this_step: Whether a cascade failure occurred this step.
            state_changed: Whether this action caused a meaningful state change.

        Returns:
            Total reward for this step (float).
        """
        # ── Outcome ────────────────────────────────────────────────────
        if failure_this_step:
            outcome = REWARD_CASCADE
        elif self._step + 1 >= EPISODE_LENGTH:
            outcome = REWARD_SHIFT_COMPLETE
        else:
            outcome = 0.0

        # ── Survival ──────────────────────────────────────────────────
        survival = REWARD_SURVIVAL_PER_STEP

        # ── Redundancy penalty ────────────────────────────────────────
        redundancy = 0.0
        if (
            self._last_action is not None
            and effective_action == self._last_action
            and not state_changed
        ):
            redundancy = REWARD_REDUNDANCY_PENALTY

        # ── Reasoning bonus ───────────────────────────────────────────
        reasoning_score = float(_score_reasoning(reasoning))

        total = outcome + survival + redundancy + reasoning_score
        return total
