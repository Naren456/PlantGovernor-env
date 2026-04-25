# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the PlantGuard-Meta Environment.

Defines Pydantic models for observations (sensor readings only — NO history),
actions (5 tools with reasoning trace), and state (episode tracking).
"""

from typing import Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


# =============================================================================
# Action Model
# =============================================================================

class PlantAction(Action):
    """
    Agent's action in the plant environment.

    Five available tools:
      - run_diagnostic:   $200, reveals hidden failure probability
      - adjust_load:      $0, reduces production intensity
      - dispatch_repair:  $1500 ($0 if spare available), fixes machine
      - order_spare_part: $100, adds one spare (max 1)
      - do_nothing:       $0, no effect

    The `reasoning` field is a free-text justification that earns bonus
    reward when it mentions domain-relevant keywords.
    """

    tool: Literal[
        "run_diagnostic",
        "adjust_load",
        "dispatch_repair",
        "order_spare_part",
        "do_nothing",
    ] = Field(
        ...,
        description=(
            "Tool to execute. One of: run_diagnostic ($200), adjust_load ($0), "
            "dispatch_repair ($1500 or $0 with spare), order_spare_part ($100), "
            "do_nothing ($0)."
        ),
    )

    reasoning: str = Field(
        default="",
        description=(
            "Free-text justification for the chosen tool. Mention keywords like "
            "'temperature', 'budget', 'cascade', 'trade-off', 'spare', 'downtime' "
            "and write >30 words for bonus reward (up to 350 pts)."
        ),
    )

    load_reduction: Optional[float] = Field(
        default=None,
        ge=0.1,
        le=0.9,
        description=(
            "Load reduction factor (0.1–0.9). Only used with adjust_load. "
            "Ignored for other tools."
        ),
    )


# =============================================================================
# Observation Model
# =============================================================================

class PlantObservation(Observation):
    """
    Current-step sensor readings plus budget.

    Memory constraint: NO past failures, NO action history, NO historical
    sensor readings. The agent must reason from a single snapshot.
    """

    air_temp: float = Field(
        default=0.0,
        description="Air temperature in Kelvin (typical range: 295–304 K)",
    )
    process_temp: float = Field(
        default=0.0,
        description="Process temperature in Kelvin (typical range: 305–313 K)",
    )
    rotational_speed: float = Field(
        default=0.0,
        description="Rotational speed in RPM (typical range: 1168–2886)",
    )
    torque: float = Field(
        default=0.0,
        description="Torque in Nm (typical range: 3.8–76.6)",
    )
    tool_wear: float = Field(
        default=0.0,
        description="Tool wear in minutes (typical range: 0–253)",
    )
    shift_hour: int = Field(
        default=0,
        ge=0,
        le=23,
        description="Current hour within the shift (0–23)",
    )
    remaining_budget: float = Field(
        default=10000.0,
        ge=0.0,
        description="Remaining maintenance budget in dollars (starts at $10,000)",
    )


# =============================================================================
# State Model
# =============================================================================

class PlantState(State):
    """
    Full episode state returned by the /state endpoint.

    Extends the base OpenEnv State with plant-specific tracking fields.
    """

    shift_complete: bool = Field(
        default=False,
        description="True if the agent survived all 720 steps without cascade",
    )
    cascade_occurred: bool = Field(
        default=False,
        description="True if a machine failure cascaded during the episode",
    )
    budget_remaining: float = Field(
        default=10000.0,
        description="Remaining maintenance budget ($)",
    )
    spare_available: bool = Field(
        default=False,
        description="True if a spare part is in inventory",
    )
