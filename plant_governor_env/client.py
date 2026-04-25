# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""PlantGuard-Meta Environment Client.

Provides a typed, WebSocket-based client for interacting with the
PlantGovernorEnvironment server. Subclasses EnvClient to convert between
PlantAction / PlantObservation / PlantState and the wire JSON format.
"""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import PlantAction, PlantObservation, PlantState


class PlantGovernorClient(
    EnvClient[PlantAction, PlantObservation, PlantState]
):
    """
    Client for the PlantGuard-Meta Environment.

    This client maintains a persistent WebSocket connection to the environment
    server, enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session.

    Example (async):
        >>> async with PlantGovernorClient(base_url="http://localhost:8000") as client:
        ...     result = await client.reset()
        ...     print(result.observation.air_temp)
        ...
        ...     action = PlantAction(tool="run_diagnostic", reasoning="Check temps")
        ...     result = await client.step(action)
        ...     print(result.observation.remaining_budget)

    Example (sync wrapper):
        >>> sync_client = PlantGovernorClient(base_url="http://localhost:8000").sync()
        >>> with sync_client:
        ...     result = sync_client.reset()
        ...     action = PlantAction(tool="do_nothing", reasoning="Observing")
        ...     result = sync_client.step(action)
    """

    def _step_payload(self, action: PlantAction) -> Dict[str, Any]:
        """
        Convert PlantAction to JSON payload for step message.

        Args:
            action: PlantAction instance with tool, reasoning, and optional
                    load_reduction.

        Returns:
            Dictionary suitable for JSON encoding.
        """
        payload: Dict[str, Any] = {
            "tool": action.tool,
            "reasoning": action.reasoning,
        }
        if action.load_reduction is not None:
            payload["load_reduction"] = action.load_reduction
        return payload

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[PlantObservation]:
        """
        Parse server response into StepResult[PlantObservation].

        Args:
            payload: JSON response data from server.

        Returns:
            StepResult wrapping a PlantObservation.
        """
        obs_data = payload.get("observation", {})

        observation = PlantObservation(
            air_temp=obs_data.get("air_temp", 0.0),
            process_temp=obs_data.get("process_temp", 0.0),
            rotational_speed=obs_data.get("rotational_speed", 0.0),
            torque=obs_data.get("torque", 0.0),
            tool_wear=obs_data.get("tool_wear", 0.0),
            shift_hour=obs_data.get("shift_hour", 0),
            remaining_budget=obs_data.get("remaining_budget", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> PlantState:
        """
        Parse server response into PlantState.

        Args:
            payload: JSON response from state request.

        Returns:
            PlantState with full episode tracking.
        """
        return PlantState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            shift_complete=payload.get("shift_complete", False),
            cascade_occurred=payload.get("cascade_occurred", False),
            budget_remaining=payload.get("budget_remaining", 10000.0),
            spare_available=payload.get("spare_available", False),
        )
