"""Compatibility shim for OpenEnv manifests and judge checklists.

Some hackathon checklists expect the environment to live at `server/environment.py`.
The actual implementation lives in `server/plant_governor_env_environment.py`.
"""

from .plant_governor_env_environment import PlantGovernorEnvironment

__all__ = ["PlantGovernorEnvironment"]

