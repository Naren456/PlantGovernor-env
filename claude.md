You are an expert AI engineer. Build the complete OpenEnv environment for **PlantGuard-Meta** according to the specification below.

## Specification

**Project name:** `plantguard-meta`  
**Theme:** #3.1 Professional Tasks (industrial plant management)  
**Key constraints:** Agent receives NO history, sparse rewards, reasoning trace bonus, spare parts ordering.

### Files to create (in correct OpenEnv structure)
- `models.py` – Pydantic models for Observation, Action, State
- `server/environment.py` – full environment logic (reset, step, state)
- `client.py` – EnvClient subclass
- `server/app.py` – FastAPI app using create_app
- `openenv.yaml` – manifest pointing to your classes
- `server/requirements.txt` – dependencies

### Detailed requirements

#### Action space (5 tools)
- `run_diagnostic` – cost $200, reveals hidden failure probability (return a random 0-100% for now, but in full version it could be based on next 12 hours of data)
- `adjust_load` – cost $0, changes production intensity (optional parameter `load_reduction` 0.1-0.9)
- `dispatch_repair` – cost $1500 (or $0 if spare available), fixes machine, causes 2h downtime unless spare used
- `order_spare_part` – cost $100, adds one spare to inventory (max 1)
- `do_nothing` – cost $0

#### Observation (NO history)
```python
air_temp: float (295‑304)
process_temp: float (305‑313)
rotational_speed: float (1168‑2886)
torque: float (3.8‑76.6)
tool_wear: float (0‑253)
shift_hour: int (0‑23)
remaining_budget: float (0‑10000)
