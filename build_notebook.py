#!/usr/bin/env python3
"""Build the Kaggle submission notebook (submission_notebook.ipynb).

The notebook is generated from policy.py so the submitted Policy class is
always identical to the tested one.  The notebook is self-contained: it
defines Policy and runs a small smoke test with a hand-built fake
observation, so "Run All" succeeds on Kaggle without any input data.

Regenerate after any policy change:
    python build_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
POLICY_SRC = (HERE / "policy.py").read_text()

SMOKE_TEST_SRC = '''# Smoke test: build a tiny fake building (2 AHUs, 4 zones) and step the
# policy through 50 synthetic observations.  This only proves the interface
# works; real scoring happens on the competition simulator.
import numpy as np

K, V = 2, 4
field_names = []
for v in range(1, V + 1):
    field_names += [
        f"VAV_{v}_zone_air_temperature_sensor",
        f"VAV_{v}_zone_air_heating_temperature_setpoint_sensor",
        f"VAV_{v}_zone_air_cooling_temperature_setpoint_sensor",
        f"VAV_{v}_supply_air_damper_percentage_command",
        f"VAV_{v}_reheat_valve_percentage_command",
        f"VAV_{v}_supply_air_flowrate_setpoint",
    ]
for k in range(1, K + 1):
    field_names += [
        f"ahs_ahu_{k}_outside_air_temperature_sensor",
        f"ahs_ahu_{k}_supply_air_temperature_sensor",
    ]
field_names += ["comfort_mode_now", "comfort_mode_soon", "num_occupants",
                "hod_sin_000", "hod_cos_000", "dow_sin_000", "dow_cos_000"]

meta = {
    "n_discrete_action": K + 1,
    "n_continuous_action": 3 * K + 2 + 2 * V,
    "temp_indices": [0, 3],
    "pressure_indices": [1, 4, 7],
    "water_indices": [6],
    "recirculation_indices": [2, 5],
    "zone_action_indices": {
        f"VAV_{v}": {
            "supply_air_damper_percentage_command": 8 + 2 * (v - 1),
            "reheat_valve_percentage_command": 9 + 2 * (v - 1),
        }
        for v in range(1, V + 1)
    },
    "observation_field_names": field_names,
    "ahu_zone_assignments": {"ahu_1": ["room_1", "room_2"],
                             "ahu_2": ["room_3", "room_4"]},
    "morning_start_hour": 6,
    "evening_start_hour": 19,
    "comfort_temp_window": (294.0, 297.0),
    "eco_temp_window": (289.0, 298.0),
    "observation_normalization_metadata": {
        "zone_air_temperature_sensor": {"sample_mean": 295.0,
                                        "sample_variance": 25.0,
                                        "sample_std": 5.0},
        "outside_air_temperature_sensor": {"sample_mean": 295.0,
                                           "sample_variance": 100.0,
                                           "sample_std": 10.0},
    },
}

policy = Policy(meta, None)
rng = np.random.default_rng(0)
for step in range(50):
    obs = np.zeros(len(field_names), dtype=np.float32)
    for v in range(1, V + 1):
        i = field_names.index(f"VAV_{v}_zone_air_temperature_sensor")
        obs[i] = rng.normal(0.0, 0.4)  # zone temps near 295 K
        obs[field_names.index(
            f"VAV_{v}_zone_air_heating_temperature_setpoint_sensor")] = 294.0
        obs[field_names.index(
            f"VAV_{v}_zone_air_cooling_temperature_setpoint_sensor")] = 297.0
    obs[field_names.index("comfort_mode_now")] = 1.0
    obs[field_names.index("num_occupants")] = 0.0
    action = policy.get_action(obs)
    assert action["continuous_action"].shape == (3 * K + 2 + 2 * V,)
    assert action["discrete_action"].shape == (K + 1,)
    assert action["continuous_action"].dtype == np.float32
    assert action["discrete_action"].dtype == np.int32
    assert np.all(action["continuous_action"] >= -1.0)
    assert np.all(action["continuous_action"] <= 1.0)
    assert set(np.unique(action["discrete_action"])) <= {0, 1}
print("smoke test passed: 50 steps, all actions valid")
'''

INTRO_MD = """# NeurIPS 2026 Smart Buildings — Adaptive Pre-conditioning Policy

Submission notebook for the **main leaderboard track** ($25,000) of the
[Smart Buildings Competition](https://www.kaggle.com/competitions/neurips-2026-smart-buildings).

## What this notebook does

It defines a single `Policy` class — the competition interface:

```python
policy = Policy(building_meta_data, action_spec)
action = policy.get_action(obs)
```

The policy is a **rule-based adaptive controller** (no learned weights, no
building-specific tuning, numpy only):

1. **Per-zone sensor bands** — each zone is judged against its own
   heating/cooling setpoint sensors (the band the scorer uses), with the
   schedule band as fallback.
2. **Worst-zone AHU loop** — each air handler's PID tracks its worst zone,
   because the comfort penalty is quadratic and unbounded past 5 K.
3. **Correct economizer** — the outside-air sensor is normalized
   (mean 295 K, std 10 K); it is denormalized before the free-cooling math.
4. **Occupancy-aware shutdown** — with zero occupants outside comfort hours
   the plant shuts down (comfort penalty is exactly zero then).
5. **Pre-conditioning** — `comfort_mode_soon` plus a cold-weather extension
   brings zones into the comfort band before occupants arrive.
6. **Proportional reheat + demand/saturation hot-water reset + pump standby.**

## How to submit

1. **Save Version** (top right) → **Save & Run All**. Wait for the run to
   finish — the smoke-test cell at the bottom must print
   `smoke test passed`.
2. Click **Submit to Competition** and confirm. The notebook is scored on
   the private leaderboard (6 unseen buildings × 5 unseen weather stations).
3. You may submit up to 5 times per day and select up to 2 final
   submissions before the deadline (**15 January 2027** — confirm the exact
   time on the competition Timeline page).

Full write-up, mock-evaluation results and honest caveats:
https://github.com/darendrarp/neurips-smart-buildings-2026
"""

NOTES_MD = """## Notes & caveats

* This policy was developed **without running the real simulator** (see the
  repo README for why): it is validated on a simplified mock simulator and
  by unit tests. Its real leaderboard score is **unverified** — the first
  Kaggle submission is the real test.
* The public leaderboard top 8 sits near **0.005** (Nov 2026) while the
  best shipped baseline scores ~0.054 on the training set, so this entry is
  a solid foundation, not a promised top-8 finish. Iterate from the
  leaderboard feedback: the repo documents the tuning knobs.
* No paper is required for the main leaderboard track.
"""


def cell_md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}


def cell_code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


def main() -> None:
    nb = {
        "cells": [
            cell_md(INTRO_MD),
            cell_code(POLICY_SRC
                      + "\n# Alias in case the grading harness looks for the "
                        "starter-kit class name.\nTransferPolicy = Policy\n"),
            cell_code(SMOKE_TEST_SRC),
            cell_md(NOTES_MD),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3",
                           "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = HERE / "submission_notebook.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"wrote {out} ({out.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
