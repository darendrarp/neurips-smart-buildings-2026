#!/usr/bin/env python3
"""Unit tests for the Smart Buildings policy (stdlib unittest, numpy only).

Run:
    python -m unittest discover -s tests -v
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from baselines_mock import MOCK_BASELINES  # noqa: E402
from mock_env import MockBuilding  # noqa: E402
from policy import Policy  # noqa: E402


def make_building(seed=1, station="atlanta", days=3, n_ahus=None, n_zones=None):
    b = MockBuilding(seed=seed, station=station, num_days=days,
                     n_ahus=n_ahus, n_zones=n_zones)
    return b


def obs_at(building, step, zone_temps=None, oat_k=None):
    """Observation at a step with controlled zone temps / OAT."""
    building._meta = building.build_metadata()
    temps = (np.full(building.n_zones, 295.0) if zone_temps is None
             else np.asarray(zone_temps, dtype=float))
    obs = building.observation(
        step, temps,
        damper_pos=np.full(building.n_zones, 0.1),
        reheat_pos=np.zeros(building.n_zones),
        supply_temps=np.full(building.n_ahus, 293.0))
    if oat_k is not None:
        for i, name in enumerate(building._meta["observation_field_names"]):
            if "outside_air_temperature_sensor" in name:
                obs[i] = (oat_k - 295.0) / 10.0
    return obs


class TestInterface(unittest.TestCase):
    def test_shapes_dtypes_ranges_many_sizes(self):
        for (k, v) in [(1, 1), (2, 4), (3, 17), (4, 33), (6, 40)]:
            with self.subTest(K=k, V=v):
                b = make_building(n_ahus=k, n_zones=v)
                meta = b.build_metadata()
                pol = Policy(meta, None)
                obs = obs_at(b, 48)  # Monday 10:00, comfort hours
                act = pol.get_action(obs)
                self.assertEqual(act["continuous_action"].shape, (3 * k + 2 + 2 * v,))
                self.assertEqual(act["discrete_action"].shape, (k + 1,))
                self.assertEqual(act["continuous_action"].dtype, np.float32)
                self.assertEqual(act["discrete_action"].dtype, np.int32)
                self.assertTrue(np.all(act["continuous_action"] >= -1.0))
                self.assertTrue(np.all(act["continuous_action"] <= 1.0))
                self.assertTrue(set(np.unique(act["discrete_action"])) <= {0, 1})
                self.assertTrue(np.all(np.isfinite(act["continuous_action"])))

    def test_hybrid_action_all_slots_written(self):
        """Every continuous slot gets a deliberate value (no accidental 0.0
        half-open damper on slots the policy forgot)."""
        b = make_building(n_ahus=3, n_zones=12)
        meta = b.build_metadata()
        pol = Policy(meta, None)
        obs = obs_at(b, 48)
        act = pol.get_action(obs)
        # Recirculation slots must be written (cap) even when idle.
        for idx in meta["recirculation_indices"]:
            self.assertEqual(act["continuous_action"][idx], 1.0)

    def test_determinism(self):
        b = make_building()
        meta = b.build_metadata()
        seq = [obs_at(b, 48 + i) for i in range(20)]
        acts1, acts2 = [], []
        for acts in (acts1, acts2):
            pol = Policy(meta, None)
            for obs in seq:
                a = pol.get_action(obs)
                acts.append((a["continuous_action"].copy(),
                             a["discrete_action"].copy()))
        for (c1, d1), (c2, d2) in zip(acts1, acts2):
            np.testing.assert_array_equal(c1, c2)
            np.testing.assert_array_equal(d1, d2)

    def test_edge_observations_do_not_crash(self):
        b = make_building()
        meta = b.build_metadata()
        pol = Policy(meta, None)
        base = obs_at(b, 48)
        variants = {
            "nan_zone": base.copy(),
            "inf_oat": base.copy(),
            "extreme": base.copy(),
            "all_zero": np.zeros_like(base),
        }
        zt = [i for i, n in enumerate(meta["observation_field_names"])
              if "zone_air_temperature" in n]
        variants["nan_zone"][zt[0]] = np.nan
        oat = [i for i, n in enumerate(meta["observation_field_names"])
               if "outside_air_temperature_sensor" in n]
        variants["inf_oat"][oat[0]] = np.inf
        variants["extreme"][:] = 1e6
        for name, obs in variants.items():
            with self.subTest(case=name):
                act = pol.get_action(obs)
                self.assertEqual(act["continuous_action"].shape,
                                 (meta["n_continuous_action"],))
                self.assertTrue(np.all(np.isfinite(act["continuous_action"])))
                self.assertTrue(set(np.unique(act["discrete_action"])) <= {0, 1})

    def test_missing_optional_metadata_fields(self):
        """No comfort flags, no OAT field, no sensor bands, no normalization
        metadata: the policy must still work (schedule reconstruction)."""
        b = make_building()
        meta = b.build_metadata()
        keep = [n for n in meta["observation_field_names"]
                if "zone_air_temperature_sensor" in n
                and "setpoint" not in n]
        meta = dict(meta)
        meta["observation_field_names"] = keep
        meta["observation_normalization_metadata"] = {}
        pol = Policy(meta, None)
        obs = np.zeros(len(keep), dtype=np.float32)  # all zones ~295 K
        act = pol.get_action(obs)
        self.assertEqual(act["continuous_action"].shape,
                         (meta["n_continuous_action"],))


class TestBehaviour(unittest.TestCase):
    def test_cold_zone_heats(self):
        b = make_building(station="fairbanks")
        meta = b.build_metadata()
        pol = Policy(meta, None)
        temps = np.full(b.n_zones, 295.0)
        temps[0] = 285.0  # 9 K below the comfort band
        act = pol.get_action(obs_at(b, 48, temps))
        ahu0 = 0  # zone 0 is served by AHU 0 in the mock layout
        self.assertEqual(act["discrete_action"][ahu0], 1)
        self.assertGreater(act["continuous_action"][meta["temp_indices"][0]], 0.0)
        za = meta["zone_action_indices"]["VAV_1"]
        self.assertGreater(
            act["continuous_action"][za["reheat_valve_percentage_command"]], -1.0)
        self.assertGreater(
            act["continuous_action"][meta["water_indices"][0]], -1.0)
        self.assertEqual(act["discrete_action"][-1], 1)  # pump on

    def test_hot_zone_cools(self):
        b = make_building(station="phoenix")
        meta = b.build_metadata()
        pol = Policy(meta, None)
        temps = np.full(b.n_zones, 295.0)
        temps[0] = 305.0  # 8 K above the comfort band
        act = pol.get_action(obs_at(b, 48, temps))
        self.assertEqual(act["discrete_action"][0], 1)
        self.assertLess(act["continuous_action"][meta["temp_indices"][0]], 0.0)
        za = meta["zone_action_indices"]["VAV_1"]
        self.assertGreater(
            act["continuous_action"][za["supply_air_damper_percentage_command"]],
            -1.0)
        # No heating: reheat shut, water cold, pump off.
        self.assertEqual(
            act["continuous_action"][za["reheat_valve_percentage_command"]], -1.0)
        self.assertEqual(act["discrete_action"][-1], 0)

    def test_in_band_idles(self):
        b = make_building()
        meta = b.build_metadata()
        pol = Policy(meta, None)
        act = pol.get_action(obs_at(b, 48, np.full(b.n_zones, 295.5)))
        self.assertTrue(np.all(act["discrete_action"] == 0))

    def test_unoccupied_night_shutdown(self):
        b = make_building()
        meta = b.build_metadata()
        pol = Policy(meta, None)
        # Monday 23:00: eco hours, zero occupants, comfort not soon.
        temps = np.full(b.n_zones, 290.0)  # below comfort, inside eco
        act = pol.get_action(obs_at(b, 204, temps))
        self.assertTrue(np.all(act["discrete_action"] == 0))
        self.assertEqual(act["continuous_action"][meta["water_indices"][0]], -1.0)

    def test_preconditioning_before_comfort_hours(self):
        b = make_building()
        meta = b.build_metadata()
        pol = Policy(meta, None)
        # Tuesday 05:00: comfort starts 06:00, so comfort_mode_soon is on.
        temps = np.full(b.n_zones, 290.0)  # cold vs comfort band
        act = pol.get_action(obs_at(b, 564, temps))
        self.assertTrue(np.any(act["discrete_action"][:-1] == 1))

    def test_economizer_engages_with_cold_outside_air(self):
        b = make_building(station="denver")
        meta = b.build_metadata()
        pol = Policy(meta, None)
        temps = np.full(b.n_zones, 295.0)
        temps[0] = 304.0  # cooling demand
        act = pol.get_action(obs_at(b, 48, temps, oat_k=273.15))  # 0 C outside
        recirc = act["continuous_action"][meta["recirculation_indices"][0]]
        self.assertLess(recirc, 1.0)

    def test_economizer_holds_cap_with_hot_outside_air(self):
        b = make_building(station="phoenix")
        meta = b.build_metadata()
        pol = Policy(meta, None)
        temps = np.full(b.n_zones, 295.0)
        temps[0] = 304.0
        act = pol.get_action(obs_at(b, 48, temps, oat_k=313.15))  # 40 C outside
        recirc = act["continuous_action"][meta["recirculation_indices"][0]]
        self.assertEqual(recirc, 1.0)

    def test_water_standby_when_no_heating(self):
        b = make_building(station="singapore")
        meta = b.build_metadata()
        pol = Policy(meta, None)
        temps = np.full(b.n_zones, 296.0)  # in band, warm climate
        act = pol.get_action(obs_at(b, 48, temps))
        self.assertEqual(act["discrete_action"][-1], 0)
        self.assertEqual(act["continuous_action"][meta["water_indices"][0]], -1.0)

    def test_worst_zone_drives_ahu_not_mean(self):
        """Mean in band, one zone far out: the AHU must still run."""
        b = make_building(n_ahus=1, n_zones=10)
        meta = b.build_metadata()
        pol = Policy(meta, None)
        temps = np.full(b.n_zones, 295.5)
        temps[0] = 288.0  # 6 K below comfort; mean still ~294.8 in band
        act = pol.get_action(obs_at(b, 48, temps))
        self.assertEqual(act["discrete_action"][0], 1)


class TestMockRollout(unittest.TestCase):
    def test_full_episode_runs_without_crash(self):
        b = make_building(station="fairbanks", days=3)
        meta = b.build_metadata()
        res = b.run(Policy(meta, None))
        self.assertEqual(res["n_steps"], 3 * 288)
        self.assertLess(res["score"], 1.0)
        self.assertTrue(np.isfinite(res["total_reward"]))

    def test_mock_baselines_run(self):
        b = make_building(station="atlanta", days=2)
        meta = b.build_metadata()
        for key, cls in MOCK_BASELINES.items():
            with self.subTest(baseline=key):
                res = b.run(cls(meta, None))
                self.assertTrue(np.isfinite(res["total_reward"]))

    def test_policy_beats_all_off_on_mock(self):
        b = make_building(station="denver", days=7)
        meta = b.build_metadata()
        ours = b.run(Policy(meta, None))["score"]
        off = b.run(MOCK_BASELINES["all_off"](meta, None))["score"]
        self.assertLess(ours, off)


class TestNotebook(unittest.TestCase):
    def test_notebook_is_valid_and_executes(self):
        nb_path = REPO / "submission_notebook.ipynb"
        nb = json.loads(nb_path.read_text())
        self.assertEqual(nb["nbformat"], 4)
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        self.assertGreaterEqual(len(code_cells), 2)
        ns: dict = {}
        for cell in code_cells:
            exec("".join(cell["source"]), ns)
        self.assertIn("Policy", ns)
        self.assertIn("TransferPolicy", ns)

    def test_notebook_policy_matches_policy_py(self):
        """The notebook's Policy cell must be exactly policy.py (plus the
        TransferPolicy alias line), so the submitted code is the tested code."""
        nb = json.loads((REPO / "submission_notebook.ipynb").read_text())
        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        code = "".join(code_cells[0]["source"])
        shipped = (REPO / "policy.py").read_text()
        self.assertTrue(code.startswith(shipped),
                        "notebook Policy cell drifted from policy.py; "
                        "re-run build_notebook.py")


if __name__ == "__main__":
    unittest.main()
