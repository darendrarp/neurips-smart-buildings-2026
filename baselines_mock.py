"""Simplified re-implementations of the shipped baseline controllers, written
from their documented descriptions, for MOCK comparison only.

These are NOT the starter-kit files (those are never imported here).  They
follow the documented behaviour of each baseline:

  * all_off:        never runs any equipment.
  * fixed_setpoint: PID on the mean zone temperature, comfort band around
                    the clock (no eco setback), binary reheat, binary water.
  * pid:            PID on the mean zone temperature, schedule band, binary
                    reheat, binary water, recirculation pinned at the cap.
  * pid_water:      pid + proportional reheat + demand/saturation hot-water
                    reset + economizer.  The economizer is implemented
                    AS DOCUMENTED in the starter kit, i.e. it reads the
                    outside-air sensor value as kelvin even though that
                    field is normalized (mean 295 K, std 10 K) -- which is
                    why it stays near the recirculation cap in practice.

Their mock scores are only meaningful relative to each other and to
``policy.Policy`` on the SAME mock, never against the real leaderboard.
"""

from __future__ import annotations

import numpy as np


class _MockBase:
    """Index mapping shared by the mock baselines (own implementation)."""

    name = "mock_base"

    def __init__(self, building_meta_data, action_spec=None,
                 kp=0.5, ki=0.02, kd=0.1):
        md = building_meta_data
        self.n_discrete = int(md["n_discrete_action"])
        self.n_continuous = int(md["n_continuous_action"])
        self.temp_idx = list(md["temp_indices"])
        self.pressure_idx = list(md["pressure_indices"])
        self.water_idx = list(md["water_indices"])
        self.recirc_idx = list(md["recirculation_indices"])
        self.zone_actions = md["zone_action_indices"]
        self.fields = list(md["observation_field_names"])
        self.assignments = md["ahu_zone_assignments"]
        self.comfort = tuple(md["comfort_temp_window"])
        self.eco = tuple(md["eco_temp_window"])
        self.morning = int(md["morning_start_hour"])
        self.evening = int(md["evening_start_hour"])
        norm = md.get("observation_normalization_metadata", {}) or {}
        zt = norm.get("zone_air_temperature_sensor", {})
        self.zt_mean = float(zt.get("sample_mean", 295.0))
        self.zt_std = float(zt.get("sample_std", 5.0))
        self.kp, self.ki, self.kd = kp, ki, kd
        self._fidx = {n: i for i, n in enumerate(self.fields)}
        self._hod_sin = self._fidx.get("hod_sin_000")
        self._hod_cos = self._fidx.get("hod_cos_000")
        self._dow_sin = self._fidx.get("dow_sin_000")
        self._dow_cos = self._fidx.get("dow_cos_000")
        self._oat = next(
            (i for i, n in enumerate(self.fields)
             if "outside_air_temperature_sensor" in n), None)
        # Per-AHU zone observation indices.
        self.ahus = []
        for i, ahu_id in enumerate(sorted(self.assignments)):
            zones = []
            for room in self.assignments[ahu_id]:
                vav = room.replace("room_", "VAV_")
                f = f"{vav}_zone_air_temperature_sensor"
                if f in self._fidx:
                    zones.append((vav, self._fidx[f],
                                  self.zone_actions.get(vav, {})))
            self.ahus.append({"temp": self.temp_idx[i],
                              "pressure": self.pressure_idx[i]
                              if i < len(self.pressure_idx) else None,
                              "recirc": self.recirc_idx[i]
                              if i < len(self.recirc_idx) else None,
                              "run": i, "zones": zones})
        self.water_temp = self.water_idx[0] if self.water_idx else None
        self.water_pressure = (
            self.pressure_idx[-1] if len(self.pressure_idx) > len(self.ahus)
            else None)
        self.water_run = self.n_discrete - 1
        self.state = {i: {"integral": 0.0, "prev": 0.0}
                      for i in range(len(self.ahus))}

    def _hour_day(self, obs):
        def cyc(s, c, period):
            a = float(np.arctan2(obs[s], obs[c]))
            if a < 0:
                a += 2 * np.pi
            return int(np.floor(period * a / (2 * np.pi)))
        return (cyc(self._hod_sin, self._hod_cos, 24),
                cyc(self._dow_sin, self._dow_cos, 7))

    def _band(self, obs):
        hour, day = self._hour_day(obs)
        if day < 5 and self.morning <= hour < self.evening:
            return self.comfort
        return self.eco

    def _temp(self, obs, idx):
        return float(obs[idx]) * self.zt_std + self.zt_mean

    @staticmethod
    def _err(temp, low, high):
        if temp < low:
            return low - temp
        if temp > high:
            return high - temp
        return 0.0

    def _zones_cmd(self, cont, zones, obs, low, high, proportional_reheat):
        for _, oidx, acts in zones:
            if not acts:
                continue
            e = self._err(self._temp(obs, oidx), low, high)
            d = min(1.0, abs(e) / 2.0)
            di = acts.get("supply_air_damper_percentage_command")
            if di is not None:
                cont[di] = 2.0 * d - 1.0
            ri = acts.get("reheat_valve_percentage_command")
            if ri is not None:
                if proportional_reheat:
                    cont[ri] = 2.0 * min(1.0, max(0.0, e) / 2.0) - 1.0
                else:
                    cont[ri] = 1.0 if e > 0 else -1.0

    def _pid_step(self, i, error):
        st = self.state[i]
        if error == 0.0:
            st["integral"] *= 0.5
            st["prev"] = 0.0
            return 0.0, True
        st["integral"] = float(np.clip(st["integral"] + error, -10.0, 10.0))
        deriv = error - st["prev"]
        st["prev"] = error
        u = float(np.clip(
            self.kp * error + self.ki * st["integral"] + self.kd * deriv,
            -1.0, 1.0))
        return u, False


class AllOffPolicy(_MockBase):
    name = "all_off"

    def get_action(self, obs):
        cont = np.full(self.n_continuous, -1.0, dtype=np.float32)
        for i in self.recirc_idx:
            cont[i] = 1.0
        return {"continuous_action": cont,
                "discrete_action": np.zeros(self.n_discrete, dtype=np.int32)}


class _PidFamily(_MockBase):
    """Shared PID loop; subclasses pick band / reheat / water behaviour."""

    use_comfort_always = False
    proportional_reheat = False
    water_reset = False
    economizer = False

    def get_action(self, obs):
        low, high = self.comfort if self.use_comfort_always else self._band(obs)
        cont = np.zeros(self.n_continuous, dtype=np.float32)
        disc = np.zeros(self.n_discrete, dtype=np.int32)
        for i in self.recirc_idx:
            cont[i] = 1.0
        any_heat = False
        any_active = False
        efforts = []
        for i, ahu in enumerate(self.ahus):
            if not ahu["zones"]:
                continue
            self._zones_cmd(cont, ahu["zones"], obs, low, high,
                            self.proportional_reheat)
            temps = [self._temp(obs, o) for _, o, _ in ahu["zones"]]
            error = self._err(float(np.mean(temps)), low, high)
            u, idle = self._pid_step(i, error)
            if idle:
                cont[ahu["temp"]] = -1.0
                if ahu["pressure"] is not None:
                    cont[ahu["pressure"]] = -1.0
                disc[ahu["run"]] = 0
                continue
            cont[ahu["temp"]] = u
            efforts.append(u)
            if ahu["pressure"] is not None:
                cont[ahu["pressure"]] = float(np.clip(2.0 * abs(u) - 1.0, -1.0, 1.0))
            disc[ahu["run"]] = 1
            if self.economizer and ahu["recirc"] is not None:
                cont[ahu["recirc"]] = self._economizer_raw_oat(
                    obs, ahu, u, low, high)
            any_active = True
            if error > 0:
                any_heat = True
        # Water loop.
        if self.water_temp is not None:
            if self.water_reset and any_heat:
                shortfalls = []
                for ahu in self.ahus:
                    for _, o, _ in ahu["zones"]:
                        shortfalls.append(max(0.0, low - self._temp(obs, o)))
                demand = (float(np.mean(shortfalls)) if shortfalls else 0.0) / 4.0
                sat = 0.0
                if efforts:
                    heat = [max(0.0, u) for u in efforts]
                    sat = float(np.quantile(heat, 0.25))
                frac = float(np.clip(max(demand, sat), 0.0, 1.0))
                cont[self.water_temp] = np.float32(2.0 * frac - 1.0)
            else:
                cont[self.water_temp] = 1.0 if any_heat else -1.0
        if self.water_pressure is not None:
            cont[self.water_pressure] = 1.0 if any_active else -1.0
        if 0 <= self.water_run < self.n_discrete:
            disc[self.water_run] = 1 if any_active else 0
        return {"continuous_action": cont, "discrete_action": disc}

    def _economizer_raw_oat(self, obs, ahu, u, low, high):
        """Economizer as documented in the starter kit: the normalized OAT
        sensor value is used as if it were kelvin."""
        if u >= 0.0 or self._oat is None:
            return 1.0
        outside = float(obs[self._oat])  # normalized value, NOT kelvin
        inside = float(np.mean(
            [self._temp(obs, o) for _, o, _ in ahu["zones"]]))
        if outside >= inside:
            return 1.0
        pressure_pa = max(
            (float(np.clip(2.0 * abs(u) - 1.0, -1.0, 1.0)) + 1.0) * 0.5 * 2000.0,
            200.0)
        if (inside - outside) <= pressure_pa * 0.00322:
            return 1.0
        target = max(285.0 + (u + 1.0) * 0.5 * 20.0, low)
        if inside <= target:
            return 1.0
        f = (inside - target) / (inside - outside)
        return 2.0 * float(np.clip(1.0 - f, 0.0, 1.0)) - 1.0


class FixedSetpointPolicy(_PidFamily):
    name = "fixed_setpoint"
    use_comfort_always = True


class PIDPolicy(_PidFamily):
    name = "pid"


class PIDWaterPolicy(_PidFamily):
    name = "pid_water"
    proportional_reheat = True
    water_reset = True
    economizer = True


MOCK_BASELINES = {
    "all_off": AllOffPolicy,
    "fixed_setpoint": FixedSetpointPolicy,
    "pid": PIDPolicy,
    "pid_water": PIDWaterPolicy,
}
