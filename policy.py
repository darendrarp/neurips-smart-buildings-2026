"""Adaptive pre-conditioning HVAC policy for the NeurIPS 2026 Smart Buildings
Competition (Kaggle code competition, main leaderboard track).

Interface contract (from the official starter kit,
https://github.com/smart-buildings-2026/starter-kit):

    policy = Policy(building_meta_data, action_spec)
    action = policy.get_action(obs)   # {"continuous_action": float32 (3K+2+2V,),
                                      #  "discrete_action":   int32 (K+1,)}

K = number of air handlers, V = number of zones (VAV boxes).  Both the
observation and the action change shape from building to building, so every
index is read from ``building_meta_data`` -- nothing is hardcoded.

Design (original implementation; ideas are documented in SUBMISSION.md):

1.  Per-zone sensor bands.  Each zone's own heating/cooling setpoint sensors
    (raw kelvin) define the band the scorer uses.  The schedule band from
    metadata is only a fallback -- and during pre-conditioning the sensors
    are deliberately ignored, because they report the *current* (eco) band
    while the target is the upcoming comfort band.
2.  Worst-zone AHU loop.  The comfort penalty is linear to 5 K outside the
    band and quadratic and unbounded past it, summed over zones, so one
    runaway zone can dominate an episode.  Each AHU's PID loop therefore
    tracks the signed error of its *worst* zone (largest magnitude outside
    its band), not the mean.
3.  Correct economizer.  The outside-air temperature sensor is *normalized*
    (mean 295 K, std 10 K in the shipped configs) -- see
    ``observation_normalization_metadata``.  This policy denormalizes it
    before doing economizer math, with a plausibility guard in case the
    field ever arrives raw.  (The shipped ``pid_water`` baseline reads the
    normalized value as kelvin, which makes its economizer nearly inert.)
4.  Occupancy-aware shutdown.  The comfort term is exactly zero when the
    building is unoccupied, so outside comfort hours, when the occupancy
    count reads zero and comfort mode is not starting within the hour, the
    whole plant shuts down instead of holding the eco band for nobody.
5.  Pre-conditioning.  ``comfort_mode_soon`` (comfort mode starting within
    60 minutes, computed by the simulator with the correct timezone and
    holidays) switches the target to the comfort band early, so zones are
    already in band when occupants arrive.  In cold weather the window is
    extended using the hour-of-day features (up to 3 h before the morning
    start when it is below -10 C outside).
6.  Proportional reheat + demand/saturation hot-water reset + pump standby.
    The water loop runs only as hot as the worst zone requires, and the pump
    runs only while some zone actually calls for heat.

No learned weights, no building-specific tuning, no randomness: the same
code runs on every building, which is what the competition scores.

Only numpy is required.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import numpy as np

# Native plant ranges (from the action normalizer in the shipped configs).
SUPPLY_AIR_MIN_K = 285.0
SUPPLY_AIR_MAX_K = 305.0
STATIC_PRESSURE_MAX_PA = 2000.0
MIN_STATIC_PRESSURE_PA = 200.0

# Kelvin of outside-air advantage needed per pascal of fan static pressure
# before drawing outside air beats recirculating (COP 3.5, fan eff 0.9):
#     3.5 / (0.9 * 1.2 * 1006) = 0.00322 K/Pa.
ECON_K_PER_PA = 0.00322

# Plausibility window for a temperature in kelvin.  Used to detect whether a
# sensor reading arrived normalized or raw.
KELVIN_PLAUSIBLE = (200.0, 330.0)

# Fallback normalization constants (shipped configs) if the metadata does
# not carry them.
ZONE_TEMP_FALLBACK = (295.0, 5.0)   # mean, std
OAT_FALLBACK = (295.0, 10.0)        # mean, std


def _numeric_suffix(name: str) -> int:
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else 0


class Policy:
    """Schedule-, occupancy- and worst-zone-aware HVAC controller."""

    name = "adaptive_precon"

    def __init__(
        self,
        building_meta_data: Mapping[str, Any],
        action_spec: Any = None,
        # PID gains for the per-AHU loop (proven values from the shipped
        # PID baseline; kept, not re-tuned, because this policy could not
        # be run against the real simulator during development).
        kp: float = 0.5,
        ki: float = 0.02,
        kd: float = 0.1,
        integral_limit: float = 10.0,
        # Zone excursion (K) at which damper / reheat saturate.
        zone_damper_full_k: float = 2.0,
        reheat_full_k: float = 2.0,
        # Hot-water reset: mean shortfall (K) that alone commands full water.
        water_reset_full_k: float = 4.0,
        water_effort_weight: float = 1.0,
        water_effort_quantile: float = 0.25,
        # Deadband (K) on the AHU run decision; avoids chattering at the edge.
        run_deadband_k: float = 0.05,
        # Extra pre-conditioning lead (hours before morning start) by outside
        # temperature.  (oat_below_C, lead_hours), evaluated in order.
        cold_precon: tuple[tuple[float, float], ...] = ((-10.0, 3.0), (2.0, 2.0)),
    ) -> None:
        self.metadata = building_meta_data
        self.action_spec = action_spec

        self.n_discrete = int(building_meta_data["n_discrete_action"])
        self.n_continuous = int(building_meta_data["n_continuous_action"])

        self.temp_indices = list(building_meta_data.get("temp_indices", []))
        self.pressure_indices = list(building_meta_data.get("pressure_indices", []))
        self.water_indices = list(building_meta_data.get("water_indices", []))
        self.recirculation_indices = list(
            building_meta_data.get("recirculation_indices", [])
        )
        self.zone_action_indices = building_meta_data.get("zone_action_indices", {})
        self.field_names = list(building_meta_data.get("observation_field_names", []))
        self.ahu_zone_assignments = building_meta_data.get("ahu_zone_assignments", {})

        self.comfort_temp_window = tuple(
            building_meta_data.get("comfort_temp_window", (294.0, 297.0))
        )
        self.eco_temp_window = tuple(
            building_meta_data.get("eco_temp_window", (289.0, 298.0))
        )
        self.morning_start_hour = int(building_meta_data.get("morning_start_hour", 6))
        self.evening_start_hour = int(building_meta_data.get("evening_start_hour", 19))

        # Normalization constants for the two temperature sensors this policy
        # reads.  Keyed by measurement name in the metadata.
        norm_meta = building_meta_data.get("observation_normalization_metadata", {}) or {}
        self._zone_temp_mean, self._zone_temp_std = self._norm_const(
            norm_meta, "zone_air_temperature_sensor", ZONE_TEMP_FALLBACK
        )
        self._oat_mean, self._oat_std = self._norm_const(
            norm_meta, "outside_air_temperature_sensor", OAT_FALLBACK
        )

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.zone_damper_full_k = zone_damper_full_k
        self.reheat_full_k = reheat_full_k
        self.water_reset_full_k = water_reset_full_k
        self.water_effort_weight = water_effort_weight
        self.water_effort_quantile = water_effort_quantile
        self.run_deadband_k = run_deadband_k
        self.cold_precon = cold_precon

        self._field_index = {name: i for i, name in enumerate(self.field_names)}
        self._comfort_now_idx = self._field_index.get("comfort_mode_now")
        self._comfort_soon_idx = self._field_index.get("comfort_mode_soon")
        self._occupancy_idx = self._field_index.get("num_occupants")
        self._hod_sin_idx = self._field_index.get("hod_sin_000")
        self._hod_cos_idx = self._field_index.get("hod_cos_000")
        self._dow_sin_idx = self._field_index.get("dow_sin_000")
        self._dow_cos_idx = self._field_index.get("dow_cos_000")
        # Outside air is the same for every AHU: one global reading suffices.
        self._oat_idx = self._find_field("outside_air_temperature_sensor")

        self._build_ahu_mapping()
        self.reset()

    # ------------------------------------------------------------------ setup

    @staticmethod
    def _norm_const(norm_meta, key, fallback):
        entry = norm_meta.get(key) or {}
        try:
            mean = float(entry["sample_mean"])
            std = float(entry["sample_std"])
            if np.isfinite(mean) and np.isfinite(std) and std > 0:
                return mean, std
        except (KeyError, TypeError, ValueError):
            pass
        return fallback

    def _find_field(self, substring: str):
        for name, idx in self._field_index.items():
            if substring in name:
                return idx
        return None

    def _build_ahu_mapping(self) -> None:
        zone_temp_field_index = {
            name: i
            for i, name in enumerate(self.field_names)
            if "zone_air_temperature" in name
        }
        ahu_ids = sorted(self.ahu_zone_assignments.keys(), key=_numeric_suffix)
        n_ahu = len(self.temp_indices)
        if ahu_ids and len(ahu_ids) != n_ahu:
            n_ahu = min(n_ahu, len(ahu_ids))
            ahu_ids = ahu_ids[:n_ahu]
        self.n_ahu = n_ahu

        self.ahu_controllers: list[dict[str, Any]] = []
        for i, ahu_id in enumerate(ahu_ids):
            rooms = self.ahu_zone_assignments[ahu_id]
            zones: list[tuple[str, int, dict[str, int]]] = []
            for room in rooms:
                vav_id = room.replace("room_", "VAV_")
                vav_field = vav_id + "_zone_air_temperature_sensor"
                if vav_field in zone_temp_field_index:
                    zones.append(
                        (
                            vav_id,
                            zone_temp_field_index[vav_field],
                            self.zone_action_indices.get(vav_id, {}),
                        )
                    )
            self.ahu_controllers.append(
                {
                    "ahu_id": ahu_id,
                    "temp_idx": self.temp_indices[i],
                    "pressure_idx": self.pressure_indices[i]
                    if i < len(self.pressure_indices)
                    else None,
                    "recirc_idx": self.recirculation_indices[i]
                    if i < len(self.recirculation_indices)
                    else None,
                    "run_discrete_idx": i,
                    "zones": zones,
                    "zone_obs_indices": [obs_idx for _, obs_idx, _ in zones],
                }
            )

        # Per-zone sensor-band indices (raw kelvin setpoint sensors).
        self._sensor_band_indices: dict[str, tuple[Any, Any]] = {}
        for ctrl in self.ahu_controllers:
            for vav_id, _, _ in ctrl["zones"]:
                self._sensor_band_indices[vav_id] = (
                    self._field_index.get(
                        f"{vav_id}_zone_air_heating_temperature_setpoint_sensor"
                    ),
                    self._field_index.get(
                        f"{vav_id}_zone_air_cooling_temperature_setpoint_sensor"
                    ),
                )

        self.water_temp_idx = self.water_indices[0] if self.water_indices else None
        # The water differential-pressure entry is the pressure index left
        # over after one per AHU (same layout the shipped baselines use).
        self.water_pressure_idx = (
            self.pressure_indices[-1]
            if len(self.pressure_indices) > self.n_ahu
            else None
        )
        self.water_discrete_idx = self.n_discrete - 1

    def reset(self) -> None:
        self.pid_state = {
            i: {"integral": 0.0, "prev_error": 0.0} for i in range(self.n_ahu)
        }
        self._last_efforts: list[float] = []
        self._error_count = 0

    # ------------------------------------------------------------- obs helpers

    @staticmethod
    def _to_kelvin(value: float, mean: float, std: float) -> float:
        """Denormalize a sensor reading, with a raw-kelvin fallback.

        If the denormalized value is outside any plausible temperature while
        the raw value is plausible, the field arrived raw and is used as-is.
        This makes the policy robust to either normalization convention.
        """
        if not np.isfinite(value):
            return float(mean)
        denorm = value * std + mean
        lo, hi = KELVIN_PLAUSIBLE
        if lo <= denorm <= hi:
            return float(denorm)
        if lo <= value <= hi:
            return float(value)
        return float(np.clip(denorm, lo, hi))

    def _zone_temp(self, obs: np.ndarray, idx: int) -> float:
        return self._to_kelvin(float(obs[idx]), self._zone_temp_mean, self._zone_temp_std)

    def _outside_temp(self, obs: np.ndarray):
        if self._oat_idx is None:
            return None
        return self._to_kelvin(float(obs[self._oat_idx]), self._oat_mean, self._oat_std)

    @staticmethod
    def _cyclic_to_index(sin_value: float, cos_value: float, period: int) -> int:
        angle = float(np.arctan2(sin_value, cos_value))
        if angle < 0:
            angle += 2 * np.pi
        return int(np.floor(period * angle / (2 * np.pi)))

    def _hour_and_day(self, obs: np.ndarray):
        if (
            self._hod_sin_idx is None
            or self._hod_cos_idx is None
            or self._dow_sin_idx is None
            or self._dow_cos_idx is None
        ):
            return None, None
        hour = self._cyclic_to_index(
            float(obs[self._hod_sin_idx]), float(obs[self._hod_cos_idx]), 24
        )
        day = self._cyclic_to_index(
            float(obs[self._dow_sin_idx]), float(obs[self._dow_cos_idx]), 7
        )
        return hour, day

    def _occupancy(self, obs: np.ndarray):
        """Building occupant count, or None if the field is absent.

        The observation is (n - 125) / 126 in the shipped configs.
        """
        if self._occupancy_idx is None:
            return None
        value = float(obs[self._occupancy_idx])
        if not np.isfinite(value):
            return None
        return value * 126.0 + 125.0

    def _zone_band(self, obs, vav_id, low: float, high: float,
                   use_sensors: bool = True):
        """This zone's band from its own setpoint sensors, else the schedule.

        The sensors report the CURRENT band, so during pre-conditioning
        (target = the upcoming comfort band) they must NOT be used: a zone
        sitting inside the eco band would read as "in band" and the
        pre-conditioning would never fire.  Callers pass use_sensors=False
        while pre-conditioning.
        """
        if not use_sensors:
            return low, high
        idx = self._sensor_band_indices.get(vav_id)
        if idx is None:
            return low, high
        lo_idx, hi_idx = idx
        lo = float(obs[lo_idx]) if lo_idx is not None else low
        hi = float(obs[hi_idx]) if hi_idx is not None else high
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return low, high
        return lo, hi

    @staticmethod
    def _deadband_error(temp: float, low: float, high: float) -> float:
        """Signed K outside the band: >0 too cold, <0 too hot, 0 inside."""
        if temp < low:
            return low - temp
        if temp > high:
            return high - temp
        return 0.0

    # ------------------------------------------------------------ control core

    def _worst_zone_error(self, obs, ctrl, low: float, high: float,
                          use_sensors: bool = True) -> float:
        worst = 0.0
        for vav_id, obs_idx, _ in ctrl["zones"]:
            lo, hi = self._zone_band(obs, vav_id, low, high, use_sensors)
            err = self._deadband_error(self._zone_temp(obs, obs_idx), lo, hi)
            if abs(err) > abs(worst):
                worst = err
        return worst

    def _command_zones(self, continuous, zones, obs, low: float, high: float,
                       use_sensors: bool = True) -> None:
        """Per-VAV damper (proportional to |error|) and reheat (proportional
        to heating shortfall only).  A zone in band sits at its ventilation
        floor with the valve shut."""
        for vav_id, obs_idx, action_indices in zones:
            if not action_indices:
                continue
            lo, hi = self._zone_band(obs, vav_id, low, high, use_sensors)
            zone_error = self._deadband_error(self._zone_temp(obs, obs_idx), lo, hi)
            damper = min(1.0, abs(zone_error) / self.zone_damper_full_k)
            damper_idx = action_indices.get("supply_air_damper_percentage_command")
            if damper_idx is not None:
                continuous[damper_idx] = 2.0 * damper - 1.0
            reheat_idx = action_indices.get("reheat_valve_percentage_command")
            if reheat_idx is not None:
                opening = min(1.0, max(0.0, zone_error) / self.reheat_full_k)
                continuous[reheat_idx] = 2.0 * opening - 1.0

    def _recirculation(self, obs, ctrl, u: float, low: float, high: float) -> float:
        """Proportional economizer.  Only while cooling and only when the
        outside air is cold enough to pay for the extra fan work."""
        if u >= 0.0:
            return 1.0  # heating (or idle): hold the heat in
        outside = self._outside_temp(obs)
        if outside is None:
            return 1.0
        inside = float(
            np.mean([self._zone_temp(obs, j) for j in ctrl["zone_obs_indices"]])
        ) if ctrl["zone_obs_indices"] else None
        if inside is None or not np.isfinite(outside) or outside >= inside:
            return 1.0
        pressure_cmd = float(np.clip(2.0 * abs(u) - 1.0, -1.0, 1.0))
        pressure_pa = max(
            (pressure_cmd + 1.0) * 0.5 * STATIC_PRESSURE_MAX_PA,
            MIN_STATIC_PRESSURE_PA,
        )
        if (inside - outside) <= pressure_pa * ECON_K_PER_PA:
            return 1.0
        target = SUPPLY_AIR_MIN_K + (u + 1.0) * 0.5 * (
            SUPPLY_AIR_MAX_K - SUPPLY_AIR_MIN_K
        )
        target = max(target, low)  # never mix below the band's lower edge
        if inside <= target:
            return 1.0
        frac_outside = (inside - target) / (inside - outside)
        recirculation = float(np.clip(1.0 - frac_outside, 0.0, 1.0))
        return 2.0 * recirculation - 1.0

    def _heating_shortfall(self, obs, low: float, high: float,
                           use_sensors: bool = True) -> float:
        """Building-wide mean heating shortfall (K below the band)."""
        shortfalls = []
        for ctrl in self.ahu_controllers:
            for vav_id, obs_idx, _ in ctrl["zones"]:
                lo, _hi = self._zone_band(obs, vav_id, low, high, use_sensors)
                shortfalls.append(max(0.0, lo - self._zone_temp(obs, obs_idx)))
        return float(np.mean(shortfalls)) if shortfalls else 0.0

    def _water_fraction(self, shortfall: float) -> float:
        """Hot-water loop temperature as a fraction of its range.

        Greater of demand (mean shortfall over the proportional band) and
        saturation (how hard the air side is already working, lower quartile
        over AHUs so one saturated handler cannot pin the loop hot).
        """
        demand = shortfall / self.water_reset_full_k
        saturation = 0.0
        if self._last_efforts and self.water_effort_weight:
            heating = [max(0.0, u) for u in self._last_efforts]
            saturation = self.water_effort_weight * float(
                np.quantile(heating, self.water_effort_quantile)
            )
        return float(np.clip(max(demand, saturation), 0.0, 1.0))

    def _schedule_mode(self, obs):
        """Return (low, high, occupied, pre_conditioning).

        Bands come from the simulator's own comfort flags when available
        (they encode weekends/holidays/timezone correctly); hour/day
        reconstruction is only used to extend pre-conditioning in the cold.
        """
        comfort_now = (
            bool(obs[self._comfort_now_idx] > 0.5)
            if self._comfort_now_idx is not None
            else None
        )
        comfort_soon = (
            bool(obs[self._comfort_soon_idx] > 0.5)
            if self._comfort_soon_idx is not None
            else None
        )
        occupants = self._occupancy(obs)
        occupied = True if occupants is None else occupants >= 0.5

        if comfort_now is None:
            # No flags: reconstruct from hour/day (weekday + work hours).
            hour, day = self._hour_and_day(obs)
            if hour is None:
                return (*self.comfort_temp_window, occupied, False)
            comfort_now = (
                day is not None
                and day < 5
                and self.morning_start_hour <= hour < self.evening_start_hour
            )
            comfort_soon = False

        pre_conditioning = bool(comfort_soon and not comfort_now)
        if comfort_now or pre_conditioning:
            return (*self.comfort_temp_window, occupied, pre_conditioning)

        # Eco hours.  In cold weather, start pre-conditioning early so a
        # cold-soaked building is back in band when occupants arrive.
        outside = self._outside_temp(obs)
        if outside is not None:
            hour, day = self._hour_and_day(obs)
            if hour is not None and day is not None and day < 5:
                hours_to_morning = self.morning_start_hour - hour
                if 0 < hours_to_morning <= 3:
                    oat_c = outside - 273.15
                    for threshold_c, lead_h in self.cold_precon:
                        if oat_c < threshold_c and hours_to_morning <= lead_h:
                            return (*self.comfort_temp_window, occupied, True)
        return (*self.eco_temp_window, occupied, False)

    def _safe_action(self) -> dict[str, np.ndarray]:
        """Fallback if anything goes wrong mid-step: run everything at
        moderate settings rather than crashing the episode."""
        continuous = np.zeros(self.n_continuous, dtype=np.float32)
        discrete = np.ones(self.n_discrete, dtype=np.int32)
        for idx in self.recirculation_indices:
            continuous[idx] = 1.0
        return {"continuous_action": continuous, "discrete_action": discrete}

    def _shutdown_action(self) -> dict[str, np.ndarray]:
        """Whole plant off: dampers at floor, valves shut, loop cold."""
        continuous = np.full(self.n_continuous, -1.0, dtype=np.float32)
        for idx in self.recirculation_indices:
            continuous[idx] = 1.0
        discrete = np.zeros(self.n_discrete, dtype=np.int32)
        return {"continuous_action": continuous, "discrete_action": discrete}

    def get_action(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        try:
            return self._get_action(obs)
        except Exception:
            self._error_count += 1
            return self._safe_action()

    def _get_action(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        obs = np.asarray(obs, dtype=np.float64)
        low_temp, high_temp, occupied, pre_conditioning = self._schedule_mode(obs)
        # The setpoint sensors report the CURRENT band.  While
        # pre-conditioning the target is the upcoming comfort band, so the
        # sensors must not override it (see _zone_band).
        use_sensors = not pre_conditioning

        # Unoccupied + eco hours + nobody arriving within the hour: the
        # comfort term is exactly zero, so conditioning burns energy and
        # carbon for nothing.  Shut the plant down.
        comfort_now = (
            bool(obs[self._comfort_now_idx] > 0.5)
            if self._comfort_now_idx is not None
            else True
        )
        comfort_soon = (
            bool(obs[self._comfort_soon_idx] > 0.5)
            if self._comfort_soon_idx is not None
            else False
        )
        if not occupied and not comfort_now and not comfort_soon:
            # ...unless the cold pre-conditioning extension already fired,
            # in which case _schedule_mode returned the comfort band.
            if (low_temp, high_temp) == tuple(self.eco_temp_window):
                return self._shutdown_action()

        continuous = np.zeros(self.n_continuous, dtype=np.float32)
        discrete = np.zeros(self.n_discrete, dtype=np.int32)
        for idx in self.recirculation_indices:
            continuous[idx] = 1.0  # ventilation cap default; loop overwrites

        any_heating = False
        any_active = False
        self._last_efforts = []

        for i, ctrl in enumerate(self.ahu_controllers):
            if not ctrl["zone_obs_indices"]:
                continue
            self._command_zones(
                continuous, ctrl["zones"], obs, low_temp, high_temp,
                use_sensors,
            )
            error = self._worst_zone_error(
                obs, ctrl, low_temp, high_temp, use_sensors
            )

            state = self.pid_state[i]
            if abs(error) <= self.run_deadband_k:
                state["integral"] *= 0.5
                state["prev_error"] = 0.0
                continuous[ctrl["temp_idx"]] = -1.0
                if ctrl["pressure_idx"] is not None:
                    continuous[ctrl["pressure_idx"]] = -1.0
                discrete[ctrl["run_discrete_idx"]] = 0
                continue

            state["integral"] = float(
                np.clip(
                    state["integral"] + error,
                    -self.integral_limit,
                    self.integral_limit,
                )
            )
            derivative = error - state["prev_error"]
            state["prev_error"] = error
            u = float(
                np.clip(
                    self.kp * error + self.ki * state["integral"] + self.kd * derivative,
                    -1.0,
                    1.0,
                )
            )

            continuous[ctrl["temp_idx"]] = u
            self._last_efforts.append(u)
            if ctrl["pressure_idx"] is not None:
                continuous[ctrl["pressure_idx"]] = float(
                    np.clip(2.0 * abs(u) - 1.0, -1.0, 1.0)
                )
            discrete[ctrl["run_discrete_idx"]] = 1
            if ctrl.get("recirc_idx") is not None:
                continuous[ctrl["recirc_idx"]] = self._recirculation(
                    obs, ctrl, u, low_temp, high_temp
                )

            any_active = True
            if error > 0:
                any_heating = True

        # Hot-water loop: demand/saturation reset while heating, cold loop
        # and pump standby otherwise.
        shortfall = self._heating_shortfall(
            obs, low_temp, high_temp, use_sensors
        )
        heating_demand = any_heating or shortfall > 0.05
        if self.water_temp_idx is not None:
            if heating_demand:
                frac = self._water_fraction(shortfall)
                continuous[self.water_temp_idx] = np.float32(2.0 * frac - 1.0)
            else:
                continuous[self.water_temp_idx] = -1.0
        if self.water_pressure_idx is not None:
            continuous[self.water_pressure_idx] = 1.0 if heating_demand else -1.0
        if 0 <= self.water_discrete_idx < self.n_discrete:
            discrete[self.water_discrete_idx] = 1 if heating_demand else 0

        return {"continuous_action": continuous, "discrete_action": discrete}
