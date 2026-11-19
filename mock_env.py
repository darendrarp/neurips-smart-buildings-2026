"""Simplified mock building simulator + reward, written from scratch for
LOCAL SANITY CHECKS ONLY.

This is NOT the competition simulator.  It is a first-order thermal model
(one temperature node per zone, AHU supply air + VAV reheat, fan/chiller/
boiler energy, and the documented reward shape: productivity regret linear
to 5 K outside the band then quadratic and unbounded, weights 0.6 / 0.2 /
0.2).  Its absolute numbers are NOT comparable to the real leaderboard or
to the shipped ``baseline_returns.json``.  It exists to:

  * prove the policy runs end-to-end over full episodes without crashing,
  * check the policy's qualitative behaviour (shutdown at night, pre-heat
    in the morning, economizer in mild weather, no runaway zones),
  * rank policy variants and simplified baselines against each other on
    the SAME mock, so tuning decisions are at least internally consistent.

The observation vector and the building metadata it produces mirror the
real interface (field names, index layout, normalization constants) so the
policy code under test is exactly the code that ships in the notebook.
"""

from __future__ import annotations

import numpy as np

STEP_MINUTES = 5
STEPS_PER_DAY = 24 * 60 // STEP_MINUTES  # 288

# Station archetypes: (mean OAT degC, diurnal amplitude degC, slow drift degC).
STATIONS = {
    "fairbanks": (-12.0, 6.0, 8.0),
    "denver": (2.0, 9.0, 6.0),
    "atlanta": (14.0, 7.0, 5.0),
    "phoenix": (30.0, 10.0, 4.0),
    "singapore": (28.0, 2.0, 1.5),
}

COMFORT_WINDOW = (294.0, 297.0)  # K (21-24 C)
ECO_WINDOW = (289.0, 298.0)      # K (16-25 C)
MORNING_HOUR = 6
EVENING_HOUR = 19

# Energy / carbon model (mock-scale, documented, not the real tariffs).
ELEC_PRICE = 0.15        # $/kWh
GAS_PRICE = 0.05         # $/kWh
ELEC_CARBON = 0.40       # kgCO2/kWh
GAS_CARBON = 0.185       # kgCO2/kWh
CHILLER_COP = 3.5
BOILER_EFF = 0.85
FAN_MAX_KW = 15.0        # per AHU at full static pressure
PUMP_KW = 2.0
PRODUCTIVITY_DEVIATION_REF_K = 5.0


def _cyclic(value: float, period: float):
    angle = 2.0 * np.pi * value / period
    return float(np.sin(angle)), float(np.cos(angle))


class MockBuilding:
    """One mock building: zones, AHUs, weather strip, schedule."""

    def __init__(self, seed: int, station: str, num_days: int = 30,
                 n_ahus: int | None = None, n_zones: int | None = None):
        rng = np.random.default_rng(seed)
        self.station = station
        self.num_days = num_days
        self.n_steps = num_days * STEPS_PER_DAY
        self.n_ahus = n_ahus if n_ahus is not None else int(rng.integers(2, 6))
        self.n_zones = n_zones if n_zones is not None else int(
            rng.integers(12, 41)
        )
        # Zone -> AHU assignment (contiguous blocks, like the real generator).
        self.zone_ahu = np.array(
            [min(i * self.n_ahus // self.n_zones, self.n_ahus - 1)
             for i in range(self.n_zones)]
        )
        # Thermal time constant per zone (hours): heavy to light construction.
        self.tau_hours = rng.uniform(2.0, 6.0, size=self.n_zones)
        # Air-side coupling: how fast supply air moves a zone (per step at
        # full flow).  Sized so full cooling moves ~1 K per 5-min step.
        self.air_gain = rng.uniform(0.10, 0.20, size=self.n_zones)
        # Internal gains when occupied (K per step).
        self.internal_gain = rng.uniform(0.02, 0.06, size=self.n_zones)
        # Which zones are ever occupied (some zones stay empty: storage etc).
        self.zone_ever_occupied = rng.random(self.n_zones) < 0.85
        # Episode starts on a Monday 06:00 (day 0 = Monday).
        self.start_hour = 6

        base_c, amp_c, drift_c = STATIONS[station]
        t_days = np.arange(self.n_steps) / STEPS_PER_DAY
        t_hours = np.arange(self.n_steps) * STEP_MINUTES / 60.0
        slow = drift_c * np.sin(2 * np.pi * t_days / 9.0 + 0.7 * seed)
        diurnal = amp_c * np.sin(2 * np.pi * (t_hours - 14.0) / 24.0)
        noise = rng.normal(0.0, 0.8, size=self.n_steps)
        self.oat_c = base_c + slow + diurnal + noise
        self.oat_k = self.oat_c + 273.15

        # Occupancy profile per step: fraction of zones occupied.
        self.occupancy_frac = np.zeros(self.n_steps)
        for s in range(self.n_steps):
            day = (s // STEPS_PER_DAY) % 7  # 0 = Monday
            hour = ((s % STEPS_PER_DAY) * STEP_MINUTES / 60.0 + self.start_hour) % 24
            if day < 5:
                if MORNING_HOUR <= hour < 9:
                    self.occupancy_frac[s] = (hour - MORNING_HOUR) / 3.0
                elif 9 <= hour < 16:
                    self.occupancy_frac[s] = 1.0
                elif 16 <= hour < EVENING_HOUR:
                    self.occupancy_frac[s] = (EVENING_HOUR - hour) / 3.0
        # Per-zone occupants (2 per occupied zone) and total count.
        self.zone_occupants = (
            self.zone_ever_occupied[None, :]
            * (self.occupancy_frac[:, None] > rng.random((1, self.n_zones)) * 0.3)
            * 2.0
        )
        # Smooth the per-zone presence a little: occupants present for whole
        # blocks rather than flickering step to step.
        self.zone_occupants = np.where(
            self.occupancy_frac[:, None] > 0.05,
            np.where(self.zone_ever_occupied[None, :], 2.0, 0.0)
            * np.clip(self.occupancy_frac[:, None] * 1.5, 0.0, 1.0),
            0.0,
        )
        self.total_occupants = self.zone_occupants.sum(axis=1)

        # Normalization maxima scale with plant size (mock analogue of the
        # real max_electricity_rate / max_natural_gas_rate).
        self.max_elec_kw = 100.0 + 20.0 * self.n_ahus
        self.max_gas_kw = 150.0 + 30.0 * self.n_ahus

    # ------------------------------------------------------------- schedule

    def _hour_day(self, step: int):
        day = (step // STEPS_PER_DAY) % 7
        hour = ((step % STEPS_PER_DAY) * STEP_MINUTES / 60.0 + self.start_hour) % 24
        return hour, day

    def is_comfort_mode(self, step: int) -> bool:
        hour, day = self._hour_day(step)
        return day < 5 and MORNING_HOUR <= hour < EVENING_HOUR

    def band(self, step: int):
        return COMFORT_WINDOW if self.is_comfort_mode(step) else ECO_WINDOW

    # ------------------------------------------------------- metadata + obs

    def build_metadata(self) -> dict:
        """Metadata dict mirroring evaluate.py's create_building_meta_data."""
        K, V = self.n_ahus, self.n_zones
        # Continuous layout (interleaved): [temp, pressure, recirc] * K,
        # then water temp + water pressure, then [damper, reheat] * V.
        temp_idx, pressure_idx, recirc_idx = [], [], []
        for k in range(K):
            temp_idx.append(3 * k)
            pressure_idx.append(3 * k + 1)
            recirc_idx.append(3 * k + 2)
        water_temp_idx = 3 * K
        water_pressure_idx = 3 * K + 1
        pressure_idx.append(water_pressure_idx)  # same catch-all as real env
        zone_action_indices = {}
        for v in range(V):
            base = 3 * K + 2 + 2 * v
            zone_action_indices[f"VAV_{v + 1}"] = {
                "supply_air_damper_percentage_command": base,
                "reheat_valve_percentage_command": base + 1,
            }
        ahu_zone_assignments = {}
        for k in range(K):
            rooms = [f"room_{v + 1}" for v in range(V) if self.zone_ahu[v] == k]
            ahu_zone_assignments[f"ahu_{k + 1}"] = rooms

        field_names = []
        for v in range(V):
            prefix = f"VAV_{v + 1}"
            field_names += [
                f"{prefix}_zone_air_temperature_sensor",
                f"{prefix}_zone_air_heating_temperature_setpoint_sensor",
                f"{prefix}_zone_air_cooling_temperature_setpoint_sensor",
                f"{prefix}_supply_air_damper_percentage_command",
                f"{prefix}_reheat_valve_percentage_command",
                f"{prefix}_supply_air_flowrate_setpoint",
            ]
        for k in range(K):
            prefix = f"ahs_ahu_{k + 1}"
            field_names += [
                f"{prefix}_outside_air_temperature_sensor",
                f"{prefix}_supply_air_temperature_sensor",
                f"{prefix}_supply_air_temperature_setpoint",
                f"{prefix}_supply_air_static_pressure_sensor",
                f"{prefix}_recirculation_percentage_command",
                f"{prefix}_run_status",
            ]
        field_names += [
            "comfort_mode_now",
            "comfort_mode_soon",
            "num_occupants",
            "hod_sin_000",
            "hod_cos_000",
            "dow_sin_000",
            "dow_cos_000",
        ]
        return {
            "n_discrete_action": K + 1,
            "n_continuous_action": 3 * K + 2 + 2 * V,
            "temp_indices": temp_idx,
            "pressure_indices": pressure_idx,
            "water_indices": [water_temp_idx],
            "recirculation_indices": recirc_idx,
            "zone_action_indices": zone_action_indices,
            "zone_temp_indices": [
                i for i, n in enumerate(field_names)
                if "zone_air_temperature" in n
            ],
            "observation_field_names": field_names,
            "floor_plan": None,
            "zone_map": None,
            "ahu_zone_assignments": ahu_zone_assignments,
            "num_floors": 3,
            "morning_start_hour": MORNING_HOUR,
            "evening_start_hour": EVENING_HOUR,
            "comfort_temp_window": COMFORT_WINDOW,
            "eco_temp_window": ECO_WINDOW,
            "time_zone": "UTC",
            "holidays": set(),
            "observation_normalization_metadata": {
                "zone_air_temperature_sensor": {
                    "sample_mean": 295.0, "sample_variance": 25.0,
                    "sample_std": 5.0,
                },
                "outside_air_temperature_sensor": {
                    "sample_mean": 295.0, "sample_variance": 100.0,
                    "sample_std": 10.0,
                },
                "supply_air_temperature_sensor": {
                    "sample_mean": 290.0, "sample_variance": 25.0,
                    "sample_std": 5.0,
                },
            },
            "action_normalization_metadata": {},
        }

    def observation(self, step: int, zone_temps: np.ndarray,
                    damper_pos: np.ndarray, reheat_pos: np.ndarray,
                    supply_temps: np.ndarray) -> np.ndarray:
        meta = self._meta
        names = meta["observation_field_names"]
        obs = np.zeros(len(names), dtype=np.float32)
        low, high = self.band(step)
        hour, day = self._hour_day(step)
        soon = self.is_comfort_mode(step + 12)  # +60 minutes
        hod_sin, hod_cos = _cyclic(hour, 24.0)
        dow_sin, dow_cos = _cyclic(day, 7.0)
        idx = {n: i for i, n in enumerate(names)}
        for v in range(self.n_zones):
            p = f"VAV_{v + 1}"
            obs[idx[f"{p}_zone_air_temperature_sensor"]] = (
                zone_temps[v] - 295.0) / 5.0
            obs[idx[f"{p}_zone_air_heating_temperature_setpoint_sensor"]] = low
            obs[idx[f"{p}_zone_air_cooling_temperature_setpoint_sensor"]] = high
            obs[idx[f"{p}_supply_air_damper_percentage_command"]] = (
                damper_pos[v] - 0.5) / 0.5
            obs[idx[f"{p}_reheat_valve_percentage_command"]] = (
                reheat_pos[v] - 0.5) / 0.5
            obs[idx[f"{p}_supply_air_flowrate_setpoint"]] = 1.0
        for k in range(self.n_ahus):
            p = f"ahs_ahu_{k + 1}"
            obs[idx[f"{p}_outside_air_temperature_sensor"]] = (
                self.oat_k[step] - 295.0) / 10.0
            obs[idx[f"{p}_supply_air_temperature_sensor"]] = (
                supply_temps[k] - 290.0) / 5.0
        obs[idx["comfort_mode_now"]] = 1.0 if self.is_comfort_mode(step) else 0.0
        obs[idx["comfort_mode_soon"]] = 1.0 if soon else 0.0
        obs[idx["num_occupants"]] = (self.total_occupants[step] - 125.0) / 126.0
        obs[idx["hod_sin_000"]] = hod_sin
        obs[idx["hod_cos_000"]] = hod_cos
        obs[idx["dow_sin_000"]] = dow_sin
        obs[idx["dow_cos_000"]] = dow_cos
        return obs

    # ------------------------------------------------------------- rollout

    def run(self, policy, steps: int | None = None) -> dict:
        """Roll out one policy; return per-step rewards and components."""
        n = self.n_steps if steps is None else min(steps, self.n_steps)
        self._meta = self.build_metadata()
        meta = self._meta
        K, V = self.n_ahus, self.n_zones
        dt_h = STEP_MINUTES / 60.0

        zone_temps = np.full(V, 295.0)  # start in band
        damper_pos = np.full(V, 0.1)
        reheat_pos = np.zeros(V)
        supply_temps = np.full(K, 293.0)

        rewards = np.zeros(n)
        comfort = np.zeros(n)
        energy = np.zeros(n)
        carbon = np.zeros(n)

        temp_idx = meta["temp_indices"]
        pressure_idx = meta["pressure_indices"][:K]
        recirc_idx = meta["recirculation_indices"]
        water_temp_idx = meta["water_indices"][0]

        for s in range(n):
            obs = self.observation(s, zone_temps, damper_pos, reheat_pos,
                                   supply_temps)
            action = policy.get_action(obs)
            cont = np.asarray(action["continuous_action"], dtype=np.float64)
            disc = np.asarray(action["discrete_action"], dtype=np.int64)
            cont = np.clip(cont, -1.0, 1.0)

            # --- decode actions to physical setpoints
            sat_set = 285.0 + (cont[temp_idx] + 1.0) * 0.5 * 20.0  # K
            pressure_pa = (cont[pressure_idx] + 1.0) * 0.5 * 2000.0
            recirc_frac = np.clip((cont[recirc_idx] + 1.0) * 0.5, 0.0, 0.7)
            water_temp = 310.0 + (cont[water_temp_idx] + 1.0) * 0.5 * 40.0
            ahu_on = disc[:K] > 0
            pump_on = disc[K] > 0

            damper_cmd = np.zeros(V)
            reheat_cmd = np.zeros(V)
            for v in range(V):
                za = meta["zone_action_indices"][f"VAV_{v + 1}"]
                damper_cmd[v] = (cont[za["supply_air_damper_percentage_command"]] + 1.0) * 0.5
                reheat_cmd[v] = (cont[za["reheat_valve_percentage_command"]] + 1.0) * 0.5
            damper_pos = np.clip(np.maximum(damper_cmd, 0.1), 0.0, 1.0)
            reheat_pos = np.clip(reheat_cmd, 0.0, 1.0)

            # --- AHU physics
            flow_frac = np.where(
                ahu_on,
                np.clip(np.maximum(pressure_pa, 200.0) / 2000.0, 0.0, 1.0),
                0.0,
            )
            fan_kw = float(np.sum(flow_frac * (np.maximum(pressure_pa, 0.0) / 2000.0)
                                  * FAN_MAX_KW * np.where(ahu_on, 1.0, 0.0)))
            chiller_kw = 0.0
            boiler_kw = 0.0
            discharge = np.zeros(V)
            for k in range(K):
                zones_k = self.zone_ahu == k
                t_return = float(np.mean(zone_temps[zones_k])) if zones_k.any() else 295.0
                mixed = recirc_frac[k] * t_return + (1.0 - recirc_frac[k]) * self.oat_k[s]
                supply = min(mixed, sat_set[k]) if ahu_on[k] else mixed
                supply_temps[k] = supply
                if ahu_on[k] and mixed > supply:
                    # Cooling delivered, kW thermal (mock scale ~ flow * dT).
                    cool_kw = flow_frac[k] * (mixed - supply) * 8.0 * max(int(zones_k.sum()), 1)
                    chiller_kw += cool_kw / CHILLER_COP
                for v in np.where(zones_k)[0]:
                    d = supply
                    if pump_on and reheat_pos[v] > 0:
                        d = min(308.15, supply + reheat_pos[v] * (water_temp - supply))
                        boiler_kw += (
                            reheat_pos[v] * flow_frac[k] * damper_pos[v]
                            * max(water_temp - supply, 0.0) * 0.35
                        )
                    discharge[v] = d
            boiler_kw /= BOILER_EFF
            pump_kw = PUMP_KW if pump_on else 0.0
            elec_kw = fan_kw + chiller_kw + pump_kw
            gas_kw = boiler_kw

            # --- zone temperature update
            oat = self.oat_k[s]
            drift = (oat - zone_temps) * (dt_h / self.tau_hours)
            flow_v = flow_frac[self.zone_ahu]
            air_effect = (
                self.air_gain * flow_v * damper_pos * (discharge - zone_temps)
            )
            occ_gain = self.internal_gain * np.clip(
                self.zone_occupants[s] / 2.0, 0.0, 1.0
            )
            zone_temps = zone_temps + drift + air_effect + occ_gain

            # --- reward (documented shape)
            low, high = self.band(s)
            dev = np.where(
                zone_temps < low, low - zone_temps,
                np.where(zone_temps > high, zone_temps - high, 0.0),
            )
            w = PRODUCTIVITY_DEVIATION_REF_K
            shortfall = np.where(
                dev <= w, dev / w, 1.0 + ((dev - w) / w) ** 2
            )
            occ = self.zone_occupants[s]
            if occ.sum() > 0:
                prod_regret = -float(np.sum(shortfall * occ) / occ.sum())
            else:
                prod_regret = 0.0
            energy_kwh_step = (elec_kw * ELEC_PRICE + gas_kw * GAS_PRICE) * dt_h
            max_cost_step = (
                self.max_elec_kw * ELEC_PRICE + self.max_gas_kw * GAS_PRICE
            ) * dt_h
            energy_norm = energy_kwh_step / max_cost_step
            carbon_step = (elec_kw * ELEC_CARBON + gas_kw * GAS_CARBON) * dt_h
            max_carbon_step = (
                self.max_elec_kw * ELEC_CARBON + self.max_gas_kw * GAS_CARBON
            ) * dt_h
            carbon_norm = carbon_step / max_carbon_step
            rewards[s] = (
                0.6 * prod_regret - 0.2 * energy_norm - 0.2 * carbon_norm
            )
            comfort[s] = prod_regret
            energy[s] = energy_norm
            carbon[s] = carbon_norm

        return {
            "total_reward": float(rewards.sum()),
            "mean_reward": float(rewards.mean()),
            "score": float(abs(rewards.mean())),
            "mean_comfort_regret": float(comfort.mean()),
            "mean_energy_norm": float(energy.mean()),
            "mean_carbon_norm": float(carbon.mean()),
            "n_steps": n,
        }
