# Submission Write-up — Adaptive Pre-conditioning Policy

NeurIPS 2026 Smart Buildings Competition, main leaderboard track.
Author: adt (GitHub: darendrarp). Code: `policy.py` (+ Kaggle notebook
`submission_notebook.ipynb`, generated from it).

## 1. Approach

A single deterministic rule-based controller, numpy only, no learned
weights, no per-building tuning. The same code runs on every building: all
indices, bands, and normalization constants are read from the
`building_meta_data` dict the competition hands the policy at construction
time, so the policy adapts to each building's AHU count, zone count, and
schedule instead of memorizing a layout.

### 1.1 Per-zone sensor bands

Each zone is judged against its own heating/cooling setpoint sensors
(raw kelvin), falling back to the schedule band from metadata when a sensor
is missing or invalid. The reward measures deviation from those sensors,
so reading them keeps the controller and the scorer agreeing on where the
band edge is — including weekends and holidays, where reconstructing the
band from hour/day can disagree with the simulator (the schedule applies
its own timezone and holiday calendar).

One exception: during pre-conditioning the sensors are ignored, because
they report the *current* (eco) band while the target is the upcoming
comfort band. A zone inside the eco band would otherwise read as "in
band" and pre-conditioning would never fire.

### 1.2 Worst-zone AHU loop

Each air handler runs an independent PID loop (kp 0.5, ki 0.02, kd 0.1,
integral clamped and decayed in band — the proven gains from the shipped
PID baseline) on the signed error of its **worst** zone: the zone furthest
outside its band, sign preserved. The comfort penalty is linear to 5 K
outside the band and quadratic and unbounded past it, summed over zones,
so one runaway zone can dominate an episode while the mean looks fine.
The shipped `pid_water` baseline tracks the mean; this policy does not.

Fan static pressure scales with control effort |u| (min at |u| = 0, max at
|u| = 1) instead of being pinned at maximum while active, since fan power
is flow × pressure and a gently recovering zone does not need full blast.
The AHU idles (discrete off) while all its zones sit inside the band
within a 0.05 K deadband.

### 1.3 Correct economizer

The outside-air temperature sensor is **normalized** (mean 295 K,
std 10 K in the shipped configs — see the `outside_air_temperature_sensor`
normalizer in any shipped `config.gin`). This policy denormalizes it using
the constants in `observation_normalization_metadata`, with a plausibility
guard that falls back to the raw value if denormalization lands outside
any plausible temperature. While cooling, if the outside air is colder
than the space by more than the fan-work breakeven
(pressure × 0.00322 K/Pa), the recirculation is solved proportionally to
land the mixed air exactly on the supply-air target — never below the
band's lower edge, so cooling is never traded for a reheat bill. While
heating, recirculation stays at the ventilation cap to hold heat in.

Note: the shipped `pid_water` baseline reads the normalized OAT value as
kelvin, which makes its economizer nearly inert (the "outside" temperature
it sees is roughly −3…+1 instead of ~273–313 K). Fixing that is one of the
expected gains over the baseline.

### 1.4 Occupancy-aware shutdown

The comfort term is exactly zero when the building is unoccupied, so
outside comfort hours, when the occupancy count reads zero and comfort
mode is not starting within the hour, the whole plant shuts down: AHUs
off, pump off, dampers at their floor, reheat valves shut, water loop
cold. The shipped baselines keep holding the eco band all night for
nobody. During comfort hours the plant always conditions normally, even
if the count momentarily reads zero (occupants may be arriving).

### 1.5 Pre-conditioning

`comfort_mode_soon` — comfort mode starting within 60 minutes, computed
by the simulator with the correct timezone and holidays — switches the
target to the comfort band early, so zones are already in band when
occupants arrive instead of starting cold at 06:00 with the penalty
clock running. In cold weather the window extends using the hour-of-day
features: 2 h before the morning start below 2 °C outside, 3 h below
−10 °C, because a cold-soaked building cannot recover in one hour.

### 1.6 Zone layer + hot-water loop

- **Damper:** proportional to the zone's own |error|, full open at 2 K.
  A zone in band sits at its 10% ventilation floor.
- **Reheat valve:** proportional to the heating shortfall only, full open
  at 2 K below the heating setpoint. A binary valve over-heats a zone
  that is only 0.3 K cold and drives it out the other side of its band.
- **Hot water:** reset to the greater of demand (mean zone shortfall over
  a 4 K band) and air-side saturation (lower quartile of per-AHU heating
  efforts — how hard the air side is already working). The loop runs only
  as hot as the worst zone requires; the pump runs only while some zone
  calls for heat (standby otherwise).

### 1.7 Robustness

- Every action slot is written every step (the zero action is the middle
  of each range, not "off").
- Non-finite observations are sanitized (a bad zone reading falls back to
  the normalization mean, i.e. "in band"); all outputs are clipped to
  their ranges; any unexpected exception in `get_action` returns a safe
  "everything on, moderate" action instead of crashing the episode.
- Fully deterministic: no randomness, no wall-clock use.

## 2. What was deliberately NOT done

- **No reinforcement learning.** An RL policy needs many simulator
  rollouts to train and to verify; the real simulator could not be run
  during development (see §4), so an RL entry would have shipped
  unverified. The rule-based policy is verifiable by construction and by
  unit tests. RL (e.g. PPO with building-metadata conditioning, trained
  on procedurally generated buildings) is the natural next step once the
  real simulator is available — see §5.
- **No per-building or per-climate tuning.** Anything fitted to the six
  training buildings would not transfer to the six unseen ones. All
  thresholds are physical (breakeven K/Pa, band edges, ventilation
  floors), not fitted.
- **No weather-label or building-id use.** The observation carries none,
  and hardcoding a reference to a held-out config is disqualifying.

## 3. Local evaluation (mock simulator)

The policy was evaluated on a simplified mock simulator written from
scratch (`mock_env.py`: first-order thermal zone model, AHU supply-air +
VAV reheat, fan/chiller/boiler energy, and the documented reward shape —
productivity regret linear to 5 K then quadratic unbounded, weights
0.6/0.2/0.2). The mock baselines (`baselines_mock.py`) are simplified
re-implementations of the shipped baselines from their documented
behaviour, including `pid_water`'s as-documented economizer.

Configuration: 6 mock buildings (seeds 11, 23, 37, 41, 53, 67; 2–5 AHUs,
12–40 zones each) × 5 station archetypes (fairbanks, denver, atlanta,
phoenix, singapore) = 30 scenarios; 7-day episodes (2,016 steps) and a
30-day confirmation run (8,640 steps, the real episode length).

Score = mean over scenarios of |mean per-step reward|, lower is better.

### 7-day episodes (2,016 steps), 30 scenarios — run 19 Nov 2026 (229.7 s)

| Policy | Score | Worst scenario | Mean comfort regret | Mean energy (norm) |
| --- | --- | --- | --- | --- |
| **policy (ours)** | **0.10886** | 0.27966 | −0.0415 | 0.2161 |
| pid_water (mock) | 0.19277 | 0.55523 | −0.0483 | 0.4250 |
| pid (mock) | 0.20770 | 0.55563 | −0.0450 | 0.4622 |
| fixed_setpoint (mock) | 0.23065 | 0.58830 | −0.0449 | 0.5193 |
| all_off (mock) | 2.12752 | 9.29309 | −3.5459 | 0.0000 |

Our policy beat the mock `pid_water` on **30/30 scenarios** and in every
station archetype:

| Station | policy | pid_water | pid | fixed_setpoint | all_off |
| --- | --- | --- | --- | --- | --- |
| atlanta | 0.0403 | 0.0446 | 0.0499 | 0.0617 | 0.3825 |
| denver | 0.0740 | 0.0925 | 0.1193 | 0.1395 | 2.4106 |
| fairbanks | 0.1208 | 0.1578 | 0.2004 | 0.2296 | 7.4093 |
| phoenix | 0.1365 | 0.3675 | 0.3674 | 0.3904 | 0.2446 |
| singapore | 0.1727 | 0.3015 | 0.3015 | 0.3320 | 0.1906 |

The win is largest in hot climates, where the baselines keep cooling all
night to hold the eco band while our policy shuts down (zero occupants)
and pre-cools before occupants arrive. Comfort is slightly better too,
not just cheaper — the worst-zone loop and pre-conditioning keep
excursions small.

### 30-day episodes (8,640 steps, the real episode length)

Pending — run in progress at the time of writing; results will be added
to `results/mock_evaluation_30d.json` and this section updated.

Full per-scenario numbers: `results/mock_evaluation_7d.json`
(and `results/mock_evaluation_30d.json` once the run finishes).

**These are mock numbers.** They rank policies against each other on the
same mock only. They are NOT comparable to the real leaderboard or to the
shipped `baseline_returns.json` (real training-set scores: bang_bang
0.22520, fixed_setpoint 0.06592, pid 0.06736, pid_water 0.05414).

## 4. Validation status (honest)

- **Unit tests:** 19/19 pass (`python -m unittest discover -s tests -v`):
  interface shapes/dtypes/ranges across building sizes (K = 1–6,
  V = 1–40), hybrid-action validity, determinism, NaN/inf/extreme
  observations, missing optional metadata, and behavioural checks
  (cold zone heats, hot zone cools, in-band idles, unoccupied-night
  shutdown, pre-conditioning, economizer on/off, water standby,
  worst-zone drive). The notebook is tested too: it must execute
  end-to-end and its Policy cell must be byte-identical to `policy.py`.
- **Real simulator:** NOT run. The starter-kit clone failed an
  execution-risk check in the development VM, so its code was read but
  never executed, imported, or installed. The policy's real-simulator
  behaviour is reasoned from the documented interface, not measured.
- **Real leaderboard:** unverified. The first Kaggle submission is the
  real test.

How to validate on the real simulator (not done here):

```bash
git clone https://github.com/smart-buildings-2026/starter-kit.git
cd starter-kit && pip install -e .
# copy policy.py to smart_buildings/baselines/adaptive_precon.py,
# register it in POLICY_MAP in smart_buildings/evaluate.py, then:
python -m smart_buildings.evaluate --policies adaptive_precon pid_water
python score_leaderboard.py smart_buildings/results --policy adaptive_precon
```

## 5. Next steps (if iterating from leaderboard feedback)

1. Submit the notebook, read the public score, and compare against the
   0.05414 `pid_water` training reference and the ~0.005 public top 8.
2. If the score is close to the baselines, the first suspects are the
   shutdown/pre-conditioning balance (too aggressive a shutdown shows up
   as morning comfort loss in cold climates) and the PID gains.
3. With the real simulator available: grid-search kp/ki and the
   pre-conditioning lead on held-out generated buildings (never the
   shipped six alone), then consider a learned residual on top of this
   controller, trained on procedurally generated buildings.
4. Keep every change general: any threshold must be justified physically
   or validated on held-out generated buildings, never fitted to the
   shipped set.

## 6. Reproduce

```bash
python -m unittest discover -s tests -v
python evaluate_mock.py --days 7 --scenarios 30
python build_notebook.py   # regenerates submission_notebook.ipynb
```

Requires only numpy (system Python 3.12 used here; the notebook needs
nothing beyond the Kaggle image).
