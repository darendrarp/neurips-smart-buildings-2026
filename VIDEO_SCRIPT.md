# Video — Not Required for the Main Track

**No video is required or requested for the main leaderboard track of the
NeurIPS 2026 Smart Buildings Competition.** It is a Kaggle code
competition: the submission is a notebook, scored purely by the private
leaderboard (rules §1.5, §2; competition page). There is no video
deliverable in the rules, the overview, or the starter kit. Do not spend
time producing one for the submission itself.

The only call-related obligation is for **winners**: after validation, a
winner *may* be asked to discuss results via a recorded or panel call
with the sponsors (rules §2.5.b, remote — not in-person). That is a
post-win step, not a submission requirement.

## Write-up draft (for the Kaggle notebook description / forum post)

Use this text when publishing the notebook or describing the entry. It is
the honest, short version of `SUBMISSION.md`.

---

**Adaptive Pre-conditioning Policy — NeurIPS 2026 Smart Buildings**

A single deterministic rule-based HVAC controller (numpy only, no learned
weights) that runs unchanged on any building in the competition:

- **Per-zone sensor bands** — each zone is judged against its own
  heating/cooling setpoint sensors, the band the scorer uses.
- **Worst-zone AHU control** — each air handler's PID tracks its worst
  zone, because the comfort penalty is quadratic and unbounded past 5 K.
- **Corrected economizer** — the outside-air sensor is normalized
  (mean 295 K, std 10 K); it is denormalized before the free-cooling
  math. The shipped `pid_water` baseline reads it as raw kelvin, which
  leaves its economizer nearly inert.
- **Occupancy-aware shutdown** — with zero occupants outside comfort
  hours, the plant shuts down (comfort penalty is exactly zero then)
  instead of holding the eco band for nobody.
- **Pre-conditioning** — `comfort_mode_soon` plus a cold-weather
  extension brings zones into the comfort band before occupants arrive.
- **Proportional reheat + demand/saturation hot-water reset + pump
  standby.**

Validation: 19/19 unit tests pass; evaluated on a simplified mock
simulator against re-implemented baselines. The real leaderboard score is
unverified — this submission is the real test. Full write-up and code:
https://github.com/darendrarp/neurips-smart-buildings-2026

---

## If a winner call is ever needed (talking points, <3 min)

1. **Problem (20 s):** one policy must control any commercial building's
   HVAC — different AHU/zone counts, climates, materials — minimizing
   energy + carbon + discomfort. The comfort penalty is unbounded, so one
   neglected zone can dominate a whole month.
2. **Idea (60 s):** a rule-based adaptive controller. Worst-zone PID per
   air handler; per-zone sensor bands; a corrected economizer (the OAT
   sensor is normalized — denormalize it first); shut the plant down when
   nobody is in the building; pre-condition before occupants arrive;
   reset the hot-water loop against demand, not a fixed temperature.
3. **Why it generalizes (40 s):** no fitted constants, no building IDs,
   no weather labels — every threshold is physical, and every index is
   read from the building metadata at runtime.
4. **Honesty (20 s):** validated on a mock simulator and unit tests; the
   leaderboard result is the real validation. The code is public and
   reproducible.
