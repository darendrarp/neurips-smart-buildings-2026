# NeurIPS 2026 Smart Buildings — Adaptive Pre-conditioning Policy

Submission package for the **main leaderboard track** ($25,000) of the
[Smart Buildings Competition](https://www.kaggle.com/competitions/neurips-2026-smart-buildings)
(NeurIPS 2026, University of Washington).

One rule-based HVAC policy (`Policy` in `policy.py`, numpy only) that
generalizes across buildings it has never seen: per-zone sensor bands,
worst-zone AHU control, a corrected economizer, occupancy-aware shutdown,
pre-conditioning, and a demand-driven hot-water reset. No learned weights,
no building-specific tuning, no randomness.

## Status: submission-ready via browser (Kaggle notebook)

| Item | Status |
| --- | --- |
| Policy (`policy.py`, interface `Policy`) | Done, 19/19 unit tests pass |
| Kaggle notebook (`submission_notebook.ipynb`) | Done, executes + smoke test passes |
| Mock evaluation vs 4 baselines | Done — see `results/` (mock only, not leaderboard) |
| Real-simulator evaluation | **NOT done** — see caveat below |
| Real leaderboard score | **Unverified** — first Kaggle submission is the real test |
| Submitted to Kaggle | No — needs the user's Kaggle account (see `SUBMIT_STEPS.md`) |

## ⚠️ Honest caveats — read before expecting a prize

1. **The policy has never run on the real simulator.** During development
   in this VM, the cloned starter-kit code failed an execution-risk check,
   so it was read (as text) but never executed, imported, or installed.
   The policy is validated against a simplified mock simulator written
   from scratch (`mock_env.py`) plus 19 unit tests. Its behaviour on the
   real simulator is reasoned from the documented interface, not measured.
   The first Kaggle submission — where the notebook runs against the real
   simulator — is the real validation. A local real-simulator run is
   possible later (clone the starter kit, `pip install -e .`,
   `python -m smart_buildings.evaluate --policies transfer` with this
   policy registered), but it was not performed here.
2. **Top 8 is a stretch.** The public leaderboard (19 Nov 2026, 836 teams,
   33% of test data) has #1 at **0.00447** and #8 at **0.00484**, while the
   best shipped baseline (`pid_water`) scores **0.05414** on the 30
   training scenarios. The leaders have iterated dozens of times each with
   sophisticated controllers. This entry is a solid, principled foundation
   that should beat all four shipped baselines — it is not a promised
   top-8 finish. Iterate from leaderboard feedback.
3. **Mock numbers are not leaderboard numbers.** The mock simulator is a
   first-order thermal model with invented energy scales. Its scores rank
   policies against each other on the same mock only.
4. **Entry deadline and exact final time are unverified.** The final date
   (15 January 2027) is confirmed in the rules and on the official site,
   but the exact time and the entry deadline live on Kaggle's
   Overview > Timeline page, which could not be read from this VM.
   Confirm both in the browser before submitting (see `SUBMIT_STEPS.md`).
   Kaggle deadlines are conventionally 11:59 PM UTC — do not rely on that
   without checking.
5. **No video is required** for the main track (Kaggle code competition).
   `VIDEO_SCRIPT.md` says so explicitly and holds a write-up draft instead.
6. **No paper is required** for the main leaderboard track. Prizes are
   decided purely by the private leaderboard.

## The policy (summary)

Full design rationale: `SUBMISSION.md`.

1. **Per-zone sensor bands** — each zone is judged against its own
   heating/cooling setpoint sensors (raw kelvin, the band the scorer
   uses); the schedule band is the fallback. During pre-conditioning the
   sensors are ignored because they report the *current* (eco) band while
   the target is the upcoming comfort band.
2. **Worst-zone AHU loop** — each AHU's PID tracks its worst zone (largest
   signed excursion), not the mean: the comfort penalty is quadratic and
   unbounded past 5 K, so one runaway zone can dominate an episode.
3. **Correct economizer** — the outside-air sensor is *normalized*
   (mean 295 K, std 10 K in the shipped configs). This policy denormalizes
   it (with a raw-kelvin plausibility fallback) before doing free-cooling
   math. The shipped `pid_water` baseline reads the normalized value as
   kelvin, which leaves its economizer nearly inert.
4. **Occupancy-aware shutdown** — the comfort term is exactly zero with
   zero occupants, so outside comfort hours, with nobody in the building
   and nobody arriving within the hour, the whole plant shuts down instead
   of holding the eco band for nobody.
5. **Pre-conditioning** — `comfort_mode_soon` (comfort starting within
   60 min, computed by the simulator with correct timezone/holidays)
   switches the target to the comfort band early; in cold weather the
   window extends to 2–3 h before the morning start.
6. **Proportional reheat + demand/saturation hot-water reset + pump
   standby** — the loop runs only as hot as the worst zone requires, and
   the pump runs only while some zone calls for heat.

## Layout

| File | What it is |
| --- | --- |
| `policy.py` | The `Policy` class (the submission). Numpy only. |
| `submission_notebook.ipynb` | Kaggle-ready notebook (generated from `policy.py` by `build_notebook.py`). |
| `build_notebook.py` | Regenerates the notebook from `policy.py`. |
| `mock_env.py` | Simplified mock simulator + reward (local sanity checks only). |
| `baselines_mock.py` | Simplified re-implementations of the 4 shipped baselines + `all_off`, for mock comparison. |
| `evaluate_mock.py` | Runs policy vs mock baselines on mock scenarios. |
| `tests/test_policy.py` | 19 unit tests (interface, behaviour, mock rollout, notebook). |
| `results/` | Mock evaluation outputs. |
| `SUBMISSION.md` | Approach write-up + mock results. |
| `SUBMIT_STEPS.md` | Step-by-step Kaggle submission instructions for the user. |
| `VIDEO_SCRIPT.md` | Video not required — write-up draft instead. |

## Quickstart

```bash
# Tests (numpy only, no pytest needed)
python -m unittest discover -s tests -v

# Mock evaluation: 30 scenarios x 7 days, policy vs all mock baselines
python evaluate_mock.py --days 7 --scenarios 30

# Full-length mock evaluation (30 days, like the real episodes)
python evaluate_mock.py --days 30 --scenarios 30 --out results/mock_evaluation_30d.json

# Regenerate the Kaggle notebook after changing policy.py
python build_notebook.py
```

## Mock evaluation results

Leaderboard-style score = mean over scenarios of |mean per-step reward|,
lower is better. **Mock only — not comparable to the real leaderboard.**

<!-- MOCK_RESULTS: 7-day run 19 Nov 2026, 229.7 s; 30-day run pending -->

| Policy | 7-day mock score (30 scenarios) | 30-day mock score |
| --- | --- | --- |
| **policy (ours)** | **0.10886** (worst 0.27966) | _pending_ |
| pid_water (mock) | 0.19277 (worst 0.55523) | _pending_ |
| pid (mock) | 0.20770 (worst 0.55563) | _pending_ |
| fixed_setpoint (mock) | 0.23065 (worst 0.58830) | _pending_ |
| bang_bang (mock) | n/a (not re-implemented) | n/a |
| all_off (mock) | 2.12752 (worst 9.29309) | _pending_ |

Our policy beat the mock `pid_water` on **30/30 scenarios** and in every
station (atlanta 0.040 vs 0.045, denver 0.074 vs 0.093, fairbanks 0.121
vs 0.158, phoenix 0.137 vs 0.368, singapore 0.173 vs 0.302). The win comes
mostly from energy (mean normalized energy 0.216 vs 0.425 — the
unoccupied-night shutdown) with slightly better comfort too (mean regret
−0.0415 vs −0.0483).

Reference — shipped baselines on the REAL training set (computed from the
starter kit's `baseline_returns.json`, 30 scenarios × 8,640 steps):

| Baseline | Training-set score |
| --- | --- |
| bang_bang | 0.22520 |
| fixed_setpoint | 0.06592 |
| pid | 0.06736 |
| pid_water | 0.05414 |

## Competition facts (verified 19 Nov 2026)

- Kaggle: https://www.kaggle.com/competitions/neurips-2026-smart-buildings
- Official site: https://smart-buildings-2026.github.io/competition/
- Starter kit: https://github.com/smart-buildings-2026/starter-kit
- Main leaderboard: **$25,000** — 1st $5,750, 2nd $3,750, 3rd $3,000,
  4th–8th $2,500 each. Paper track $15,000 (not our target).
- Final deadline: **15 January 2027** (exact time + entry deadline:
  confirm on the Timeline page).
- Metric: mean absolute error of per-scenario mean per-step reward vs 0,
  over 6 unseen buildings × 5 unseen weather stations (30 scenarios).
  Lower is better. Reward weights: comfort 0.6, energy 0.2, carbon 0.2.
- Limits: 5 submissions/day, 2 final submissions. Team max 5; solo OK.
  Free entry, no purchase necessary. Winner license CC-BY 4.0.
- Winner verification: prize-acceptance documents + W-8BEN (for
  non-US winners) — approved by the user.

## License

MIT (this package only). The starter kit is Apache 2.0 and is NOT
included here — it is referenced by URL only.
