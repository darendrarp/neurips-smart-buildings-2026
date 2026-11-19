#!/usr/bin/env python3
"""Run the policy and the mock baselines on mock scenarios and report the
leaderboard-style score (mean over scenarios of |mean per-step reward|,
lower is better).

MOCK ONLY -- see mock_env.py.  Absolute numbers are not comparable to the
real leaderboard or to the shipped baseline_returns.json.  The comparison
is only valid between policies run here, on the same mock scenarios.

Usage:
    python evaluate_mock.py [--days 7] [--scenarios 30] [--out results/...]
    python evaluate_mock.py --policies policy pid_water --days 30
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from baselines_mock import MOCK_BASELINES
from mock_env import STATIONS, MockBuilding
from policy import Policy

POLICIES = {"policy": Policy, **MOCK_BASELINES}

# 6 building seeds x 5 stations = 30 scenarios, mirroring the shipped set.
BUILDING_SEEDS = [11, 23, 37, 41, 53, 67]


def build_scenarios(count: int, days: int):
    scenarios = []
    for seed in BUILDING_SEEDS:
        for station in STATIONS:
            scenarios.append((seed, station))
    scenarios = scenarios[:count]
    return [(f"mock_b{seed}_{station}",
             MockBuilding(seed=seed, station=station, num_days=days))
            for seed, station in scenarios]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--scenarios", type=int, default=30)
    ap.add_argument("--policies", nargs="+", default=list(POLICIES),
                    choices=list(POLICIES))
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "results" / "mock_evaluation.json")
    args = ap.parse_args()

    scenarios = build_scenarios(args.scenarios, args.days)
    print(f"mock eval: {len(scenarios)} scenarios x {args.days} days "
          f"({args.days * 288} steps each)")
    results: dict[str, dict] = {}
    t_start = time.time()
    for name, building in scenarios:
        meta = building.build_metadata()
        row = {}
        for key in args.policies:
            policy = POLICIES[key](meta, None)
            res = building.run(policy)
            row[key] = res
        results[name] = row
        scores = "  ".join(f"{k}={row[k]['score']:.4f}" for k in args.policies)
        print(f"  {name:<24} {scores}")
    elapsed = time.time() - t_start

    summary = {}
    for key in args.policies:
        scores = [results[s][key]["score"] for s in results]
        summary[key] = {
            "score": sum(scores) / len(scores),
            "worst_scenario": max(scores),
            "best_scenario": min(scores),
            "mean_comfort_regret": sum(
                results[s][key]["mean_comfort_regret"] for s in results
            ) / len(results),
            "mean_energy_norm": sum(
                results[s][key]["mean_energy_norm"] for s in results
            ) / len(results),
            "mean_carbon_norm": sum(
                results[s][key]["mean_carbon_norm"] for s in results
            ) / len(results),
        }

    print("\n=== MOCK leaderboard-style score (lower is better) ===")
    for key in sorted(summary, key=lambda k: summary[k]["score"]):
        s = summary[key]
        print(f"  {key:<15} {s['score']:.5f}  "
              f"(worst {s['worst_scenario']:.5f}, "
              f"comfort {s['mean_comfort_regret']:.4f}, "
              f"energy {s['mean_energy_norm']:.4f}, "
              f"carbon {s['mean_carbon_norm']:.4f})")
    print(f"\nelapsed {elapsed:.1f}s")

    payload = {
        "note": "MOCK evaluation only. Not comparable to the real leaderboard.",
        "days": args.days,
        "n_scenarios": len(scenarios),
        "elapsed_seconds": round(elapsed, 1),
        "summary": summary,
        "per_scenario": {
            name: {k: v["score"] for k, v in row.items()}
            for name, row in results.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
