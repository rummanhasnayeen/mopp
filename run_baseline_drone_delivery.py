"""Baseline brute-force experiment for the drone delivery case study (no Gurobi needed)."""
from __future__ import annotations

import itertools
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple

OUTPUT_DIR = os.path.join("output_mpcwpc", "baseline_drone_delivery")


# ── Instance definition ───────────────────────────────────────────────────────

OBJECTIVES = [
    "flight_safety", "battery_life", "delivery_speed", "noise_pollution",
    "weather_resilience", "cargo_integrity", "airspace_compliance", "maintenance_cost",
]

PLAN_VALUES = {
    "route_highway":     {"flight_safety": 9, "battery_life": 10, "delivery_speed": 9, "noise_pollution": 9, "weather_resilience": 10, "cargo_integrity": 5,  "airspace_compliance": 4,  "maintenance_cost": 10},
    "route_residential": {"flight_safety": 9, "battery_life": 4,  "delivery_speed": 3, "noise_pollution": 9, "weather_resilience": 6,  "cargo_integrity": 4,  "airspace_compliance": 3,  "maintenance_cost": 10},
    "route_park":        {"flight_safety": 2, "battery_life": 8,  "delivery_speed": 9, "noise_pollution": 4, "weather_resilience": 2,  "cargo_integrity": 10, "airspace_compliance": 3,  "maintenance_cost": 2},
    "route_direct":      {"flight_safety": 2, "battery_life": 5,  "delivery_speed": 5, "noise_pollution": 2, "weather_resilience": 9,  "cargo_integrity": 7,  "airspace_compliance": 9,  "maintenance_cost": 5},
    "route_cautious":    {"flight_safety": 10, "battery_life": 5, "delivery_speed": 6, "noise_pollution": 9, "weather_resilience": 2,  "cargo_integrity": 3,  "airspace_compliance": 9,  "maintenance_cost": 6},
    "route_express":     {"flight_safety": 8, "battery_life": 10, "delivery_speed": 3, "noise_pollution": 6, "weather_resilience": 7,  "cargo_integrity": 5,  "airspace_compliance": 10, "maintenance_cost": 6},
}

GROUND_TRUTH_HASSE = [
    ("flight_safety",     "airspace_compliance"),
    ("cargo_integrity",   "flight_safety"),
    ("weather_resilience","flight_safety"),
    ("battery_life",      "weather_resilience"),
    ("delivery_speed",    "battery_life"),
    ("maintenance_cost",  "battery_life"),
    ("noise_pollution",   "maintenance_cost"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _generate_transitive_closure(
    objectives: List[str],
    hasse_edges: List[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    reachable = {o: {o} for o in objectives}
    for o, o_p in hasse_edges:
        reachable[o].add(o_p)
    changed = True
    while changed:
        changed = False
        for o in objectives:
            new = set()
            for o_p in list(reachable[o]):
                new |= reachable.get(o_p, set())
            if not new.issubset(reachable[o]):
                reachable[o] |= new
                changed = True
    return [(o, o_p) for o in objectives for o_p in reachable[o] if o_p != o]


def _determine_comparison_label(
    lifted_pi: Dict[str, float],
    lifted_pip: Dict[str, float],
    objectives: List[str],
) -> Tuple[object, bool]:
    pi_ge  = all(lifted_pi[o] >= lifted_pip[o] for o in objectives)
    pi_gt  = any(lifted_pi[o] >  lifted_pip[o] for o in objectives)
    pip_ge = all(lifted_pip[o] >= lifted_pi[o] for o in objectives)
    pip_gt = any(lifted_pip[o] >  lifted_pi[o] for o in objectives)
    if pi_ge and pi_gt:
        return 1, False
    if pip_ge and pip_gt:
        return 1, True
    if pi_ge and pip_ge:
        return 0, False
    return '?', False


def _build_comparisons(
    objectives: List[str],
    plan_values: Dict[str, Dict[str, float]],
    preorder_edges: List[Tuple[str, str]],
) -> List[Tuple[str, str, object]]:
    upper_closure = {o: {o} for o in objectives}
    for o, o_p in preorder_edges:
        upper_closure[o].add(o_p)

    plans = list(plan_values.keys())
    comparisons = []
    for pi, pip in itertools.combinations(plans, 2):
        lifted_pi  = {o: sum(plan_values[pi][o_p]  for o_p in upper_closure[o]) for o in objectives}
        lifted_pip = {o: sum(plan_values[pip][o_p] for o_p in upper_closure[o]) for o in objectives}
        r, swap = _determine_comparison_label(lifted_pi, lifted_pip, objectives)
        if swap:
            comparisons.append((pip, pi, r))
        else:
            comparisons.append((pi, pip, r))
    return comparisons


def _check_consistency(
    objectives: List[str],
    plan_values: Dict[str, Dict[str, float]],
    preorder_edges: List[Tuple[str, str]],
    comparisons: List[Tuple[str, str, object]],
) -> bool:
    upper_closure = {o: {o} for o in objectives}
    for o, o_p in preorder_edges:
        upper_closure[o].add(o_p)
    for pi_name, pip_name, r in comparisons:
        lifted_pi  = {o: sum(plan_values[pi_name][o_p]  for o_p in upper_closure[o]) for o in objectives}
        lifted_pip = {o: sum(plan_values[pip_name][o_p] for o_p in upper_closure[o]) for o in objectives}
        actual_r, _ = _determine_comparison_label(lifted_pi, lifted_pip, objectives)
        if actual_r != r:
            return False
    return True


# ── Experiment ────────────────────────────────────────────────────────────────

def run_drone_delivery_baseline(time_limit: float = 600.0):
    ground_truth_preorder = _generate_transitive_closure(OBJECTIVES, GROUND_TRUTH_HASSE)
    comparisons = _build_comparisons(OBJECTIVES, PLAN_VALUES, ground_truth_preorder)

    all_directed_edges = [
        (o_i, o_j) for o_i in OBJECTIVES for o_j in OBJECTIVES if o_i != o_j
    ]

    print(f"Drone delivery baseline")
    print(f"  objectives      : {len(OBJECTIVES)}")
    print(f"  comparisons     : {len(comparisons)}")
    print(f"  directed edges  : {len(all_directed_edges)}")
    print(f"  ground truth    : {len(GROUND_TRUTH_HASSE)} Hasse / {len(ground_truth_preorder)} transitive edges")
    print(f"  time limit      : {time_limit}s")
    print()

    found = False
    result_preorder_size = None
    result_preorder_edges = None
    status = "TIME_LIMIT"
    start_time = time.perf_counter()

    last_k = 0
    total_checked = 0
    checked_at_last_k = 0

    outer_break = False
    for k in range(len(all_directed_edges) + 1):
        if time.perf_counter() - start_time >= time_limit:
            outer_break = True
            break

        last_k = k
        checked_at_last_k = 0

        for edge_subset in itertools.combinations(all_directed_edges, k):
            if time.perf_counter() - start_time >= time_limit:
                outer_break = True
                break

            total_checked += 1
            checked_at_last_k += 1

            tc_edges = _generate_transitive_closure(OBJECTIVES, list(edge_subset))

            if _check_consistency(OBJECTIVES, PLAN_VALUES, tc_edges, comparisons):
                result_preorder_size = len(tc_edges)
                result_preorder_edges = tc_edges
                found = True
                status = "OPTIMAL"
                break

        if found or outer_break:
            break

    elapsed = time.perf_counter() - start_time

    size_str = str(result_preorder_size) if found else "---"
    print(f"  status                  : {status}")
    print(f"  preorder_size           : {size_str}")
    print(f"  elapsed                 : {elapsed:.3f}s")
    print(f"  last_k_reached          : {last_k}")
    print(f"  total_candidates_checked: {total_checked:,}")
    print(f"  checked_at_last_k       : {checked_at_last_k:,}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "timestamp": ts_str,
        "instance": "drone_delivery",
        "num_objectives": len(OBJECTIVES),
        "num_comparisons": len(comparisons),
        "num_directed_edges": len(all_directed_edges),
        "ground_truth_hasse_size": len(GROUND_TRUTH_HASSE),
        "ground_truth_preorder_size": len(ground_truth_preorder),
        "time_limit_sec": time_limit,
        "status": status,
        "preorder_size": result_preorder_size,
        "preorder_edges": result_preorder_edges,
        "elapsed_sec": round(elapsed, 3),
        "progress_at_timeout": {
            "last_k_reached": last_k,
            "total_candidates_checked": total_checked,
            "candidates_checked_at_last_k": checked_at_last_k,
        },
    }

    out_path = os.path.join(OUTPUT_DIR, f"baseline_result_{ts_str}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Result saved to: {out_path}")
    return result


if __name__ == "__main__":
    run_drone_delivery_baseline()
