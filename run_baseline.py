"""Standalone runner for the scalability baseline experiment (no Gurobi needed)."""
from __future__ import annotations

import itertools
import json
import os
import random
import time
from datetime import datetime
from typing import Dict, List, Tuple

OUTPUT_BASE_DIR = "output_mpcwpc"


# ── Helpers copied from main_mpcwpc.py (no ILP / Gurobi dependency) ──────────

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


def _compute_lifted_values(
    objectives: List[str],
    plan_vals: Dict[str, float],
    preorder_edges: List[Tuple[str, str]],
) -> Dict[str, float]:
    upper_closure = {o: {o} for o in objectives}
    for o, o_p in preorder_edges:
        upper_closure[o].add(o_p)
    return {o: sum(plan_vals[o_p] for o_p in upper_closure[o]) for o in objectives}


def _determine_comparison_label(
    lifted_pi: Dict[str, float],
    lifted_pip: Dict[str, float],
    objectives: List[str],
) -> Tuple[object, bool]:
    pi_ge = all(lifted_pi[o] >= lifted_pip[o] for o in objectives)
    pi_gt = any(lifted_pi[o] > lifted_pip[o] for o in objectives)
    pip_ge = all(lifted_pip[o] >= lifted_pi[o] for o in objectives)
    pip_gt = any(lifted_pip[o] > lifted_pi[o] for o in objectives)
    if pi_ge and pi_gt:
        return 1, False
    if pip_ge and pip_gt:
        return 1, True
    if pi_ge and pip_ge:
        return 0, False
    return '?', False


def _build_scalability_instance(
    num_objectives: int,
    num_comparisons: int,
    value_range: Tuple[int, int] = (1, 10),
    seed: int = 42,
):
    rng = random.Random(seed)
    objectives = [f"o_{i+1}" for i in range(num_objectives)]
    hasse_edges = [(objectives[i], objectives[0]) for i in range(1, num_objectives)]
    preorder_edges = _generate_transitive_closure(objectives, hasse_edges)

    plan_values: Dict[str, Dict[str, float]] = {}
    comparisons = []
    plan_counter = 0

    for _ in range(num_comparisons):
        plan_counter += 1
        pi_name = f"pi_{plan_counter}"
        plan_counter += 1
        pip_name = f"pi_{plan_counter}"

        plan_values[pi_name] = {o: rng.randint(*value_range) for o in objectives}
        plan_values[pip_name] = {o: rng.randint(*value_range) for o in objectives}

        lifted_pi = _compute_lifted_values(objectives, plan_values[pi_name], preorder_edges)
        lifted_pip = _compute_lifted_values(objectives, plan_values[pip_name], preorder_edges)
        r, swap = _determine_comparison_label(lifted_pi, lifted_pip, objectives)

        if swap:
            comparisons.append((pip_name, pi_name, r))
        else:
            comparisons.append((pi_name, pip_name, r))

    return objectives, plan_values, comparisons


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
        lifted_pi = {o: sum(plan_values[pi_name][o_p] for o_p in upper_closure[o]) for o in objectives}
        lifted_pip = {o: sum(plan_values[pip_name][o_p] for o_p in upper_closure[o]) for o in objectives}
        actual_r, _ = _determine_comparison_label(lifted_pi, lifted_pip, objectives)
        if actual_r != r:
            return False
    return True


# ── Main experiment ───────────────────────────────────────────────────────────

def run_scalability_baseline_experiments(
    objective_range: List[int] = None,
    comparisons_multiplier: int = 3,
    time_limit: float = 600.0,
    seed_base: int = 42,
):
    if objective_range is None:
        objective_range = list(range(10, 101, 5))

    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_BASE_DIR, "scalability_baseline", f"increasing_objectives_{ts_str}")
    os.makedirs(run_dir, exist_ok=True)

    all_results = []

    for step_idx, n_obj in enumerate(objective_range):
        n_comp = n_obj * comparisons_multiplier
        objectives, plan_values, comparisons = _build_scalability_instance(
            num_objectives=n_obj,
            num_comparisons=n_comp,
            seed=seed_base + step_idx,
        )

        all_directed_edges = [
            (o_i, o_j) for o_i in objectives for o_j in objectives if o_i != o_j
        ]

        found = False
        result_preorder_size = None
        status = "TIME_LIMIT"
        start_time = time.perf_counter()

        outer_break = False
        for k in range(len(all_directed_edges) + 1):
            if time.perf_counter() - start_time >= time_limit:
                outer_break = True
                break

            for edge_subset in itertools.combinations(all_directed_edges, k):
                if time.perf_counter() - start_time >= time_limit:
                    outer_break = True
                    break

                tc_edges = _generate_transitive_closure(objectives, list(edge_subset))

                if _check_consistency(objectives, plan_values, tc_edges, comparisons):
                    result_preorder_size = len(tc_edges)
                    found = True
                    status = "OPTIMAL"
                    break

            if found or outer_break:
                break

        elapsed = time.perf_counter() - start_time

        entry = {
            "n_obj": n_obj,
            "n_comp": n_comp,
            "status": status,
            "preorder_size": result_preorder_size,
            "elapsed_sec": round(elapsed, 3),
        }
        all_results.append(entry)

        size_str = str(result_preorder_size) if found else "---"
        print(f"  >> n={n_obj:3d} comp={n_comp:3d}  status={status:<10s}  "
              f"preorder_size={size_str:<6s}  time={elapsed:.2f}s", flush=True)

        # Save incrementally after each instance
        summary_path = os.path.join(run_dir, "scalability_baseline_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": ts_str,
                "time_limit_sec": time_limit,
                "experiments": all_results,
            }, f, indent=2)

    print(f"\nBaseline summary saved to: {run_dir}/scalability_baseline_summary.json")
    return all_results


if __name__ == "__main__":
    run_scalability_baseline_experiments()
