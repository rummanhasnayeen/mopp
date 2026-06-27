from __future__ import annotations

import io
import os
import json
import time
import random
from datetime import datetime
from contextlib import redirect_stdout
from typing import List, Tuple, Dict

from Models.mpcwpc_instance import MPCwPCInstance
from Solvers.mpcwpc_ilp_solver import MPCwPCILPSolver

TEXT_LOG_DIR = "text_mpcwpc"
JSON_LOG_DIR = "json_mpcwpc"


def ensure_output_dirs():
    os.makedirs(TEXT_LOG_DIR, exist_ok=True)
    os.makedirs(JSON_LOG_DIR, exist_ok=True)

def build_paper_hasse_example() -> MPCwPCInstance:
    objectives = ["o1", "o2", "o3"]

    plan_values = {
        "pi":  {"o1": 4, "o2": 2, "o3": 6},
        "pi_prime": {"o1": 2, "o2": 5, "o3": 3},
    }

    comparisons = [
        ("pi_prime", "pi", 1),
    ]

    return MPCwPCInstance(
        objectives=objectives,
        plan_values=plan_values,
        comparisons=comparisons,
        k=3,
    )


def build_autonomous_vehicle_example() -> MPCwPCInstance:
    objectives = [
        "safety", "energy", "time", "comfort", "cost",
        "legal", "robust", "emissions", "maint", "payload",
    ]

    plan_values = {
        "route_fast":     {"safety": 5, "energy": 4, "time": 9, "comfort": 6, "cost": 5, "legal": 8, "robust": 6, "emissions": 4, "maint": 6, "payload": 7},
        "route_safe":     {"safety": 9, "energy": 5, "time": 5, "comfort": 7, "cost": 6, "legal": 9, "robust": 8, "emissions": 6, "maint": 7, "payload": 6},
        "route_green":    {"safety": 7, "energy": 9, "time": 4, "comfort": 6, "cost": 7, "legal": 8, "robust": 7, "emissions": 9, "maint": 6, "payload": 5},
        "route_balanced": {"safety": 8, "energy": 8, "time": 8, "comfort": 8, "cost": 7, "legal": 9, "robust": 8, "emissions": 7, "maint": 8, "payload": 7},
        "route_budget":   {"safety": 6, "energy": 6, "time": 6, "comfort": 5, "cost": 9, "legal": 7, "robust": 6, "emissions": 6, "maint": 7, "payload": 6},
    }

    comparisons = [
        ("route_safe",     "route_fast",   1),
        ("route_fast",     "route_green",  '?'),
        ("route_balanced", "route_fast",   1),
        ("route_fast",     "route_budget", '?'),
        ("route_safe",     "route_green",  '?'),
        ("route_safe",     "route_balanced",'?'),
        ("route_safe",     "route_budget",  1),
        ("route_green",    "route_balanced",'?'),
        ("route_green",    "route_budget",  '?'),
        ("route_balanced", "route_budget",  1),
    ]

    return MPCwPCInstance(
        objectives=objectives,
        plan_values=plan_values,
        comparisons=comparisons,
        k=5,
    )


def run_autonomous_vehicle_example():
    instance = build_autonomous_vehicle_example()
    run_single_experiment(
        instance,
        title="Autonomous Vehicle (10 objectives, 5 plans)",
        use_big_m=True,
    )


def build_coloring_gadget_example() -> MPCwPCInstance:
    vertices = ["A", "B", "C"]
    colors = ["R", "G", "Bl"]
    edges = [("A", "B"), ("B", "C"), ("A", "C")]

    obj_names = [f"o_{v}" for v in vertices] + [f"o_{c}" for c in colors]
    objectives = obj_names

    INF_VAL = 1000
    NEG_INF_VAL = -1000

    plan_values = {}
    comparisons = []
    plan_counter = [0]

    def new_plan(values_dict: Dict[str, float]) -> str:
        plan_counter[0] += 1
        name = f"plan_{plan_counter[0]}"
        plan_values[name] = values_dict
        return name

    # block outgoing edges from color
    for t in colors:
        o_t = f"o_{t}"
        v_plus = {o: 1 for o in objectives}
        v_plus[o_t] = 0
        v_minus = {o: 0 for o in objectives}
        v_minus[o_t] = 1
        p_plus = new_plan(v_plus)
        p_minus = new_plan(v_minus)
        comparisons.append((p_plus, p_minus, '?'))

    # block edges vertex
    for v in vertices:
        for v_p in vertices:
            if v == v_p:
                continue
            o_v = f"o_{v}"
            o_vp = f"o_{v_p}"
            v_plus = {o: 0 for o in objectives}
            v_plus[o_vp] = 1
            v_minus = {o: NEG_INF_VAL for o in objectives}
            v_minus[o_v] = 1
            v_minus[o_vp] = 0
            for c in colors:
                v_minus[f"o_{c}"] = 0
            p_plus = new_plan(v_plus)
            p_minus = new_plan(v_minus)
            comparisons.append((p_plus, p_minus, '?'))

    #  at least one color
    #   at most one color
    for v in vertices:
        o_v = f"o_{v}"
        v_plus = {o: INF_VAL for o in objectives}
        v_plus[o_v] = 0
        for c in colors:
            v_plus[f"o_{c}"] = 1

        v_minus = {o: INF_VAL for o in objectives}
        v_minus[o_v] = 1
        for c in colors:
            v_minus[f"o_{c}"] = 0

        p_plus = new_plan(v_plus)
        p_minus = new_plan(v_minus)
        comparisons.append((p_plus, p_minus, 1))

        # at most one color
        v_star = {o: INF_VAL for o in objectives}
        v_star[o_v] = 2
        for c in colors:
            v_star[f"o_{c}"] = 0
        p_star = new_plan(v_star)
        comparisons.append((p_plus, p_star, '?'))

    # adjacent vertices cannot share color
    for (u, w) in edges:
        o_u = f"o_{u}"
        o_w = f"o_{w}"
        v_minus_edge = {o: INF_VAL for o in objectives}
        v_minus_edge[o_u] = 1
        v_minus_edge[o_w] = 1
        for c in colors:
            v_minus_edge[f"o_{c}"] = 0
        p_minus_edge = new_plan(v_minus_edge)

        for t in colors:
            o_t = f"o_{t}"
            v_plus_edge = {o: INF_VAL for o in objectives}
            v_plus_edge[o_u] = 0
            v_plus_edge[o_w] = 0
            for c in colors:
                v_plus_edge[f"o_{c}"] = 0
            v_plus_edge[o_t] = 1
            p_plus_edge = new_plan(v_plus_edge)
            comparisons.append((p_plus_edge, p_minus_edge, '?'))

    return MPCwPCInstance(
        objectives=objectives,
        plan_values=plan_values,
        comparisons=comparisons,
        k=len(vertices),
    )


def _generate_random_preorder_edges(
    objectives: List[str],
    num_edges: int,
    rng: random.Random,
) -> List[Tuple[str, str]]:
    n = len(objectives)
    possible = [(o, o_p) for o in objectives for o_p in objectives if o != o_p]
    rng.shuffle(possible)
    hasse_edges = possible[:min(num_edges, len(possible))]

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

    all_edges = []
    for o in objectives:
        for o_p in reachable[o]:
            if o_p != o:
                all_edges.append((o, o_p))
    return all_edges


def _compute_lifted_values(
    objectives: List[str],
    plan_vals: Dict[str, float],
    preorder_edges: List[Tuple[str, str]],
) -> Dict[str, float]:
    upper_closure = {o: {o} for o in objectives}
    for o, o_p in preorder_edges:
        upper_closure[o].add(o_p)

    lifted = {}
    for o in objectives:
        lifted[o] = sum(plan_vals[o_p] for o_p in upper_closure[o])
    return lifted


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


def build_random_instance(
    num_objectives: int,
    num_comparisons: int,
    k: int,
    num_preorder_edges: int = None,
    value_range: Tuple[int, int] = (1, 100),
    seed: int = 42,
) -> MPCwPCInstance:
    rng = random.Random(seed)
    objectives = [f"o_{i+1}" for i in range(num_objectives)]

    if num_preorder_edges is None:
        num_preorder_edges = max(1, num_objectives // 3)

    preorder_edges = _generate_random_preorder_edges(
        objectives, num_preorder_edges, rng)

    plan_values = {}
    comparisons = []
    plan_counter = 0

    for _ in range(num_comparisons):
        plan_counter += 1
        pi_name = f"pi_{plan_counter}"
        plan_counter += 1
        pip_name = f"pi_{plan_counter}"

        pi_vals = {o: rng.randint(*value_range) for o in objectives}
        pip_vals = {o: rng.randint(*value_range) for o in objectives}
        plan_values[pi_name] = pi_vals
        plan_values[pip_name] = pip_vals

        lifted_pi = _compute_lifted_values(objectives, pi_vals, preorder_edges)
        lifted_pip = _compute_lifted_values(objectives, pip_vals, preorder_edges)
        r, swap = _determine_comparison_label(lifted_pi, lifted_pip, objectives)

        if swap:
            comparisons.append((pip_name, pi_name, r))
        else:
            comparisons.append((pi_name, pip_name, r))

    return MPCwPCInstance(
        objectives=objectives,
        plan_values=plan_values,
        comparisons=comparisons,
        k=k,
    )


def run_single_experiment(
    instance: MPCwPCInstance,
    title: str,
    use_big_m: bool = True,
    big_m: float = 1e4,
    epsilon: float = 1e-3,
    time_limit: float = 300.0,
    verbose_gurobi: bool = False,
) -> dict:
    total_start = time.perf_counter()
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_buffer = io.StringIO()

    with redirect_stdout(log_buffer):
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)
        print(f"Timestamp: {ts_str}")
        print(f"Mode: {'Big-M' if use_big_m else 'Indicator Constraints'}")
        print(f"k (budget): {instance.k}")
        print(f"Objectives ({len(instance.objectives)}): {instance.objectives}")
        print(f"Plans: {instance.plans}")
        print(f"Comparisons ({len(instance.comparisons)}):")
        for pi, pi_p, r in instance.comparisons:
            label = {0: 'indifferent', 1: 'preferred', '?': 'incomparable'}[r]
            print(f"  ({pi}, {pi_p}, {r}) [{label}]")

        print("\nPlan values:")
        for p, vals in instance.plan_values.items():
            print(f"  {p}: {vals}")

        t_build_start = time.perf_counter()
        solver = MPCwPCILPSolver(
            instance,
            use_big_m=use_big_m,
            big_m=big_m,
            epsilon=epsilon,
            time_limit=time_limit,
            verbose=verbose_gurobi,
        )
        solver.build_model()
        t_build_end = time.perf_counter()

        print(f"\nModel built in {t_build_end - t_build_start:.6f} sec")
        print(f"  Variables: {solver.model.NumVars}")
        print(f"  Constraints: {solver.model.NumConstrs}")

        t_solve_start = time.perf_counter()
        result = solver.solve()
        t_solve_end = time.perf_counter()

        total_end = time.perf_counter()

        construction_time = t_build_end - t_build_start
        solver_time = t_solve_end - t_solve_start
        total_time = total_end - total_start

        print(f"\nSolver finished in {solver_time:.6f} sec")
        print(f"  Status: {result['status_name']}")
        print(f"  Feasible: {result['feasible']}")
        print(f"  Optimal: {result['optimal']}")

        if result['feasible']:
            print(f"  Objective value (preorder size): {result['objective_value']}")
            print(f"  Preorder edges ({result['preorder_size']}):")
            for o, o_p in result['preorder_edges']:
                print(f"    {o_p} >= {o}")

        print(f"\nTiming:")
        print(f"  Construction time: {construction_time:.6f} sec")
        print(f"  Solver time:       {solver_time:.6f} sec")
        print(f"  Total time:        {total_time:.6f} sec")

        for label, interval in sorted(result['intervals'].items()):
            print(f"  {label}: {interval:.6f} sec")

    text_log = log_buffer.getvalue()
    print(text_log)

    json_data = {
        "timestamp": ts_str,
        "title": title,
        "mode": "big_m" if use_big_m else "indicator",
        "inputs": {
            "num_objectives": len(instance.objectives),
            "objectives": instance.objectives,
            "num_plans": len(instance.plans),
            "num_comparisons": len(instance.comparisons),
            "k": instance.k,
        },
        "model_stats": {
            "num_variables": result["num_variables"],
            "num_constraints": result["num_constraints"],
        },
        "result": {
            "status": result["status_name"],
            "feasible": result["feasible"],
            "optimal": result["optimal"],
            "objective_value": result["objective_value"],
            "preorder_size": result["preorder_size"],
            "preorder_edges": result["preorder_edges"],
        },
        "timing": {
            "construction_time_sec": construction_time,
            "solver_time_sec": solver_time,
            "total_time_sec": total_time,
            "detailed_intervals": result["intervals"],
        },
    }

    ensure_output_dirs()
    safe_title = title.replace(" ", "_").replace("/", "_")[:50]
    text_path = os.path.join(TEXT_LOG_DIR, f"{safe_title}_{ts_str}.txt")
    json_path = os.path.join(JSON_LOG_DIR, f"{safe_title}_{ts_str}.json")

    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text_log)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    print(f"Text log saved to: {text_path}")
    print(f"JSON saved to: {json_path}")

    return json_data


def run_paper_example():
    instance = build_paper_hasse_example()
    run_single_experiment(
        instance,
        title="Paper Hasse Diagram Example (3 objectives)",
        use_big_m=True,
    )


def run_coloring_example():
    instance = build_coloring_gadget_example()
    run_single_experiment(
        instance,
        title="3-Coloring Gadget (A-B-C triangle)",
        use_big_m=True,
    )


def run_scalability_experiments(
    objective_range: List[int] = None,
    k_base: int = 3,
    comparisons_per_step: int = 5,
    use_big_m: bool = True,
    time_limit: float = 300.0,
    seed_base: int = 42,
):
    if objective_range is None:
        objective_range = [3, 5, 8, 10, 15]

    all_results = []
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    for step_idx, n_obj in enumerate(objective_range):
        n_comp = max(3, n_obj * comparisons_per_step // 3)
        k = k_base + step_idx
        instance = build_random_instance(
            num_objectives=n_obj,
            num_comparisons=n_comp,
            k=k,
            seed=seed_base + step_idx,
        )

        result = run_single_experiment(
            instance,
            title=f"Scalability n_obj={n_obj} n_comp={n_comp}",
            use_big_m=use_big_m,
            time_limit=time_limit,
        )
        all_results.append(result)

    ensure_output_dirs()
    summary_path = os.path.join(JSON_LOG_DIR, f"scalability_summary_{ts_str}.json")
    summary = {
        "timestamp": ts_str,
        "mode": "big_m" if use_big_m else "indicator",
        "experiments": [
            {
                "num_objectives": r["inputs"]["num_objectives"],
                "num_comparisons": r["inputs"]["num_comparisons"],
                "num_plans": r["inputs"]["num_plans"],
                "k": r["inputs"]["k"],
                "num_variables": r["model_stats"]["num_variables"],
                "num_constraints": r["model_stats"]["num_constraints"],
                "status": r["result"]["status"],
                "preorder_size": r["result"]["preorder_size"],
                "construction_time_sec": r["timing"]["construction_time_sec"],
                "solver_time_sec": r["timing"]["solver_time_sec"],
                "total_time_sec": r["timing"]["total_time_sec"],
            }
            for r in all_results
        ],
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nScalability summary saved to: {summary_path}")


if __name__ == "__main__":
    run_autonomous_vehicle_example()
    # run_paper_example()
    # run_scalability_experiments(objective_range=[3, 5, 8, 10, 15, 20])
    # run_coloring_example()