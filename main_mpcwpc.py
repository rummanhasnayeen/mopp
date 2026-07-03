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

OUTPUT_BASE_DIR = "output_mpcwpc"


def _make_run_dir(run_type: str, identifier: str, ts_str: str) -> str:
    safe_id = identifier.replace(" ", "_").replace("/", "_")[:60]
    run_dir = os.path.join(OUTPUT_BASE_DIR, run_type, f"{safe_id}_{ts_str}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def build_paper_hasse_example() -> MPCwPCInstance:
    objectives = ["o1", "o2", "o3"]

    plan_values = {
        "pi":  {"o1": 4, "o2": 2, "o3": 6},
        "pi_prime": {"o1": 2, "o2": 5, "o3": 3},
    }

    comparisons = [
        ("pi_prime", "pi", 1),
    ]

    hasse = [("o2", "o1"), ("o2", "o3")]
    preorder = _generate_transitive_closure(objectives, hasse)

    return MPCwPCInstance(
        objectives=objectives,
        plan_values=plan_values,
        comparisons=comparisons,
        ground_truth_hasse=hasse,
        ground_truth_preorder=preorder,
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
    )


def build_autonomous_vehicle_hierarchy_example() -> MPCwPCInstance:
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

    hasse_edges = [
        ("legal", "safety"),
        ("emissions", "safety"),
        ("time", "safety"),
        ( "maint", "safety"),

        ("payload", "cost"),
        ("emissions", "cost"),

        ("energy", "emissions"),

        ("cost", "time"),

        ("robust", "maint"),

        ("comfort", "maint"),
    ]

    preorder_edges = _generate_transitive_closure(objectives, hasse_edges)

    plans = list(plan_values.keys())
    comparisons = []
    from itertools import combinations
    for pi, pip in combinations(plans, 2):
        lifted_pi = _compute_lifted_values(objectives, plan_values[pi], preorder_edges)
        lifted_pip = _compute_lifted_values(objectives, plan_values[pip], preorder_edges)
        r, swap = _determine_comparison_label(lifted_pi, lifted_pip, objectives)
        if swap:
            comparisons.append((pip, pi, r))
        else:
            comparisons.append((pi, pip, r))

    return MPCwPCInstance(
        objectives=objectives,
        plan_values=plan_values,
        comparisons=comparisons,
        ground_truth_hasse=hasse_edges,
        ground_truth_preorder=preorder_edges,
    )


def build_drone_delivery_example() -> MPCwPCInstance:
    objectives = [
        "flight_safety", "battery_life", "delivery_speed", "noise_pollution",
        "weather_resilience", "cargo_integrity", "airspace_compliance", "maintenance_cost",
    ]

    plan_values = {
        "route_highway":     {"flight_safety": 9, "battery_life": 10, "delivery_speed": 9, "noise_pollution": 9, "weather_resilience": 10, "cargo_integrity": 5, "airspace_compliance": 4, "maintenance_cost": 10},
        "route_residential": {"flight_safety": 9, "battery_life": 4,  "delivery_speed": 3, "noise_pollution": 9, "weather_resilience": 6,  "cargo_integrity": 4, "airspace_compliance": 3, "maintenance_cost": 10},
        "route_park":        {"flight_safety": 2, "battery_life": 8,  "delivery_speed": 9, "noise_pollution": 4, "weather_resilience": 2,  "cargo_integrity": 10,"airspace_compliance": 3, "maintenance_cost": 2},
        "route_direct":      {"flight_safety": 2, "battery_life": 5,  "delivery_speed": 5, "noise_pollution": 2, "weather_resilience": 9,  "cargo_integrity": 7, "airspace_compliance": 9, "maintenance_cost": 5},
        "route_cautious":    {"flight_safety": 10,"battery_life": 5,  "delivery_speed": 6, "noise_pollution": 9, "weather_resilience": 2,  "cargo_integrity": 3, "airspace_compliance": 9, "maintenance_cost": 6},
        "route_express":     {"flight_safety": 8, "battery_life": 10, "delivery_speed": 3, "noise_pollution": 6, "weather_resilience": 7,  "cargo_integrity": 5, "airspace_compliance": 10,"maintenance_cost": 6},
    }

    hasse_edges = [
        ("flight_safety", "airspace_compliance"),
        ("cargo_integrity", "flight_safety"),
        ("weather_resilience", "flight_safety"),
        ("battery_life", "weather_resilience"),
        ("delivery_speed", "battery_life"),
        ("maintenance_cost", "battery_life"),
        ("noise_pollution", "maintenance_cost"),
    ]

    preorder_edges = _generate_transitive_closure(objectives, hasse_edges)

    plans = list(plan_values.keys())
    comparisons = []
    from itertools import combinations
    for pi, pip in combinations(plans, 2):
        lifted_pi = _compute_lifted_values(objectives, plan_values[pi], preorder_edges)
        lifted_pip = _compute_lifted_values(objectives, plan_values[pip], preorder_edges)
        r, swap = _determine_comparison_label(lifted_pi, lifted_pip, objectives)
        if swap:
            comparisons.append((pip, pi, r))
        else:
            comparisons.append((pi, pip, r))

    return MPCwPCInstance(
        objectives=objectives,
        plan_values=plan_values,
        comparisons=comparisons,
        ground_truth_hasse=hasse_edges,
        ground_truth_preorder=preorder_edges,
    )


def run_drone_delivery_example():
    instance = build_drone_delivery_example()
    run_single_experiment(
        instance,
        title="Drone Delivery Fleet (8 objectives, 6 plans)",
        run_type="case_study",
        use_big_m=True,
    )


def run_autonomous_vehicle_example():
    instance = build_autonomous_vehicle_example()
    run_single_experiment(
        instance,
        title="Autonomous Vehicle (10 objectives, 5 plans)",
        run_type="case_study",
        use_big_m=True,
    )


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
    all_edges = []
    for o in objectives:
        for o_p in reachable[o]:
            if o_p != o:
                all_edges.append((o, o_p))
    return all_edges


def run_autonomous_vehicle_hierarchy_example():
    instance = build_autonomous_vehicle_hierarchy_example()
    run_single_experiment(
        instance,
        title="Autonomous Vehicle Hierarchy (10 objectives, 5 plans, deep preorder)",
        run_type="case_study",
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
        ground_truth_preorder=preorder_edges,
    )


def run_single_experiment(
    instance: MPCwPCInstance,
    title: str,
    run_type: str = "single",
    run_dir: str = None,
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
        print(f"Objectives ({len(instance.objectives)}): {instance.objectives}")
        print(f"Plans: {instance.plans}")
        print(f"Comparisons ({len(instance.comparisons)}):")
        for pi, pi_p, r in instance.comparisons:
            label = {0: 'indifferent', 1: 'preferred', '?': 'incomparable'}[r]
            print(f"  ({pi}, {pi_p}, {r}) [{label}]")

        if instance.ground_truth_hasse is not None:
            print(f"\nGround truth Hasse edges ({len(instance.ground_truth_hasse)}):")
            for a, b in instance.ground_truth_hasse:
                print(f"  {a} ≽ {b}")
        if instance.ground_truth_preorder is not None:
            print(f"\nGround truth preorder after transitive closure ({len(instance.ground_truth_preorder)} edges):")
            for a, b in sorted(instance.ground_truth_preorder):
                print(f"  {a} ≽ {b}")

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

        verification = None
        if result['feasible']:
            verification = solver.verify(result)
            print(f"\nVerification:")
            print(f"  Verified: {verification['verified']}")
            print(f"  Comparisons checked: {verification['num_comparisons_checked']}")
            print(f"  Violations: {verification['num_violations']}")
            if not verification['verified']:
                for v in verification['violations']:
                    pi, pip, r = v['comparison']
                    label = {0: 'indifferent', 1: 'preferred', '?': 'incomparable'}[r]
                    print(f"    FAIL: ({pi}, {pip}, {r}) [{label}]")
                    print(f"      Left wins on: {v['left_wins_on']}")
                    print(f"      Right wins on: {v['right_wins_on']}")

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
            "ground_truth_hasse": instance.ground_truth_hasse,
            "ground_truth_hasse_size": len(instance.ground_truth_hasse) if instance.ground_truth_hasse is not None else None,
            "ground_truth_preorder": instance.ground_truth_preorder,
            "ground_truth_preorder_size": len(instance.ground_truth_preorder) if instance.ground_truth_preorder is not None else None,
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
        "verification": verification,
        "timing": {
            "construction_time_sec": construction_time,
            "solver_time_sec": solver_time,
            "total_time_sec": total_time,
            "detailed_intervals": result["intervals"],
        },
    }

    if run_dir is None:
        run_dir = _make_run_dir(run_type, title, ts_str)
    safe_title = title.replace(" ", "_").replace("/", "_")[:50]
    text_path = os.path.join(run_dir, f"{safe_title}.txt")
    json_path = os.path.join(run_dir, f"{safe_title}.json")

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
        run_type="case_study",
        use_big_m=True,
    )


def run_coloring_example():
    instance = build_coloring_gadget_example()
    run_single_experiment(
        instance,
        title="3-Coloring Gadget (A-B-C triangle)",
        run_type="case_study",
        use_big_m=True,
    )


def _build_scalability_instance(
    num_objectives: int,
    num_comparisons: int,
    value_range: Tuple[int, int] = (1, 10),
    seed: int = 42,
) -> MPCwPCInstance:
    rng = random.Random(seed)
    objectives = [f"o_{i+1}" for i in range(num_objectives)]

    hasse_edges = [(objectives[i], objectives[0]) for i in range(1, num_objectives)]
    preorder_edges = _generate_transitive_closure(objectives, hasse_edges)

    plan_values = {}
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

    return MPCwPCInstance(
        objectives=objectives,
        plan_values=plan_values,
        comparisons=comparisons,
        ground_truth_hasse=hasse_edges,
        ground_truth_preorder=preorder_edges,
    )


def run_scalability_experiments(
    objective_range: List[int] = None,
    comparisons_multiplier: int = 3,
    use_big_m: bool = True,
    time_limit: float = 300.0,
    seed_base: int = 42,
):
    if objective_range is None:
        objective_range = list(range(10, 101, 5))

    all_results = []
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _make_run_dir("scalability_a", "increasing_objectives", ts_str)

    for step_idx, n_obj in enumerate(objective_range):
        n_comp = n_obj * comparisons_multiplier
        instance = _build_scalability_instance(
            num_objectives=n_obj,
            num_comparisons=n_comp,
            seed=seed_base + step_idx,
        )

        counts = {0: 0, 1: 0, '?': 0}
        for _, _, r in instance.comparisons:
            counts[r] += 1

        result = run_single_experiment(
            instance,
            title=f"Scalability_A_n_obj={n_obj}_n_comp={n_comp}",
            run_type="scalability_a",
            run_dir=run_dir,
            use_big_m=use_big_m,
            time_limit=time_limit,
        )

        result["comparison_distribution"] = {
            "preferred": counts[1],
            "incomparable": counts['?'],
            "indifferent": counts[0],
        }
        all_results.append(result)

        status = result["result"]["status"]
        build_t = result["timing"]["construction_time_sec"]
        solve_t = result["timing"]["solver_time_sec"]
        preorder = result["result"]["preorder_size"]
        verified = (result.get("verification") or {}).get("verified", "N/A")
        print(f"  >> n={n_obj} comp={n_comp} pref={counts[1]} incomp={counts['?']} "
              f"build={build_t:.1f}s solve={solve_t:.1f}s preorder={preorder} "
              f"verified={verified} status={status}")

        if status == "TIME_LIMIT":
            print(f"  >> Time limit reached at n_obj={n_obj}. Stopping.")
            break

    summary_path = os.path.join(run_dir, f"scalability_a_summary.json")
    summary = {
        "timestamp": ts_str,
        "mode": "big_m" if use_big_m else "indicator",
        "time_limit_sec": time_limit,
        "experiments": [
            {
                "num_objectives": r["inputs"]["num_objectives"],
                "num_comparisons": r["inputs"]["num_comparisons"],
                "num_plans": r["inputs"]["num_plans"],
                "num_variables": r["model_stats"]["num_variables"],
                "num_constraints": r["model_stats"]["num_constraints"],
                "status": r["result"]["status"],
                "preorder_size": r["result"]["preorder_size"],
                "preferred_count": r["comparison_distribution"]["preferred"],
                "incomparable_count": r["comparison_distribution"]["incomparable"],
                "indifferent_count": r["comparison_distribution"]["indifferent"],
                "construction_time_sec": r["timing"]["construction_time_sec"],
                "solver_time_sec": r["timing"]["solver_time_sec"],
                "total_time_sec": r["timing"]["total_time_sec"],
                "verified": (r.get("verification") or {}).get("verified"),
            }
            for r in all_results
        ],
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nScalability summary saved to: {summary_path}")


def _build_scalability_b_instance(
    num_objectives: int,
    num_comparisons: int,
    hasse_edges: List[Tuple[str, str]],
    fixed_plan_values: Dict[str, Dict[str, float]],
    fixed_pairs: List[Tuple[str, str]],
) -> MPCwPCInstance:
    objectives = [f"o_{i+1}" for i in range(num_objectives)]
    preorder_edges = _generate_transitive_closure(objectives, hasse_edges)

    comparisons = []
    for pi_name, pip_name in fixed_pairs:
        lifted_pi  = _compute_lifted_values(objectives, fixed_plan_values[pi_name],  preorder_edges)
        lifted_pip = _compute_lifted_values(objectives, fixed_plan_values[pip_name], preorder_edges)
        r, swap = _determine_comparison_label(lifted_pi, lifted_pip, objectives)
        if swap:
            comparisons.append((pip_name, pi_name, r))
        else:
            comparisons.append((pi_name, pip_name, r))

    return MPCwPCInstance(
        objectives=objectives,
        plan_values=fixed_plan_values,
        comparisons=comparisons,
        ground_truth_hasse=hasse_edges,
        ground_truth_preorder=preorder_edges,
    )


def run_scalability_b_experiments(
    num_objectives: int = 20,
    num_comparisons: int = 100,
    initial_edges: int = 5,
    edge_increment: int = 5,
    max_edges: int = None,
    use_big_m: bool = True,
    time_limit: float = 300.0,
    seed: int = 42,
):
    rng = random.Random(seed)
    objectives = [f"o_{i+1}" for i in range(num_objectives)]

    if max_edges is None:
        max_edges = num_objectives * (num_objectives - 1)

    # Generate fixed pool of plans and pairs once
    fixed_plan_values: Dict[str, Dict[str, float]] = {}
    fixed_pairs: List[Tuple[str, str]] = []
    for idx in range(num_comparisons):
        pi_name  = f"pi_{2*idx+1}"
        pip_name = f"pi_{2*idx+2}"
        fixed_plan_values[pi_name]  = {o: rng.randint(1, 10) for o in objectives}
        fixed_plan_values[pip_name] = {o: rng.randint(1, 10) for o in objectives}
        fixed_pairs.append((pi_name, pip_name))

    all_used = set()
    hasse_edges: List[Tuple[str, str]] = []

    all_results = []
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _make_run_dir("scalability_b", f"fixed_{num_objectives}obj_growing_preorder", ts_str)

    num_to_add = initial_edges
    iteration = 0

    while len(hasse_edges) < max_edges:
        iteration += 1

        added = 0
        attempts = 0
        while added < num_to_add and attempts < num_objectives * num_objectives:
            attempts += 1
            a = rng.choice(objectives)
            b = rng.choice(objectives)
            if a == b:
                continue
            candidate = (a, b)
            if candidate in all_used or (b, a) in all_used:
                continue
            all_used.add(candidate)
            hasse_edges.append(candidate)
            added += 1

        if added == 0:
            print(f"  >> No more edges to add. Stopping at iteration {iteration}.")
            break

        preorder_edges = _generate_transitive_closure(objectives, hasse_edges)
        num_hasse = len(hasse_edges)
        num_transitive = len(preorder_edges)

        instance = _build_scalability_b_instance(
            num_objectives=num_objectives,
            num_comparisons=num_comparisons,
            hasse_edges=hasse_edges,
            fixed_plan_values=fixed_plan_values,
            fixed_pairs=fixed_pairs,
        )

        counts = {0: 0, 1: 0, '?': 0}
        for _, _, r in instance.comparisons:
            counts[r] += 1

        result = run_single_experiment(
            instance,
            title=f"ScalabilityB_hasse={num_hasse}_trans={num_transitive}",
            run_type="scalability_b",
            run_dir=run_dir,
            use_big_m=use_big_m,
            time_limit=time_limit,
        )

        result["comparison_distribution"] = {
            "preferred": counts[1],
            "incomparable": counts['?'],
            "indifferent": counts[0],
        }
        result["preorder_input"] = {
            "num_hasse_edges": num_hasse,
            "num_transitive_edges": num_transitive,
            "hasse_edges": hasse_edges[:],
        }
        all_results.append(result)

        status = result["result"]["status"]
        build_t = result["timing"]["construction_time_sec"]
        solve_t = result["timing"]["solver_time_sec"]
        preorder = result["result"]["preorder_size"]
        verified = (result.get("verification") or {}).get("verified", "N/A")
        print(f"  >> iter={iteration} hasse={num_hasse} trans={num_transitive} "
              f"pref={counts[1]} incomp={counts['?']} "
              f"build={build_t:.1f}s solve={solve_t:.1f}s preorder={preorder} "
              f"verified={verified} status={status}")

        if status == "TIME_LIMIT":
            print(f"  >> Time limit reached at {num_hasse} hasse edges. Stopping.")
            break

        num_to_add = edge_increment

    summary_path = os.path.join(run_dir, f"scalability_b_summary.json")
    summary = {
        "timestamp": ts_str,
        "variant": "B",
        "mode": "big_m" if use_big_m else "indicator",
        "fixed_objectives": num_objectives,
        "fixed_comparisons": num_comparisons,
        "time_limit_sec": time_limit,
        "experiments": [
            {
                "iteration": i + 1,
                "num_hasse_edges": r["preorder_input"]["num_hasse_edges"],
                "num_transitive_edges": r["preorder_input"]["num_transitive_edges"],
                "num_comparisons": r["inputs"]["num_comparisons"],
                "num_plans": r["inputs"]["num_plans"],
                "num_variables": r["model_stats"]["num_variables"],
                "num_constraints": r["model_stats"]["num_constraints"],
                "status": r["result"]["status"],
                "solver_preorder_size": r["result"]["preorder_size"],
                "preferred_count": r["comparison_distribution"]["preferred"],
                "incomparable_count": r["comparison_distribution"]["incomparable"],
                "indifferent_count": r["comparison_distribution"]["indifferent"],
                "construction_time_sec": r["timing"]["construction_time_sec"],
                "solver_time_sec": r["timing"]["solver_time_sec"],
                "total_time_sec": r["timing"]["total_time_sec"],
                "verified": (r.get("verification") or {}).get("verified"),
            }
            for i, r in enumerate(all_results)
        ],
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nScalability B summary saved to: {summary_path}")


def _generate_informative_pair(
    objectives: List[str],
    preorder_edges: List[Tuple[str, str]],
    rng: random.Random,
    value_range: Tuple[int, int] = (1, 10),
    max_attempts: int = 500,
):
    for _ in range(max_attempts):
        pi_vals  = {o: rng.randint(*value_range) for o in objectives}
        pip_vals = {o: rng.randint(*value_range) for o in objectives}

        lifted_pi  = _compute_lifted_values(objectives, pi_vals,  preorder_edges)
        lifted_pip = _compute_lifted_values(objectives, pip_vals, preorder_edges)
        r, swap = _determine_comparison_label(lifted_pi, lifted_pip, objectives)
        if r != 1:
            continue

        # Reject if raw values alone already explain the preference
        r_raw, _ = _determine_comparison_label(pi_vals, pip_vals, objectives)
        if r_raw == 1:
            continue

        return pi_vals, pip_vals, swap

    return None


def _build_scalability_b_informed_instance(
    num_objectives: int,
    num_comparisons: int,
    hasse_edges: List[Tuple[str, str]],
    fixed_plan_values: Dict[str, Dict[str, float]],
    fixed_pairs: List[Tuple[str, str]],
    informative_plan_values: Dict[str, Dict[str, float]],
    informative_pairs: List[Tuple[str, str]],
) -> MPCwPCInstance:
    objectives = [f"o_{i+1}" for i in range(num_objectives)]
    preorder_edges = _generate_transitive_closure(objectives, hasse_edges)

    all_plan_values = {**fixed_plan_values, **informative_plan_values}
    comparisons = []

    for pi_name, pip_name in fixed_pairs:
        lifted_pi  = _compute_lifted_values(objectives, all_plan_values[pi_name],  preorder_edges)
        lifted_pip = _compute_lifted_values(objectives, all_plan_values[pip_name], preorder_edges)
        r, swap = _determine_comparison_label(lifted_pi, lifted_pip, objectives)
        if swap:
            comparisons.append((pip_name, pi_name, r))
        else:
            comparisons.append((pi_name, pip_name, r))

    for pi_name, pip_name in informative_pairs:
        lifted_pi  = _compute_lifted_values(objectives, all_plan_values[pi_name],  preorder_edges)
        lifted_pip = _compute_lifted_values(objectives, all_plan_values[pip_name], preorder_edges)
        r, swap = _determine_comparison_label(lifted_pi, lifted_pip, objectives)
        if swap:
            comparisons.append((pip_name, pi_name, r))
        else:
            comparisons.append((pi_name, pip_name, r))

    return MPCwPCInstance(
        objectives=objectives,
        plan_values=all_plan_values,
        comparisons=comparisons,
        ground_truth_hasse=hasse_edges,
        ground_truth_preorder=preorder_edges,
    )


def run_scalability_b_informed_experiments(
    num_objectives: int = 20,
    num_comparisons: int = 100,
    num_informative: int = 5,
    initial_edges: int = 5,
    edge_increment: int = 5,
    max_edges: int = None,
    use_big_m: bool = True,
    time_limit: float = 300.0,
    seed: int = 42,
):
    rng = random.Random(seed)
    objectives = [f"o_{i+1}" for i in range(num_objectives)]

    if max_edges is None:
        max_edges = num_objectives * (num_objectives - 1)

    fixed_plan_values: Dict[str, Dict[str, float]] = {}
    fixed_pairs: List[Tuple[str, str]] = []
    for idx in range(num_comparisons):
        pi_name  = f"pi_{2*idx+1}"
        pip_name = f"pi_{2*idx+2}"
        fixed_plan_values[pi_name]  = {o: rng.randint(1, 10) for o in objectives}
        fixed_plan_values[pip_name] = {o: rng.randint(1, 10) for o in objectives}
        fixed_pairs.append((pi_name, pip_name))

    all_used = set()
    hasse_edges: List[Tuple[str, str]] = []

    all_results = []
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _make_run_dir(
        "scalability_b_informed",
        f"fixed_{num_objectives}obj_growing_preorder_informed",
        ts_str,
    )

    num_to_add = initial_edges
    iteration = 0

    while len(hasse_edges) < max_edges:
        iteration += 1

        added = 0
        attempts = 0
        while added < num_to_add and attempts < num_objectives * num_objectives:
            attempts += 1
            a = rng.choice(objectives)
            b = rng.choice(objectives)
            if a == b:
                continue
            candidate = (a, b)
            if candidate in all_used or (b, a) in all_used:
                continue
            all_used.add(candidate)
            hasse_edges.append(candidate)
            added += 1

        if added == 0:
            print(f"  >> No more edges to add. Stopping at iteration {iteration}.")
            break

        preorder_edges = _generate_transitive_closure(objectives, hasse_edges)
        num_hasse = len(hasse_edges)
        num_transitive = len(preorder_edges)

        informative_plan_values: Dict[str, Dict[str, float]] = {}
        informative_pairs: List[Tuple[str, str]] = []
        inf_counter = 0
        for _ in range(num_informative):
            result = _generate_informative_pair(objectives, preorder_edges, rng)
            if result is None:
                print(f"  >> Warning: could not generate informative pair at iter={iteration}")
                continue
            pi_vals, pip_vals, swap = result
            inf_counter += 1
            pi_name  = f"inf_{iteration}_{inf_counter}_a"
            pip_name = f"inf_{iteration}_{inf_counter}_b"
            informative_plan_values[pi_name]  = pi_vals
            informative_plan_values[pip_name] = pip_vals
            if swap:
                informative_pairs.append((pip_name, pi_name))
            else:
                informative_pairs.append((pi_name, pip_name))

        instance = _build_scalability_b_informed_instance(
            num_objectives=num_objectives,
            num_comparisons=num_comparisons,
            hasse_edges=hasse_edges,
            fixed_plan_values=fixed_plan_values,
            fixed_pairs=fixed_pairs,
            informative_plan_values=informative_plan_values,
            informative_pairs=informative_pairs,
        )

        counts = {0: 0, 1: 0, '?': 0}
        for _, _, r in instance.comparisons:
            counts[r] += 1

        result = run_single_experiment(
            instance,
            title=f"ScalabilityB_Informed_hasse={num_hasse}_trans={num_transitive}",
            run_type="scalability_b_informed",
            run_dir=run_dir,
            use_big_m=use_big_m,
            time_limit=time_limit,
        )

        result["comparison_distribution"] = {
            "preferred": counts[1],
            "incomparable": counts['?'],
            "indifferent": counts[0],
        }
        result["preorder_input"] = {
            "num_hasse_edges": num_hasse,
            "num_transitive_edges": num_transitive,
            "hasse_edges": hasse_edges[:],
        }
        result["num_informative_pairs"] = len(informative_pairs)
        all_results.append(result)

        status = result["result"]["status"]
        build_t = result["timing"]["construction_time_sec"]
        solve_t = result["timing"]["solver_time_sec"]
        preorder = result["result"]["preorder_size"]
        verified = (result.get("verification") or {}).get("verified", "N/A")
        print(f"  >> iter={iteration} hasse={num_hasse} trans={num_transitive} "
              f"pref={counts[1]} incomp={counts['?']} inf={len(informative_pairs)} "
              f"build={build_t:.1f}s solve={solve_t:.1f}s preorder={preorder} "
              f"verified={verified} status={status}")

        if status == "TIME_LIMIT":
            print(f"  >> Time limit reached at {num_hasse} hasse edges. Stopping.")
            break

        num_to_add = edge_increment

    summary_path = os.path.join(run_dir, "scalability_b_informed_summary.json")
    summary = {
        "timestamp": ts_str,
        "variant": "B_informed",
        "mode": "big_m" if use_big_m else "indicator",
        "fixed_objectives": num_objectives,
        "fixed_comparisons": num_comparisons,
        "num_informative_pairs": num_informative,
        "time_limit_sec": time_limit,
        "experiments": [
            {
                "iteration": i + 1,
                "num_hasse_edges": r["preorder_input"]["num_hasse_edges"],
                "num_transitive_edges": r["preorder_input"]["num_transitive_edges"],
                "num_comparisons": r["inputs"]["num_comparisons"],
                "num_plans": r["inputs"]["num_plans"],
                "num_informative_pairs": r["num_informative_pairs"],
                "num_variables": r["model_stats"]["num_variables"],
                "num_constraints": r["model_stats"]["num_constraints"],
                "status": r["result"]["status"],
                "solver_preorder_size": r["result"]["preorder_size"],
                "preferred_count": r["comparison_distribution"]["preferred"],
                "incomparable_count": r["comparison_distribution"]["incomparable"],
                "indifferent_count": r["comparison_distribution"]["indifferent"],
                "construction_time_sec": r["timing"]["construction_time_sec"],
                "solver_time_sec": r["timing"]["solver_time_sec"],
                "total_time_sec": r["timing"]["total_time_sec"],
                "verified": (r.get("verification") or {}).get("verified"),
            }
            for i, r in enumerate(all_results)
        ],
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nScalability B (informed) summary saved to: {summary_path}")


if __name__ == "__main__":
    run_scalability_b_informed_experiments(
        num_objectives=20,
        num_comparisons=100,
        num_informative=3,
        initial_edges=15,
        edge_increment=5,
        seed=42,
    )