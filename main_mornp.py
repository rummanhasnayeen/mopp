from __future__ import annotations

import io
import os
from typing import Dict, List, Any

import json
import time
import random
from datetime import datetime
from contextlib import redirect_stdout

from Models.mornp_instance import MORNPInstance
from Solvers.mornpdec_sat_solver import MORNPDECSATSolver

RANDOM_VALUE_MIN = 1
RANDOM_VALUE_MAX = 100
MAX_PLAN_BUILD_ATTEMPTS = 5000
MIN_NEGATIVE_PLANS_REQUIRED = 5
MIN_POSITIVE_PLANS_REQUIRED = 1
NEGATIVE_BIAS_PROBABILITY = 0.5

TEXT_LOG_DIR = "mornp_text"
JSON_LOG_DIR = "mornp_json"
SUMMARY_LOG_DIR = "mornp_summary"

# stress test constants
STRESS_OBJECTIVE_INCREMENT = 15
STRESS_PLAN_INCREMENT = 20
STRESS_K_INCREMENT = 5
STRESS_MIN_PLAN_INCREMENT = 5

STRESS_END_OBJECTIVE_COUNT = 80


def build_small_example() -> MORNPInstance:
    objectives = ["o1", "o2", "o3"]

    plan_values = {
        "p_pos": {"o1": 8, "o2": 5, "o3": 4},
        "p_neg": {"o1": 6, "o2": 7, "o3": 4},
    }

    positive_plans = ["p_pos"]
    negative_plans = ["p_neg"]
    k = 1

    return MORNPInstance(
        objectives=objectives,
        plan_values=plan_values,
        positive_plans=positive_plans,
        negative_plans=negative_plans,
        k=k,
    )


def build_small_example_equality_unsat() -> MORNPInstance:
    objectives = ["o1", "o2"]

    plan_values = {
        "p_pos": {"o1": 5, "o2": 7},
        "p_neg": {"o1": 5, "o2": 7},
    }

    positive_plans = ["p_pos"]
    negative_plans = ["p_neg"]
    k = 2

    return MORNPInstance(
        objectives=objectives,
        plan_values=plan_values,
        positive_plans=positive_plans,
        negative_plans=negative_plans,
        k=k,
    )


def run_instance(instance: MORNPInstance, title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print("Objectives:", instance.objectives)
    print("Positive plans:", instance.positive_plans)
    print("Negative plans:", instance.negative_plans)
    print("k =", instance.k)

    print("\nPlan values:")
    for p, vals in instance.plan_values.items():
        print(f"  {p}: {vals}")

    solver = MORNPDECSATSolver(instance)
    solver.build_formula()

    print("\nCNF stats:")
    print("  #variables =", solver.vpool.top)
    print("  #clauses   =", len(solver.cnf.clauses))

    solver.print_pairwise_debug()

    result = solver.solve()

    print("\nSolve result:")
    if result["sat"]:
        print("  SAT = YES")
        print("  Selected objectives =", result["selected_objectives"])
    else:
        print("  SAT = NO")


def dominates(plan_a: dict, plan_b: dict, selected_objectives: list[str]) -> bool:
    """
    plan_a dominates plan_b iff:
      - plan_a[obj] >= plan_b[obj] for all selected objectives
      - and > on at least one selected objective
    large values better
    """
    all_ge = all(plan_a[obj] >= plan_b[obj] for obj in selected_objectives)
    one_gt = any(plan_a[obj] > plan_b[obj] for obj in selected_objectives)
    return all_ge and one_gt


def dominated_by_all_positive(candidate: dict, positive_values: list[dict], selected_objectives: list[str]) -> bool:
    return all(dominates(pos, candidate, selected_objectives) for pos in positive_values)


def non_dominated_wrt_all_positive(candidate: dict, positive_values: list[dict],
                                   selected_objectives: list[str]) -> bool:
    return all(not dominates(pos, candidate, selected_objectives) for pos in positive_values)


def generate_random_plan_values_for_all_objectives(all_objectives: list[str]) -> dict[str, float]:
    return {
        obj: random.randint(RANDOM_VALUE_MIN, RANDOM_VALUE_MAX)
        for obj in all_objectives
    }

def generate_negative_biased_plan_values(
    all_objectives: list[str],
    selected_objectives: list[str],
    positive_values: list[dict[str, float]],
) -> dict[str, float]:
    """
    Generate a candidate that is likely to be dominated by all current positive plans.

    Strategy:
    - Assign random values to all objectives first.
    - For selected objectives only, push the candidate down to be <= the
      coordinate-wise minimum over all positive plans.
    - Make at least one selected objective strictly smaller than that minimum
      whenever possible, so every positive plan will dominate the candidate.
    """
    full_values = generate_random_plan_values_for_all_objectives(all_objectives)

    min_per_obj = {
        obj: min(p[obj] for p in positive_values)
        for obj in selected_objectives
    }

    strictly_lower_possible = [obj for obj in selected_objectives if min_per_obj[obj] > RANDOM_VALUE_MIN]

    # First make candidate <= min positive value on every selected objective
    for obj in selected_objectives:
        full_values[obj] = random.randint(RANDOM_VALUE_MIN, min_per_obj[obj])

    # Then force strict inequality on at least one selected objective if possible
    if strictly_lower_possible:
        obj = random.choice(strictly_lower_possible)
        full_values[obj] = random.randint(RANDOM_VALUE_MIN, min_per_obj[obj] - 1)

    return full_values


def ensure_output_dirs() -> None:
    os.makedirs(TEXT_LOG_DIR, exist_ok=True)
    os.makedirs(JSON_LOG_DIR, exist_ok=True)
    os.makedirs(SUMMARY_LOG_DIR, exist_ok=True)


def build_random_floorplan_mornp_instance(
        total_objectives: list[str],
        target_num_plans: int,
        k: int,
        seed: int | None = None,
) -> tuple[MORNPInstance, dict]:
    """
    Build a MORNP instance using random plan construction.

    Rules:
    - Randomly select k objectives from total_objectives
    - Every generated plan gets random values for ALL objectives
    - Dominance checks use ONLY the selected k objectives
    - First accepted plan -> positive_plans
    - Dominated by all positive plans -> negative_plans
    - Non-dominated w.r.t. all positive plans -> positive_plans
    - Otherwise discard
    """
    if seed is not None:
        random.seed(seed)

    if k <= 0:
        raise ValueError("k must be positive")

    if k > len(total_objectives):
        raise ValueError("k cannot exceed the total number of available objectives")

    build_start = time.perf_counter()

    selected_objectives = random.sample(total_objectives, k)

    # Stores full values for all generated/accepted plans
    plan_values: dict[str, dict[str, float]] = {}
    positive_plans: list[str] = []
    negative_plans: list[str] = []

    attempts = 0
    discarded_count = 0
    next_plan_id = 1
    dominated_all_count = 0
    nondominated_count = 0

    while attempts < MAX_PLAN_BUILD_ATTEMPTS:
        attempts += 1

        accepted_count = len(positive_plans) + len(negative_plans)
        if (
                accepted_count >= target_num_plans
                and len(positive_plans) >= MIN_POSITIVE_PLANS_REQUIRED
                and len(negative_plans) >= MIN_NEGATIVE_PLANS_REQUIRED
        ): #TODO:: can explode to max attempts. need to fix
            break

        plan_name = f"p_{next_plan_id}"
        next_plan_id += 1

        # Assign values for ALL objectives
        # full_values = generate_random_plan_values_for_all_objectives(total_objectives)

        if not positive_plans:
            full_values = generate_random_plan_values_for_all_objectives(total_objectives)
        else:
            positive_values = [plan_values[p] for p in positive_plans]

            should_bias_negative = (
                len(negative_plans) < MIN_NEGATIVE_PLANS_REQUIRED
                and random.random() < NEGATIVE_BIAS_PROBABILITY
            )

            if should_bias_negative:
                full_values = generate_negative_biased_plan_values(
                    all_objectives=total_objectives,
                    selected_objectives=selected_objectives,
                    positive_values=positive_values,
                )
            else:
                full_values = generate_random_plan_values_for_all_objectives(total_objectives)

        # First accepted plan always goes to positive
        if not positive_plans:
            plan_values[plan_name] = full_values
            positive_plans.append(plan_name)
            continue

        # positive_values = [plan_values[p] for p in positive_plans]

        if dominated_by_all_positive(full_values, positive_values, selected_objectives):
            plan_values[plan_name] = full_values
            negative_plans.append(plan_name)
            dominated_all_count += 1
        elif non_dominated_wrt_all_positive(full_values, positive_values, selected_objectives):
            plan_values[plan_name] = full_values
            positive_plans.append(plan_name)
            nondominated_count += 1
        else:
            discarded_count += 1

    build_end = time.perf_counter()
    construction_time = build_end - build_start

    instance = MORNPInstance(
        objectives=selected_objectives,
        plan_values=plan_values,
        positive_plans=positive_plans,
        negative_plans=negative_plans,
        k=k,
    )

    construction_success = (
            len(positive_plans) >= MIN_POSITIVE_PLANS_REQUIRED
            and len(negative_plans) >= MIN_NEGATIVE_PLANS_REQUIRED
    )

    failure_reason = None
    if len(positive_plans) < MIN_POSITIVE_PLANS_REQUIRED:
        failure_reason = "Not enough positive plans were constructed."
    elif len(negative_plans) < MIN_NEGATIVE_PLANS_REQUIRED:
        failure_reason = "No negative plans were constructed within the attempt limit."

    metadata = {
        "seed": seed,
        "total_objectives": total_objectives,
        "selected_objectives": selected_objectives,
        "target_num_plans": target_num_plans,
        "constructed_positive_count": len(positive_plans),
        "constructed_negative_count": len(negative_plans),
        "constructed_total_count": len(positive_plans) + len(negative_plans),
        "discarded_count": discarded_count,
        "attempts": attempts,
        "max_attempts": MAX_PLAN_BUILD_ATTEMPTS,
        "construction_time_seconds": construction_time,
        "construction_success": construction_success,
        "failure_reason": failure_reason,
        "dominated_all_count": dominated_all_count,
        "nondominated_count": nondominated_count,
    }

    return instance, metadata


def save_experiment_outputs(timestamp_str: str, text_log: str, json_data: dict) -> tuple[str, str]:
    ensure_output_dirs()

    text_path = os.path.join(TEXT_LOG_DIR, f"{timestamp_str}.txt")
    json_path = os.path.join(JSON_LOG_DIR, f"{timestamp_str}.json")

    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text_log)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    return text_path, json_path


def build_experiment_summary(json_data: dict) -> dict:
    construction_meta = json_data["construction_metadata"]
    instance_data = json_data["instance"]
    solver_data = json_data["solver"]
    timing_data = json_data["timing"]

    verification = solver_data.get("verification")

    return {
        "timestamp": json_data["timestamp"],
        "seed": json_data["seed"],
        "objective_count": len(json_data["inputs"]["total_objectives"]),
        "selected_objective_count": len(instance_data["selected_objectives"]),
        "target_num_plans": json_data["inputs"]["target_num_plans"],
        "k": json_data["inputs"]["k"],
        "constructed_positive_count": construction_meta["constructed_positive_count"],
        "constructed_negative_count": construction_meta["constructed_negative_count"],
        "constructed_total_count": construction_meta["constructed_total_count"],
        "discarded_count": construction_meta["discarded_count"],
        "attempts": construction_meta["attempts"],
        "construction_success": construction_meta["construction_success"],
        "failure_reason": construction_meta["failure_reason"],
        "dominated_all_count": construction_meta.get("dominated_all_count"),
        "nondominated_count": construction_meta.get("nondominated_count"),
        "sat": solver_data["sat"],
        "selected_objectives": solver_data["selected_objectives"],
        "cnf_variables": solver_data["cnf_variables"],
        "cnf_clauses": solver_data["cnf_clauses"],
        "verification_verified": None if verification is None else verification["verified"],
        "verification_cardinality_ok": None if verification is None else verification["cardinality_ok"],
        "verification_nondominance_ok": None if verification is None else verification["nondominance_ok"],
        "construction_time_seconds": timing_data["construction_time_seconds"],
        "execution_time_seconds": timing_data["execution_time_seconds"],
        "total_run_time_seconds": timing_data["total_run_time_seconds"],
    }


def save_summary_output(timestamp_str: str, summary_data: dict, prefix: str = "") -> str:
    ensure_output_dirs()

    filename = f"{prefix}{timestamp_str}.json" if prefix else f"{timestamp_str}.json"
    summary_path = os.path.join(SUMMARY_LOG_DIR, filename)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    return summary_path


def run_floorplan_random_experiment(
        total_objectives: list[str],
        target_num_plans: int,
        k: int,
        seed: int | None = None,
) -> dict[str, str | dict[
    str, dict[str, list[str] | dict[str, dict[str, float]] | int] | int | None | dict[str, int | list[str]] | dict[
        str, float | Any] | dict | str | dict[str, list[Any] | None | int | Any]]]:
    total_start = time.perf_counter()
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_buffer = io.StringIO()

    with redirect_stdout(log_buffer):
        print("\n" + "=" * 70)
        print("FLOOR PLAN RANDOM MORNP EXPERIMENT")
        print("=" * 70)
        print(f"Timestamp: {timestamp_str}")
        print(f"Seed: {seed}")
        print(f"Total objective pool: {total_objectives}")
        print(f"Requested number of plans: {target_num_plans}")
        print(f"k: {k}")
        print(f"Max build attempts: {MAX_PLAN_BUILD_ATTEMPTS}")

        instance, construction_meta = build_random_floorplan_mornp_instance(
            total_objectives=total_objectives,
            target_num_plans=target_num_plans,
            k=k,
            seed=seed,
        )

        print("\nConstruction metadata:")
        for key, value in construction_meta.items():
            print(f"  {key}: {value}")

        print("\nConstructed MORNP instance:")
        print("Selected objectives used in instance:", instance.objectives)
        print("Positive plans:", instance.positive_plans)
        print("Negative plans:", instance.negative_plans)
        print("k =", instance.k)

        print("\nFull plan values (all objectives):")
        for p, vals in instance.plan_values.items():
            print(f"  {p}: {vals}")

        if not construction_meta["construction_success"]:
            print("\nConstruction status:")
            print("  Construction failed to produce a meaningful MORNP instance.")
            print(f"  Reason: {construction_meta['failure_reason']}")
            print("  Solver execution skipped.")

            execution_time = 0.0
            result = {
                "sat": None,
                "selected_objectives": [],
            }
            cnf_variables = None
            cnf_clauses = None

        else:
            execution_start = time.perf_counter()

            solver = MORNPDECSATSolver(instance)
            solver.build_formula()

            print("\nCNF stats:")
            print("  #variables =", solver.vpool.top)
            print("  #clauses   =", len(solver.cnf.clauses))

            solver.print_pairwise_debug()

            # result = solver.solve()
            result = solver.solve_and_verify()

            if result["sat"] and result.get("verification") is not None:
                verification = result["verification"]

                print("\nCertificate verification:")
                print("  Verified =", verification["verified"])
                print("  Cardinality OK =", verification["cardinality_ok"])
                print("  Non-dominance OK =", verification["nondominance_ok"])
                print("  Verifier time complexity =", verification["time_complexity"])

                if not verification["verified"]:
                    print("  Reason =", verification["reason"])
                    print("  Violations:")
                    for violation in verification["violations"]:
                        print(
                            f"    {violation['dominating_plan']} dominates "
                            f"{violation['positive_plan']}"
                        )

            execution_end = time.perf_counter()
            execution_time = execution_end - execution_start

            cnf_variables = solver.vpool.top
            cnf_clauses = len(solver.cnf.clauses)

            print("\nSolve result:")
            if result["sat"]:
                print("  SAT = YES")
                print("  Selected objectives =", result["selected_objectives"])
            else:
                print("  SAT = NO")
                print("  Selected objectives = []")

        total_end = time.perf_counter()
        total_time = total_end - total_start

        print("\nTiming:")
        print(f"  Construction time = {construction_meta['construction_time_seconds']:.6f} seconds")
        print(f"  Execution time    = {execution_time:.6f} seconds")
        print(f"  Total run time    = {total_time:.6f} seconds")

    text_log = log_buffer.getvalue()
    print(text_log)

    json_data = {
        "timestamp": timestamp_str,
        "seed": seed,
        "inputs": {
            "total_objectives": total_objectives,
            "target_num_plans": target_num_plans,
            "k": k,
            "max_build_attempts": MAX_PLAN_BUILD_ATTEMPTS,
            "random_value_min": RANDOM_VALUE_MIN,
            "random_value_max": RANDOM_VALUE_MAX,
        },
        "construction_metadata": construction_meta,
        "instance": {
            "selected_objectives": instance.objectives,
            "positive_plans": instance.positive_plans,
            "negative_plans": instance.negative_plans,
            "plan_values_all_objectives": instance.plan_values,
            "k": instance.k,
        },
        "solver": {
            "sat": result["sat"],
            "selected_objectives": result["selected_objectives"],
            "cnf_variables": cnf_variables,
            "cnf_clauses": cnf_clauses,
            "verification": result.get("verification"),
        },
        "timing": {
            "construction_time_seconds": construction_meta["construction_time_seconds"],
            "execution_time_seconds": execution_time,
            "total_run_time_seconds": total_time,
        },
    }

    text_path, json_path = save_experiment_outputs("single_instance_" + timestamp_str, text_log, json_data)

    print(f"Text log saved to: {text_path}")
    print(f"JSON stats saved to: {json_path}")

    summary_data = build_experiment_summary(json_data)
    summary_path = save_summary_output("single_instance_" + timestamp_str, summary_data)

    print(f"Summary log saved to: {summary_path}")

    return {
        "text_path": text_path,
        "json_path": json_path,
        "summary_path": summary_path,
        "text_log": text_log,
        "json_data": json_data,
        "summary_data": summary_data,
    }


def build_stress_test_summary(
    stress_timestamp: str,
    status: str,
    successful_iterations: list[dict],
    exception_type: str | None = None,
    exception_message: str | None = None,
) -> dict:
    return {
        "stress_test_timestamp": stress_timestamp,
        "status": status,
        "iteration_count": len(successful_iterations),
        "iterations": successful_iterations,
        "exception_type": exception_type,
        "exception_message": exception_message,
    }

def extend_objective_pool(base_objectives: list[str], target_count: int) -> list[str]:
    """
    Extend the objective list up to target_count with synthetic objective names.
    """
    if target_count <= len(base_objectives):
        return base_objectives[:target_count]

    extended = list(base_objectives)
    next_index = 1

    while len(extended) < target_count:
        candidate = f"extra_obj_{next_index}"
        if candidate not in extended:
            extended.append(candidate)
        next_index += 1

    return extended


def save_stress_test_outputs(timestamp_str: str, combined_text_log: str, combined_json_data: dict) -> tuple[str, str]:
    ensure_output_dirs()

    text_path = os.path.join(TEXT_LOG_DIR, f"stress_{timestamp_str}.txt")
    json_path = os.path.join(JSON_LOG_DIR, f"stress_{timestamp_str}.json")

    with open(text_path, "w", encoding="utf-8") as f:
        f.write(combined_text_log)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(combined_json_data, f, indent=2)

    return text_path, json_path


def run_floorplan_experiment() -> None:
    total_objectives = [
        "q1_tile",
        "q2_carpet",
        "q3_concrete",
        "q4_wood",
        "distance",
        "time",
        "energy",
        "risk",
    ]

    target_num_plans = 10
    k = 4
    seed = 42

    run_floorplan_random_experiment(
        total_objectives=total_objectives,
        target_num_plans=target_num_plans,
        k=k,
        seed=seed,
    )


def run_default_stress_test_experiment() -> None:
    initial_config = {
        "total_objectives": [
            "q1_tile",
            "q2_carpet",
            "q3_concrete",
            "q4_wood",
            "distance",
            "time",
            "energy",
            "risk",
        ],
        "target_num_plans": 10,
        "k": 4,
        "seed": 42,
    }

    run_stress_test_experiment(initial_config)


def run_stress_test_experiment(initial_config: dict) -> None:
    """
    Run repeated floorplan experiments with gradually increasing size.

    initial_config must contain:
      - total_objectives: list[str]
      - target_num_plans: int
      - k: int
      - seed: int
    """
    global MIN_NEGATIVE_PLANS_REQUIRED
    global MIN_POSITIVE_PLANS_REQUIRED

    ensure_output_dirs()

    stress_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    combined_text_parts: list[str] = []
    combined_json_runs: list[dict] = []
    combined_summary_runs: list[dict] = []

    base_total_objectives = list(initial_config["total_objectives"])
    current_target_objective_count = len(base_total_objectives)
    current_target_num_plans = initial_config["target_num_plans"]
    current_k = initial_config["k"]
    current_seed = initial_config["seed"]

    original_min_negative = MIN_NEGATIVE_PLANS_REQUIRED
    original_min_positive = MIN_POSITIVE_PLANS_REQUIRED

    iteration_index = 1

    try:
        while current_target_objective_count <= STRESS_END_OBJECTIVE_COUNT:
            total_objectives_for_iteration = extend_objective_pool(
                base_objectives=base_total_objectives,
                target_count=current_target_objective_count,
            )

            combined_text_parts.append("\n" + "#" * 90)
            combined_text_parts.append(f"STRESS TEST ITERATION {iteration_index}")
            combined_text_parts.append("#" * 90 + "\n")

            combined_text_parts.append(
                f"Iteration inputs:\n"
                f"  objective_count = {len(total_objectives_for_iteration)}\n"
                f"  target_num_plans = {current_target_num_plans}\n"
                f"  k = {current_k}\n"
                f"  seed = {current_seed}\n"
                f"  MIN_NEGATIVE_PLANS_REQUIRED = {MIN_NEGATIVE_PLANS_REQUIRED}\n"
                f"  MIN_POSITIVE_PLANS_REQUIRED = {MIN_POSITIVE_PLANS_REQUIRED}\n"
            )

            # Guard against invalid k
            if current_k > len(total_objectives_for_iteration):
                combined_text_parts.append(
                    f"Skipping iteration {iteration_index}: "
                    f"k={current_k} exceeds objective_count={len(total_objectives_for_iteration)}\n"
                )
                break

            run_output = run_floorplan_random_experiment(
                total_objectives=total_objectives_for_iteration,
                target_num_plans=current_target_num_plans,
                k=current_k,
                seed=current_seed,
            )

            combined_text_parts.append(run_output["text_log"])
            combined_json_runs.append({
                "iteration_index": iteration_index,
                "iteration_inputs": {
                    "objective_count": len(total_objectives_for_iteration),
                    "target_num_plans": current_target_num_plans,
                    "k": current_k,
                    "seed": current_seed,
                    "min_negative_plans_required": MIN_NEGATIVE_PLANS_REQUIRED,
                    "min_positive_plans_required": MIN_POSITIVE_PLANS_REQUIRED,
                },
                "run_output": run_output["json_data"],
            })

            combined_summary_runs.append({
                "iteration_index": iteration_index,
                "iteration_inputs": {
                    "objective_count": len(total_objectives_for_iteration),
                    "target_num_plans": current_target_num_plans,
                    "k": current_k,
                    "seed": current_seed,
                    "min_negative_plans_required": MIN_NEGATIVE_PLANS_REQUIRED,
                    "min_positive_plans_required": MIN_POSITIVE_PLANS_REQUIRED,
                },
                "run_summary": run_output["summary_data"],
            })

            # next iteration increments
            current_target_objective_count += STRESS_OBJECTIVE_INCREMENT
            current_target_num_plans += STRESS_PLAN_INCREMENT
            current_k += STRESS_K_INCREMENT
            current_seed += 1

            MIN_NEGATIVE_PLANS_REQUIRED += STRESS_MIN_PLAN_INCREMENT
            MIN_POSITIVE_PLANS_REQUIRED += STRESS_MIN_PLAN_INCREMENT

            iteration_index += 1

    except Exception as exc:
        combined_text_parts.append("\n" + "!" * 90 + "\n")
        combined_text_parts.append("STRESS TEST TERMINATED UNEXPECTEDLY\n")
        combined_text_parts.append(f"Exception type: {type(exc).__name__}\n")
        combined_text_parts.append(f"Exception message: {exc}\n")

        combined_json_data = {
            "stress_test_timestamp": stress_timestamp,
            "status": "terminated_unexpectedly",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "successful_iterations": combined_json_runs,
        }

        stress_summary_data = build_stress_test_summary(
            stress_timestamp=stress_timestamp,
            status="terminated_unexpectedly",
            successful_iterations=combined_summary_runs,
            exception_type=type(exc).__name__,
            exception_message=str(exc),
        )

        combined_text_log = "\n".join(combined_text_parts)
        text_path, json_path = save_stress_test_outputs(
            stress_timestamp,
            combined_text_log,
            combined_json_data,
        )

        summary_path = save_summary_output(f"stress_{stress_timestamp}", stress_summary_data)

        print(f"Combined stress summary log saved to: {summary_path}")

        print(f"Stress test terminated unexpectedly.")
        print(f"Combined stress text log saved to: {text_path}")
        print(f"Combined stress JSON log saved to: {json_path}")
        raise

    finally:
        MIN_NEGATIVE_PLANS_REQUIRED = original_min_negative
        MIN_POSITIVE_PLANS_REQUIRED = original_min_positive

    combined_json_data = {
        "stress_test_timestamp": stress_timestamp,
        "status": "completed",
        "successful_iterations": combined_json_runs,
    }

    stress_summary_data = build_stress_test_summary(
        stress_timestamp=stress_timestamp,
        status="completed",
        successful_iterations=combined_summary_runs,
    )

    combined_text_log = "\n".join(combined_text_parts)
    text_path, json_path = save_stress_test_outputs(
        stress_timestamp,
        combined_text_log,
        combined_json_data,
    )

    summary_path = save_summary_output(f"stress_{stress_timestamp}", stress_summary_data)

    print(f"Combined stress summary log saved to: {summary_path}")

    print(f"Stress test completed.")
    print(f"Combined stress text log saved to: {text_path}")
    print(f"Combined stress JSON log saved to: {json_path}")


if __name__ == "__main__":
    # inst1 = build_small_example()
    # run_instance(inst1, "Example 1: Small example")
    #
    # inst2 = build_small_example_equality_unsat()
    # run_instance(inst2, "Example 2: equality UNSAT")

    # run_floorplan_experiment()
    run_default_stress_test_experiment()