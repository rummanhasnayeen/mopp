from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime
from typing import Dict, List, Tuple

from Models.mornp_instance import MORNPInstance


# =========================
# Configuration
# =========================
RANDOM_VALUE_MIN = 1
RANDOM_VALUE_MAX = 100
MAX_PLAN_BUILD_ATTEMPTS = 100000

OBJECTIVE_COUNT = 20
TARGET_NUM_PLANS = 1000
K = 12

# Pick a value in the requested 100-200 range
MIN_NEGATIVE_PLANS_REQUIRED = 150
MIN_POSITIVE_PLANS_REQUIRED = 5

NEGATIVE_BIAS_PROBABILITY = 0.70

OUTPUT_DIR = "seed_data_for_generative"


# =========================
# Dominance helpers
# =========================
def dominates(plan_a: dict, plan_b: dict, selected_objectives: List[str]) -> bool:
    """
    plan_a dominates plan_b iff:
      - plan_a[obj] >= plan_b[obj] for all selected objectives
      - and > on at least one selected objective
    Larger values are better.
    """
    all_ge = all(plan_a[obj] >= plan_b[obj] for obj in selected_objectives)
    one_gt = any(plan_a[obj] > plan_b[obj] for obj in selected_objectives)
    return all_ge and one_gt

def dominates_under_selected_objectives(
    plan_a_name: str,
    plan_b_name: str,
    plan_values: Dict[str, Dict[str, float]],
    selected_objectives: List[str],
) -> bool:
    """
    Return True iff plan_a Pareto-dominates plan_b under selected objectives.
    """
    plan_a = plan_values[plan_a_name]
    plan_b = plan_values[plan_b_name]

    all_ge = all(plan_a[obj] >= plan_b[obj] for obj in selected_objectives)
    one_gt = any(plan_a[obj] > plan_b[obj] for obj in selected_objectives)
    return all_ge and one_gt


def compute_final_frontier_partition(
    plan_values: Dict[str, Dict[str, float]],
    selected_objectives: List[str],
) -> Tuple[List[str], List[str]]:
    """
    Partition the accepted sample into:
      - positive_plans: plans not dominated by any other accepted sample plan
      - negative_plans: plans dominated by at least one other accepted sample plan

    This matches the final nondominance condition that the solver verifies.
    """
    all_plans = list(plan_values.keys())

    positive_plans = []
    negative_plans = []

    for pi in all_plans:
        dominated = False

        for pi_prime in all_plans:
            if pi == pi_prime:
                continue

            if dominates_under_selected_objectives(
                plan_a_name=pi_prime,
                plan_b_name=pi,
                plan_values=plan_values,
                selected_objectives=selected_objectives,
            ):
                dominated = True
                break

        if dominated:
            negative_plans.append(pi)
        else:
            positive_plans.append(pi)

    return positive_plans, negative_plans

def final_negative_count_ok(
    plan_values: Dict[str, Dict[str, float]],
    selected_objectives: List[str],
    min_negative_plans_required: int,
) -> bool:
    """
    Check whether the final frontier partition yields enough negatives.
    """
    _, final_negative_plans = compute_final_frontier_partition(
        plan_values=plan_values,
        selected_objectives=selected_objectives,
    )
    return len(final_negative_plans) >= min_negative_plans_required

def verify_generated_dataset(instance: MORNPInstance) -> dict:
    """
    Verify the generated dataset in two ways:

    1. MORNPDEC formulation check (matches current solver/verifier):
       For every pi in P+, no pi' in P+ U P- dominates pi
       under the selected objectives.

    2. Stronger negative check (optional dataset sanity check):
       Every pi in P- is dominated by at least one plan in P+ U P-.

    Returns a JSON-friendly verification report.
    """
    selected_objectives = instance.objectives
    all_sample_plans = instance.all_sample_plans
    positive_plans = instance.positive_plans
    negative_plans = instance.negative_plans
    plan_values = instance.plan_values

    positive_violations = []
    negative_violations = []

    # -------------------------------------------------
    # A. Verify formulation used by current MORNPDEC code
    #    No positive plan may be dominated by any sample plan
    # -------------------------------------------------
    for pi in positive_plans:
        for pi_prime in all_sample_plans:
            if pi == pi_prime:
                continue

            if dominates_under_selected_objectives(
                plan_a_name=pi_prime,
                plan_b_name=pi,
                plan_values=plan_values,
                selected_objectives=selected_objectives,
            ):
                positive_violations.append({
                    "positive_plan": pi,
                    "dominating_plan": pi_prime,
                })

    positive_nondominance_ok = len(positive_violations) == 0

    # -------------------------------------------------
    # B. Optional stronger negative consistency check
    #    Each negative plan should be dominated by at least one sample plan
    # -------------------------------------------------
    for pi_neg in negative_plans:
        dominated_by = []

        for pi_other in all_sample_plans:
            if pi_neg == pi_other:
                continue

            if dominates_under_selected_objectives(
                plan_a_name=pi_other,
                plan_b_name=pi_neg,
                plan_values=plan_values,
                selected_objectives=selected_objectives,
            ):
                dominated_by.append(pi_other)

        if not dominated_by:
            negative_violations.append({
                "negative_plan": pi_neg,
                "dominated_by": [],
            })

    negative_dominance_ok = len(negative_violations) == 0

    return {
        "formulation_check": {
            "description": (
                "For every pi in P+, no pi' in P+ U P- dominates pi "
                "under the selected objectives."
            ),
            "passed": positive_nondominance_ok,
            "violations": positive_violations,
        },
        "negative_consistency_check": {
            "description": (
                "Every pi in P- is dominated by at least one plan in P+ U P- "
                "under the selected objectives."
            ),
            "passed": negative_dominance_ok,
            "violations": negative_violations,
        },
        "overall_passed": positive_nondominance_ok and negative_dominance_ok,
    }


def dominated_by_all_positive(
    candidate: dict,
    positive_values: List[dict],
    selected_objectives: List[str],
) -> bool:
    return all(dominates(pos, candidate, selected_objectives) for pos in positive_values)


def non_dominated_wrt_all_positive(
    candidate: dict,
    positive_values: List[dict],
    selected_objectives: List[str],
) -> bool:
    return all(not dominates(pos, candidate, selected_objectives) for pos in positive_values)


# =========================
# Data generation helpers
# =========================
def build_objective_pool(objective_count: int) -> List[str]:
    """
    Build a pool of generic objective names.
    """
    return [f"obj_{i}" for i in range(1, objective_count + 1)]


def generate_random_plan_values_for_all_objectives(
    all_objectives: List[str],
) -> Dict[str, float]:
    return {
        obj: random.randint(RANDOM_VALUE_MIN, RANDOM_VALUE_MAX)
        for obj in all_objectives
    }


def generate_negative_biased_plan_values(
    all_objectives: List[str],
    selected_objectives: List[str],
    positive_values: List[Dict[str, float]],
) -> Dict[str, float]:
    """
    Generate a candidate that is likely to be dominated by all current positive plans.

    Strategy:
    - Randomly initialize all objectives
    - For selected objectives only, push values downward
      to be <= the coordinate-wise minimum over all positive plans
    - Force at least one strict decrease when possible
    """
    full_values = generate_random_plan_values_for_all_objectives(all_objectives)

    min_per_obj = {
        obj: min(p[obj] for p in positive_values)
        for obj in selected_objectives
    }

    strictly_lower_possible = [
        obj for obj in selected_objectives
        if min_per_obj[obj] > RANDOM_VALUE_MIN
    ]

    for obj in selected_objectives:
        full_values[obj] = random.randint(RANDOM_VALUE_MIN, min_per_obj[obj])

    if strictly_lower_possible:
        obj = random.choice(strictly_lower_possible)
        full_values[obj] = random.randint(RANDOM_VALUE_MIN, min_per_obj[obj] - 1)

    return full_values


def build_random_mornp_seed_instance(
    total_objectives: List[str],
    target_num_plans: int,
    k: int,
    min_negative_plans_required: int,
    min_positive_plans_required: int,
    seed: int | None = None,
) -> Tuple[MORNPInstance, dict]:
    """
    Uses the same construction strategy as your current experiment builder:
    - randomly select k objectives
    - generate plans over ALL objectives
    - first accepted plan -> P+
    - dominated by all positive -> P-
    - non-dominated wrt all positive -> P+
    - otherwise discard
    """
    if seed is not None:
        random.seed(seed)

    if k <= 0:
        raise ValueError("k must be positive")

    if k > len(total_objectives):
        raise ValueError("k cannot exceed total objective count")

    build_start = time.perf_counter()

    selected_objectives = random.sample(total_objectives, k)

    plan_values: Dict[str, Dict[str, float]] = {}
    positive_plans: List[str] = []
    negative_plans: List[str] = []

    attempts = 0
    discarded_count = 0
    next_plan_id = 1
    dominated_all_count = 0
    nondominated_count = 0

    while attempts < MAX_PLAN_BUILD_ATTEMPTS:
        attempts += 1

        accepted_count = len(plan_values)

        if accepted_count >= target_num_plans:
            if final_negative_count_ok(
                plan_values=plan_values,
                selected_objectives=selected_objectives,
                min_negative_plans_required=min_negative_plans_required,
            ):
                break

        plan_name = f"p_{next_plan_id}"
        next_plan_id += 1

        if not positive_plans:
            full_values = generate_random_plan_values_for_all_objectives(total_objectives)
            plan_values[plan_name] = full_values
            positive_plans.append(plan_name)
            continue

        positive_values = [plan_values[p] for p in positive_plans]

        should_bias_negative = (
            len(negative_plans) < min_negative_plans_required
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

    # -------------------------------------------------
    # Recompute final P+ / P- over the accepted sample
    # so that labels satisfy the same final nondominance
    # condition used by the MORNPDEC solver verifier.
    # -------------------------------------------------
    final_positive_plans, final_negative_plans = compute_final_frontier_partition(
        plan_values=plan_values,
        selected_objectives=selected_objectives,
    )

    build_end = time.perf_counter()

    instance = MORNPInstance(
        objectives=selected_objectives,
        plan_values=plan_values,
        positive_plans=final_positive_plans,
        negative_plans=final_negative_plans,
        k=k,
    )

    construction_success = (
        len(final_positive_plans) >= min_positive_plans_required
        and len(final_negative_plans) >= min_negative_plans_required
        and (len(final_positive_plans) + len(final_negative_plans)) >= target_num_plans
    )

    failure_reason = None
    if len(final_positive_plans) < min_positive_plans_required:
        failure_reason = "Not enough final positive plans were constructed."
    elif len(final_negative_plans) < min_negative_plans_required:
        failure_reason = "Not enough final negative plans were constructed."
    elif (len(final_positive_plans) + len(final_negative_plans)) < target_num_plans:
        failure_reason = "Target number of plans was not reached within attempt limit."

    metadata = {
        "seed": seed,
        "total_objectives": total_objectives,
        "selected_objectives": selected_objectives,
        "target_num_plans": target_num_plans,
        "constructed_positive_count": len(final_positive_plans),
        "constructed_negative_count": len(final_negative_plans),
        "constructed_total_count": len(final_positive_plans) + len(final_negative_plans),
        "temporary_positive_count_before_final_partition": len(positive_plans),
        "temporary_negative_count_before_final_partition": len(negative_plans),
        "discarded_count": discarded_count,
        "attempts": attempts,
        "max_attempts": MAX_PLAN_BUILD_ATTEMPTS,
        "construction_time_seconds": build_end - build_start,
        "construction_success": construction_success,
        "failure_reason": failure_reason,
        "dominated_all_count": dominated_all_count,
        "nondominated_count": nondominated_count,
        "min_negative_plans_required": min_negative_plans_required,
        "min_positive_plans_required": min_positive_plans_required,
        "objective_count": len(total_objectives),
        "k": k,
    }

    return instance, metadata


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_output_payload(instance: MORNPInstance, metadata: dict) -> dict:
    """
    Create a JSON-friendly payload for generative-model training.
    """
    records = []
    for plan_name, values in instance.plan_values.items():
        label = 1 if plan_name in instance.positive_plans else 0

        records.append({
            "plan_name": plan_name,
            "label": label,
            "set_name": "P+" if label == 1 else "P-",
            "objective_values": values,
            "feature_vector_order": list(values.keys()),
            "feature_vector": [values[obj] for obj in values.keys()],
        })

    return {
        "dataset_type": "MORNP_seed_data_for_generative_modeling",
        "created_at_readable": datetime.now().isoformat(),
        "metadata": metadata,
        "instance_summary": {
            "selected_objectives": instance.objectives,
            "k": instance.k,
            "positive_plans": instance.positive_plans,
            "negative_plans": instance.negative_plans,
        },
        "records": records,
    }


def save_payload(payload: dict) -> str:
    ensure_output_dir()

    unix_ts = int(time.time())
    filename = f"generative_data_seed{unix_ts}.json"
    output_path = os.path.join(OUTPUT_DIR, filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return output_path


def main() -> None:
    # Use unix timestamp as seed for reproducibility and filename traceability
    seed = int(time.time())

    total_objectives = build_objective_pool(OBJECTIVE_COUNT)

    instance, metadata = build_random_mornp_seed_instance(
        total_objectives=total_objectives,
        target_num_plans=TARGET_NUM_PLANS,
        k=K,
        min_negative_plans_required=MIN_NEGATIVE_PLANS_REQUIRED,
        min_positive_plans_required=MIN_POSITIVE_PLANS_REQUIRED,
        seed=seed,
    )

    if not metadata["construction_success"]:
        raise RuntimeError(
            f"Construction failed: {metadata['failure_reason']} | "
            f"positive={metadata['constructed_positive_count']} | "
            f"negative={metadata['constructed_negative_count']} | "
            f"total={metadata['constructed_total_count']}"
        )

    payload = build_output_payload(instance, metadata)
    verification_report = verify_generated_dataset(instance)
    payload["verification"] = verification_report
    output_path = save_payload(payload)

    print("=" * 70)
    print("GENERATIVE SEED DATA GENERATED")
    print("=" * 70)
    print(f"Seed: {seed}")
    print(f"Output file: {output_path}")
    print(f"Selected objectives: {instance.objectives}")
    print(f"Positive plans: {len(instance.positive_plans)}")
    print(f"Negative plans: {len(instance.negative_plans)}")
    print(f"Total accepted plans: {len(instance.positive_plans) + len(instance.negative_plans)}")
    print(f"Discarded plans: {metadata['discarded_count']}")
    print(f"Construction time: {metadata['construction_time_seconds']:.6f} seconds")

    print("\nVerification summary:")
    print(
        f"  Formulation check passed: "
        f"{verification_report['formulation_check']['passed']}"
    )
    print(
        f"  Negative consistency check passed: "
        f"{verification_report['negative_consistency_check']['passed']}"
    )
    print(f"  Overall passed: {verification_report['overall_passed']}")

    print(
        f"  Positive formulation violations: "
        f"{len(verification_report['formulation_check']['violations'])}"
    )
    print(
        f"  Negative consistency violations: "
        f"{len(verification_report['negative_consistency_check']['violations'])}"
    )
    #
    # if not verification_report["overall_passed"]:
    #     raise RuntimeError(
    #         "Generated dataset failed verification. "
    #         "Check payload['verification'] in the output JSON."
    #     )


if __name__ == "__main__":
    main()