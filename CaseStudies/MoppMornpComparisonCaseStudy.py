"""
Shared case-study generator for the MOPP-DEC vs MORNP-DEC head-to-head
comparison experiment (added per ICTAI review R1's request: compare the two
formulations on the SAME underlying instance, to show how MOPP's richer
pairwise-comparison supervision affects the recovered |Omega| relative to
MORNP's weaker nondominated/dominated labels).

Pipeline:
  1) build_shared_pool(...)     -> ONE pool of objectives, plans, plan
                                    values, and a hidden ground-truth
                                    Omega*, shared by both derivations below.
  2) derive_mopp_instance(pool) -> a plan-comparison sample (pi, pi', r)
                                    with r in {1, 0, "?"}, labeled under
                                    Omega*, with a controlled label mix.
  3) derive_mornp_instance(pool)-> a positive/negative split: EVERY plan in
                                    the pool is classified as nondominated
                                    (P+) or dominated (P-) under Omega*,
                                    checked against the full population.
                                    No plans are discarded.

IMPORTANT ordering requirement: derive_mopp_instance() may mutate a small
number of plans' values (see the note in that function for why), so it must
be called BEFORE derive_mornp_instance() if you want both derivations to
reflect the identical, final, shared ground truth. The comparison-experiment
driver (main_compare_mopp_mornp.py) calls them in this order.

This file is purely additive -- it does not modify DynamicRandomCaseStudy.py,
mopp_instance.py, or mornp_instance.py, though it mirrors some of the
generation techniques already used in DynamicRandomCaseStudy for consistency.
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

from Models.mopp_instance import MOPPInstance
from Models.mornp_instance import MORNPInstance

ComparisonLabel = Union[int, str]  # 1, 0, or "?"

PLAN_VALUE_MIN = 1
PLAN_VALUE_MAX = 10


@dataclass
class SharedPool:
    objectives: List[str]
    plans: List[str]
    values: Dict[str, Dict[str, int]]
    omega_star: List[str]
    seed: int


def build_shared_pool(
    num_objectives: int,
    num_plans: int,
    omega_star_size: int,
    seed: int = 42,
) -> SharedPool:
    """
    Build the single shared pool of objectives, plans, values, and hidden
    ground-truth Omega* used by both derivations below.
    """
    if omega_star_size > num_objectives:
        raise ValueError("omega_star_size cannot exceed num_objectives")
    if omega_star_size < 1:
        raise ValueError("omega_star_size must be >= 1")

    rng = random.Random(seed)

    objectives = [f"o{i}" for i in range(1, num_objectives + 1)]
    plans = [f"p{i}" for i in range(1, num_plans + 1)]

    values = {
        p: {o: rng.randint(PLAN_VALUE_MIN, PLAN_VALUE_MAX) for o in objectives}
        for p in plans
    }

    omega_star = rng.sample(objectives, omega_star_size)

    return SharedPool(
        objectives=objectives,
        plans=plans,
        values=values,
        omega_star=omega_star,
        seed=seed,
    )


def _relation_under_omega(
    values: Dict[str, Dict[str, int]],
    omega: List[str],
    p: str,
    q: str,
) -> ComparisonLabel:
    """
    Relation of p vs q restricted to the objectives in `omega`:
      1    if p strictly Pareto-dominates q
      -1   if q strictly Pareto-dominates p
      0    if p and q are equal on every objective in omega
      "?"  otherwise (incomparable)
    """
    ge_all = all(values[p][o] >= values[q][o] for o in omega)
    le_all = all(values[p][o] <= values[q][o] for o in omega)
    gt_any = any(values[p][o] > values[q][o] for o in omega)
    lt_any = any(values[p][o] < values[q][o] for o in omega)

    if ge_all and gt_any:
        return 1
    if le_all and lt_any:
        return -1
    if not gt_any and not lt_any:
        return 0
    return "?"


def derive_mopp_instance(
    pool: SharedPool,
    num_comparisons: int,
    label_mix: Tuple[float, float, float] = (0.30, 0.20, 0.50),
    seed: int = None,
) -> Tuple[MOPPInstance, Dict]:
    """
    Sample `num_comparisons` plan pairs from the shared pool and label each
    under pool.omega_star, targeting the requested (r=1, r=0, r=?)
    percentage mix given in `label_mix`.

    NOTE on r=0 (indifference): with wide-range random values (1-10) over
    |omega_star| >= 2 objectives, an exact tie on EVERY selected objective
    is naturally extremely rare (roughly (1/10)^|omega_star| per random
    pair) -- for the default omega_star_size=6 that's about a
    1-in-a-million chance, so a meaningful r=0 count is not reachable by
    pure random rejection sampling. To honor the requested label mix, this
    function DELIBERATELY CONSTRUCTS the required number of indifferent
    pairs by choosing `target_zeros` disjoint pairs of previously-random
    plans and setting the second plan's values equal to the first's on
    every objective in omega_star (mirroring the technique already used by
    DynamicRandomCaseStudy._create_indifference_pairs). This mutates
    pool.values for those specific plans -- which is why this function must
    run BEFORE derive_mornp_instance() if both derivations are meant to see
    the same final, shared values.

    r=1 and r=? are both naturally abundant under random values (roughly
    5-6% and >90% of random pairs respectively, for 6 objectives), so both
    are filled via ordinary rejection sampling, no construction needed.

    Returns the MOPPInstance plus a metadata dict recording the ACHIEVED
    label counts/percentages and the full comparison list, for reporting.
    """
    rng = random.Random(seed if seed is not None else pool.seed + 1)

    frac_one, frac_zero, frac_q = label_mix
    if abs((frac_one + frac_zero + frac_q) - 1.0) > 1e-6:
        raise ValueError("label_mix fractions must sum to 1.0")

    target_ones = int(round(frac_one * num_comparisons))
    target_zeros = int(round(frac_zero * num_comparisons))
    target_q = num_comparisons - target_ones - target_zeros

    omega = pool.omega_star
    seen = set()

    # ---- Step 1: deliberately construct the requested number of exact
    #      ties (r=0), since these are naturally unobservable at random ----
    zeros: List[Tuple[str, str, ComparisonLabel]] = []
    if target_zeros > 0:
        if 2 * target_zeros > len(pool.plans):
            raise ValueError(
                "num_plans too small to construct the requested number of "
                "disjoint indifference pairs; increase num_plans or lower "
                "the r=0 target fraction."
            )
        shuffled_plans = pool.plans[:]
        rng.shuffle(shuffled_plans)
        for idx in range(target_zeros):
            a, b = shuffled_plans[2 * idx], shuffled_plans[2 * idx + 1]
            for o in omega:
                pool.values[b][o] = pool.values[a][o]
            zeros.append((a, b, 0))
            seen.add((a, b))

    # ---- Step 2: naturally sample toward the r=1 and r=? targets ----
    ones: List[Tuple[str, str, ComparisonLabel]] = []
    qs: List[Tuple[str, str, ComparisonLabel]] = []

    max_attempts = max(200_000, num_comparisons * 400)
    attempts = 0
    while len(ones) + len(qs) < target_ones + target_q and attempts < max_attempts:
        attempts += 1
        p, q = rng.sample(pool.plans, 2)
        if (p, q) in seen or (q, p) in seen:
            continue

        rel = _relation_under_omega(pool.values, omega, p, q)
        if rel == 1 and len(ones) < target_ones:
            ones.append((p, q, 1)); seen.add((p, q))
        elif rel == -1 and len(ones) < target_ones:
            ones.append((q, p, 1)); seen.add((q, p))
        elif rel == "?" and len(qs) < target_q:
            qs.append((p, q, "?")); seen.add((p, q))
        # rel == 0 here is astronomically unlikely and, if it ever occurs,
        # is simply skipped (the r=0 bucket is already filled by construction)

    comparisons = zeros + ones + qs
    if len(comparisons) < num_comparisons:
        raise RuntimeError(
            f"Could only generate {len(comparisons)}/{num_comparisons} "
            f"comparisons after {attempts} sampling attempts "
            f"(r=1: {len(ones)}/{target_ones}, r=?: {len(qs)}/{target_q}). "
            "Try increasing num_plans, lowering num_comparisons, or "
            "adjusting the label mix."
        )
    rng.shuffle(comparisons)

    total = len(comparisons)
    count_r1 = sum(1 for _, _, r in comparisons if r == 1)
    count_r0 = sum(1 for _, _, r in comparisons if r == 0)
    count_rq = sum(1 for _, _, r in comparisons if r == "?")

    achieved = {
        "num_comparisons": total,
        "count_r1": count_r1,
        "count_r0": count_r0,
        "count_rq": count_rq,
        "pct_r1": count_r1 / total,
        "pct_r0": count_r0 / total,
        "pct_rq": count_rq / total,
    }

    instance = MOPPInstance(
        objectives=pool.objectives,
        plans=pool.plans,
        values=pool.values,
        comparisons=comparisons,
        k=len(pool.objectives),  # placeholder; overwritten by the optimal-t search
    )

    metadata = {
        "target_label_mix": {"r1": frac_one, "r0": frac_zero, "rq": frac_q},
        "achieved_label_mix": achieved,
        "comparisons": comparisons,
        "forced_indifference_pairs": zeros,
    }

    return instance, metadata


def derive_mornp_instance(pool: SharedPool) -> Tuple[MORNPInstance, Dict]:
    """
    Classify EVERY plan in the shared pool as positive (nondominated under
    Omega*, checked against the full pool) or negative (dominated by at
    least one other plan in the pool under Omega*). No plans are discarded
    -- under plain Pareto dominance every plan is exactly one or the other.

    Must be called AFTER derive_mopp_instance() if you want this to reflect
    the same final values (see the note in that function).
    """
    plans = pool.plans
    values = pool.values
    omega = pool.omega_star

    # Precompute each plan's values over omega once, to speed up the O(n^2)
    # pairwise dominance check below.
    omega_values = {p: [values[p][o] for o in omega] for p in plans}

    def dominates(a_vals: List[int], b_vals: List[int]) -> bool:
        ge_all = True
        gt_any = False
        for av, bv in zip(a_vals, b_vals):
            if av < bv:
                return False
            if av > bv:
                gt_any = True
        return ge_all and gt_any

    positive_plans: List[str] = []
    negative_plans: List[str] = []

    for p in plans:
        p_vals = omega_values[p]
        is_dominated = any(
            dominates(omega_values[q], p_vals) for q in plans if q != p
        )
        if is_dominated:
            negative_plans.append(p)
        else:
            positive_plans.append(p)

    instance = MORNPInstance(
        objectives=pool.objectives,
        plan_values=values,
        positive_plans=positive_plans,
        negative_plans=negative_plans,
        k=len(pool.objectives),  # placeholder; overwritten by the optimal-t search
    )

    metadata = {
        "positive_plans": positive_plans,
        "negative_plans": negative_plans,
        "positive_count": len(positive_plans),
        "negative_count": len(negative_plans),
    }

    return instance, metadata
