import io
import os
import sys
import json
import time
from contextlib import contextmanager, redirect_stdout
from datetime import datetime

from CaseStudies.MoppMornpComparisonCaseStudy import (
    build_shared_pool,
    derive_mopp_instance,
    derive_mornp_instance,
)
from Solvers.moppdec_sat_solver import solve_with_optimal_k
from Solvers.mornpdec_optimal_search import solve_mornpdec_with_optimal_t

# experiment set up
NUM_OBJECTIVES = 20
NUM_PLANS = 700
OMEGA_STAR_SIZE = 6
NUM_COMPARISONS = 1000
LABEL_MIX = (0.30, 0.20, 0.50)  # (1,0,?)
SEED = 42

MORNP_TIME_LIMIT_SEC = 300

TEXT_LOG_DIR = "compare_text"
JSON_LOG_DIR = "compare_json"
SUMMARY_LOG_DIR = "compare_summary"


def _ensure_output_dirs() -> None:
    os.makedirs(TEXT_LOG_DIR, exist_ok=True)
    os.makedirs(JSON_LOG_DIR, exist_ok=True)
    os.makedirs(SUMMARY_LOG_DIR, exist_ok=True)


@contextmanager
def _tee_stdout(buffer: io.StringIO):
    original = sys.stdout

    class Tee:
        def write(self, s):
            original.write(s)
            buffer.write(s)

        def flush(self):
            original.flush()

    sys.stdout = Tee()
    try:
        yield
    finally:
        sys.stdout = original


def _extract_solver_summary(optimal_search_result: dict) -> dict:

    optimal = optimal_search_result.get("optimal")
    certified_minimal = optimal_search_result.get("certified_minimal")
    timed_out = optimal_search_result.get("timed_out")

    if optimal is None:
        return {
            "optimal_t": None,
            "omega_size": None,
            "omega": None,
            "solve_time_sec_at_optimal": None,
            "num_vars_at_optimal": None,
            "num_clauses_at_optimal": None,
            "certified_minimal": certified_minimal,
            "timed_out": timed_out,
        }
    return {
        "optimal_t": optimal["t"] if "t" in optimal else optimal.get("k"),
        "omega_size": len(optimal["solution"]) if optimal["solution"] is not None else None,
        "omega": optimal["solution"],
        "solve_time_sec_at_optimal": optimal["solve_time_sec"],
        "num_vars_at_optimal": optimal.get("num_vars"),
        "num_clauses_at_optimal": optimal.get("num_clauses"),
        "certified_minimal": certified_minimal,
        "timed_out": timed_out,
    }


def run_comparison_experiment() -> dict:
    _ensure_output_dirs()
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_buffer = io.StringIO()
    total_start = time.perf_counter()

    with _tee_stdout(log_buffer):
        print("=" * 70)
        print("MOPP-DEC vs MORNP-DEC COMPARISON EXPERIMENT")
        print("=" * 70)
        print(f"Timestamp: {timestamp_str}")
        print(f"Objectives: {NUM_OBJECTIVES}, Plans: {NUM_PLANS}, "
              f"|Omega*|: {OMEGA_STAR_SIZE}, Seed: {SEED}")
        print(f"MOPP comparisons requested: {NUM_COMPARISONS}, "
              f"target label mix (r=1/0/?): {LABEL_MIX}")

        pool_start = time.perf_counter()
        pool = build_shared_pool(
            num_objectives=NUM_OBJECTIVES,
            num_plans=NUM_PLANS,
            omega_star_size=OMEGA_STAR_SIZE,
            seed=SEED,
        )
        pool_end = time.perf_counter()
        pool_construction_time = pool_end - pool_start

        print(f"\nShared pool built in {pool_construction_time:.6f} sec.")
        print(f"Hidden ground-truth Omega*: {pool.omega_star}")

        mopp_derive_start = time.perf_counter()
        mopp_instance, mopp_derive_meta = derive_mopp_instance(
            pool=pool,
            num_comparisons=NUM_COMPARISONS,
            label_mix=LABEL_MIX,
            seed=SEED + 1,
        )
        mopp_derive_end = time.perf_counter()
        mopp_derive_time = mopp_derive_end - mopp_derive_start

        print(f"\nMOPP-DEC comparison sample derived in {mopp_derive_time:.6f} sec.")
        print("Achieved label mix:", mopp_derive_meta["achieved_label_mix"])
        print(f"Forced indifference pairs constructed: "
              f"{len(mopp_derive_meta['forced_indifference_pairs'])}")

        mornp_derive_start = time.perf_counter()
        mornp_instance, mornp_derive_meta = derive_mornp_instance(pool=pool)
        mornp_derive_end = time.perf_counter()
        mornp_derive_time = mornp_derive_end - mornp_derive_start

        print(f"\nMORNP-DEC classification derived in {mornp_derive_time:.6f} sec.")
        print(f"Positive plans: {mornp_derive_meta['positive_count']}, "
              f"Negative plans: {mornp_derive_meta['negative_count']} "
              f"(all {NUM_PLANS} plans classified, none discarded)")

        print("\n" + "#" * 70)
        print("Solving MOPP-DEC (halving + binary search for optimal t)")
        print("#" * 70)
        mopp_solve_start = time.perf_counter()
        mopp_search_result = solve_with_optimal_k(mopp_instance, verbose=True)
        mopp_solve_end = time.perf_counter()
        mopp_total_solve_time = mopp_solve_end - mopp_solve_start

        print("\n" + "#" * 70)
        print("Solving MORNP-DEC (halving + binary search for optimal t)")
        print("#" * 70)
        mornp_solve_start = time.perf_counter()
        mornp_search_result = solve_mornpdec_with_optimal_t(
            mornp_instance, verbose=True, time_limit_sec=MORNP_TIME_LIMIT_SEC
        )
        mornp_solve_end = time.perf_counter()
        mornp_total_solve_time = mornp_solve_end - mornp_solve_start

        mopp_summary = _extract_solver_summary(mopp_search_result)
        mornp_summary = _extract_solver_summary(mornp_search_result)

        print("\n" + "=" * 70)
        print("HEAD-TO-HEAD RESULT")
        print("=" * 70)
        print(f"MOPP-DEC : optimal t = {mopp_summary['optimal_t']}, "
              f"|Omega| = {mopp_summary['omega_size']}, "
              f"Omega = {mopp_summary['omega']}")
        print(f"MORNP-DEC: optimal t = {mornp_summary['optimal_t']}, "
              f"|Omega| = {mornp_summary['omega_size']}, "
              f"Omega = {mornp_summary['omega']}, "
              f"certified_minimal = {mornp_summary['certified_minimal']}, "
              f"timed_out = {mornp_summary['timed_out']}")
        print(f"Ground-truth Omega* used to generate the instance: {pool.omega_star}")

    total_end = time.perf_counter()
    total_time = total_end - total_start

    text_log = log_buffer.getvalue()

    json_data = {
        "timestamp": timestamp_str,
        "parameters": {
            "num_objectives": NUM_OBJECTIVES,
            "num_plans": NUM_PLANS,
            "omega_star_size": OMEGA_STAR_SIZE,
            "num_comparisons_requested": NUM_COMPARISONS,
            "target_label_mix": {"r1": LABEL_MIX[0], "r0": LABEL_MIX[1], "rq": LABEL_MIX[2]},
            "seed": SEED,
        },
        "shared_pool": {
            "objectives": pool.objectives,
            "plans": pool.plans,
            "plan_values": pool.values,
            "omega_star": pool.omega_star,
            "construction_time_sec": pool_construction_time,
        },
        "mopp": {
            "comparisons": mopp_derive_meta["comparisons"],
            "achieved_label_mix": mopp_derive_meta["achieved_label_mix"],
            "forced_indifference_pairs": mopp_derive_meta["forced_indifference_pairs"],
            "derivation_time_sec": mopp_derive_time,
            "optimal_t": mopp_summary["optimal_t"],
            "omega_size": mopp_summary["omega_size"],
            "omega": mopp_summary["omega"],
            "total_solve_time_sec": mopp_total_solve_time,
            "num_vars_at_optimal": mopp_summary["num_vars_at_optimal"],
            "num_clauses_at_optimal": mopp_summary["num_clauses_at_optimal"],
            "halving_iterations": mopp_search_result["halving"]["iterations"],
            "binary_search_iterations": mopp_search_result["binary_iterations"],
        },
        "mornp": {
            "positive_plans": mornp_derive_meta["positive_plans"],
            "negative_plans": mornp_derive_meta["negative_plans"],
            "positive_count": mornp_derive_meta["positive_count"],
            "negative_count": mornp_derive_meta["negative_count"],
            "derivation_time_sec": mornp_derive_time,
            "optimal_t": mornp_summary["optimal_t"],
            "omega_size": mornp_summary["omega_size"],
            "omega": mornp_summary["omega"],
            "total_solve_time_sec": mornp_total_solve_time,
            "num_vars_at_optimal": mornp_summary["num_vars_at_optimal"],
            "num_clauses_at_optimal": mornp_summary["num_clauses_at_optimal"],
            "halving_iterations": mornp_search_result["halving"]["iterations"],
            "binary_search_iterations": mornp_search_result["binary_iterations"],
        },
        "total_experiment_time_sec": total_time,
    }

    summary_data = {
        "timestamp": timestamp_str,
        "num_objectives": NUM_OBJECTIVES,
        "num_plans": NUM_PLANS,
        "omega_star_size": OMEGA_STAR_SIZE,
        "omega_star": pool.omega_star,
        "mopp_optimal_t": mopp_summary["optimal_t"],
        "mopp_omega_size": mopp_summary["omega_size"],
        "mopp_omega": mopp_summary["omega"],
        "mopp_achieved_label_mix": mopp_derive_meta["achieved_label_mix"],
        "mopp_total_solve_time_sec": mopp_total_solve_time,
        "mopp_num_vars": mopp_summary["num_vars_at_optimal"],
        "mopp_num_clauses": mopp_summary["num_clauses_at_optimal"],
        "mornp_optimal_t": mornp_summary["optimal_t"],
        "mornp_omega_size": mornp_summary["omega_size"],
        "mornp_omega": mornp_summary["omega"],
        "mornp_certified_minimal": mornp_summary["certified_minimal"],
        "mornp_timed_out": mornp_summary["timed_out"],
        "mornp_positive_count": mornp_derive_meta["positive_count"],
        "mornp_negative_count": mornp_derive_meta["negative_count"],
        "mornp_total_solve_time_sec": mornp_total_solve_time,
        "mornp_num_vars": mornp_summary["num_vars_at_optimal"],
        "mornp_num_clauses": mornp_summary["num_clauses_at_optimal"],
        "mornp_time_limit_sec_per_solve": MORNP_TIME_LIMIT_SEC,
        "total_experiment_time_sec": total_time,
    }

    text_path = os.path.join(TEXT_LOG_DIR, f"{timestamp_str}.txt")
    json_path = os.path.join(JSON_LOG_DIR, f"{timestamp_str}.json")
    summary_path = os.path.join(SUMMARY_LOG_DIR, f"{timestamp_str}.json")

    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text_log)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nText log saved to:    {text_path}")
    print(f"Full JSON saved to:   {json_path}")
    print(f"Summary JSON saved to: {summary_path}")

    return {
        "text_path": text_path,
        "json_path": json_path,
        "summary_path": summary_path,
        "json_data": json_data,
        "summary_data": summary_data,
    }


def run_comparison_sweep(
    num_rounds: int,
    *,
    base_num_objectives: int = NUM_OBJECTIVES,
    objectives_increment: int = 0,
    base_num_plans: int = NUM_PLANS,
    plans_increment: int = 0,
    base_omega_star_size: int = OMEGA_STAR_SIZE,
    omega_star_size_increment: int = 0,
    base_num_comparisons: int = NUM_COMPARISONS,
    comparisons_increment: int = 0,
    comparisons_equal_plans: bool = False,
    label_mix=LABEL_MIX,
    seed: int = SEED,
    mornp_time_limit_sec: float = MORNP_TIME_LIMIT_SEC,
    verbose: bool = True,
) -> dict:

    global NUM_OBJECTIVES, NUM_PLANS, OMEGA_STAR_SIZE, NUM_COMPARISONS
    global LABEL_MIX, SEED, MORNP_TIME_LIMIT_SEC

    orig = dict(
        NUM_OBJECTIVES=NUM_OBJECTIVES, NUM_PLANS=NUM_PLANS,
        OMEGA_STAR_SIZE=OMEGA_STAR_SIZE, NUM_COMPARISONS=NUM_COMPARISONS,
        LABEL_MIX=LABEL_MIX, SEED=SEED,
        MORNP_TIME_LIMIT_SEC=MORNP_TIME_LIMIT_SEC,
    )

    sweep_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_results = []

    try:
        for i in range(num_rounds):
            if i > 0:
                time.sleep(1.1)

            NUM_OBJECTIVES = base_num_objectives + i * objectives_increment
            NUM_PLANS = base_num_plans + i * plans_increment
            OMEGA_STAR_SIZE = base_omega_star_size + i * omega_star_size_increment
            NUM_COMPARISONS = (
                NUM_PLANS if comparisons_equal_plans
                else base_num_comparisons + i * comparisons_increment
            )
            LABEL_MIX = label_mix
            SEED = seed
            MORNP_TIME_LIMIT_SEC = mornp_time_limit_sec

            if verbose:
                print("\n" + "#" * 70)
                print(f"[Sweep round {i + 1}/{num_rounds}] "
                      f"objectives={NUM_OBJECTIVES}, plans={NUM_PLANS}, "
                      f"omega*_size={OMEGA_STAR_SIZE}, "
                      f"comparisons={NUM_COMPARISONS}")
                print("#" * 70)

            round_result = run_comparison_experiment()
            sweep_results.append({
                "round": i + 1,
                "params": {
                    "num_objectives": NUM_OBJECTIVES,
                    "num_plans": NUM_PLANS,
                    "omega_star_size": OMEGA_STAR_SIZE,
                    "num_comparisons": NUM_COMPARISONS,
                },
                "summary_data": round_result["summary_data"],
                "json_path": round_result["json_path"],
                "summary_path": round_result["summary_path"],
            })
    finally:
        NUM_OBJECTIVES = orig["NUM_OBJECTIVES"]
        NUM_PLANS = orig["NUM_PLANS"]
        OMEGA_STAR_SIZE = orig["OMEGA_STAR_SIZE"]
        NUM_COMPARISONS = orig["NUM_COMPARISONS"]
        LABEL_MIX = orig["LABEL_MIX"]
        SEED = orig["SEED"]
        MORNP_TIME_LIMIT_SEC = orig["MORNP_TIME_LIMIT_SEC"]

    _ensure_output_dirs()
    sweep_summary_path = os.path.join(SUMMARY_LOG_DIR, f"sweep_{sweep_timestamp}.json")
    with open(sweep_summary_path, "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2)

    if verbose:
        print(f"\nSweep summary saved to: {sweep_summary_path}")

    return {"sweep_summary_path": sweep_summary_path, "rounds": sweep_results}


if __name__ == "__main__":
    # run_comparison_experiment()


    #
    run_comparison_sweep(
        num_rounds=5,
        base_num_plans=20,
        plans_increment=40,
        comparisons_equal_plans=True,
    )
