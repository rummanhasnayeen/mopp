import time
import  datetime
import sys
import json
import os
from contextlib import contextmanager

from Solvers.moppdec_sat_solver import MOPPDECSATSolver, solve_with_halving_k, solve_with_optimal_k
from CaseStudies.car_example import CarPreferenceExample
from CaseStudies.autonomous_delivery_vehicle import AutonomousDeliveryVehicleCaseStudy
from CaseStudies.AutonomousVehicle10ObjectiveCaseStudy import AutonomousVehicle10ObjectiveCaseStudy
from CaseStudies.DynamicRandomCaseStudy import DynamicRandomCaseStudy


text_dir = "text"
json_dir = "json"

os.makedirs(text_dir, exist_ok=True)
os.makedirs(json_dir, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

log_file_path = os.path.join(text_dir, f"experiment_log_{timestamp}.txt")
json_file_path = os.path.join(json_dir, f"experiment_report_{timestamp}.json")

@contextmanager
def tee_stdout(path: str):
    original = sys.stdout
    with open(path, "w") as f:
        class Tee:
            def write(self, s):
                original.write(s)
                f.write(s)
            def flush(self):
                original.flush()
                f.flush()
        sys.stdout = Tee()
        try:
            yield
        finally:
            sys.stdout = original


def run_experiments(file_ts):
    results = []
    base_plans = 30
    base_k = 5

    step_idx = 0
    for n_obj in range(50, 201, 50):
        step_idx += 1
        n_plans = base_plans * step_idx
        n_comp = 2 * n_plans
        k = base_k + (step_idx - 1)

        print("\n" + "#" * 70)
        print(f"Experiment {step_idx}: n_obj={n_obj}, n_plans={n_plans}, n_comp={n_comp}, k={k}")
        print("#" * 70)

        # construction time
        t_construct_start = time.perf_counter()
        case_study = DynamicRandomCaseStudy(
            num_objectives=n_obj,
            num_plans=n_plans,
            num_comparisons=n_comp,
            k=k,
            seed=42 + step_idx,
        )
        t_construct_end = time.perf_counter()

        case_study.print_summary()
        instance = case_study.get_instance()

        # solve time
        t_solve_start = time.perf_counter()
        report = solve_with_optimal_k(instance, verbose=True)
        t_solve_end = time.perf_counter()

        construction_time = t_construct_end - t_construct_start
        sat_solver_time = t_solve_end - t_solve_start

        optimal = report["optimal"]
        selected_omega = None if optimal is None else optimal["solution"]
        optimal_k = None if optimal is None else optimal["k"]

        results.append({
            "experiment": step_idx,
            "num_objectives": n_obj,
            "num_plans": n_plans,
            "num_comparisons": n_comp,
            "k_input": k,
            "construction_time_sec": construction_time,
            "sat_solver_time_sec": sat_solver_time,
            "optimal_found": optimal is not None,
            "optimal_k": optimal_k,
            "selected_omega": selected_omega,
        })

    with open(json_file_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved JSON: dynamic_random_experiments_{time.time()}.json")



def main():
    t_total_start = time.perf_counter()
    print("=" * 60)
    print("Running MOPP-DEC SAT Solver ")
    print("=" * 60)

    # construction time
    t_construct_start = time.perf_counter()
    # instance = CarPreferenceExample().get_instance()
    # instance = AutonomousDeliveryVehicleCaseStudy().get_instance()
    # instance = AutonomousVehicle10ObjectiveCaseStudy().get_instance()

    # below gives No
    # case_study = DynamicRandomCaseStudy(
    #     num_objectives=10,
    #     num_plans=10,
    #     num_comparisons=15,
    #     k=3,
    #     seed=42
    # )

    #below gives YES
    # case_study = DynamicRandomCaseStudy(
    #     num_objectives=80,
    #     num_plans=50,
    #     num_comparisons=100,
    #     k=10,
    #     seed=42
    # )

    case_study = DynamicRandomCaseStudy(
        num_objectives=150,
        num_plans=100,
        num_comparisons=200,
        k=10,
        seed=42
    )

    t_construct_end = time.perf_counter()
    case_study.print_summary()

    instance = case_study.get_instance()

    # solve time- with k
    # solver = MOPPDECSATSolver(instance)
    # t_solve_start = time.perf_counter()
    # solution = solver.solve()

    # solve without k
    t_solve_start = time.perf_counter()
    # report = solve_with_halving_k(instance, verbose=True)
    report = solve_with_optimal_k(instance, verbose=True)

    print("Optimal:", report["optimal"])

    t_solve_end = time.perf_counter()

    t_total_end = time.perf_counter()

    # timing
    construction_time = t_construct_end - t_construct_start
    sat_solver_time = t_solve_end - t_solve_start
    total_time = t_total_end - t_total_start

    print("\n========== Timing ==========")
    print(f"Construction time: {construction_time:.6f} sec")
    print(f"SAT solver time:   {sat_solver_time:.6f} sec")
    print(f"Total time:        {total_time:.6f} sec")
    print("============================\n")

    # if solution is None:
    #     print("NO: No objective subset satisfies the comparisons.")
    # else:
    #     print("YES: Found consistent objective subset Ω")
    #     print("Selected objectives:", solution)

    last_yes = report["halving"]["last_yes"]
    if last_yes is None:
        print("Final result: NO for all tried k values.")
    else:
        print("Best k found:", last_yes["k"])
        print("Selected objectives:", last_yes["solution"])


if __name__ == "__main__":
    # main()
    file_time_stamp = time.time()
    with tee_stdout(log_file_path):
        run_experiments(file_time_stamp)
