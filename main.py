import time

from Solvers.moppdec_sat_solver import MOPPDECSATSolver
from CaseStudies.car_example import CarPreferenceExample
from CaseStudies.autonomous_delivery_vehicle import AutonomousDeliveryVehicleCaseStudy
from CaseStudies.AutonomousVehicle10ObjectiveCaseStudy import AutonomousVehicle10ObjectiveCaseStudy
from CaseStudies.DynamicRandomCaseStudy import DynamicRandomCaseStudy

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
    case_study = DynamicRandomCaseStudy(
        num_objectives=30,
        num_plans=50,
        num_comparisons=35,
        k=10,
        seed=42
    )

    t_construct_end = time.perf_counter()
    case_study.print_summary()

    instance = case_study.get_instance()

    # solve time
    solver = MOPPDECSATSolver(instance)
    t_solve_start = time.perf_counter()
    solution = solver.solve()

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

    if solution is None:
        print("NO: No objective subset satisfies the comparisons.")
    else:
        print("YES: Found consistent objective subset Ω")
        print("Selected objectives:", solution)


if __name__ == "__main__":
    main()
