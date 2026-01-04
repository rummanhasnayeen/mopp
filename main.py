from Solvers.moppdec_sat_solver import MOPPDECSATSolver
from CaseStudies.car_example import CarPreferenceExample
from CaseStudies.autonomous_delivery_vehicle import AutonomousDeliveryVehicleCaseStudy

def main():
    print("=" * 60)
    print("Running MOPP-DEC SAT Solver ")
    print("=" * 60)

    # instance = CarPreferenceExample().get_instance()
    instance = AutonomousDeliveryVehicleCaseStudy().get_instance()

    solver = MOPPDECSATSolver(instance)
    solution = solver.solve()

    if solution is None:
        print("NO: No objective subset satisfies the comparisons.")
    else:
        print("YES: Found consistent objective subset Ω")
        print("Selected objectives:", solution)


if __name__ == "__main__":
    main()
