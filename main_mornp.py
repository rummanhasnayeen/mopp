from Models.mornp_instance import MORNPInstance
from Solvers.mornpdec_sat_solver import MORNPDECSATSolver


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


if __name__ == "__main__":
    inst1 = build_small_example()
    run_instance(inst1, "Example 1: Small example")

    inst2 = build_small_example_equality_unsat()
    run_instance(inst2, "Example 2: equality UNSAT")