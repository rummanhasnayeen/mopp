from Models.mopp_instance import MOPPInstance
from typing import Dict, List, Tuple


class AutonomousDeliveryVehicleCaseStudy:

    def __init__(self):

        self.objectives: List[str] = [
            "travel_time", # Faster is better
            "energy_efficiency", # Lower is better
            "safety", # Higher is better
            "traffic_compliance", # Higher is better
            "passenger_comfort", # Higher is better
        ]

        self.plans: List[str] = [
            "pi1",
            "pi2",
            "pi3",
            "pi4",
            "pi5",
        ]

        self.values: Dict[str, Dict[str, int]] = {
            "pi1": {
                "travel_time": 9,
                "energy_efficiency": 7,
                "safety": 6,
                "traffic_compliance": 8,
                "passenger_comfort": 5,
            },
            "pi2": {
                "travel_time": 7,
                "energy_efficiency": 9,
                "safety": 7,
                "traffic_compliance": 9,
                "passenger_comfort": 6,
            },
            "pi3": {
                "travel_time": 6,
                "energy_efficiency": 6,
                "safety": 9,
                "traffic_compliance": 9,
                "passenger_comfort": 8,
            },
            "pi4": {
                "travel_time": 8,
                "energy_efficiency": 8,
                "safety": 6,
                "traffic_compliance": 7,
                "passenger_comfort": 7,
            },
            "pi5": {
                "travel_time": 5,
                "energy_efficiency": 5,
                "safety": 8,
                "traffic_compliance": 6,
                "passenger_comfort": 9,
            },
        }

        self.comparisons: List[Tuple[str, str, object]] = [
            # ("pi1", "pi2", "?"),  # Trade-off between speed and efficiency
            ("pi3", "pi1", 1),    # Safety-focused dominates fast plan
            ("pi4", "pi2", "?"),  # Balanced vs efficiency
            ("pi5", "pi3", "?"),  # Comfort vs safety
            ("pi2", "pi1", 1),    # Efficiency dominates speed
        ]

        self.k: int = 2

    def get_objectives(self) -> List[str]:
        """Return the list of objective names."""
        return self.objectives

    def get_plans(self) -> Dict[str, List[int]]:
        """Return the plan-to-objective-value mapping."""
        return self.plans

    def get_comparisons(self) -> List[Tuple[str, str, object]]:
        """Return the plan comparison sample C."""
        return self.comparisons

    def get_k(self) -> int:
        """Return the cardinality bound k."""
        return self.k

    def num_objectives(self) -> int:
        """Return the number of objectives."""
        return len(self.objectives)

    def num_plans(self) -> int:
        """Return the number of plans."""
        return len(self.plans)

    def describe(self):
        """Print a human-readable description of the case study."""
        print("\n=== Autonomous Delivery Vehicle Case Study ===")
        print(f"Objectives ({len(self.objectives)}):")
        for i, obj in enumerate(self.objectives, start=1):
            print(f"  o{i}: {obj}")

        print("\nPlans:")
        for p, values in self.plans.items():
            print(f"  {p}: {values}")

        print("\nPlan comparisons:")
        for (pi, pj, r) in self.comparisons:
            print(f"  ({pi}, {pj}, {r})")

        print(f"\nCardinality bound k = {self.k}")
        print("=============================================\n")


    def get_instance(self):
        return MOPPInstance(
            objectives=self.objectives,
            plans=self.plans,
            values=self.values,
            comparisons=self.comparisons,
            k=self.k
        )
