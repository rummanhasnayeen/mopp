from Models.mopp_instance import MOPPInstance


class CarPreferenceExample:
    """
    Car preference planning case study.

    This class encodes a small multi-objective planning problem
    used to test the MOPP-DEC SAT formulation.
    """

    def __init__(self):
        """
        Initialize the car preference example.
        """

        # -----------------------------
        # Objectives
        # -----------------------------
        self.objectives = [
            "cost",       # lower is better
            "comfort",    # higher is better
            "speed"       # higher is better
        ]

        # -----------------------------
        # Plans
        # -----------------------------
        self.plans = [
            "car_A",
            "car_B",
            "car_C"
        ]

        # -----------------------------
        # Objective values
        # Each plan maps objective -> value
        # -----------------------------
        self.values = {
            "car_A": {"cost": 3, "comfort": 7, "speed": 6},
            "car_B": {"cost": 5, "comfort": 6, "speed": 8},
            "car_C": {"cost": 4, "comfort": 8, "speed": 5},
        }

        # -----------------------------
        # Plan comparison sample C
        # (pi, pi', r) where r ∈ {1, 0, '?'}
        # -----------------------------
        self.comparisons = [
            ("car_A", "car_B", "?"),   # incomparable
            ("car_C", "car_A", 1),     # car_C preferred over car_A
            ("car_B", "car_C", "?")    # incomparable
        ]

        # -----------------------------
        # Cardinality bound |Ω| ≤ k
        # -----------------------------
        self.k = 2

    def get_instance(self):
        """
        Build and return a MOPPInstance corresponding to this case study.
        """
        return MOPPInstance(
            objectives=self.objectives,
            plans=self.plans,
            values=self.values,
            comparisons=self.comparisons,
            k=self.k
        )
