from Models.mopp_instance import MOPPInstance

class CarPreferenceExample:

    def __init__(self):
        self.objectives = [
            "cost",       # lower is better
            "comfort",    # higher is better
            "speed"       # higher is better
        ]

        self.plans = [
            "car_A",
            "car_B",
            "car_C"
        ]

        self.values = {
            "car_A": {"cost": 3, "comfort": 7, "speed": 6},
            "car_B": {"cost": 5, "comfort": 6, "speed": 8},
            "car_C": {"cost": 4, "comfort": 8, "speed": 5},
        }

        self.comparisons = [
            ("car_A", "car_B", "?"),   # incomparable
            ("car_C", "car_A", 1),     # car_C preferred over car_A
            ("car_B", "car_C", "?")    # incomparable
        ]

        self.k = 2

    def get_instance(self):
        return MOPPInstance(
            objectives=self.objectives,
            plans=self.plans,
            values=self.values,
            comparisons=self.comparisons,
            k=self.k
        )
