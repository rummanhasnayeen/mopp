from Models.mopp_instance import MOPPInstance


class AutonomousVehicle10ObjectiveCaseStudy:
    """
    10-objective autonomous vehicle preference case study.

    Design goal:
      - Ensure there exists a subset Ω of size <= k that satisfies all comparisons (YES-instance).
      - We make Ω = {safety, energy_efficiency, time_efficiency} work, and set k = 3.

    Notes:
      - This case study assumes "higher is better" for all objective values,
        matching the solver's use of > and < when building B_c and W_c.
    """

    def __init__(self):
        # 10 objectives (all treated as "higher is better" in the solver)
        self.objectives = [
            "safety",
            "energy_efficiency",
            "time_efficiency",
            "comfort",
            "cost_efficiency",
            "legal_compliance",
            "robustness",
            "emissions",
            "maintenance_ease",
            "payload_capacity",
        ]

        # 5 plans (routes / driving policies)
        self.plans = [
            "route_fast",
            "route_safe",
            "route_green",
            "route_balanced",
            "route_budget",
        ]

        # Values: plan -> objective -> value
        # Chosen so that Ω = {safety, energy_efficiency, time_efficiency} can satisfy all comparisons.
        self.values = {
            "route_fast": {
                "safety": 5,
                "energy_efficiency": 4,
                "time_efficiency": 9,
                "comfort": 6,
                "cost_efficiency": 5,
                "legal_compliance": 8,
                "robustness": 6,
                "emissions": 4,
                "maintenance_ease": 6,
                "payload_capacity": 7,
            },
            "route_safe": {
                "safety": 9,
                "energy_efficiency": 5,
                "time_efficiency": 5,
                "comfort": 7,
                "cost_efficiency": 6,
                "legal_compliance": 9,
                "robustness": 8,
                "emissions": 6,
                "maintenance_ease": 7,
                "payload_capacity": 6,
            },
            "route_green": {
                "safety": 7,
                "energy_efficiency": 9,
                "time_efficiency": 4,
                "comfort": 6,
                "cost_efficiency": 7,
                "legal_compliance": 8,
                "robustness": 7,
                "emissions": 9,
                "maintenance_ease": 6,
                "payload_capacity": 5,
            },
            "route_balanced": {
                "safety": 8,
                "energy_efficiency": 8,
                "time_efficiency": 8,
                "comfort": 8,
                "cost_efficiency": 7,
                "legal_compliance": 9,
                "robustness": 8,
                "emissions": 7,
                "maintenance_ease": 8,
                "payload_capacity": 7,
            },
            "route_budget": {
                "safety": 6,
                "energy_efficiency": 6,
                "time_efficiency": 6,
                "comfort": 5,
                "cost_efficiency": 9,
                "legal_compliance": 7,
                "robustness": 6,
                "emissions": 6,
                "maintenance_ease": 7,
                "payload_capacity": 6,
            },
        }

        # 10 comparisons (r in {1, 0, "?"})
        # We ensure all "?" comparisons are truly incomparable under Ω by making each one have:
        #   at least one objective in Ω where left is better, and at least one objective in Ω where left is worse.
        #
        # And we include one r=1 comparison that is satisfied by Ω:
        #   route_balanced dominates route_budget on Ω (8>=6, 8>=6, 8>=6 and strictly >).
        # self.comparisons = [
        #     ("route_balanced", "route_budget", 1),   # dominance under Ω
        #
        #     ("route_safe", "route_fast", "?"),       # safe better safety, worse time
        #     ("route_green", "route_safe", "?"),      # green better energy, worse safety/time
        #     ("route_fast", "route_green", "?"),      # fast better time, worse safety/energy
        #
        #     ("route_balanced", "route_safe", "?"),   # balanced better energy/time, worse safety
        #     ("route_balanced", "route_green", "?"),  # balanced better safety/time, worse energy
        #
        #     ("route_balanced", "route_fast", "?"),   # balanced better safety/energy, worse time
        #     ("route_budget", "route_fast", "?"),     # budget better safety/energy, worse time
        #
        #     ("route_budget", "route_green", "?"),    # budget better time, worse safety/energy
        #     ("route_safe", "route_budget", "?"),     # safe better safety, worse energy/time
        # ]

        self.comparisons = [
            # r = 0
            ("route_safe", "route_budget", 0),

            # r = 1
            ("route_balanced", "route_fast", 1),
            ("route_balanced", "route_budget", 1),
            ("route_balanced", "route_safe", 1),

            # r = ?
            ("route_safe", "route_fast", "?"),
            ("route_green", "route_safe", "?"),
            ("route_fast", "route_green", "?"),
            ("route_balanced", "route_green", "?"),
            ("route_budget", "route_fast", "?"),
            ("route_green", "route_budget", "?"),
        ]

        self.k = 3

    def get_instance(self) -> MOPPInstance:
        return MOPPInstance(
            objectives=self.objectives,
            plans=self.plans,
            values=self.values,
            comparisons=self.comparisons,
            k=self.k,
        )
