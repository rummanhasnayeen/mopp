class MOPPInstance:
    def __init__(self, objectives, plans, values, comparisons, k):
        """
        objectives: list[str]
        plans: list[str]
        values: dict[plan][objective] -> numeric value
        comparisons: list of tuples (pi, pi_prime, r) where r in {1,0,'?'}
        k: int
        """
        self.objectives = objectives
        self.plans = plans
        self.values = values
        self.comparisons = comparisons
        self.k = k

    def B(self, pi, pj):
        """Indices of objectives where pi is strictly better than pj"""
        return [
            i for i, o in enumerate(self.objectives)
            if self.values[pi][o] > self.values[pj][o]
        ]

    def W(self, pi, pj):
        """Indices of objectives where pi is strictly worse than pj"""
        return [
            i for i, o in enumerate(self.objectives)
            if self.values[pi][o] < self.values[pj][o]
        ]
