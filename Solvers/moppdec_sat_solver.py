from pysat.formula import CNF
from pysat.solvers import Solver


class MOPPDECSATSolver:
    """
    SAT solver
    x_i  : objective-selection
    t_i_j: cardinality
    """

    def __init__(self, instance):
        """
        - instance.objectives : list of objective names
        - instance.plans      : list of plan identifiers
        - instance.values     : dict plan -> dict objective -> value
        - instance.comparisons: list of (pi, pi', r)
        - instance.k          : cardinality bound
        """
        self.instance = instance
        self.objectives = instance.objectives
        self.plans = instance.plans
        self.values = instance.values
        self.comparisons = instance.comparisons
        self.k = instance.k

        self.n = len(self.objectives)

        self.cnf = CNF()
        self.var_id = 1

        # Variable maps
        self.x = {}       # x_i
        self.t = {}       # t_{i,j}

        self._create_objective_variables()
        self._create_cardinality_variables()

    # Variable creation
    def _new_var(self):
        v = self.var_id
        self.var_id += 1
        return v

    def _create_objective_variables(self):
        """
        Create Boolean variables x_i for objectives.
        """
        for i in range(1, self.n + 1):
            self.x[i] = self._new_var()

    def _create_cardinality_variables(self):
        """
        Create sequential-counter variables t_{i,j}.
        """
        for i in range(1, self.n + 1):
            for j in range(1, self.k + 1):
                self.t[(i, j)] = self._new_var()

    def _compute_B_W(self, pi, pi_p):
        """
        B_c and W_c
        """
        B = set()
        W = set()

        for idx, o in enumerate(self.objectives, start=1):
            v1 = self.values[pi][o]
            v2 = self.values[pi_p][o]
            if v1 > v2:
                B.add(idx)
            elif v1 < v2:
                W.add(idx)

        return B, W

    # Constraint r = 1
    def _add_r1_constraints(self, B, W):
        # ∧_{i ∈ W} ¬x_i
        for i in W:
            self.cnf.append([-self.x[i]])

        # ∨_{i ∈ B} x_i
        if not B:
            # impossible to satisfy dominance
            self.cnf.append([])  # empty clause → UNSAT
        else:
            self.cnf.append([self.x[i] for i in B])

    # Constraints for r = 0
    def _add_r0_constraints(self, B, W):
        # ∧_{i ∈ B ∪ W} ¬x_i
        for i in B.union(W):
            self.cnf.append([-self.x[i]])

    # Constraints for r = ?
    def _add_rq_constraints(self, pi, pi_p, B, W):
        # Prevent π ≻ π'
        for j in B:
            clause = [-self.x[j]]
            if W:
                clause += [self.x[i] for i in W]
            self.cnf.append(clause)

        # Prevent π' ≻ π
        Bp, Wp = self._compute_B_W(pi_p, pi)
        for j in Bp:
            clause = [-self.x[j]]
            if Wp:
                clause += [self.x[i] for i in Wp]
            self.cnf.append(clause)

    # Cardinality constraint (sequential counter)
    def _add_cardinality_constraints(self):
        n = self.n
        k = self.k

        # ¬x_i ∨ t_{i,1}
        for i in range(1, n + 1):
            self.cnf.append([-self.x[i], self.t[(i, 1)]])

        # ¬t_{i-1,j} ∨ t_{i,j}
        for i in range(2, n + 1):
            for j in range(1, k + 1):
                self.cnf.append([-self.t[(i - 1, j)], self.t[(i, j)]])

        # ¬x_i ∨ ¬t_{i-1,j-1} ∨ t_{i,j}
        for i in range(2, n + 1):
            for j in range(2, k + 1):
                self.cnf.append([
                    -self.x[i],
                    -self.t[(i - 1, j - 1)],
                    self.t[(i, j)]
                ])

        # ¬x_i ∨ ¬t_{i-1,k}
        for i in range(2, n + 1):
            self.cnf.append([-self.x[i], -self.t[(i - 1, k)]])

    # --------------------------------------------------
    # Build Φ_MOPPDEC
    # --------------------------------------------------

    def build_formula(self):
        """
        MOPPDEC = phi_C ∧ phi_{≤k}
        """
        for (pi, pi_p, r) in self.comparisons:
            B, W = self._compute_B_W(pi, pi_p)

            if r == 1:
                self._add_r1_constraints(B, W)
            elif r == 0:
                self._add_r0_constraints(B, W)
            elif r == "?":
                self._add_rq_constraints(pi, pi_p, B, W)
            else:
                raise ValueError(f"Unknown comparison label: {r}")

        self._add_cardinality_constraints()

    # --------------------------------------------------
    # Solve
    # --------------------------------------------------

    def solve(self):
        """
        Solve the SAT instance.

        Returns:
            list of selected objective names if SAT,
            None otherwise.
        """
        self.build_formula()

        with Solver(bootstrap_with=self.cnf) as solver:
            if not solver.solve():
                return None

            model = solver.get_model()

        selected = []
        for i, obj in enumerate(self.objectives, start=1):
            if self.x[i] in model:
                selected.append(obj)

        return selected
