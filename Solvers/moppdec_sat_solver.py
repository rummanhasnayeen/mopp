from pysat.formula import CNF
from pysat.solvers import Solver

import time
import copy


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

    # Cardinality constraint
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

    # Build phi_MOPPDEC
    def build_formula(self):
        """
        MOPPDEC = phi_C ∧ phi_{<=k}
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
        Solve the SAT instance
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


def solve_with_halving_k(instance, *, verbose=True):
    """
    Iteratively solve MOPPDEC by halving k each iteration:
      k0 = n, k1 = floor(n/2), k2 = floor(k1/2), ...
    Stop when the solver first returns NO after a YES.
    Return a dict with the full iteration log and the last YES solution.

    NOTE: We rebuild the solver each iteration because cardinality vars t_{i,j}
    depend on k (your solver creates them in __init__).
    """

    n = len(instance.objectives)
    if n == 0:
        raise ValueError("Instance has zero objectives.")

    # Halving schedule: n, n//2, n//4, ...
    k_values = []
    k = n
    while k >= 1:
        k_values.append(k)
        next_k = k // 2
        if next_k == k:  # safety (shouldn't happen for ints)
            break
        k = next_k

    iteration_log = []
    last_yes = None  # {"k": ..., "solution": [...], "solve_time": ...}

    for it, k in enumerate(k_values, start=1):
        inst_k = copy.copy(instance)
        inst_k.k = k

        if verbose:
            print("\n" + "=" * 60)
            print(f"[Halving-k Iteration {it}] Trying k = {k} (n = {n})")
            print("=" * 60)

        # Build + solve timing (solver.build_formula is called inside solve())
        t_solve_start = time.perf_counter()
        solver = MOPPDECSATSolver(inst_k)
        solution = solver.solve()
        t_solve_end = time.perf_counter()

        solve_time = t_solve_end - t_solve_start

        result = {
            "iteration": it,
            "k": k,
            "is_sat": solution is not None,
            "solution": solution,
            "solve_time_sec": solve_time,
        }
        iteration_log.append(result)

        if verbose:
            print(f"SAT solver time: {solve_time:.6f} sec")
            if solution is None:
                print("NO: No objective subset satisfies the comparisons.")
            else:
                print("YES: Found consistent objective subset Ω")
                print("Selected objectives:", solution)

        # Stop condition: first NO after having seen a YES
        if solution is None and last_yes is not None:
            if verbose:
                print("\n--- Transition detected: YES -> NO ---")
                print(f"Sub-optimal (last YES) k = {last_yes['k']}")
                print("Sub-optimal selected objectives:", last_yes["solution"])
            break

        # Update last YES if sat
        if solution is not None:
            last_yes = {
                "k": k,
                "solution": solution,
                "solve_time_sec": solve_time,
                "iteration": it,
            }

    return {
        "n": n,
        "iterations": iteration_log,
        "last_yes": last_yes,  # None if never SAT
        "k_schedule": k_values,
    }
