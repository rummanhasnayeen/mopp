from pysat.formula import CNF
from pysat.solvers import Solver

import time
import copy
import threading

TIME_LIMIT = 300  # 5 minutes

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

    def solve(self, time_limit_sec=None):
        """
        Solve the SAT instance
        """
        self.build_formula()

        with Solver(bootstrap_with=self.cnf) as solver:
            timed_out = False

            timer = None
            if time_limit_sec is not None:
                def _timeout():
                    nonlocal timed_out
                    timed_out = True
                    try:
                        solver.interrupt()
                    except Exception:
                        pass

                timer = threading.Timer(time_limit_sec, _timeout)
                timer.daemon = True
                timer.start()

            try:
                sat = solver.solve()
            finally:
                if timer is not None:
                    timer.cancel()

            if timed_out:
                return "TIMEOUT"

            if not sat:
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
    first_no = None

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
        solution = solver.solve(time_limit_sec=TIME_LIMIT)
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
            first_no = {  # <--- ADD THIS BLOCK
                "k": k,
                "solve_time_sec": solve_time,
                "iteration": it,
            }
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
        "first_no": first_no,
        "k_schedule": k_values,
    }


def solve_with_optimal_k(instance, *, verbose=True):
    """
    Find the *minimal* k that yields SAT (optimal k),
    using:
      1) halving schedule to find a YES/NO bracket
      2) binary search between first NO and last YES

    Stops when: (k_no + 1 == k_yes) i.e. consecutive NO then YES.
    Returns a dict with full logs and the optimal solution.
    """
    # 1) Get initial bracket from halving
    halving_res = solve_with_halving_k(instance, verbose=verbose)

    last_yes = halving_res["last_yes"]
    first_no = halving_res["first_no"]

    # If never SAT even at k = n, there is no solution at all
    if last_yes is None:
        if verbose:
            print("\nNo SAT solution found even with k = n.")
        return {
            "mode": "halving+binary",
            "halving": halving_res,
            "binary_iterations": [],
            "optimal": None,
        }

    # If we never hit NO in halving schedule, we only know it's SAT down to k=1 in that schedule.
    # In that case, just test k=1 and decide.
    if first_no is None:
        if verbose:
            print("\nHalving never produced NO. Refining by directly testing k=1.")
        inst1 = copy.copy(instance)
        inst1.k = 1
        t0 = time.perf_counter()
        sol1 = MOPPDECSATSolver(inst1).solve()
        t1 = time.perf_counter()

        if sol1 is not None:
            if verbose:
                print("YES at k=1 => optimal k = 1")
            return {
                "mode": "halving+binary",
                "halving": halving_res,
                "binary_iterations": [{
                    "k": 1, "is_sat": True, "solution": sol1, "solve_time_sec": (t1 - t0)
                }],
                "optimal": {"k": 1, "solution": sol1, "solve_time_sec": (t1 - t0)},
            }
        else:
            # SAT at last_yes.k but UNSAT at 1; bracket is [1, last_yes.k]
            if verbose:
                print("NO at k=1. Using bracket [k_no=1, k_yes=last_yes.k] for binary search.")
            k_no = 1
            k_yes = last_yes["k"]
            best_sol = last_yes["solution"]
            best_time = last_yes["solve_time_sec"]
            binary_log = []
    else:
        # Normal bracket from halving: first NO is smaller, last YES is larger
        k_no = first_no["k"]
        k_yes = last_yes["k"]
        best_sol = last_yes["solution"]
        best_time = last_yes["solve_time_sec"]
        binary_log = []

    if verbose:
        print("\n" + "=" * 60)
        print(f"[Binary Search Refinement] Bracket: NO at k={k_no}, YES at k={k_yes}")
        print("=" * 60)

    # 2) Binary search for minimal SAT k
    while (k_yes - k_no) > 1:
        mid = (k_yes + k_no) // 2

        inst_mid = copy.copy(instance)
        inst_mid.k = mid

        if verbose:
            print("\n" + "-" * 60)
            print(f"Trying mid k = {mid} (current bracket: NO={k_no}, YES={k_yes})")
            print("-" * 60)

        t0 = time.perf_counter()
        sol_mid = MOPPDECSATSolver(inst_mid).solve(time_limit_sec=TIME_LIMIT)
        t1 = time.perf_counter()
        solve_time = t1 - t0

        entry = {
            "k": mid,
            "is_sat": sol_mid is not None,
            "solution": sol_mid,
            "solve_time_sec": solve_time,
        }
        binary_log.append(entry)

        if verbose:
            print(f"SAT solver time: {solve_time:.6f} sec")
            if sol_mid is None:
                print("NO at k =", mid)
            else:
                print("YES at k =", mid)
                print("Selected objectives:", sol_mid)

        if sol_mid is not None:
            # SAT: move YES boundary down
            k_yes = mid
            best_sol = sol_mid
            best_time = solve_time
        else:
            # UNSAT: move NO boundary up
            k_no = mid

    # Now they are consecutive: NO at k_no and YES at k_yes
    optimal = {"k": k_yes, "solution": best_sol, "solve_time_sec": best_time}

    if verbose:
        print("\n" + "=" * 60)
        print("[Optimal k Found]")
        print(f"NO at k = {k_no}")
        print(f"YES at k = {k_yes}  <-- optimal (minimal SAT k)")
        print("Optimal selected objectives:", best_sol)
        print("=" * 60)

    return {
        "mode": "halving+binary",
        "halving": halving_res,
        "binary_iterations": binary_log,
        "optimal": optimal,
    }
