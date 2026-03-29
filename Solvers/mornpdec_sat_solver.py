from typing import Dict, List, Optional, Tuple

# from pysat.card import CardEnc
from pysat.formula import CNF, IDPool
from pysat.solvers import Minisat22

from Models.mornp_instance import MORNPInstance


class MORNPDECSATSolver:
    def __init__(self, instance: MORNPInstance):
        self.instance = instance
        self.instance.validate()

        self.cnf = CNF()
        self.vpool = IDPool()

        self.objectives = instance.objectives
        self.positive_plans = instance.positive_plans
        self.sample_plans = instance.all_sample_plans
        self.k = instance.k

        # variable ids
        self.x_vars: Dict[str, int] = {}  # x_i
        self.r_vars: Dict[Tuple[str, str, str], int] = {}  # r_(pi,pi',obj)
        self.t_vars: Dict[Tuple[int, int], int] = {}  # t_(i,j) for cardinality encoding

        self._init_x_vars()

    def _x(self, obj: str) -> int:
        return self.x_vars[obj]

    def _r(self, pi: str, pi_prime: str, obj: str) -> int:
        key = (pi, pi_prime, obj)
        if key not in self.r_vars:
            self.r_vars[key] = self.vpool.id(f"r::{pi}::{pi_prime}::{obj}")
        return self.r_vars[key]

    def _t(self, i: int, j: int) -> int:
        key = (i, j)
        if key not in self.t_vars:
            self.t_vars[key] = self.vpool.id(f"t::{i}::{j}")
        return self.t_vars[key]

    def _init_x_vars(self) -> None:
        for obj in self.objectives:
            self.x_vars[obj] = self.vpool.id(f"x::{obj}")

    def g_value(self, pi: str, pi_prime: str, obj: str) -> int:
        # g_(pi,pi',obj) = 1 iff value(pi,obj) > value(pi_prime,obj)
        v_pi = self.instance.plan_values[pi][obj]
        v_pi_prime = self.instance.plan_values[pi_prime][obj]
        return 1 if v_pi > v_pi_prime else 0

    def build_formula(self) -> None:
        #  Phi_S AND Phi_<=k
        self._add_pairwise_mornp_constraints()
        self._add_cardinality_constraint()

    def _add_pairwise_mornp_constraints(self) -> None:
        # Phi(pi, pi'
        for pi in self.positive_plans:
            for pi_prime in self.sample_plans:
                if pi == pi_prime:
                    continue
                r_clause: List[int] = []

                for obj in self.objectives:
                    x_i = self._x(obj)
                    r_i = self._r(pi, pi_prime, obj)
                    g_i = self.g_value(pi, pi_prime, obj)

                    # r_i participates in the big OR
                    r_clause.append(r_i)

                    # (!r_i ∨ x_i)
                    self.cnf.append([-r_i, x_i])

                    if g_i == 1:
                        # (!x_i ∨ !g_i ∨ r_i) becomes (!x_i ∨ r_i)
                        self.cnf.append([-x_i, r_i])
                    else:
                        # (!r_i ∨ g_i) becomes (!r_i)
                        self.cnf.append([-r_i])

                self.cnf.append(r_clause)

    def _add_cardinality_constraint(self) -> None:
        # t_(i,j)
        n = len(self.objectives)
        k = self.k

        # k >= n
        if k >= n:
            return

        def x_at(i: int) -> int:
            obj = self.objectives[i - 1]
            return self._x(obj)

        #   (!x_i ∨ t_{i,1})
        for i in range(1, n + 1):
            self.cnf.append([-x_at(i), self._t(i, 1)])

        #   (!t_{i-1,j} ∨ t_{i,j})
        for i in range(2, n + 1):
            for j in range(1, k + 1):
                self.cnf.append([-self._t(i - 1, j), self._t(i, j)])

        #   (!x_i ∨ !t_{i-1,j-1} ∨ t_{i,j})
        for i in range(2, n + 1):
            for j in range(2, k + 1):
                self.cnf.append([-x_at(i), -self._t(i - 1, j - 1), self._t(i, j)])

        #   (!x_i ∨ !t_{i-1,k})
        for i in range(2, n + 1):
            self.cnf.append([-x_at(i), -self._t(i - 1, k)])

    def solve(self) -> Dict:
        with Minisat22(bootstrap_with=self.cnf.clauses) as solver:  # Minisat because its lightweight(gemini)
            sat = solver.solve()
            if not sat:
                return {
                    "sat": False,
                    "selected_objectives": [],
                    "model": None,
                }

            model = solver.get_model()
            model_set = set(model)

            selected = [
                obj for obj in self.objectives
                if self._x(obj) in model_set
            ]

            # DEBUG PRINT
            self.print_cardinality_debug(model)

            return {
                "sat": True,
                "selected_objectives": selected,
                "model": model,
            }

    def explain_pair_constants(self, pi: str, pi_prime: str) -> Dict[str, int]:
        # return g_(pi,pi',i)

        return {
            obj: self.g_value(pi, pi_prime, obj)
            for obj in self.objectives
        }

    def print_pairwise_debug(self) -> None:
        print("=== MORNP Pairwise g(pi, pi', i) constants ===")
        for pi in self.positive_plans:
            for pi_prime in self.sample_plans:
                g_map = self.explain_pair_constants(pi, pi_prime)
                print(f"\nPair (pi={pi}, pi'={pi_prime})")
                for obj in self.objectives:
                    print(f"  g[{pi},{pi_prime},{obj}] = {g_map[obj]}")

    def print_cardinality_debug(self, model: List[int]) -> None:
        # t_(i,j) vars
        print("\n=== Cardinality t(i,j) variable values ===")

        model_set = set(model)

        n = len(self.objectives)
        k = self.k

        if k >= n:
            print("Cardinality constraint inactive (k >= n)")
            return

        for i in range(1, n + 1):
            row = []
            for j in range(1, k + 1):
                t_var = self._t(i, j)
                val = 1 if t_var in model_set else 0
                row.append(f"t({i},{j})={val}")
            print("  " + ", ".join(row))

    def _dominates_under_objectives(
        self,
        plan_a: str,
        plan_b: str,
        selected_objectives: List[str],
    ) -> bool:
        """
        Return True iff plan_a Pareto-dominates plan_b
        under the selected objectives.
        """
        if not selected_objectives:
            return False

        values = self.instance.plan_values

        all_ge = all(
            values[plan_a][obj] >= values[plan_b][obj]
            for obj in selected_objectives
        )
        one_gt = any(
            values[plan_a][obj] > values[plan_b][obj]
            for obj in selected_objectives
        )
        return all_ge and one_gt

    def verify_certificate(self, selected_objectives: List[str]) -> Dict:
        """
        Verify the MORNP-DEC certificate Ω in polynomial time.

        Checks:
          1. |Ω| <= k
          2. For every pi in P+, no pi' in P+ U P- dominates pi under Ω
        """
        unknown = [obj for obj in selected_objectives if obj not in self.objectives]
        if unknown:
            return {
                "verified": False,
                "cardinality_ok": False,
                "nondominance_ok": False,
                "reason": f"Unknown objectives in certificate: {unknown}",
                "violations": [],
                "time_complexity": "O(1) early reject",
            }

        if len(selected_objectives) > self.k:
            return {
                "verified": False,
                "cardinality_ok": False,
                "nondominance_ok": False,
                "reason": f"Certificate size {len(selected_objectives)} exceeds k={self.k}",
                "violations": [],
                "time_complexity": "O(1)",
            }

        violations = []

        for pi in self.positive_plans:
            for pi_prime in self.sample_plans:
                if pi == pi_prime:
                    continue

                if self._dominates_under_objectives(pi_prime, pi, selected_objectives):
                    violations.append({
                        "positive_plan": pi,
                        "dominating_plan": pi_prime,
                    })

        nondominance_ok = len(violations) == 0

        return {
            "verified": nondominance_ok,
            "cardinality_ok": True,
            "nondominance_ok": nondominance_ok,
            "reason": None if nondominance_ok else "Some positive plan is dominated under the returned certificate.",
            "violations": violations,
            "time_complexity": "O(|P+| * |P+ U P-| * |O|)",
        }

    def solve_and_verify(self) -> Dict:
        """
        Solve first, then verify the returned certificate.
        Returns everything solve() returns, plus 'verification'.
        """
        result = self.solve()

        if not result["sat"]:
            result["verification"] = None
            return result

        result["verification"] = self.verify_certificate(
            result["selected_objectives"]
        )
        return result
