from typing import Dict, List, Tuple, Optional
import time

import gurobipy as gp
from gurobipy import GRB

from Models.mpcwpc_instance import MPCwPCInstance


class MPCwPCILPSolver:
    def __init__(self, instance: MPCwPCInstance, use_big_m: bool = True,
                 big_m: float = 1e4, epsilon: float = 1e-3,
                 time_limit: float = 300.0, verbose: bool = False):
        self.instance = instance
        self.instance.validate()

        self.use_big_m = use_big_m
        self.M = big_m
        self.epsilon = epsilon
        self.time_limit = time_limit
        self.verbose = verbose

        self.objectives = instance.objectives
        self.plan_values = instance.plan_values
        self.comparisons = instance.comparisons
        self.n = len(self.objectives)

        self.all_plans: List[str] = []
        for pi, pi_p, _ in self.comparisons:
            if pi not in self.all_plans:
                self.all_plans.append(pi)
            if pi_p not in self.all_plans:
                self.all_plans.append(pi_p)

        self.model: Optional[gp.Model] = None
        self.x_vars: Dict[Tuple[str, str], gp.Var] = {}
        self.y_vars: Dict[Tuple[str, str], gp.Var] = {}
        self.z_vars: Dict[Tuple[str, str, str], gp.Var] = {}
        self.g_vars: Dict[Tuple[str, str, str], gp.Var] = {}

        self.timestamps: Dict[str, float] = {}
        self.intervals: Dict[str, float] = {}

    def _record_ts(self, label: str):
        self.timestamps[label] = time.perf_counter()

    def _record_interval(self, label: str, start_label: str, end_label: str):
        self.intervals[label] = self.timestamps[end_label] - self.timestamps[start_label]

    def build_model(self):
        self._record_ts("build_start")

        try:
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 1 if self.verbose else 0)
            env.start()
        except gp.GurobiError as e:
            raise RuntimeError(
                f"Gurobi license error: {e}. "
                "Please renew Gurobi license at https://www.gurobi.com/downloads/end-user-license-agreement-academic/"
            ) from e
        self.model = gp.Model("MPCwPC", env=env)
        self.model.setParam("TimeLimit", self.time_limit)

        self._create_variables()
        self._record_ts("variables_done")
        self._record_interval("variable_creation", "build_start", "variables_done")

        self._add_constraints()
        self._record_ts("constraints_done")
        self._record_interval("constraint_addition", "variables_done", "constraints_done")

        self._set_objective()
        self._record_ts("objective_done")

        self.model.update()
        self._record_ts("build_end")
        self._record_interval("total_build", "build_start", "build_end")

    def _create_variables(self):
        objs = self.objectives

        # x[o, o']
        for o in objs:
            for o_p in objs:
                if o != o_p:
                    self.x_vars[o, o_p] = self.model.addVar(
                        vtype=GRB.BINARY, name=f"x_{o}_{o_p}")

        # y[o, pi]
        for o in objs:
            for pi in self.all_plans:
                self.y_vars[o, pi] = self.model.addVar(
                    vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name=f"y_{o}_{pi}")

        # z[o, o', pi] : x[o,o'] * y[o', pi]
        for o in objs:
            for o_p in objs:
                if o != o_p:
                    for pi in self.all_plans:
                        self.z_vars[o, o_p, pi] = self.model.addVar(
                            vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name=f"z_{o}_{o_p}_{pi}")

        # g[o', pi, pi']; 1 = y[o', pi] > y[o', pi']
        comparisons_needing_g = [(pi, pi_p) for pi, pi_p, r in self.comparisons if r in (1, '?')]
        for pi, pi_p in comparisons_needing_g:
            for o_p in objs:
                self.g_vars[o_p, pi, pi_p] = self.model.addVar(
                    vtype=GRB.BINARY, name=f"g_{o_p}_{pi}_{pi_p}")
        #  r='?' g reverse
        for pi, pi_p, r in self.comparisons:
            if r == '?':
                for o_p in objs:
                    if (o_p, pi_p, pi) not in self.g_vars:
                        self.g_vars[o_p, pi_p, pi] = self.model.addVar(
                            vtype=GRB.BINARY, name=f"g_{o_p}_{pi_p}_{pi}")

    def _add_constraints(self):
        self._add_transitivity_constraints()
        self._record_ts("transitivity_done")
        self._record_interval("transitivity_constraints", "variables_done", "transitivity_done")

        self._add_y_computation_constraints()
        self._record_ts("y_computation_done")
        self._record_interval("y_computation_constraints", "transitivity_done", "y_computation_done")

        self._add_comparison_constraints()
        self._record_ts("comparison_done")
        self._record_interval("comparison_constraints", "y_computation_done", "comparison_done")

    def _add_transitivity_constraints(self):
        objs = self.objectives
        for o in objs:
            for o_p in objs:
                if o_p == o:
                    continue
                for o_pp in objs:
                    if o_pp == o or o_pp == o_p:
                        continue
                    self.model.addConstr(
                        self.x_vars[o, o_pp] >= self.x_vars[o, o_p] + self.x_vars[o_p, o_pp] - 1,
                        name=f"trans_{o}_{o_p}_{o_pp}")

    def _add_y_computation_constraints(self):
        objs = self.objectives

        for o in objs:
            for pi in self.all_plans:
                base_val = self.plan_values[pi][o]

                for o_p in objs:
                    if o_p == o:
                        continue
                    z = self.z_vars[o, o_p, pi]
                    x = self.x_vars[o, o_p]
                    v_op_pi = self.plan_values[pi][o_p]

                    if self.use_big_m:
                        # z <= M * x
                        self.model.addConstr(z <= self.M * x,
                                             name=f"z_ub_Mx_{o}_{o_p}_{pi}")
                        # z >= -M * x
                        self.model.addConstr(z >= -self.M * x,
                                             name=f"z_lb_Mx_{o}_{o_p}_{pi}")
                        # z <= v + M*(1 - x)   => when x=1, z <= v
                        self.model.addConstr(z <= v_op_pi + self.M * (1 - x),
                                             name=f"z_ub_val_{o}_{o_p}_{pi}")
                        # z >= v - M*(1 - x)   => when x=1, z >= v
                        self.model.addConstr(z >= v_op_pi - self.M * (1 - x),
                                             name=f"z_lb_val_{o}_{o_p}_{pi}")
                    else:
                        # Indicator: x=1 => z = v_op_pi
                        self.model.addGenConstrIndicator(
                            x, True, z == v_op_pi,
                            name=f"z_ind_on_{o}_{o_p}_{pi}")
                        # Indicator: x=0 => z = 0
                        self.model.addGenConstrIndicator(
                            x, False, z == 0.0,
                            name=f"z_ind_off_{o}_{o_p}_{pi}")

                # y[o, pi] = V^pi_o + sum z[o, o', pi]
                self.model.addConstr(
                    self.y_vars[o, pi] == base_val + gp.quicksum(
                        self.z_vars[o, o_p, pi] for o_p in objs if o_p != o),
                    name=f"y_def_{o}_{pi}")

    def _add_comparison_constraints(self):
        for pi, pi_p, r in self.comparisons:
            if r == 0:
                self._add_indifference_constraints(pi, pi_p)
            elif r == 1:
                self._add_preference_constraints(pi, pi_p)
            elif r == '?':
                self._add_incomparability_constraints(pi, pi_p)

    def _add_indifference_constraints(self, pi: str, pi_p: str):
        for o in self.objectives:
            self.model.addConstr(
                self.y_vars[o, pi] == self.y_vars[o, pi_p],
                name=f"indf_{o}_{pi}_{pi_p}")

    def _add_preference_constraints(self, pi: str, pi_p: str):
        objs = self.objectives

        # all o: y[o, pi] >= y[o, pi']
        for o in objs:
            self.model.addConstr(
                self.y_vars[o, pi] >= self.y_vars[o, pi_p],
                name=f"pref_ge_{o}_{pi}_{pi_p}")

        # g[o', pi, pi'] inequality
        if self.use_big_m:
            for o_p in objs:
                g = self.g_vars[o_p, pi, pi_p]
                diff = self.y_vars[o_p, pi] - self.y_vars[o_p, pi_p]
                # diff <= M * g
                self.model.addConstr(diff <= self.M * g,
                                     name=f"pref_g_ub_{o_p}_{pi}_{pi_p}")
                # diff >= epsilon * g - M * (1 - g)
                self.model.addConstr(diff >= self.epsilon * g - self.M * (1 - g),
                                     name=f"pref_g_lb_{o_p}_{pi}_{pi_p}")
        else:
            for o_p in objs:
                g = self.g_vars[o_p, pi, pi_p]
                diff = self.y_vars[o_p, pi] - self.y_vars[o_p, pi_p]
                # g=1 => diff >= epsilon
                self.model.addGenConstrIndicator(
                    g, True, diff >= self.epsilon,
                    name=f"pref_g_ind_on_{o_p}_{pi}_{pi_p}")
                # g=0 => diff <= 0
                self.model.addGenConstrIndicator(
                    g, False, diff <= 0.0,
                    name=f"pref_g_ind_off_{o_p}_{pi}_{pi_p}")

        # sum g >= 1
        self.model.addConstr(
            gp.quicksum(self.g_vars[o_p, pi, pi_p] for o_p in objs) >= 1,
            name=f"pref_exists_{pi}_{pi_p}")

    def _add_incomparability_constraints(self, pi: str, pi_p: str):
        objs = self.objectives

        # g[o', pi, pi']
        if self.use_big_m:
            for o_p in objs:
                g = self.g_vars[o_p, pi, pi_p]
                diff = self.y_vars[o_p, pi] - self.y_vars[o_p, pi_p]
                self.model.addConstr(diff <= self.M * g,
                                     name=f"inc_fwd_g_ub_{o_p}_{pi}_{pi_p}")
                self.model.addConstr(diff >= self.epsilon * g - self.M * (1 - g),
                                     name=f"inc_fwd_g_lb_{o_p}_{pi}_{pi_p}")
        else:
            for o_p in objs:
                g = self.g_vars[o_p, pi, pi_p]
                diff = self.y_vars[o_p, pi] - self.y_vars[o_p, pi_p]
                self.model.addGenConstrIndicator(
                    g, True, diff >= self.epsilon,
                    name=f"inc_fwd_g_ind_on_{o_p}_{pi}_{pi_p}")
                self.model.addGenConstrIndicator(
                    g, False, diff <= 0.0,
                    name=f"inc_fwd_g_ind_off_{o_p}_{pi}_{pi_p}")

        # sum g >= 1 (forward)
        self.model.addConstr(
            gp.quicksum(self.g_vars[o_p, pi, pi_p] for o_p in objs) >= 1,
            name=f"inc_fwd_exists_{pi}_{pi_p}")

        # reverse g[o', pi', pi]
        if self.use_big_m:
            for o_p in objs:
                g = self.g_vars[o_p, pi_p, pi]
                diff = self.y_vars[o_p, pi_p] - self.y_vars[o_p, pi]
                self.model.addConstr(diff <= self.M * g,
                                     name=f"inc_rev_g_ub_{o_p}_{pi_p}_{pi}")
                self.model.addConstr(diff >= self.epsilon * g - self.M * (1 - g),
                                     name=f"inc_rev_g_lb_{o_p}_{pi_p}_{pi}")
        else:
            for o_p in objs:
                g = self.g_vars[o_p, pi_p, pi]
                diff = self.y_vars[o_p, pi_p] - self.y_vars[o_p, pi]
                self.model.addGenConstrIndicator(
                    g, True, diff >= self.epsilon,
                    name=f"inc_rev_g_ind_on_{o_p}_{pi_p}_{pi}")
                self.model.addGenConstrIndicator(
                    g, False, diff <= 0.0,
                    name=f"inc_rev_g_ind_off_{o_p}_{pi_p}_{pi}")

        # sum g >= 1 (reverse)
        self.model.addConstr(
            gp.quicksum(self.g_vars[o_p, pi_p, pi] for o_p in objs) >= 1,
            name=f"inc_rev_exists_{pi_p}_{pi}")

    def _set_objective(self):
        self.model.setObjective(
            gp.quicksum(self.x_vars[o, o_p]
                        for o in self.objectives
                        for o_p in self.objectives if o != o_p),
            GRB.MINIMIZE)

    def solve(self) -> dict:
        if self.model is None:
            self.build_model()

        self._record_ts("solve_start")
        self.model.optimize()
        self._record_ts("solve_end")
        self._record_interval("solver_time", "solve_start", "solve_end")

        status = self.model.Status

        result = {
            "status": status,
            "status_name": self._status_name(status),
            "optimal": status == GRB.OPTIMAL,
            "feasible": status in (GRB.OPTIMAL, GRB.SUBOPTIMAL,
                                    GRB.SOLUTION_LIMIT, GRB.TIME_LIMIT)
                        and self.model.SolCount > 0,
            "preorder_edges": [],
            "preorder_size": 0,
            "objective_value": None,
            "y_values": {},
            "timestamps": dict(self.timestamps),
            "intervals": dict(self.intervals),
            "num_variables": self.model.NumVars,
            "num_constraints": self.model.NumConstrs,
        }

        if result["feasible"]:
            result["objective_value"] = self.model.ObjVal

            edges = []
            for (o, o_p), var in self.x_vars.items():
                if var.X > 0.5:
                    edges.append((o, o_p))
            result["preorder_edges"] = edges
            result["preorder_size"] = len(edges)

            y_vals = {}
            for (o, pi), var in self.y_vars.items():
                y_vals.setdefault(pi, {})[o] = var.X
            result["y_values"] = y_vals

        return result

    def verify(self, result: dict) -> dict:
        """
        Verify that the solver's preorder satisfies all comparisons
        under weak-stochastic dominance.

        Steps:
          1. Compute transitive closure of the recovered preorder edges
          2. Compute lifted values for all plans
          3. Check each comparison label against Pareto dominance on lifted values
        """
        if not result["feasible"]:
            return {"verified": False, "reason": "No feasible solution to verify"}

        edges = result["preorder_edges"]
        objs = self.objectives

        upper = {o: {o} for o in objs}
        for o, o_p in edges:
            upper[o].add(o_p)
        changed = True
        while changed:
            changed = False
            for o in objs:
                new = set()
                for o_p in list(upper[o]):
                    new |= upper.get(o_p, set())
                if not new.issubset(upper[o]):
                    upper[o] |= new
                    changed = True

        lifted = {}
        for pi in self.all_plans:
            lifted[pi] = {}
            for o in objs:
                lifted[pi][o] = sum(self.plan_values[pi][o_p] for o_p in upper[o])

        violations = []
        for pi, pi_p, r in self.comparisons:
            l_pi = lifted[pi]
            l_pip = lifted[pi_p]

            pi_ge = all(l_pi[o] >= l_pip[o] for o in objs)
            pi_gt = any(l_pi[o] > l_pip[o] for o in objs)
            pip_ge = all(l_pip[o] >= l_pi[o] for o in objs)
            pip_gt = any(l_pip[o] > l_pi[o] for o in objs)

            if r == 1:
                ok = pi_ge and pi_gt
            elif r == 0:
                ok = pi_ge and pip_ge and not pi_gt and not pip_gt
            elif r == '?':
                ok = not (pi_ge and pi_gt) and not (pip_ge and pip_gt)
            else:
                ok = False

            if not ok:
                pi_wins = [o for o in objs if l_pi[o] > l_pip[o]]
                pip_wins = [o for o in objs if l_pip[o] > l_pi[o]]
                violations.append({
                    "comparison": (pi, pi_p, r),
                    "expected_label": r,
                    "left_wins_on": pi_wins,
                    "right_wins_on": pip_wins,
                })

        verified = len(violations) == 0
        return {
            "verified": verified,
            "preorder_size": len(edges),
            "num_comparisons_checked": len(self.comparisons),
            "violations": violations,
            "num_violations": len(violations),
        }

    @staticmethod
    def _status_name(status: int) -> str:
        names = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INF_OR_UNBD: "INF_OR_UNBD",
            GRB.UNBOUNDED: "UNBOUNDED",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
            GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        }
        return names.get(status, f"UNKNOWN({status})")
