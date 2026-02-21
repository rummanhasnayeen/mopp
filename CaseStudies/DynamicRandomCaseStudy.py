import random
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Set
from collections import defaultdict, deque

ComparisonLabel = Union[int, str]  # 0, 1, "?"
PLAN_VECTOR_LOWER_BOUND = 1
PLAN_VECTOR_UPPER_BOUND = 10

@dataclass
class SimpleMOPPInstance:
    objectives: List[str]
    plans: List[str]
    values: Dict[str, Dict[str, int]]
    comparisons: List[Tuple[str, str, ComparisonLabel]]
    k: int


class DynamicRandomCaseStudy:
    """
    Dynamic case study generator.

    Constructor args:
      - num_objectives: n
      - num_plans: m
      - num_comparisons: c
      - k: bound on |Omega|
      - seed: optional for reproducibility

    Generates:
      - objectives: ["o1", ..., "on"]
      - plans: ["p1", ..., "pm"]
      - values: random ints in [1,10]
      - comparisons: exactly num_comparisons, includes at least one of each label: 0, 1, "?"
        and (crucially) ALL comparisons are consistent with a hidden Omega* of size <= k,
        so the SAT instance is a YES-instance (assuming k >= 2 if you require "?").
    """

    def __init__(
        self,
        num_objectives: int,
        num_plans: int,
        num_comparisons: int,
        k: int,
        seed: int = 7,
        omega_ratio: float = 0.15,
        omega_min: int = 2
    ):
        if num_objectives <= 0:
            raise ValueError("num_objectives must be >= 1")
        if num_plans <= 1:
            raise ValueError("num_plans must be >= 2")
        if num_comparisons <= 0:
            raise ValueError("num_comparisons must be >= 1")
        if k < 0:
            raise ValueError("k must be >= 0")

        # If user asks for "?" to exist, Pareto incomparability needs >= 2 selected objectives.
        # We still *generate* "?" comparisons, but guaranteeing SAT=YES requires k>=2.
        self._require_incomparability = True

        self.num_objectives = num_objectives
        self.num_plans = num_plans
        self.num_comparisons = num_comparisons
        self.k = k
        self.seed = seed
        self.omega_ratio = omega_ratio
        self.omega_min = omega_min

        # random.seed(seed)
        self.rng = random.Random(seed)

        self.objectives = [f"o{i}" for i in range(1, num_objectives + 1)]
        self.plans = [f"p{i}" for i in range(1, num_plans + 1)]

        # Hidden "true" objective subset Omega* used to generate consistent comparisons
        self._omega_star = self._pick_omega_star()

        # Values: plan -> objective -> int in [1,10]
        self.values = {
            p: {o: self.rng.randint(PLAN_VECTOR_LOWER_BOUND, PLAN_VECTOR_UPPER_BOUND) for o in self.objectives}
            for p in self.plans
        }

        # ============================
        # Force at least one example of each label under Omega* (0,1,"?")--commented--
        self.comparisons = []
        self._force_one_of_each_label()
        remaining = self.num_comparisons - len(self.comparisons)
        if remaining > 0:
            extra = self._generate_comparisons_mixture(self.values, self._omega_star, remaining)
            self.comparisons.extend(extra)

        # self.values, self.comparisons = self._generate_graph_first_then_values()
        # ============================
        # assert self._debug_check_consistency_with_true_omega()

        rep = self._sanity_check_transitivity(raise_on_violation=False)
        print(rep["details"])
        if not rep["ok"]:
            print("Cycle:", rep["cycle"])

    def _build_plan_order(self) -> List[str]:
        plans = self.plans[:]
        self.rng.shuffle(plans)
        return plans

    def _build_dominance_dag(self, order: List[str], edge_budget: int) -> Set[tuple]:
        edges = set()
        m = len(order)
        # only allow edges from earlier -> later to avoid cycles
        candidates = [(order[i], order[j]) for i in range(m) for j in range(i + 1, m)]
        self.rng.shuffle(candidates)
        for (u, v) in candidates[:edge_budget]:
            edges.add((u, v))
        return edges

    def _init_random_values(self):
        values = {p: {} for p in self.plans}
        for p in self.plans:
            for o in self.objectives:
                values[p][o] = self.rng.randint(PLAN_VECTOR_LOWER_BOUND, PLAN_VECTOR_UPPER_BOUND)
        return values

    def _enforce_dominance_on_omega(self, values, omega_star, edges):
        omega_list = list(omega_star)
        for (u, v) in edges:
            # make u >= v on all omega objectives
            for o in omega_list:
                if values[u][o] < values[v][o]:
                    values[u][o] = values[v][o]
            # ensure at least one strict improvement
            o_strict = self.rng.choice(omega_list)
            if values[u][o_strict] == values[v][o_strict]:
                if values[u][o_strict] < 10:
                    values[u][o_strict] += 1
                elif values[v][o_strict] > 1:
                    values[v][o_strict] -= 1
                # else extremely rare corner; you can reroll that objective

    def _create_indifference_pairs(self, values, omega_star, num_zero_pairs):
        omega_list = list(omega_star)
        pairs = set()
        attempts = 0
        while len(pairs) < num_zero_pairs and attempts < 10_000:
            a, b = self.rng.sample(self.plans, 2)
            attempts += 1
            if a == b:
                continue
            # force equality on omega objectives
            for o in omega_list:
                values[b][o] = values[a][o]
            pairs.add((a, b))
        return pairs

    def _create_incomparable_pairs(self, values, omega_star, num_q_pairs):
        omega_list = list(omega_star)
        if len(omega_list) < 2:
            return set()

        pairs = set()
        attempts = 0
        while len(pairs) < num_q_pairs and attempts < 10_000:
            a, b = self.rng.sample(self.plans, 2)
            attempts += 1
            if a == b:
                continue

            o1, o2 = self.rng.sample(omega_list, 2)

            # enforce a better than b on o1
            if values[a][o1] <= values[b][o1]:
                values[a][o1] = min(10, values[b][o1] + 1)

            # enforce a worse than b on o2
            if values[a][o2] >= values[b][o2]:
                values[a][o2] = max(1, values[b][o2] - 1)

            pairs.add((a, b))
        return pairs

    # --added--
    def _generate_graph_first_then_values(self):
        """
        Pipeline:
          1) Build a DAG over plans (for r=1 dominance constraints)
          2) Choose comparison pairs for r=1, r=0, r=?
          3) Initialize random values, then enforce constraints so ALL chosen comparisons are consistent under Omega*
        """
        omega = list(self._omega_star)

        # ------------- 1) Decide how many of each label we want -------------
        # Keep a real mixture: e.g., 30% "1", 20% "0", rest "?"
        total = self.num_comparisons
        num_1 = max(2, int(0.30 * total))
        num_0 = max(2, int(0.20 * total))
        num_q = total - num_1 - num_0

        # If omega too small, "?" isn't satisfiable in Pareto sense with your encoding
        # (needs >=2 selected objectives to force a tradeoff).
        if self._require_incomparability and len(omega) < 2:
            # fallback: no "?" and keep instance satisfiable
            num_q = 0
            # distribute remaining to 0/1
            num_1 = max(1, total - num_0)

        # ------------- 2) Build DAG first (these will become r=1 comparisons) -------------
        order = self._build_plan_order()
        edges = self._build_dominance_dag(order, edge_budget=num_1)

        # ------------- 3) Init values (random) -------------
        values = self._init_random_values()

        # ------------- 4) Enforce "1" edges as Pareto dominance over Omega* -------------
        if len(omega) > 0:
            self._enforce_dominance_on_omega(values, omega, edges)
        else:
            # if omega empty, ONLY r=0 is satisfiable; so force all comparisons to 0
            num_0 = total
            num_1 = 0
            num_q = 0
            edges = set()

        # ------------- 5) Create r=0 pairs (indifference) and enforce equality on Omega* -------------
        zero_pairs = self._create_indifference_pairs(values, omega, num_zero_pairs=num_0)

        # ------------- 6) Create r=? pairs and enforce incomparability (tradeoff on Omega*) -------------
        q_pairs = self._create_incomparable_pairs(values, omega, num_q_pairs=num_q)

        # ------------- 7) Build comparisons list from those pairs -------------
        comparisons = []

        # r=1 from DAG edges
        for (a, b) in edges:
            comparisons.append((a, b, 1))

        # r=0 from zero_pairs
        for (a, b) in zero_pairs:
            comparisons.append((a, b, 0))

        # r=? from q_pairs
        for (a, b) in q_pairs:
            comparisons.append((a, b, "?"))

        # Ensure exact length and uniqueness (trim or top-up)
        # If duplicates reduced count, top-up using consistent labels derived from omega
        seen = set()
        uniq = []
        for t in comparisons:
            if t not in seen:
                uniq.append(t)
                seen.add(t)

        # Top-up if needed: sample random pairs and label them using the values we constructed
        # NOTE: set self.values temporarily so your existing _relation_under_omega works unchanged.
        self.values = values
        max_attempts = 50000
        attempts = 0
        while len(uniq) < total and attempts < max_attempts:
            attempts += 1
            a, b = self.rng.sample(self.plans, 2)
            r = self._relation_under_omega(a, b, omega)

            # keep mixture but don't break satisfiability rules
            if r == "?" and len(omega) < 2:
                continue

            triple = (a, b, r)
            if triple not in seen:
                uniq.append(triple)
                seen.add(triple)

        if len(uniq) < total:
            raise RuntimeError(
                "Could not generate enough unique comparisons. Increase num_plans or reduce num_comparisons.")

        # If we overfilled, trim
        uniq = uniq[:total]

        return values, uniq

    # def _label_under_omega(self, values, omega_star, p, q):
    #     better = False
    #     worse = False
    #     for o in omega_star:
    #         if values[p][o] > values[q][o]:
    #             better = True
    #         elif values[p][o] < values[q][o]:
    #             worse = True
    #     if not better and not worse:
    #         return 0
    #     if better and not worse:
    #         return 1
    #     if worse and not better:
    #         return 1  # but note: this would mean q dominates p; handle direction outside
    #     return "?"

    def _relation_under_omega(self, values, omega_star, p, q):
        ge_all = all(values[p][o] >= values[q][o] for o in omega_star)
        le_all = all(values[p][o] <= values[q][o] for o in omega_star)
        gt_any = any(values[p][o] > values[q][o] for o in omega_star)
        lt_any = any(values[p][o] < values[q][o] for o in omega_star)

        if ge_all and gt_any:
            return 1  # p ≻ q
        if le_all and lt_any:
            return -1  # q ≻ p
        if not gt_any and not lt_any:
            return 0  # equal
        return "?"  # incomparable

    def _generate_comparisons_mixture(self, values, omega_star, num_comparisons):
        target_ones = max(2, int(0.30 * num_comparisons))
        target_zeros = max(2, int(0.20 * num_comparisons))
        target_q = num_comparisons - target_ones - target_zeros

        ones, zeros, qs = [], [], []
        seen = set()
        attempts = 0

        while len(ones) + len(zeros) + len(qs) < num_comparisons and attempts < 200_000:
            p, q = self.rng.sample(self.plans, 2)
            if (p, q) in seen:
                attempts += 1
                continue

            rel = self._relation_under_omega(values, omega_star, p, q)

            if rel == 1 and len(ones) < target_ones:
                ones.append((p, q, 1));
                seen.add((p, q))
            elif rel == -1 and len(ones) < target_ones:
                ones.append((q, p, 1));
                seen.add((q, p))  # flip
            elif rel == 0 and len(zeros) < target_zeros:
                zeros.append((p, q, 0));
                seen.add((p, q))
            elif rel == "?" and len(qs) < target_q:
                qs.append((p, q, "?"));
                seen.add((p, q))

            attempts += 1

        # If you couldn’t fill buckets, it means your value construction didn’t create enough of some type.
        # In that case: increase the forced zero/? pairs in Step 2.3/2.4.
        return ones + zeros + qs

    # def _pick_omega_star(self) -> List[str]:
    #     if self.k == 0:
    #         # With empty Omega r=1 would be impossible
    #         return []
    #     size = min(self.k, self.num_objectives)
    #     # If we want "?" to be possible, we need at least 2 in Omega*.
    #     if self._require_incomparability and size == 1 and self.num_objectives >= 2:
    #         # keep Omega* size 2, but this violates k=1; can't guarantee YES with "?" required.
    #         # We'll keep size=1 to respect k, but that means "?" can't be generated consistently.
    #         pass
    #     return self.rng.sample(self.objectives, size)

    def _pick_omega_star(self) -> List[str]:
        # Grow Ω* with number of objectives, independent of provided k
        # size = int(math.ceil(self.omega_ratio * self.num_objectives))
        # size = max(self.omega_min, size)
        # size = min(size, self.num_objectives)
        # # grows like sqrt(n)
        # size = max(2, int(round(0.6 * (self.num_objectives ** 0.5))))
        # size = min(size, self.num_objectives)
        # grows linearly with n
        size = max(2, int(round(0.05 * self.num_objectives)))  # 5%
        size = min(size, self.num_objectives)
        return self.rng.sample(self.objectives, size)

    def _relation_under_omega_self_values(self, p: str, q: str, omega: List[str]) -> ComparisonLabel:
        """
        Returns the label r consistent with Pareto relation under omega:
          - 1 if p strictly dominates q
          - 0 if equal on all omega objectives
          - "?" otherwise (incomparable)
        """
        if not omega:
            return 0  # empty vector equals empty vector

        better = False
        worse = False
        for o in omega:
            vp = self.values[p][o]
            vq = self.values[q][o]
            if vp > vq:
                better = True
            elif vp < vq:
                worse = True

        if (not worse) and better:
            return 1
        if (not better) and (not worse):
            return 0
        return "?"

    def _force_one_of_each_label(self) -> None:
        """
        GPT v
        We hard-construct 3 comparisons consistent with Omega*:
          - one r=0 (equality on Omega*)
          - one r=1 (dominance on Omega*)
          - one r="?" (incomparability on Omega*), only feasible if |Omega*| >= 2
        """
        omega = self._omega_star

        # --- Force r=0
        p_eq, q_eq = self.plans[0], self.plans[1]
        for o in omega:
            self.values[q_eq][o] = self.values[p_eq][o]  # equalize on Omega*
        self.comparisons.append((p_eq, q_eq, 0))

        # --- Force r=1 (p_dom dominates q_dom)
        # Use next pair if available, else reuse.
        p_dom = self.plans[2] if self.num_plans >= 3 else self.plans[0]
        q_dom = self.plans[3] if self.num_plans >= 4 else self.plans[1]

        if omega:
            # Make p_dom >= q_dom on all omega, and strictly > on at least one
            # (set p_dom high, q_dom low)
            for o in omega:
                self.values[p_dom][o] = self.rng.randint(6, 10)
                self.values[q_dom][o] = self.rng.randint(1, 5)
            # Ensure strictness
            self.values[p_dom][omega[0]] = min(10, self.values[q_dom][omega[0]] + 1)

            self.comparisons.append((p_dom, q_dom, 1))
        else:
            # If omega is empty (k=0), r=1 cannot be satisfied. We won’t add it.
            pass

        # --- Force r="?"
        # Only meaningful if omega has at least 2 objectives.
        if len(omega) >= 2 and self.num_plans >= 6:
            p_inc, q_inc = self.plans[4], self.plans[5]
            o1, o2 = omega[0], omega[1]

            # p_inc better on o1 but worse on o2 vs q_inc
            base1 = self.rng.randint(4, 7)
            base2 = self.rng.randint(4, 7)

            self.values[p_inc][o1] = min(10, base1 + 2)
            self.values[q_inc][o1] = max(1, base1 - 2)

            self.values[p_inc][o2] = max(1, base2 - 2)
            self.values[q_inc][o2] = min(10, base2 + 2)

            self.comparisons.append((p_inc, q_inc, "?"))
        else:
            # k<2
            pass

        # Remove duplicates
        seen = set()
        unique = []
        for (a, b, r) in self.comparisons:
            key = (a, b, r)
            if key not in seen:
                seen.add(key)
                unique.append((a, b, r))
        self.comparisons = unique

    def _fill_remaining_comparisons(self) -> None:
        """
        Fill until reaching num_comparisons by sampling plan pairs and labeling them
        using the true Omega* relation, guaranteeing consistency (YES-instance).
        """
        omega = self._omega_star

        #GPT v
        # If omega empty: only r=0 comparisons are consistent. We will fill with r=0 only.
        # Also note: if you demanded r=1 or "?" in this scenario, it can become UNSAT.
        max_attempts = 20000
        attempts = 0

        # Track label exists
        have_0 = any(r == 0 for _, _, r in self.comparisons)
        have_1 = any(r == 1 for _, _, r in self.comparisons)
        have_q = any(r == "?" for _, _, r in self.comparisons)

        while len(self.comparisons) < self.num_comparisons and attempts < max_attempts:
            attempts += 1
            a, b = self.rng.sample(self.plans, 2)
            r = self._relation_under_omega(self.values, omega, a, b)

            #  k<2
            if r == "?" and len(omega) < 2:
                continue

            # balance labels to get 0/1/? mix
            if len(omega) >= 2:
                # enforce at least one of each label
                if not have_0 and r != 0:
                    continue
                if not have_1 and r != 1:
                    continue
                if not have_q and r != "?":
                    continue

            triple = (a, b, r)
            if triple not in self.comparisons:
                self.comparisons.append(triple)
                have_0 = have_0 or (r == 0)
                have_1 = have_1 or (r == 1)
                have_q = have_q or (r == "?")

        if len(self.comparisons) < self.num_comparisons:
            raise RuntimeError(
                f"Could not generate {self.num_comparisons} unique comparisons "
                f"after {max_attempts} attempts. Try increasing num_plans or lowering num_comparisons."
            )


    def print_summary(self):
        print("\n========== Dynamic Random Case Study ==========")
        print(f"Number of objectives: {len(self.objectives)}")
        print(f"Objectives: {self.objectives}")

        print(f"\nNumber of plans: {len(self.plans)}")
        print(f"Plans: {self.plans}")

        print(f"\nCardinality bound k = {self.k}")

        print("\n--- Objective values per plan ---")
        for p in self.plans:
            print(f"{p}: {self.values[p]}")

        print("\n--- Plan comparisons (pi, pj, r) ---")
        print(f"\nNumber of comparisons: {len(self.comparisons)}")
        for (p1, p2, r) in self.comparisons:
            print(f"({p1}, {p2}, {r})")

        # # debug
        if hasattr(self, "_omega_star"):
            print(f"\n[Debug] Hidden true Ω* used for generation: {self._omega_star}")

        print("==============================================\n")


    def get_instance(self) -> SimpleMOPPInstance:
        return SimpleMOPPInstance(
            objectives=self.objectives,
            plans=self.plans,
            values=self.values,
            comparisons=self.comparisons,
            k=self.k,
        )

    def debug_true_omega(self) -> List[str]:
        return list(self._omega_star)

    def _debug_check_consistency_with_true_omega(self):
        omega = list(self._omega_star)
        for (a, b, r) in self.comparisons:
            rr = self._relation_under_omega(a, b, omega)
            if rr != r:
                print("Mismatch!", (a, b, r), "but computed:", rr)
                return False
        return True

    def _sanity_check_transitivity(self, raise_on_violation: bool = False):
        """
        Strict-preference transitivity sanity check.

        Interpretation:
          - r = 1 : edge a -> b (a strictly preferred to b)
          - r = 0 : a and b are equivalent (union-find collapse)
          - r = ? : ignored for transitivity (incomparability)

        A contradiction (for strict order) is a directed cycle among the collapsed nodes.

        Returns:
          dict with:
            ok: bool
            cycle: list[str] | None   # representative node names in cycle order (if found)
            details: str
        """

        comparisons = self.comparisons  # list[(planA, planB, r)]

        # ---------- 1) Union-Find to collapse r=0 equivalences ----------
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        # Collect all plan names
        plans = set()
        for a, b, r in comparisons:
            plans.add(a)
            plans.add(b)
            if r == 0:
                union(a, b)

        # Ensure all plans in UF
        for p in plans:
            find(p)

        # ---------- 2) Build strict edges (r=1) between reps ----------
        adj = defaultdict(set)
        indeg = defaultdict(int)

        # Initialize nodes
        reps = {find(p) for p in plans}
        for rep in reps:
            indeg[rep] = 0

        # Add edges
        for a, b, r in comparisons:
            if r != 1:
                continue
            ra, rb = find(a), find(b)

            # If a ~ b but labeled as strict preference, that's an immediate inconsistency
            if ra == rb:
                report = {
                    "ok": False,
                    "cycle": [ra],
                    "details": f"Immediate violation: '{a}' ~ '{b}' via r=0, but also has r=1 edge."
                }
                if raise_on_violation:
                    raise AssertionError(report["details"])
                return report

            if rb not in adj[ra]:
                adj[ra].add(rb)
                indeg[rb] += 1

        # ---------- 3) Detect cycle via Kahn (topological check) ----------
        q = deque([n for n in reps if indeg[n] == 0])
        topo = []

        while q:
            n = q.popleft()
            topo.append(n)
            for nei in adj[n]:
                indeg[nei] -= 1
                if indeg[nei] == 0:
                    q.append(nei)

        if len(topo) == len(reps):
            # No cycle => transitivity can hold as a strict partial order over the observed r=1 edges
            return {"ok": True, "cycle": None,
                    "details": "No directed cycle found in r=1 edges (after collapsing r=0)."}

        # ---------- 4) Extract one concrete cycle (DFS on remaining nodes) ----------
        remaining = set(reps) - set(topo)

        # DFS to find a cycle
        color = {n: 0 for n in reps}  # 0=unvisited, 1=visiting, 2=done
        stack = []
        stack_pos = {}

        def dfs(u):
            color[u] = 1
            stack_pos[u] = len(stack)
            stack.append(u)

            for v in adj[u]:
                if v not in remaining:
                    continue
                if color[v] == 0:
                    cyc = dfs(v)
                    if cyc:
                        return cyc
                elif color[v] == 1:
                    # Found a back-edge u -> v, cycle is stack[pos[v]:] + [v]
                    start = stack_pos[v]
                    cycle_nodes = stack[start:] + [v]
                    return cycle_nodes

            color[u] = 2
            stack_pos.pop(u, None)
            stack.pop()
            return None

        cycle = None
        for node in list(remaining):
            if color[node] == 0:
                cycle = dfs(node)
                if cycle:
                    break

        report = {
            "ok": False,
            "cycle": cycle,
            "details": "Directed cycle found in r=1 edges (after collapsing r=0). This violates strict-order transitivity."
        }
        if raise_on_violation:
            raise AssertionError(f"{report['details']} Cycle: {cycle}")
        return report
