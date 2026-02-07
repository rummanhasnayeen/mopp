import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Set

ComparisonLabel = Union[int, str]  # 0, 1, "?"


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

        random.seed(seed)

        self.objectives = [f"o{i}" for i in range(1, num_objectives + 1)]
        self.plans = [f"p{i}" for i in range(1, num_plans + 1)]

        # Hidden "true" objective subset Omega* used to generate consistent comparisons
        self._omega_star = self._pick_omega_star()

        # Values: plan -> objective -> int in [1,10]
        self.values = {
            p: {o: random.randint(1, 10) for o in self.objectives}
            for p in self.plans
        }

        # Force at least one example of each label under Omega* (0,1,"?")
        self.comparisons: List[Tuple[str, str, ComparisonLabel]] = []
        self._force_one_of_each_label()
        self._fill_remaining_comparisons()

    def _pick_omega_star(self) -> List[str]:
        if self.k == 0:
            # With empty Omega r=1 would be impossible
            return []
        size = min(self.k, self.num_objectives)
        # If we want "?" to be possible, we need at least 2 in Omega*.
        if self._require_incomparability and size == 1 and self.num_objectives >= 2:
            # keep Omega* size 2, but this violates k=1; can't guarantee YES with "?" required.
            # We'll keep size=1 to respect k, but that means "?" can't be generated consistently.
            pass
        return random.sample(self.objectives, size)

    def _relation_under_omega(self, p: str, q: str, omega: List[str]) -> ComparisonLabel:
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
                self.values[p_dom][o] = random.randint(6, 10)
                self.values[q_dom][o] = random.randint(1, 5)
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
            base1 = random.randint(4, 7)
            base2 = random.randint(4, 7)

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
            a, b = random.sample(self.plans, 2)
            r = self._relation_under_omega(a, b, omega)

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
