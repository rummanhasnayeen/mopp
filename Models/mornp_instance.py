from dataclasses import dataclass
from typing import Dict, List


@dataclass
class MORNPInstance:
    """
    Instance for MORNP-DEC.

    objectives: list of obj
    plan_values: plan -> mapping obj -> numeric value
    positive_plans: plan in P+
    negative_plans: plan in P-
    k: max number of selected obj
    """
    objectives: List[str]
    plan_values: Dict[str, Dict[str, float]]
    positive_plans: List[str]
    negative_plans: List[str]
    k: int

    @property
    def all_sample_plans(self) -> List[str]:
        return self.positive_plans + self.negative_plans

    def validate(self) -> None:
        # Check every plan has values for every objective
        for plan, values in self.plan_values.items():
            for obj in self.objectives:
                if obj not in values:
                    raise ValueError(f"Plan '{plan}' missing value for objective '{obj}'")

        # Check all positive/negative plans exist
        for p in self.positive_plans + self.negative_plans:
            if p not in self.plan_values:
                raise ValueError(f"Plan '{p}' listed in sample but missing in plan_values")

        if self.k < 0:
            raise ValueError("k must be nonnegative")