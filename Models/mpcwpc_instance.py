from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class MPCwPCInstance:
    objectives: List[str]
    plan_values: Dict[str, Dict[str, float]]
    comparisons: List[Tuple[str, str, object]]

    @property
    def plans(self) -> List[str]:
        seen = []
        for pi, pi_p, _ in self.comparisons:
            if pi not in seen:
                seen.append(pi)
            if pi_p not in seen:
                seen.append(pi_p)
        return seen

    def validate(self) -> None:
        for plan, values in self.plan_values.items():
            for obj in self.objectives:
                if obj not in values:
                    raise ValueError(f"Plan '{plan}' missing value for objective '{obj}'")

        for pi, pi_p, r in self.comparisons:
            if pi not in self.plan_values:
                raise ValueError(f"Plan '{pi}' in comparisons but missing in plan_values")
            if pi_p not in self.plan_values:
                raise ValueError(f"Plan '{pi_p}' in comparisons but missing in plan_values")
            if r not in (0, 1, '?'):
                raise ValueError(f"Invalid comparison label: {r}")
