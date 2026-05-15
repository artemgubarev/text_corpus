"""Блок 5: функциональный статус (ECOG + Карновский).

Зависит от: stage_group (на ранних стадиях ECOG 4 практически не встречается).
Устанавливает: ecog, karnofsky.
"""

import random
from core.state import ClinicalCase, weighted_choice


class EcogBlock:
    def __init__(self, schemas: dict):
        self.s = schemas["ecog"]

    def fill(self, case: ClinicalCase, rng: random.Random) -> None:
        ecog_dist = self.s["ecog_distribution_by_stage_group"]
        dist = ecog_dist.get(case.stage_group, ecog_dist["locally_advanced_operable"])
        case.ecog = int(weighted_choice(dist, rng))

        scale = next(s for s in self.s["ecog_scale"] if s["ecog"] == case.ecog)
        karn_lo, karn_hi = scale["karnofsky_range"]
        case.karnofsky = rng.choice([karn_lo, karn_hi])
