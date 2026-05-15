"""Блок 1: выбор диагноза, подтипа, grade.

Первый в pipeline — все последующие блоки опираются на этот.
"""

import random
from core.state import ClinicalCase, weighted_choice, maybe


class DiagnosisBlock:
    """Заполняет: diagnosis_id, diagnosis_name(_lat), diagnosis_category,
    icd_o_3, subtype_name(_lat), grade."""

    def __init__(self, schemas: dict):
        self.s = schemas["diagnoses"]

    def fill(self, case: ClinicalCase, rng: random.Random) -> None:
        # 1. Основной диагноз
        diag_id = weighted_choice(self.s["diagnosis_distribution"], rng)
        diag = next(d for d in self.s["diagnoses"] if d["id"] == diag_id)
        case.diagnosis_id = diag_id
        case.diagnosis_name = diag["name"]
        case.diagnosis_name_lat = diag["name_lat"]
        case.diagnosis_category = diag["category"]
        case.icd_o_3 = diag["icd_o_3"]

        # 2. Подтип (только для аденокарциномы)
        if diag.get("subtype_pool") == "adeno":
            self._maybe_pick_adeno_subtype(case, rng)

        # 3. Grade
        if diag["gradable"] and self._subtype_allows_grade(case):
            case.grade = rng.choices(
                ["G1", "G2", "G3"], weights=[0.20, 0.45, 0.35], k=1
            )[0]

    def _maybe_pick_adeno_subtype(self, case: ClinicalCase, rng: random.Random) -> None:
        if not maybe(0.7, rng):
            return
        subtypes = self.s["adenocarcinoma_subtypes"]
        weights = [1.0 if s.get("frequency") == "high" else 0.3 for s in subtypes]
        sub = rng.choices(subtypes, weights=weights, k=1)[0]
        case.subtype_name = sub["name"]
        case.subtype_name_lat = sub.get("name_lat")

    def _subtype_allows_grade(self, case: ClinicalCase) -> bool:
        """AIS/MIA не grade-уются, у других подтипов и без подтипа — grade применим."""
        if case.diagnosis_id != "D1" or not case.subtype_name:
            return True
        sub = next(
            (s for s in self.s["adenocarcinoma_subtypes"] if s["name"] == case.subtype_name),
            None,
        )
        return bool(sub and sub.get("gradable", True))
