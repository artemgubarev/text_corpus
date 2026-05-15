"""Блок 3: морфологическая характеристика.

Зависит от: diagnosis_id, grade, t_code, n_code, stage_group, invasion_t_extension.
Устанавливает: specimen_*, паттерны (для D1), lvi/pleural/perineural_status,
margin_status, margin_distance_mm, macro_comment, high_grade_share.
"""

import random
from core.state import ClinicalCase, weighted_choice, maybe


class MorphologyBlock:
    def __init__(self, schemas: dict):
        self.morph = schemas["morphology"]

    def fill(self, case: ClinicalCase, rng: random.Random) -> None:
        self._pick_specimen(case, rng)
        if case.diagnosis_id == "D1" and case.grade:
            self._pick_adeno_patterns(case, rng)
        self._pick_invasions(case, rng)
        self._pick_margins(case, rng)
        case.macro_comment = rng.choice(self.morph["macro_comments"])

    # --- Specimen ---

    def _pick_specimen(self, case: ClinicalCase, rng: random.Random) -> None:
        is_resection_likely = case.stage_group in (
            "locally_advanced_operable", "locally_advanced_borderline"
        )
        if is_resection_likely and maybe(0.50, rng):
            pool = [s for s in self.morph["specimen_types"] if s["is_resection"]]
        elif case.stage_group in ("early",):
            pool = [s for s in self.morph["specimen_types"] if not s["is_resection"]]
        elif case.stage_group == "metastatic":
            pool = [s for s in self.morph["specimen_types"] if not s["is_resection"]]
        else:
            pool = self.morph["specimen_types"]
        spec = rng.choice(pool)
        case.specimen_text = spec["text"]
        case.specimen_is_resection = spec["is_resection"]

    # --- Паттерны (только для аденокарциномы) ---

    def _pick_adeno_patterns(self, case: ClinicalCase, rng: random.Random) -> None:
        rules = self.morph["grade_rules"]
        patterns = self.morph["adeno_patterns"]

        # Доминирующий по grade
        candidates_ids = rules["dominant_pattern_by_grade"][case.grade]
        dom_id = rng.choice(candidates_ids)
        case.dominant_pattern = next(p for p in patterns if p["id"] == dom_id)

        dom_range = rules[f"dominant_proportion_range_{case.grade.lower()}"]
        dom = rng.randint(*dom_range)
        case.dominant_proportion = int(round(dom / 10) * 10)

        # Вторичный — обычно есть
        if maybe(0.85, rng):
            sec_pool = [p for p in patterns if p["id"] != dom_id]
            # Для G3 вторичный часто из low-grade пула; для G1 — из low-grade тоже
            if case.grade in ("G3", "G1") and maybe(0.55, rng):
                sec_pool = [p for p in sec_pool if not p["is_high_grade"]]
            case.secondary_pattern = rng.choice(sec_pool)
            sec = min(100 - case.dominant_proportion, rng.randint(10, 40))
            case.secondary_proportion = int(round(sec / 10) * 10)

        # Расчёт high-grade share
        hg = 0
        if case.dominant_pattern["is_high_grade"]:
            hg += case.dominant_proportion
        if case.secondary_pattern and case.secondary_pattern["is_high_grade"]:
            hg += case.secondary_proportion
        case.high_grade_share = hg

    # --- Инвазии ---

    def _pick_invasions(self, case: ClinicalCase, rng: random.Random) -> None:
        # LVI — частота связана с N-статусом
        if case.n_code != "N0":
            case.lvi_status = rng.choices(
                ["positive", "negative", "indeterminate"],
                weights=[0.55, 0.30, 0.15], k=1
            )[0]
        else:
            case.lvi_status = rng.choices(
                ["positive", "negative", "indeterminate"],
                weights=[0.15, 0.70, 0.15], k=1
            )[0]

        # Плевральная — согласована с T-extension
        if case.invasion_t_extension and "висцеральн" in case.invasion_t_extension:
            case.pleural_status = rng.choice(["PL1", "PL2"])
        elif case.invasion_t_extension and "париетальн" in case.invasion_t_extension:
            case.pleural_status = "PL3"
        elif case.specimen_is_resection:
            case.pleural_status = rng.choices(
                ["PL0", "PL1", "PL2"], weights=[0.80, 0.15, 0.05], k=1
            )[0]
        else:
            # Биопсия — плевра обычно не доступна для оценки
            case.pleural_status = "not_applicable"

        # PNI
        case.perineural_status = rng.choices(
            ["positive", "negative"], weights=[0.10, 0.90], k=1
        )[0]

    # --- Края резекции ---

    def _pick_margins(self, case: ClinicalCase, rng: random.Random) -> None:
        if not case.specimen_is_resection:
            case.margin_status = "not_applicable"
            return
        r_dist = {"R0": 0.85, "R0_distance": 0.10, "R1": 0.04, "R2": 0.01}
        case.margin_status = weighted_choice(r_dist, rng)
        if case.margin_status == "R0_distance":
            case.margin_distance_mm = rng.randint(2, 25)
