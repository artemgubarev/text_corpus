"""Блок 6: рекомендация по лечению.

ПОЛНОСТЬЮ ДЕТЕРМИНИРОВАННЫЙ. На основе всех уже заполненных полей state
выбирает первое сработавшее правило (по убыванию priority) и заполняет
recommendation_text + follow_up_text.

Это место, где гарантируется отсутствие противоречий между ECOG, стадией,
драйверами и финальной рекомендацией.
"""

from core.state import ClinicalCase


class RecommendationBlock:
    N_ORDER = ["N0", "N1", "N2a", "N2b", "N3"]

    def __init__(self, schemas: dict):
        rec = schemas["recommendations"]
        # Сортируем правила по убыванию priority один раз при инициализации
        self.rules = sorted(rec["rules"], key=lambda r: -r["priority"])
        self.follow_ups = rec["follow_up_templates"]

    def fill(self, case: ClinicalCase, rng=None) -> None:
        # rng не используется — это полностью детерминированный блок
        for rule in self.rules:
            if self._matches(rule, case):
                self._apply(case, rule)
                return
        # Не должно случаться благодаря fallback-правилам
        case.recommendation_text = (
            "Тактика лечения требует обсуждения на мультидисциплинарном консилиуме."
        )
        case.treatment_intent = "uncertain"

    def _matches(self, rule: dict, case: ClinicalCase) -> bool:
        cond = rule["condition"]

        # _default: матчится только проверкой stage_group
        if cond.get("_default"):
            if "stage_group_in" in cond:
                return case.stage_group in cond["stage_group_in"]
            if "stage_group" in cond:
                return case.stage_group == cond["stage_group"]
            return True

        # ECOG
        if "ecog" in cond and case.ecog != cond["ecog"]:
            return False
        if "ecog_max" in cond and case.ecog > cond["ecog_max"]:
            return False

        # Stage
        if "stage" in cond and case.stage not in cond["stage"]:
            return False
        if "stage_group" in cond and case.stage_group != cond["stage_group"]:
            return False
        if "stage_group_in" in cond and case.stage_group not in cond["stage_group_in"]:
            return False

        # N
        if "n_max" in cond:
            if self.N_ORDER.index(case.n_code) > self.N_ORDER.index(cond["n_max"]):
                return False

        # Диагноз
        if "category" in cond and case.diagnosis_category != cond["category"]:
            return False
        if "diagnosis_in" in cond and case.diagnosis_id not in cond["diagnosis_in"]:
            return False

        # Драйверы
        if "actionable_driver" in cond and case.actionable_driver != cond["actionable_driver"]:
            return False
        if "actionable_driver_gene" in cond and case.actionable_gene != cond["actionable_driver_gene"]:
            return False
        if "actionable_variant_in" in cond:
            if case.actionable_variant_short not in cond["actionable_variant_in"]:
                return False

        # PD-L1
        if "pdl1_min" in cond and case.pdl1_tps < cond["pdl1_min"]:
            return False
        if "pdl1_max" in cond and case.pdl1_tps > cond["pdl1_max"]:
            return False

        return True

    def _apply(self, case: ClinicalCase, rule: dict) -> None:
        case.matched_rule_id = rule["id"]
        case.treatment_intent = rule["intent"]

        # Подстановка переменных в шаблон
        text = rule["recommendation"]
        text = text.replace("{stage}", case.stage)
        text = text.replace("{pdl1}", str(case.pdl1_tps))
        text = text.replace("{variant}", case.actionable_variant_short or "")
        if case.actionable_drugs_first_line:
            text = text.replace("{drug}", " / ".join(case.actionable_drugs_first_line))
        elif case.actionable_drugs_second_line:
            text = text.replace("{drug}", " / ".join(case.actionable_drugs_second_line))
        case.recommendation_text = text
        case.follow_up_text = self.follow_ups.get(rule["intent"], "")
