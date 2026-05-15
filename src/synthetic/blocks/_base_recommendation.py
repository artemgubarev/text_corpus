"""Базовый класс для всех 4 блоков рекомендаций.

Реализует общую логику матчинга правил по condition.
Каждый конкретный блок (surgery/radiotherapy/systemic/supportive)
наследуется от него и переопределяет только подстановку переменных и
запись результата в нужные поля state.
"""

from core.state import ClinicalCase


class BaseRecommendationBlock:
    """Базовая логика для блока рекомендаций.

    Подклассы должны определить:
        SCHEMA_KEY        — ключ в schemas (например, 'surgery')
        STATE_FIELDS      — имена полей в case для записи результата:
                            (rule_id_field, intent_field, text_field)
    """

    N_ORDER = ["N0", "N1", "N2a", "N2b", "N3"]
    SCHEMA_KEY: str = ""
    STATE_FIELDS: tuple = ()

    def __init__(self, schemas: dict):
        schema = schemas[self.SCHEMA_KEY]
        # Сортируем правила по убыванию priority один раз при инициализации
        self.rules = sorted(schema["rules"], key=lambda r: -r["priority"])
        self.header = schema.get("section_header", "")

    def fill(self, case: ClinicalCase, rng=None) -> None:
        """Детерминированный матчинг правил. rng не используется."""
        for rule in self.rules:
            if self._matches(rule, case):
                self._apply(case, rule)
                return

    def _matches(self, rule: dict, case: ClinicalCase) -> bool:
        cond = rule["condition"]

        # _default матчится только проверкой stage_group (если задана)
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
        if "stage_in" in cond and case.stage not in cond["stage_in"]:
            return False
        if "stage_group" in cond and case.stage_group != cond["stage_group"]:
            return False
        if "stage_group_in" in cond and case.stage_group not in cond["stage_group_in"]:
            return False

        # N
        if "n_code" in cond and case.n_code != cond["n_code"]:
            return False
        if "n_max" in cond:
            if self.N_ORDER.index(case.n_code) > self.N_ORDER.index(cond["n_max"]):
                return False

        # T
        if "t_code" in cond and case.t_code != cond["t_code"]:
            return False
        if "t_in" in cond and case.t_code not in cond["t_in"]:
            return False

        # M
        if "m_code" in cond and case.m_code != cond["m_code"]:
            return False

        # Размер опухоли
        if "tumor_size_max_mm" in cond and case.tumor_size_mm > cond["tumor_size_max_mm"]:
            return False
        if "tumor_size_min_mm" in cond and case.tumor_size_mm < cond["tumor_size_min_mm"]:
            return False

        # Категория и диагноз
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
        if "tki_eligible" in cond:
            # tki_eligible = True если есть actionable_driver и есть first_line или second_line препараты
            has_tki = case.actionable_driver and (
                bool(case.actionable_drugs_first_line) or bool(case.actionable_drugs_second_line)
            )
            if has_tki != cond["tki_eligible"]:
                return False

        # PD-L1
        if "pdl1_min" in cond and case.pdl1_tps < cond["pdl1_min"]:
            return False
        if "pdl1_max" in cond and case.pdl1_tps > cond["pdl1_max"]:
            return False

        # Метастазы по локализациям
        if "has_brain_metastases" in cond and case.has_brain_metastases != cond["has_brain_metastases"]:
            return False
        if "has_bone_metastases" in cond and case.has_bone_metastases != cond["has_bone_metastases"]:
            return False

        # Композитные intent-флаги для supportive
        if "_curative_intent" in cond:
            curative_intents = {
                "lobectomy_curative", "sublobar_curative", "sublobar_or_lobectomy",
                "adjuvant_chemo_io", "adjuvant_chemo_then_tki", "neoadjuvant_chemo_io",
                "chemo_radiation_consolidation", "radical_crt",
            }
            is_curative = (
                case.surgery_intent in curative_intents or
                case.systemic_intent in curative_intents or
                case.radiotherapy_intent == "radical_crt"
            )
            if is_curative != cond["_curative_intent"]:
                return False

        if "_io_intent" in cond:
            io_intents = {"io_monotherapy", "chemo_io_combo", "neoadjuvant_chemo_io"}
            has_io = case.systemic_intent in io_intents
            if has_io != cond["_io_intent"]:
                return False

        return True

    def _apply(self, case: ClinicalCase, rule: dict) -> None:
        rule_id_field, intent_field, text_field = self.STATE_FIELDS
        setattr(case, rule_id_field, rule["id"])
        setattr(case, intent_field, rule["intent"])

        text = rule["text"]
        text = self._substitute_variables(text, case)
        setattr(case, text_field, text)

    def _substitute_variables(self, text: str, case: ClinicalCase) -> str:
        """Подстановка переменных. Подклассы могут расширять."""
        text = text.replace("{stage}", case.stage)
        text = text.replace("{pdl1}", str(case.pdl1_tps))
        text = text.replace("{variant}", case.actionable_variant_short or "")
        if case.actionable_drugs_first_line:
            text = text.replace("{drug}", " / ".join(case.actionable_drugs_first_line))
        elif case.actionable_drugs_second_line:
            text = text.replace("{drug}", " / ".join(case.actionable_drugs_second_line))
        return text
