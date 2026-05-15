"""Текстовый рендерер. Превращает ClinicalCase в человекочитаемый текст.

Никаких случайных выборов — только форматирование уже зафиксированных полей.
Любая правка стиля текста живёт здесь, не трогая логику в blocks/.
"""

from core.state import ClinicalCase


class TextRenderer:
    GRADE_DESCRIPTIONS = {
        "G1": "высокодифференцированная",
        "G2": "умеренно дифференцированная",
        "G3": "плохо дифференцированная",
    }

    PLEURAL_TEXTS = {
        "PL0": "плевральная инвазия — PL0 (отсутствует)",
        "PL1": "плевральная инвазия — PL1 (висцеральная плевра)",
        "PL2": "плевральная инвазия — PL2 (поверхность висцеральной плевры)",
        "PL3": "плевральная инвазия — PL3 (париетальная плевра)",
    }

    def __init__(self, schemas: dict):
        self.schemas = schemas
        self.ecog_scale = schemas["ecog"]["ecog_scale"]

    def render(self, case: ClinicalCase) -> str:
        blocks = [
            self._diagnosis_header(case),
            self._morphology(case),
            self._tnm(case),
        ]
        if case.molecular_included:
            blocks.append(self._molecular(case))
        blocks.append(self._ecog(case))
        blocks.append(self._recommendation(case))
        return "\n\n".join(b for b in blocks if b)

    # ---- Блоки ----

    def _diagnosis_header(self, case: ClinicalCase) -> str:
        lines = [f"Тип материала: {case.specimen_text}."]

        diag_line = f"Диагноз: {case.diagnosis_name}"
        if case.subtype_name:
            diag_line += f", {case.subtype_name}"
            if case.subtype_name_lat:
                diag_line += f" ({case.subtype_name_lat})"
        elif case.diagnosis_name_lat:
            diag_line += f" ({case.diagnosis_name_lat})"
        if case.grade:
            diag_line += f". Степень дифференцировки: {case.grade}"
            diag_line += f" ({self.GRADE_DESCRIPTIONS[case.grade]})"
        diag_line += "."
        lines.append(diag_line)

        if case.icd_o_3:
            lines.append(f"ICD-O-3: {case.icd_o_3}.")
        return "\n".join(lines)

    def _morphology(self, case: ClinicalCase) -> str:
        lines = ["Морфологическая характеристика."]

        # Паттерны
        if case.dominant_pattern:
            dom = case.dominant_pattern
            lines.append(
                f"Доминирующий паттерн: {dom['name_ru'].lower()} "
                f"({dom['name_en']}) — около {case.dominant_proportion}%."
            )
            if case.secondary_pattern:
                sec = case.secondary_pattern
                lines.append(
                    f"Второстепенный паттерн: {sec['name_ru'].lower()} "
                    f"({sec['name_en']}) — около {case.secondary_proportion}%."
                )
            if case.grade == "G3" and case.high_grade_share >= 20:
                lines.append(
                    f"Доля high-grade компонентов: {case.high_grade_share}% "
                    f"(критерий G3 выполнен по IASLC 2020)."
                )

        # Инвазии
        inv_parts = []
        if case.lvi_status == "positive":
            inv_parts.append("лимфоваскулярная инвазия (LVI) — обнаружена")
        elif case.lvi_status == "negative":
            inv_parts.append("лимфоваскулярная инвазия (LVI) — не обнаружена")
        else:
            inv_parts.append("LVI — не оценивается")

        if case.pleural_status in self.PLEURAL_TEXTS:
            inv_parts.append(self.PLEURAL_TEXTS[case.pleural_status])

        if case.perineural_status == "positive":
            inv_parts.append("периневральная инвазия (PNI) — обнаружена")
        else:
            inv_parts.append("периневральная инвазия (PNI) — не обнаружена")

        lines.append("Инвазия: " + "; ".join(inv_parts) + ".")

        # Края
        if case.margin_status == "not_applicable":
            lines.append("Резекционные края: не оценены (биопсия).")
        elif case.margin_status == "R0":
            lines.append("Резекционные края: R0 — свободны от опухоли.")
        elif case.margin_status == "R0_distance":
            lines.append(
                f"Резекционные края: R0 — ближайший край резекции "
                f"на расстоянии {case.margin_distance_mm} мм."
            )
        elif case.margin_status == "R1":
            lines.append("Резекционные края: R1 — микроскопический позитивный край.")
        elif case.margin_status == "R2":
            lines.append("Резекционные края: R2 — макроскопически позитивный край.")

        if case.macro_comment:
            lines.append(case.macro_comment)
        return "\n".join(lines)

    def _tnm(self, case: ClinicalCase) -> str:
        lines = [
            "Стадирование по TNM (9-е изд., 2025).",
            "Оценка проведена по данным КТ ОГК и ПЭТ/КТ.",
        ]
        # T
        t_line = f"T — {case.t_code}"
        if case.tumor_size_mm > 0:
            t_line += f" (размер опухоли {case.tumor_size_mm} мм)"
        if case.invasion_t_extension:
            t_line += f"; {case.invasion_t_extension}"
        t_line += "."
        lines.append(t_line)

        # N
        if case.n_code == "N0":
            lines.append("N — N0, метастазов в регионарных лимфоузлах не выявлено.")
        else:
            if case.n_stations:
                stations_str = "; ".join(
                    f"станция {s['station']} ({s['name']}) — {s['size_mm']} мм, SUVmax {s['suv']}"
                    for s in case.n_stations
                )
                lines.append(f"N — {case.n_code}. Поражены: {stations_str}.")
            else:
                lines.append(f"N — {case.n_code}.")

        # M
        if case.m_code == "M0":
            lines.append("M — M0, отдалённых метастазов не выявлено.")
        else:
            lines.append(f"M — {case.m_code}. Локализация метастазов: {', '.join(case.m_sites)}.")

        lines.append(f"Заключение: стадия {case.stage} ({case.t_code}{case.n_code}{case.m_code}).")
        return "\n".join(lines)

    def _molecular(self, case: ClinicalCase) -> str:
        lines = [
            f"Молекулярно-генетическое исследование. Метод: {case.molecular_method}. "
            f"Материал: {case.molecular_material}. "
            f"Чувствительность метода: {case.molecular_sensitivity_pct}% мутантного аллеля."
        ]

        positive, negative = [], []
        for gene, res in case.gene_results.items():
            if res["result"] == "positive":
                vaf_str = f", VAF {res['vaf']}%" if res.get("vaf") else ""
                tier_str = f" [уровень {res['tier']}]" if res.get("tier") else ""
                positive.append(f"{gene} — {res['label']}{vaf_str}{tier_str}")
            else:
                negative.append(gene)

        if positive:
            lines.append("Выявленные мутации: " + "; ".join(positive) + ".")
        if negative:
            lines.append(f"Мутации не выявлены: {', '.join(negative)}.")

        if case.tp53_result == "positive":
            vaf_str = f", VAF {case.tp53_vaf}%" if case.tp53_vaf else ""
            lines.append(f"TP53 — мутация {case.tp53_variant}{vaf_str} (VUS, уровень 3).")
        elif case.tp53_result == "negative":
            lines.append("TP53 — мутации не выявлены.")

        if case.pdl1_tps == 0:
            lines.append("PD-L1 (ИГХ, клон 22C3): отрицательный (TPS <1%).")
        else:
            lines.append(f"PD-L1 (ИГХ, клон 22C3): TPS {case.pdl1_tps}%.")
        return "\n".join(lines)

    def _ecog(self, case: ClinicalCase) -> str:
        # Плюрализация баллов
        if case.ecog == 1:
            ball = "балл"
        elif case.ecog in (2, 3, 4):
            ball = "балла"
        else:
            ball = "баллов"

        lines = [
            f"Функциональный статус по ECOG — {case.ecog} {ball} "
            f"(по Карновскому — {case.karnofsky}%)."
        ]
        scale = next(s for s in self.ecog_scale if s["ecog"] == case.ecog)
        interp = scale["interpretations"][:2]
        lines.append(" ".join(interp))
        return "\n".join(lines)

    def _recommendation(self, case: ClinicalCase) -> str:
        lines = ["Рекомендация по лечению.", case.recommendation_text]
        if case.follow_up_text:
            lines.append(case.follow_up_text)
        return "\n".join(lines)
