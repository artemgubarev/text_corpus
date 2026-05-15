"""Блок 4: молекулярно-генетический профиль.

Зависит от: diagnosis_id, diagnosis_category, t_code.
Устанавливает: molecular_*, gene_results, actionable_*, tp53_*, pdl1_*.

Ключевой инвариант: actionable драйверы (EGFR, ALK, ROS1, KRAS G12C, BRAF, MET,
RET, NTRK, HER2) ВЗАИМОИСКЛЮЧАЮЩИЕ. Как только зафиксирован первый положительный
драйвер, все последующие принудительно negative.
"""

import random
from core.state import ClinicalCase, weighted_choice, maybe


class MolecularBlock:
    def __init__(self, schemas: dict):
        self.mol = schemas["molecular"]

    def fill(self, case: ClinicalCase, rng: random.Random) -> None:
        if not self._is_applicable(case):
            case.molecular_included = False
            return
        if not maybe(0.65, rng):
            case.molecular_included = False
            return

        case.molecular_included = True
        self._pick_method(case, rng)
        self._pick_drivers(case, rng)
        self._pick_tp53(case, rng)
        self._pick_pdl1(case, rng)

    def _is_applicable(self, case: ClinicalCase) -> bool:
        # Карциноиды и SCLC — без NGS
        if case.diagnosis_category in ("carcinoid", "SCLC"):
            return False
        # Tis/T1mi — обычно не тестируют
        if case.t_code in ("Tis", "T1mi"):
            return False
        return True

    def _pick_method(self, case: ClinicalCase, rng: random.Random) -> None:
        method = rng.choice(self.mol["methods"])
        case.molecular_method = method["text"]
        case.molecular_sensitivity_pct = method["sensitivity_pct"]
        case.molecular_material = rng.choice(self.mol["materials"])

    def _pick_drivers(self, case: ClinicalCase, rng: random.Random) -> None:
        driver_found = False  # глобальный замок: после первого + все остальные negative

        for driver in self.mol["drivers"]:
            gene = driver["gene"]
            negative_var = next(v for v in driver["variants"] if v["result"] == "negative")

            if driver_found:
                chosen = negative_var
            else:
                w_pos = driver["weight_pos_by_diagnosis"].get(
                    case.diagnosis_id,
                    driver["weight_pos_by_diagnosis"].get("default", 0.01),
                )
                if rng.random() < w_pos:
                    pos_variants = [v for v in driver["variants"] if v["result"] == "positive"]
                    chosen = rng.choice(pos_variants)
                    driver_found = True
                else:
                    chosen = negative_var

            entry = {
                "result": chosen["result"],
                "label": chosen["label"],
                "tier": chosen.get("tier"),
                "variant_short": chosen.get("variant_short"),
            }
            if chosen["result"] == "positive":
                lo, hi = self.mol["vaf_range_positive"]
                entry["vaf"] = round(rng.uniform(lo, hi), 1)
                # Если есть препараты — отмечаем как actionable
                if "drugs_first_line" in chosen or "drugs_second_line" in chosen:
                    case.actionable_driver = True
                    case.actionable_gene = gene
                    case.actionable_variant_short = chosen.get("variant_short")
                    case.actionable_tier = chosen.get("tier")
                    case.actionable_drugs_first_line = chosen.get("drugs_first_line", [])
                    case.actionable_drugs_alt = chosen.get("drugs_alt", [])
                    case.actionable_drugs_second_line = chosen.get("drugs_second_line", [])
            case.gene_results[gene] = entry

    def _pick_tp53(self, case: ClinicalCase, rng: random.Random) -> None:
        if not maybe(self.mol["tp53"]["test_probability"], rng):
            case.tp53_result = "not_tested"
            return
        variants = self.mol["tp53"]["variants"]
        chosen = rng.choices(variants, weights=[v["prob"] for v in variants], k=1)[0]
        case.tp53_result = chosen["result"]
        if chosen["result"] == "positive":
            case.tp53_variant = chosen["label"]
            lo, hi = self.mol["vaf_range_positive"]
            case.tp53_vaf = round(rng.uniform(lo, hi), 1)

    def _pick_pdl1(self, case: ClinicalCase, rng: random.Random) -> None:
        case.pdl1_category = weighted_choice(self.mol["pdl1"]["distribution"], rng)
        lo, hi = self.mol["pdl1"]["ranges"][case.pdl1_category]
        case.pdl1_tps = lo if lo == hi else rng.randint(lo, hi)
