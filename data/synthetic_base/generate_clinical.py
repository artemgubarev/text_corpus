"""
Этап 2 + 3: генераторы для TNM, молекулярного профиля, ECOG/функционального статуса,
сборка полного клинического отчёта и валидация медицинских противоречий с автофиксом.

Использование:
    python generate_clinical.py -n 50 -a 3 -o clinical_reports.jsonl

Если есть morpho-блоки (от generate.py), их можно загрузить через --morph-jsonl
и приклеить к каждому отчёту.
"""

import argparse
import json
import random
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional


# =========================================================================
# УТИЛИТЫ
# =========================================================================

class WeightedChoice:
    def __init__(self, distribution: dict, rng: random.Random):
        keys = list(distribution.keys())
        weights = list(distribution.values())
        total = sum(weights)
        if total <= 0:
            raise ValueError(f"Сумма весов <= 0: {distribution}")
        self.keys = keys
        self.weights = [w / total for w in weights]
        self.rng = rng

    def sample(self) -> str:
        return self.rng.choices(self.keys, weights=self.weights, k=1)[0]


def pick(items: list, rng: random.Random):
    return rng.choice(items)


def maybe(prob: float, rng: random.Random) -> bool:
    return rng.random() < prob


# =========================================================================
# СТРУКТУРЫ
# =========================================================================

@dataclass
class TNMResult:
    text: str
    t_code: str
    n_code: str
    m_code: str
    stage: str
    tumor_size_mm: int
    metastatic_sites: list = field(default_factory=list)
    n_stations: list = field(default_factory=list)


@dataclass
class MolecularResult:
    text: str
    included: bool                          # включён ли блок в отчёт
    actionable_mutation: Optional[str] = None    # ключ для interpretation
    actionable_drugs: list = field(default_factory=list)
    egfr_status: str = "negative"
    kras_status: str = "negative"
    kras_variant: Optional[str] = None
    alk_status: str = "negative"
    ros1_status: str = "negative"
    braf_status: str = "negative"
    other_actionable: list = field(default_factory=list)
    tp53_status: str = "negative"
    pdl1_tps: int = 0
    pdl1_category: str = "negative"


@dataclass
class FunctionalStatusResult:
    text: str
    ecog: int
    karnofsky: int


# =========================================================================
# ГЕНЕРАТОРЫ
# =========================================================================

class TNMGenerator:
    def __init__(self, schema: dict, rng: random.Random):
        self.s = schema["tnm_staging"]
        self.rng = rng
        self.rules = self.s["generation_rules"]
        self._t_chooser = WeightedChoice(self.rules["t_distribution"], rng)
        self._m_chooser = WeightedChoice(self.rules["m_distribution"], rng)

    def _pick_t(self) -> dict:
        code = self._t_chooser.sample()
        return next(t for t in self.s["ts_c_t_categories"] if t["code"] == code)

    def _t_group(self, t_code: str) -> str:
        groups = self.rules["t_category_groups"]
        for g, codes in groups.items():
            if t_code in codes:
                return g
        return "intermediate"

    def _pick_n(self, t_code: str) -> dict:
        group = self._t_group(t_code)
        dist = self.rules["n_distribution_by_t"][group]
        code = WeightedChoice(dist, self.rng).sample()
        return next(n for n in self.s["ts_d_n_categories"] if n["code"] == code)

    def _pick_m(self) -> dict:
        code = self._m_chooser.sample()
        return next(m for m in self.s["ts_e_m_categories"] if m["code"] == code)

    def _stage_key(self, t: str, n: str, m: str) -> str:
        """Вычисление ключа для stage_table."""
        if m != "M0":
            return f"AnyT_AnyN_{m}"
        if n == "N3":
            return f"AnyT_N3_M0"
        return f"{t}_{n}_M0"

    def _generate_tumor_size(self, t_data: dict) -> int:
        rng = t_data.get("size_range_mm")
        if rng is None:
            return 0
        return self.rng.randint(rng[0], rng[1])

    def _format_lymph_node_detail(self, n_data: dict) -> tuple[str, list]:
        """Возвращает текст детализации + список станций."""
        stations = n_data.get("stations", [])
        if not stations:
            return "", []

        if n_data["code"] == "N1":
            n_pick = self.rng.randint(1, 2)
        elif n_data["code"] == "N2a":
            n_pick = 1
        elif n_data["code"] == "N2b":
            n_pick = self.rng.randint(2, 3)
        else:
            n_pick = self.rng.randint(1, 3)

        n_pick = min(n_pick, len(stations))
        picked = self.rng.sample(stations, n_pick)

        size_lo, size_hi = self.rules["lymph_node_size_range_mm"]
        suv_lo, suv_hi = self.rules["lymph_node_suv_range"]

        station_info = {s["station"]: s["name"] for s in self.s["ts_d_lymph_node_stations"]}
        lines = []
        for st in picked:
            name = station_info.get(st, "регионарный")
            size = self.rng.randint(size_lo, size_hi)
            suv = round(self.rng.uniform(suv_lo, suv_hi), 1)
            tpl = pick(self.s["ts_g_lymph_node_detail_format"], self.rng)["template"]
            lines.append("        * " + tpl.format(station=st, name=name, size=size, suv=suv))

        return "\n".join(lines), picked

    def _generate_metastatic_sites(self, m_code: str) -> list:
        if m_code == "M0":
            return []
        dist = self.rules["metastatic_sites_distribution"]
        if m_code == "M1b":
            return [WeightedChoice(dist, self.rng).sample()]
        if m_code == "M1c1":
            site = WeightedChoice(dist, self.rng).sample()
            return [site, site]
        if m_code == "M1c2":
            sites = list(dist.keys())
            n_sites = self.rng.randint(2, 4)
            return self.rng.sample(sites, min(n_sites, len(sites)))
        if m_code == "M1a":
            return ["pleura"] if self.rng.random() < 0.5 else ["contralateral_lung"]
        return []

    def generate(self) -> TNMResult:
        t = self._pick_t()
        n = self._pick_n(t["code"])
        m = self._pick_m()

        size_mm = self._generate_tumor_size(t)
        stage_key = self._stage_key(t["code"], n["code"], m["code"])
        stage = self.s["ts_f_stage_table_tnm9"].get(stage_key, "X")

        lines = []
        # Короткий заголовок — одна строка, без рамок
        lines.append("Стадирование по TNM (9-е изд., 2025).")

        if maybe(self.rules["assessment_basis_probability"], self.rng):
            basis = pick(self.s["ts_b_assessment_basis"], self.rng)["text"]
            lines.append(f"{basis}.")

        # T — собираем в одно предложение
        t_parts = [f"T — {t['code']}"]
        if size_mm > 0:
            t_parts[0] += f" (размер {size_mm} мм)"

        invasion_text = None
        if t["code"] in ("T2a", "T3", "T4") and maybe(self.rules["invasion_feature_probability"], self.rng):
            feats = [f for f in self.s["ts_c_t_invasion_features"]
                     if self._compare_t_level(f["min_t_category"], t["code"]) <= 0]
            if feats:
                feat = pick(feats, self.rng)
                invasion_text = feat["text"].lower()

        if invasion_text:
            lines.append(f"{t_parts[0]}; {invasion_text}.")
        else:
            lines.append(f"{t_parts[0]}.")

        # N — описание
        nodes_detail, picked_stations = "", []
        if n["code"] == "N0":
            lines.append("N — N0, метастазов в регионарных лимфоузлах не выявлено.")
        else:
            n_descriptions = {
                "N1": "метастазы в ипсилатеральных перибронхиальных/корневых лимфоузлах",
                "N2a": "метастаз в одной станции медиастинальных лимфоузлов",
                "N2b": "метастазы в нескольких станциях медиастинальных лимфоузлов",
                "N3": "метастазы в контралатеральных медиастинальных лимфоузлах",
            }
            # Пробуем сделать детализацию (только если есть stations и роли разрешает)
            if (n.get("stations") and
                maybe(self.rules["lymph_node_detail_probability"], self.rng)):
                nodes_detail, picked_stations = self._format_lymph_node_detail_natural(n)
                if nodes_detail:
                    if n["code"] == "N2b":
                        lines.append(f"N — N2b. Поражены множественные станции медиастинальных лимфоузлов: {nodes_detail}.")
                    elif n["code"] == "N2a":
                        lines.append(f"N — N2a. Поражена одна станция медиастинальных лимфоузлов: {nodes_detail}.")
                    elif n["code"] == "N1":
                        lines.append(f"N — N1. Метастазы в ипсилатеральных перибронхиальных/корневых лимфоузлах: {nodes_detail}.")
                    elif n["code"] == "N3":
                        lines.append(f"N — N3. Метастазы в контралатеральных медиастинальных лимфоузлах: {nodes_detail}.")
                else:
                    # На случай если детализация пустая (для N3 без станций в схеме)
                    lines.append(f"N — {n['code']}, {n_descriptions.get(n['code'], '')}.")
            else:
                lines.append(f"N — {n['code']}, {n_descriptions.get(n['code'], '')}.")

        # M
        m_sites = self._generate_metastatic_sites(m["code"])
        if m["code"] == "M0":
            lines.append("M — M0, отдалённых метастазов не выявлено.")
        else:
            site_names = {s["site"]: s["text"] for s in self.s["ts_e_metastatic_sites"]}
            ru_sites = [site_names.get(s, s) for s in m_sites]
            uniq = list(dict.fromkeys(ru_sites))
            lines.append(f"M — {m['code']}. Локализация метастазов: {', '.join(uniq)}.")

        # Итог — естественной фразой
        lines.append(f"Заключение: стадия {stage} ({t['code']}{n['code']}{m['code']}).")

        return TNMResult(
            text="\n".join(lines),
            t_code=t["code"], n_code=n["code"], m_code=m["code"],
            stage=stage, tumor_size_mm=size_mm,
            metastatic_sites=m_sites,
            n_stations=picked_stations,
        )

    def _format_lymph_node_detail_natural(self, n_data: dict) -> tuple[str, list]:
        """Естественный формат — через запятую, без * и табуляции."""
        stations = n_data.get("stations", [])
        if not stations:
            return "", []

        if n_data["code"] == "N1":
            n_pick = self.rng.randint(1, 2)
        elif n_data["code"] == "N2a":
            n_pick = 1
        elif n_data["code"] == "N2b":
            n_pick = self.rng.randint(2, 3)
        else:
            n_pick = self.rng.randint(1, 3)

        n_pick = min(n_pick, len(stations))
        picked = self.rng.sample(stations, n_pick)

        size_lo, size_hi = self.rules["lymph_node_size_range_mm"]
        suv_lo, suv_hi = self.rules["lymph_node_suv_range"]

        station_info = {s["station"]: s["name"] for s in self.s["ts_d_lymph_node_stations"]}
        items = []
        for st in picked:
            name = station_info.get(st, "регионарный")
            size = self.rng.randint(size_lo, size_hi)
            suv = round(self.rng.uniform(suv_lo, suv_hi), 1)
            items.append(f"станция {st} ({name}) — {size} мм, SUVmax {suv}")

        return "; ".join(items), picked

    @staticmethod
    def _compare_t_level(a: str, b: str) -> int:
        order = ["T1a", "T1b", "T1c", "T2a", "T2b", "T3", "T4"]
        try:
            return order.index(a) - order.index(b)
        except ValueError:
            return 0


class MolecularGenerator:
    """Генерирует молекулярный профиль с обеспечением взаимоисключения драйверных мутаций."""

    # Порядок проверки драйверов: первый положительный фиксируется, остальные принудительно negative
    ACTIONABLE_DRIVERS = ["EGFR", "ALK", "ROS1", "KRAS_G12C", "BRAF", "MET_ex14", "RET", "NTRK", "HER2"]

    def __init__(self, schema: dict, rng: random.Random):
        self.s = schema["molecular_profile"]
        self.rng = rng
        self.rules = self.s["generation_rules"]

    def _pick_variant_by_diagnosis(self, variants: list, diagnosis_id: str) -> dict:
        weights = []
        for v in variants:
            w_map = v.get("weight_by_diagnosis", {"default": 0.01})
            w = w_map.get(diagnosis_id, w_map.get("default", 0.01))
            weights.append(max(w, 1e-6))
        return self.rng.choices(variants, weights=weights, k=1)[0]

    def _force_negative(self, gene_variants: list) -> dict:
        return next(v for v in gene_variants if v["result"] == "negative")

    def generate(self, diagnosis_id: str) -> MolecularResult:
        # 1. Включён ли блок вообще?
        if diagnosis_id not in self.rules["applicable_to_diagnosis"]:
            return MolecularResult(text="", included=False)
        if not maybe(self.rules["inclusion_probability"], self.rng):
            return MolecularResult(text="", included=False)

        result = MolecularResult(text="", included=True)

        # 2. Сначала выбираем драйверы — с принудительной взаимоисключаемостью
        driver_found = False

        # EGFR
        egfr = self._pick_variant_by_diagnosis(self.s["mp_d_egfr_variants"], diagnosis_id)
        if egfr["result"] == "positive":
            result.egfr_status = egfr["variant"]
            result.actionable_mutation = f"egfr_{egfr['variant']}"
            result.actionable_drugs = egfr.get("drugs", [])
            driver_found = True

        # ALK
        if not driver_found:
            alk = self._pick_variant_by_diagnosis(self.s["mp_d_alk"], diagnosis_id)
            if alk["result"] == "positive":
                result.alk_status = "positive"
                result.actionable_mutation = "alk_positive"
                result.actionable_drugs = alk.get("drugs", [])
                driver_found = True
        else:
            alk = self._force_negative(self.s["mp_d_alk"])

        # ROS1
        if not driver_found:
            ros1 = self._pick_variant_by_diagnosis(self.s["mp_d_ros1"], diagnosis_id)
            if ros1["result"] == "positive":
                result.ros1_status = "positive"
                result.actionable_mutation = "ros1_positive"
                result.actionable_drugs = ros1.get("drugs", [])
                driver_found = True
        else:
            ros1 = self._force_negative(self.s["mp_d_ros1"])

        # KRAS
        if not driver_found:
            kras = self._pick_variant_by_diagnosis(self.s["mp_d_kras_variants"], diagnosis_id)
            if kras["result"] == "positive":
                result.kras_status = "positive"
                result.kras_variant = kras["variant"]
                if kras["variant"] == "G12C":
                    result.actionable_mutation = "kras_g12c"
                    result.actionable_drugs = kras.get("drugs", [])
                    driver_found = True
        else:
            kras = self._force_negative(self.s["mp_d_kras_variants"])

        # BRAF
        if not driver_found:
            braf = self._pick_variant_by_diagnosis(self.s["mp_d_braf"], diagnosis_id)
            if braf["result"] == "positive":
                result.braf_status = "V600E"
                result.actionable_mutation = "braf_v600e"
                result.actionable_drugs = braf.get("drugs", [])
                driver_found = True
        else:
            braf = self._force_negative(self.s["mp_d_braf"])

        # Other genes (MET/RET/NTRK/HER2)
        other_genes_results = {}
        for gene_data in self.s["mp_d_other_genes"]:
            gene = gene_data["gene"]
            if not driver_found:
                # шанс положительного варианта
                pos_variants = gene_data["positive_variants"]
                roll = self.rng.random()
                acc = 0.0
                chosen_pos = None
                for v in pos_variants:
                    acc += v["prob"]
                    if roll < acc:
                        chosen_pos = v
                        break
                if chosen_pos:
                    other_genes_results[gene] = {"result": "positive", "text": chosen_pos["text"], "tier": chosen_pos["tier"]}
                    result.other_actionable.append(gene)
                    result.actionable_mutation = self._other_gene_to_key(gene, chosen_pos["text"])
                    result.actionable_drugs = chosen_pos.get("drugs", [])
                    driver_found = True
                else:
                    other_genes_results[gene] = {"result": "negative", "text": gene_data["negative_text"], "tier": None}
            else:
                other_genes_results[gene] = {"result": "negative", "text": gene_data["negative_text"], "tier": None}

        # TP53 — независим
        tp53 = self._pick_tp53()
        result.tp53_status = "positive" if tp53["result"] == "positive" else "negative"

        # PD-L1 — независим
        pdl1_cat = WeightedChoice(self.rules["pdl1_distribution"], self.rng).sample()
        pdl1 = next(p for p in self.s["mp_f_pdl1"] if p["result"] == pdl1_cat)
        if pdl1["tps_range"][1] > 0:
            tps = self.rng.randint(*pdl1["tps_range"])
        else:
            tps = 0
        result.pdl1_tps = tps
        result.pdl1_category = pdl1["result"]

        # === Формируем текст ===
        result.text = self._format_text(
            diagnosis_id, egfr, kras, alk, ros1, braf, other_genes_results, tp53, pdl1, tps, result
        )
        return result

    def _other_gene_to_key(self, gene: str, text: str) -> str:
        if "MET" in gene and "Exon 14" in text: return "met_exon14"
        if gene == "RET": return "ret_rearr"
        if gene == "NTRK": return "ntrk_rearr"
        if "HER2" in gene: return "her2_exon20"
        return "other"

    def _pick_tp53(self) -> dict:
        if not maybe(self.rules["tp53_test_probability"], self.rng):
            return self._force_negative_simple(self.s["mp_e_tp53"])
        items = self.s["mp_e_tp53"]
        weights = [v["probability"] for v in items]
        return self.rng.choices(items, weights=weights, k=1)[0]

    @staticmethod
    def _force_negative_simple(items):
        return next(i for i in items if i["result"] == "negative")

    def _format_text(self, diagnosis_id, egfr, kras, alk, ros1, braf,
                     other_genes, tp53, pdl1, tps, result) -> "MolecularResult":
        lines = []
        # Шапка — одно предложение
        method = pick(self.s["mp_b_method"], self.rng)
        material = pick(self.s["mp_c_material"], self.rng)["text"]
        lines.append(
            f"Молекулярно-генетическое исследование. Метод: {method['text']}. "
            f"Материал: {material}. Чувствительность: {method['sensitivity_percent']}% мутантного аллеля."
        )

        def gen_vaf():
            lo, hi = self.rules["vaf_range_for_positive"]
            return self.rng.uniform(lo, hi)

        # Положительные находки — отдельным абзацем, естественной фразой
        positive_findings = []
        if egfr["result"] == "positive":
            positive_findings.append(f"EGFR — {egfr['text']}, VAF {gen_vaf():.1f}%")
        if kras["result"] == "positive":
            positive_findings.append(f"KRAS — {kras['text']}, VAF {gen_vaf():.1f}%")
        if alk["result"] == "positive":
            positive_findings.append(f"ALK — {alk['text']}")
        if ros1["result"] == "positive":
            positive_findings.append(f"ROS1 — {ros1['text']}")
        if braf["result"] == "positive":
            positive_findings.append(f"BRAF — {braf['text']}, VAF {gen_vaf():.1f}%")
        for gene_data in self.s["mp_d_other_genes"]:
            gene = gene_data["gene"]
            g_res = other_genes[gene]
            if g_res["result"] == "positive":
                positive_findings.append(f"{gene} — {g_res['text']}, VAF {gen_vaf():.1f}%")
        if tp53["result"] == "positive":
            # Из "Обнаружена (R273H)" вытаскиваем "R273H" если в скобках
            import re as _re
            m = _re.search(r"\(([^)]+)\)", tp53["text"])
            tp53_variant = m.group(1) if m else tp53["text"]
            positive_findings.append(f"TP53 — мутация {tp53_variant}, VAF {gen_vaf():.1f}%")

        # Перечисление отрицательных — одной строкой
        negative_genes = []
        if egfr["result"] == "negative": negative_genes.append("EGFR")
        if kras["result"] == "negative": negative_genes.append("KRAS")
        if alk["result"] == "negative": negative_genes.append("ALK")
        if ros1["result"] == "negative": negative_genes.append("ROS1")
        if braf["result"] == "negative": negative_genes.append("BRAF")
        for gene_data in self.s["mp_d_other_genes"]:
            if other_genes[gene_data["gene"]]["result"] == "negative":
                negative_genes.append(gene_data["gene"])

        if positive_findings:
            lines.append("Выявленные находки: " + "; ".join(positive_findings) + ".")

        if negative_genes:
            lines.append(f"Мутации не выявлены в генах: {', '.join(negative_genes)}.")

        # TP53 если отрицательный — отдельной фразой только если был протестирован
        if tp53["result"] == "negative":
            lines.append("TP53 — мутации не выявлены.")

        # PD-L1
        if tps == 0:
            lines.append("PD-L1 (ИГХ, клон 22C3): отрицательный (TPS <1%).")
        else:
            lines.append(f"PD-L1 (ИГХ, клон 22C3): TPS {tps}%.")

        # Интерпретация — связным абзацем
        interp_tpls = self.s["mp_h_interpretation_templates"]
        interp_parts = []
        if result.actionable_mutation and result.actionable_mutation in interp_tpls:
            interp_parts.append(interp_tpls[result.actionable_mutation])
        if result.tp53_status == "positive":
            interp_parts.append(interp_tpls["tp53_vus"])
        if tps == 0:
            interp_parts.append(interp_tpls["pdl1_negative"])
        elif tps < 50:
            interp_parts.append(interp_tpls["pdl1_low"].format(tps=tps))
        else:
            interp_parts.append(interp_tpls["pdl1_high"].format(tps=tps))

        if interp_parts:
            lines.append("Клиническая интерпретация. " + " ".join(interp_parts))

        # Рекомендация — одной фразой
        rec_tpls = self.s["mp_i_recommendation_templates"]
        if result.actionable_mutation == "kras_g12c":
            rec_text = rec_tpls["kras_g12c_second_line"]
        elif result.actionable_mutation and "egfr" in result.actionable_mutation:
            drugs = ", ".join(result.actionable_drugs)
            rec_text = rec_tpls["actionable_first_line"].format(drug=drugs)
        elif result.actionable_mutation in ("alk_positive", "ros1_positive", "braf_v600e",
                                            "met_exon14", "ret_rearr", "ntrk_rearr", "her2_exon20"):
            drugs = ", ".join(result.actionable_drugs)
            rec_text = rec_tpls["actionable_first_line"].format(drug=drugs)
        elif tps >= 50:
            rec_text = rec_tpls["io_mono"]
        elif tps >= 1:
            rec_text = rec_tpls["io_chemo_combo"]
        else:
            rec_text = rec_tpls["chemo_only"]
        lines.append(f"Рекомендация. {rec_text}")

        return "\n".join(lines)


class FunctionalStatusGenerator:
    def __init__(self, schema: dict, rng: random.Random):
        self.s = schema["functional_status"]
        self.rng = rng
        self.rules = self.s["generation_rules"]
        # ecog в JSON — строки ("0", "1", ...); распределение тоже строковое
        self._ecog_chooser = WeightedChoice(self.rules["ecog_distribution"], rng)

    def generate(self) -> FunctionalStatusResult:
        ecog = int(self._ecog_chooser.sample())
        scale = next(s for s in self.s["fs_b_ecog_scale"] if s["ecog"] == ecog)
        karn_lo, karn_hi = scale["karnofsky"]
        karn = self.rng.choice([karn_lo, karn_hi])

        # Плюрализация: 0 баллов, 1 балл, 2-4 балла, 5+ баллов
        if ecog == 1:
            ball_word = "балл"
        elif ecog in (2, 3, 4):
            ball_word = "балла"
        else:  # 0 и 5+
            ball_word = "баллов"

        # Собираем одной-двумя фразами
        ecog_phrase = f"Функциональный статус по ECOG — {ecog} {ball_word}"
        if maybe(self.rules["karnofsky_inclusion_probability"], self.rng):
            ecog_phrase += f" (по Карновскому — {karn}%)"
        ecog_phrase += "."

        lines = [ecog_phrase]

        # Интерпретация — связным абзацем, без стрелочек
        if maybe(self.rules["interpretation_probability"], self.rng):
            interp_options = self.s["fs_e_ecog_interpretation_extended"][str(ecog)]
            n_lines = self.rng.randint(
                self.rules["interpretation_lines_count"]["min"],
                min(self.rules["interpretation_lines_count"]["max"], len(interp_options))
            )
            picked_interp = self.rng.sample(interp_options, n_lines)
            interp_text = " ".join(picked_interp)
            lines.append(interp_text)

        return FunctionalStatusResult(
            text="\n".join(lines),
            ecog=ecog, karnofsky=karn,
        )


# =========================================================================
# ВАЛИДАТОР ПРОТИВОРЕЧИЙ
# =========================================================================

@dataclass
class ValidationIssue:
    severity: str        # "warning" | "error"
    block: str           # tnm/molecular/ecog
    description: str


class ClinicalConsistencyValidator:
    """Проверяет медицинские противоречия между блоками."""

    # Допустимые комбинации (M0 → нет метастазов; M1x → есть)
    def validate(self,
                 diagnosis_id: str,
                 grade: Optional[str],
                 tnm: Optional[TNMResult],
                 molecular: Optional[MolecularResult],
                 ecog: Optional[FunctionalStatusResult]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        # === TNM ↔ Diagnosis ===
        if tnm:
            # T-категория и размер должны быть согласованы (это уже обеспечено генератором,
            # но проверим инвариант)
            if tnm.t_code in ("T1a",) and tnm.tumor_size_mm > 10:
                issues.append(ValidationIssue("error", "tnm",
                    f"T1a с размером {tnm.tumor_size_mm} мм (должно быть ≤10 мм)"))
            if tnm.t_code == "T2a" and tnm.tumor_size_mm > 40:
                issues.append(ValidationIssue("error", "tnm",
                    f"T2a с размером {tnm.tumor_size_mm} мм (должно быть ≤40 мм)"))
            if tnm.t_code == "T4" and 0 < tnm.tumor_size_mm < 71:
                # T4 может быть и при <70 мм если есть инвазия — но если только размер, это ошибка
                issues.append(ValidationIssue("warning", "tnm",
                    f"T4 с размером {tnm.tumor_size_mm} мм без явного признака инвазии"))

            # M0 не должно иметь метастазов
            if tnm.m_code == "M0" and tnm.metastatic_sites:
                issues.append(ValidationIssue("error", "tnm",
                    f"M0 но указаны метастазы: {tnm.metastatic_sites}"))
            if tnm.m_code != "M0" and not tnm.metastatic_sites:
                issues.append(ValidationIssue("warning", "tnm",
                    f"{tnm.m_code} без указанных локализаций метастазов"))

        # === TNM ↔ ECOG ===
        if tnm and ecog:
            # Tis/T1mi/T1a с N0 M0 — ранняя стадия, ECOG 3-4 неестественен
            if tnm.stage in ("0", "IA1", "IA2", "IA3") and ecog.ecog >= 3:
                issues.append(ValidationIssue("warning", "ecog",
                    f"Ранняя стадия ({tnm.stage}) с ECOG {ecog.ecog} — клинически маловероятно"))
            # M1c2 (множественные метастазы) с ECOG 0 — тоже редко
            if tnm.m_code == "M1c2" and ecog.ecog == 0:
                issues.append(ValidationIssue("warning", "ecog",
                    f"M1c2 (множественные метастазы) с ECOG 0 — клинически редко"))

        # === Molecular ↔ Diagnosis ===
        if molecular and molecular.included:
            # SCC (D2) с EGFR ex19/L858R — крайне редко, флаг
            if diagnosis_id == "D2" and molecular.egfr_status in ("exon19_del", "L858R"):
                issues.append(ValidationIssue("warning", "molecular",
                    f"Плоскоклеточный рак с EGFR {molecular.egfr_status} — клинически редко"))
            # ALK fusion в SCLC (D6) — практически не встречается
            if diagnosis_id == "D6" and molecular.alk_status == "positive":
                issues.append(ValidationIssue("warning", "molecular",
                    "Мелкоклеточный рак с ALK-fusion — клинически не встречается"))
            # Несколько драйверов одновременно
            drivers_positive = sum([
                molecular.egfr_status != "negative",
                molecular.alk_status == "positive",
                molecular.ros1_status == "positive",
                molecular.kras_status == "positive" and molecular.kras_variant == "G12C",
                molecular.braf_status != "negative",
                bool(molecular.other_actionable),
            ])
            if drivers_positive > 1:
                issues.append(ValidationIssue("error", "molecular",
                    f"Найдено {drivers_positive} actionable драйверов одновременно (взаимоисключающие)"))

        # === Molecular ↔ TNM (косвенно) ===
        # Здесь нет прямого противоречия, только статистическая редкость, пропускаем.

        return issues


# =========================================================================
# СБОРКА КЛИНИЧЕСКОГО ОТЧЁТА
# =========================================================================

@dataclass
class ClinicalReport:
    text: str
    diagnosis_id: str
    diagnosis_name: str
    grade: Optional[str]

    # tnm
    t_code: str
    n_code: str
    m_code: str
    stage: str
    tumor_size_mm: int

    # molecular
    molecular_included: bool
    actionable_mutation: Optional[str]
    egfr_status: str
    kras_status: str
    kras_variant: Optional[str]
    alk_status: str
    ros1_status: str
    braf_status: str
    tp53_status: str
    pdl1_tps: int

    # ecog
    ecog: int
    karnofsky: int

    # валидация
    has_contradictions: bool
    validation_issues: list  # list[dict]
    autofix_attempts: int = 0


class ClinicalReportAssembler:
    DIAGNOSIS_NAMES = {
        "D1": "Аденокарцинома",
        "D2": "Плоскоклеточный рак",
        "D3": "Крупноклеточный рак",
        "D4": "Аденосквамозный рак",
        "D5": "Немелкоклеточный рак БДУ",
        "D6": "Мелкоклеточный рак",
        "D7": "Саркоматоидный рак",
        "D11": "SMARCA4-дефицитная недифференцированная опухоль",
    }
    GRADABLE_DIAGS = {"D1", "D2", "D4"}

    MAX_AUTOFIX_ATTEMPTS = 5

    def __init__(self, schemas_dir: Path, rng: random.Random):
        self.rng = rng
        tnm_schema = self._load(schemas_dir / "tnm_staging.json")
        mol_schema = self._load(schemas_dir / "molecular_profile.json")
        fs_schema = self._load(schemas_dir / "functional_status.json")
        self.tnm_gen = TNMGenerator(tnm_schema, rng)
        self.mol_gen = MolecularGenerator(mol_schema, rng)
        self.fs_gen = FunctionalStatusGenerator(fs_schema, rng)
        self.validator = ClinicalConsistencyValidator()

    @staticmethod
    def _load(path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(
                f"Не найден файл схемы: {path}\n"
                f"Положите его рядом со скриптом или укажите путь через --schemas-dir"
            )
        if path.stat().st_size == 0:
            raise ValueError(
                f"Файл схемы пустой: {path}\n"
                f"Пересохраните файл — он должен содержать валидный JSON, начинающийся с '{{'"
            )
        with open(path, encoding="utf-8-sig") as f:  # utf-8-sig снимает BOM, если он есть
            try:
                return json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Невалидный JSON в файле {path}\n"
                    f"Ошибка: {e}\n"
                    f"Первые 100 символов файла: {path.read_text(encoding='utf-8', errors='replace')[:100]!r}"
                ) from e

    def _pick_diagnosis(self) -> tuple[str, Optional[str]]:
        diag = self.rng.choice(list(self.DIAGNOSIS_NAMES.keys()))
        grade = None
        if diag in self.GRADABLE_DIAGS:
            grade = self.rng.choices(["G1", "G2", "G3"], weights=[0.2, 0.45, 0.35], k=1)[0]
        return diag, grade

    def generate(self) -> ClinicalReport:
        diagnosis_id, grade = self._pick_diagnosis()

        # Генерация с автофиксом: до MAX_AUTOFIX_ATTEMPTS попыток
        autofix_count = 0
        tnm = self.tnm_gen.generate()
        molecular = self.mol_gen.generate(diagnosis_id)
        ecog = self.fs_gen.generate()

        issues = self.validator.validate(diagnosis_id, grade, tnm, molecular, ecog)
        errors = [i for i in issues if i.severity == "error"]

        while errors and autofix_count < self.MAX_AUTOFIX_ATTEMPTS:
            autofix_count += 1
            # Перегенерируем тот блок, в котором ошибка
            blocks_to_fix = {i.block for i in errors}
            if "tnm" in blocks_to_fix:
                tnm = self.tnm_gen.generate()
            if "molecular" in blocks_to_fix:
                molecular = self.mol_gen.generate(diagnosis_id)
            if "ecog" in blocks_to_fix:
                ecog = self.fs_gen.generate()
            issues = self.validator.validate(diagnosis_id, grade, tnm, molecular, ecog)
            errors = [i for i in issues if i.severity == "error"]

        has_contradictions = bool(errors)  # warnings оставляем, errors после autofix считаем

        # Сборка текста
        diag_name = self.DIAGNOSIS_NAMES[diagnosis_id]
        diag_line = f"Диагноз: {diag_name}"
        if grade:
            diag_line += f", {grade}"

        sections = [diag_line, "", tnm.text]
        if molecular.included:
            sections.append("")
            sections.append(molecular.text)
        sections.append("")
        sections.append(ecog.text)

        text = "\n".join(sections)

        return ClinicalReport(
            text=text,
            diagnosis_id=diagnosis_id,
            diagnosis_name=diag_name,
            grade=grade,
            t_code=tnm.t_code, n_code=tnm.n_code, m_code=tnm.m_code,
            stage=tnm.stage, tumor_size_mm=tnm.tumor_size_mm,
            molecular_included=molecular.included,
            actionable_mutation=molecular.actionable_mutation,
            egfr_status=molecular.egfr_status,
            kras_status=molecular.kras_status,
            kras_variant=molecular.kras_variant,
            alk_status=molecular.alk_status,
            ros1_status=molecular.ros1_status,
            braf_status=molecular.braf_status,
            tp53_status=molecular.tp53_status,
            pdl1_tps=molecular.pdl1_tps,
            ecog=ecog.ecog,
            karnofsky=ecog.karnofsky,
            has_contradictions=has_contradictions,
            validation_issues=[asdict(i) for i in issues],
            autofix_attempts=autofix_count,
        )


# =========================================================================
# ЛЕКСИЧЕСКАЯ АУГМЕНТАЦИЯ (упрощённая, как в generate.py)
# =========================================================================

import re

SYNONYM_GROUPS = [
    ["обнаружена", "выявлена", "определена"],
    ["обнаружено", "выявлено", "определено"],
    ["показана", "рекомендована", "целесообразна"],
    ["показан", "рекомендован", "целесообразен"],
    ["умеренная", "средней степени"],
    ["умеренный", "промежуточный"],
    ["выраженная", "значительная", "отчётливая"],
    ["выраженный", "значительный", "отчётливый"],
    ["амбулаторен", "ходячий"],
    ["опухоль", "новообразование"],
]
_SYN_MAP: dict[str, list[str]] = {}
for group in SYNONYM_GROUPS:
    for word in group:
        _SYN_MAP[word.lower()] = group

PROTECTED_TERMS = [
    "Аденокарцинома", "Плоскоклеточный рак", "Крупноклеточный рак", "Мелкоклеточный рак",
    "Аденосквамозный рак", "Немелкоклеточный рак", "Саркоматоидный рак",
    "SMARCA4-дефицитная", "ECOG", "PS", "TNM", "IASLC", "FFPE", "NGS", "FISH",
    "EGFR", "KRAS", "ALK", "ROS1", "BRAF", "MET", "RET", "NTRK", "HER2", "ERBB2",
    "TP53", "PD-L1", "Ki-67", "TPS", "VAF", "LUMYKRAS",
    "осимертиниб", "соторасиб", "адаграсиб", "алектиниб", "лорлатиниб",
    "пембролизумаб", "энтректиниб", "кризотиниб", "капматиниб", "тепотиниб",
    "Karnofsky", "Карновскому",
]


def lex_augment(text: str, rng: random.Random, syn_prob: float = 0.3) -> str:
    """Лексическая аугментация — синонимы + защита терминов."""
    masks: dict[str, str] = {}
    masked = text
    for i, phrase in enumerate(PROTECTED_TERMS):
        if phrase in masked:
            mask = f"\x00P{i}\x00"
            masks[mask] = phrase
            masked = masked.replace(phrase, mask)

    def repl(m):
        word = m.group(0)
        lower = word.lower()
        if lower in _SYN_MAP and rng.random() < syn_prob:
            cands = [w for w in _SYN_MAP[lower] if w.lower() != lower]
            if cands:
                new = rng.choice(cands)
                if word[0].isupper():
                    new = new[0].upper() + new[1:]
                return new
        return word

    masked = re.sub(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)?", repl, masked)
    for mask, phrase in masks.items():
        masked = masked.replace(mask, phrase)
    return masked


# =========================================================================
# CLI
# =========================================================================

def run(n_samples: int, n_variants: int, seed: int, schemas_dir: Path, output: Path,
        show: int = 0, morph_jsonl: Optional[Path] = None) -> None:
    rng = random.Random(seed)
    assembler = ClinicalReportAssembler(schemas_dir, rng)

    morph_lookup: dict[int, str] = {}
    if morph_jsonl and morph_jsonl.exists():
        with open(morph_jsonl, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("variant", 0) == 0:  # только оригиналы
                    morph_lookup[r["base_id"]] = r["text"]

    records: list[dict] = []
    n_contradictions = 0
    n_autofixed = 0

    for base_idx in range(n_samples):
        report = assembler.generate()
        if report.autofix_attempts > 0:
            n_autofixed += 1
        if report.has_contradictions:
            n_contradictions += 1

        # Если есть morpho-блок — клеим перед клиникой
        full_text = report.text
        if morph_lookup and base_idx in morph_lookup:
            full_text = morph_lookup[base_idx] + "\n\n" + full_text

        base_record = {
            "id": f"clin_{base_idx:05d}_v0",
            "base_id": base_idx,
            "variant": 0,
            "is_augmented": False,
            "text": full_text,
            "diagnosis_id": report.diagnosis_id,
            "diagnosis_name": report.diagnosis_name,
            "grade": report.grade,
            "t_code": report.t_code, "n_code": report.n_code, "m_code": report.m_code,
            "stage": report.stage, "tumor_size_mm": report.tumor_size_mm,
            "molecular_included": report.molecular_included,
            "actionable_mutation": report.actionable_mutation,
            "egfr_status": report.egfr_status,
            "kras_status": report.kras_status, "kras_variant": report.kras_variant,
            "alk_status": report.alk_status, "ros1_status": report.ros1_status,
            "braf_status": report.braf_status, "tp53_status": report.tp53_status,
            "pdl1_tps": report.pdl1_tps,
            "ecog": report.ecog, "karnofsky": report.karnofsky,
            "has_contradictions": report.has_contradictions,
            "validation_issues": report.validation_issues,
            "autofix_attempts": report.autofix_attempts,
        }
        records.append(base_record)

        # Аугментации
        for v in range(1, n_variants):
            aug_text = lex_augment(full_text, rng)
            aug_record = dict(base_record)
            aug_record["id"] = f"clin_{base_idx:05d}_v{v}"
            aug_record["variant"] = v
            aug_record["is_augmented"] = True
            aug_record["text"] = aug_text
            records.append(aug_record)

    with open(output, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✓ Записей: {len(records)} ({n_samples} базовых × {n_variants})")
    print(f"  Автофиксов выполнено: {n_autofixed}/{n_samples}")
    print(f"  Записей с warning-противоречиями (после автофикса): {sum(1 for r in records if r['validation_issues'] and not r['is_augmented'])}")
    print(f"  → {output}")

    if show:
        with open(output, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= show:
                    break
                r = json.loads(line)
                print("=" * 70)
                print(f"{r['id']} | dx={r['diagnosis_id']} | stage={r['stage']} | "
                      f"ecog={r['ecog']} | aug={r['is_augmented']}")
                if r["validation_issues"]:
                    print(f"  ⚠ warnings: {[i['description'] for i in r['validation_issues']]}")
                print("=" * 70)
                print(r["text"])
                print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-n", "--num-samples", type=int, default=10)
    p.add_argument("-a", "--augmentations", type=int, default=3)
    p.add_argument("-s", "--seed", type=int, default=42)
    p.add_argument("--schemas-dir", type=str, default=".")
    p.add_argument("-o", "--output", type=str, default="clinical_reports.jsonl")
    p.add_argument("--show", type=int, default=0)
    p.add_argument("--morph-jsonl", type=str, default=None,
                   help="JSONL с morpho-блоками от generate.py (приклеит перед клиникой)")
    args = p.parse_args()

    run(args.num_samples, args.augmentations, args.seed,
        Path(args.schemas_dir), Path(args.output), args.show,
        Path(args.morph_jsonl) if args.morph_jsonl else None)


if __name__ == "__main__":
    main()
