"""Единый ClinicalCase — состояние пациента, через которое общаются все блоки.

Никакой блок не делает выборов параллельно. Все блоки только читают и пишут
в этот объект, в строго определённом порядке (см. builder.py).
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ClinicalCase:
    """Полное состояние одного синтетического случая."""

    # --- Диагноз ---
    diagnosis_id: str = ""
    diagnosis_name: str = ""
    diagnosis_name_lat: str = ""
    diagnosis_category: str = ""        # NSCLC | SCLC | carcinoid
    icd_o_3: str = ""
    grade: Optional[str] = None         # G1 | G2 | G3 | None
    subtype_name: Optional[str] = None
    subtype_name_lat: Optional[str] = None

    # --- Морфология ---
    specimen_text: str = ""
    specimen_is_resection: bool = False
    dominant_pattern: Optional[dict] = None
    secondary_pattern: Optional[dict] = None
    dominant_proportion: int = 0
    secondary_proportion: int = 0
    high_grade_share: int = 0
    lvi_status: str = "negative"        # positive | negative | indeterminate
    pleural_status: str = "PL0"         # PL0..PL3 | not_applicable
    perineural_status: str = "negative"
    margin_status: str = "not_applicable"
    margin_distance_mm: Optional[int] = None
    macro_comment: str = ""

    # --- TNM ---
    t_code: str = ""
    n_code: str = ""
    m_code: str = ""
    tumor_size_mm: int = 0
    n_stations: list = field(default_factory=list)
    m_sites: list = field(default_factory=list)
    stage: str = ""
    stage_group: str = ""
    invasion_t_extension: Optional[str] = None

    # --- Молекулярка ---
    molecular_included: bool = False
    molecular_method: str = ""
    molecular_material: str = ""
    molecular_sensitivity_pct: int = 5
    gene_results: dict = field(default_factory=dict)
    actionable_driver: bool = False
    actionable_gene: Optional[str] = None
    actionable_variant_short: Optional[str] = None
    actionable_tier: Optional[str] = None
    actionable_drugs_first_line: list = field(default_factory=list)
    actionable_drugs_alt: list = field(default_factory=list)
    actionable_drugs_second_line: list = field(default_factory=list)
    tp53_result: str = "not_tested"
    tp53_variant: Optional[str] = None
    tp53_vaf: Optional[float] = None
    pdl1_tps: int = 0
    pdl1_category: str = "<1"

    # --- ECOG ---
    ecog: int = 0
    karnofsky: int = 100

    # --- Рекомендации (4 блока, каждый учится отдельной моделью) ---
    # Хирургия
    surgery_rule_id: Optional[str] = None
    surgery_intent: str = ""
    surgery_text: str = ""
    # Лучевая терапия
    radiotherapy_rule_id: Optional[str] = None
    radiotherapy_intent: str = ""
    radiotherapy_text: str = ""
    # Системная терапия
    systemic_rule_id: Optional[str] = None
    systemic_intent: str = ""
    systemic_text: str = ""
    # Поддерживающая терапия и наблюдение
    supportive_rule_id: Optional[str] = None
    supportive_intent: str = ""
    supportive_text: str = ""

    # --- Вычисляемые флаги для условий (используются блоками рекомендаций) ---
    has_brain_metastases: bool = False
    has_bone_metastases: bool = False

    # --- Финальный текст ---
    full_text: str = ""


# ======================================================================
# Утилиты выбора
# ======================================================================

def weighted_choice(dist: dict, rng: random.Random) -> str:
    """Выбор ключа по распределению (веса нормируются автоматически)."""
    keys = list(dist.keys())
    weights = list(dist.values())
    total = sum(weights)
    if total <= 0:
        raise ValueError(f"Сумма весов <= 0: {dist}")
    return rng.choices(keys, weights=[w / total for w in weights], k=1)[0]


def maybe(prob: float, rng: random.Random) -> bool:
    return rng.random() < prob


# ======================================================================
# Загрузка схем
# ======================================================================

SCHEMA_FILES = [
    "diagnoses", "morphology", "tnm", "molecular", "ecog",
    "surgery", "radiotherapy", "systemic_therapy", "supportive_care",
]


def load_schemas(schemas_dir: Path) -> dict:
    """Загрузить все JSON-схемы. utf-8-sig снимает BOM, если есть."""
    out = {}
    for name in SCHEMA_FILES:
        path = schemas_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Не найден файл схемы: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"Файл схемы пустой: {path}")
        with open(path, encoding="utf-8-sig") as f:
            try:
                out[name] = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Невалидный JSON в {path}: {e}") from e
    return out


# ======================================================================
# Валидатор согласованности
# ======================================================================

def validate_case(case: ClinicalCase) -> list[str]:
    """Sanity-check итогового state. Возвращает список ошибок."""
    errors = []

    # ECOG 4 → во всех 4 блоках должен быть отказ от активного лечения
    if case.ecog == 4:
        if case.surgery_intent not in ("not_indicated_ps",):
            errors.append(f"ECOG 4 но surgery_intent={case.surgery_intent}")
        if case.radiotherapy_intent not in ("palliative_only",):
            errors.append(f"ECOG 4 но radiotherapy_intent={case.radiotherapy_intent}")
        if case.systemic_intent not in ("not_indicated_ps",):
            errors.append(f"ECOG 4 но systemic_intent={case.systemic_intent}")
        if case.supportive_intent != "bsc":
            errors.append(f"ECOG 4 но supportive_intent={case.supportive_intent}")

    # ECOG 3 → системная химия/IO противопоказана
    if case.ecog == 3 and case.systemic_intent in (
            "chemo_io_combo", "neoadjuvant_chemo_io", "chemo_radiation_concurrent",
            "chemo_radiation_consolidation", "adjuvant_chemo_io"):
        errors.append(f"ECOG 3 но systemic_intent={case.systemic_intent}")

    # Несколько actionable драйверов
    actionable_count = sum(
        1 for r in case.gene_results.values()
        if r["result"] == "positive" and r.get("tier") in ("1 (FDA)", "2")
    )
    if actionable_count > 1:
        errors.append(f"Несколько actionable драйверов одновременно: {actionable_count}")

    if case.m_code == "M0" and case.m_sites:
        errors.append(f"M0 но указаны метастазы: {case.m_sites}")

    if case.t_code == "T1a" and case.tumor_size_mm > 10:
        errors.append(f"T1a с размером {case.tumor_size_mm} мм")
    if case.t_code == "T2a" and case.tumor_size_mm > 40 and not case.invasion_t_extension:
        errors.append(f"T2a {case.tumor_size_mm} мм без признака инвазии")

    return errors
