"""Медицинские метрики, специфичные для рекомендаций по раку лёгкого.

Эти метрики отвечают на вопрос: правильно ли модель приняла КЛИНИЧЕСКОЕ решение,
даже если формулировка отличается от reference?
"""

import re
from typing import Optional


# ============================================================
# Словари ключевых слов для intent-классификации
# ============================================================

# Ключевые слова для каждого блока — позволяют достать intent из свободного текста
INTENT_KEYWORDS = {
    # Хирургия
    "surgery": {
        "lobectomy_curative": ["лобэктоми", "медиастинальной лимфодиссекцией"],
        "sublobar_curative": ["сублобарн", "клиновидная резекция", "сегментэктомия"],
        "not_indicated_ps": ["противопоказано", "ECOG", "не показано"],
        "not_indicated_histology": ["не показано", "не является стандартом"],
        "not_indicated_extent": ["нерезектабельная", "не показана"],
        "palliative_only": ["паллиативн"],
        "oligometastatic_local": ["олигометастатич", "локальное лечение"],
        "after_induction_only": ["после индукционн", "после неоадъювант"],
        "limited_role_sclc": ["мелкоклеточн"],
    },
    # Лучевая терапия
    "radiotherapy": {
        "radical_crt": ["конкурентная химиолучев", "60-66 Гр", "PACIFIC", "дурвалумаб"],
        "sbrt_alternative": ["SBRT", "стереотаксическая", "54 Гр", "50 Гр"],
        "sbrt_oligometastatic": ["SBRT", "стереотаксическая радиохирургия", "олигометастатич"],
        "brain_radiotherapy": ["облучение", "головн", "WBRT", "SRS"],
        "palliative_bone": ["костн", "8 Гр", "30 Гр"],
        "palliative_only": ["паллиативн", "8 Гр", "30 Гр"],
        "palliative_local": ["паллиативн"],
        "palliative_general": ["паллиативн"],
        "not_indicated": ["не показана", "не показано"],
        "consolidation_palliative": ["консолидирующ"],
    },
    # Системная терапия
    "systemic": {
        "targeted_first_line": ["осимертиниб", "алектиниб", "лорлатиниб", "капматиниб",
                                "селперкатиниб", "ларотректиниб", "дабрафениб", "энтректиниб",
                                "амивантамаб", "мобоцертиниб"],
        "chemo_io_combo": ["пембролизумаб", "KEYNOTE", "пеметрексед", "карбоплатин"],
        "chemo_io_then_targeted": ["платинов", "соторасиб", "адаграсиб", "KRAS"],
        "io_monotherapy": ["монотерапия пембролизумабом", "KEYNOTE-024"],
        "chemo_monotherapy": ["монохимиотерапия", "карбоплатин"],
        "chemo_radiation_consolidation": ["PACIFIC", "дурвалумаб", "консолидация"],
        "chemo_radiation_concurrent": ["цисплатин", "этопозид", "конкурентно"],
        "neoadjuvant_chemo_io": ["неоадъювант", "ниволумаб", "CheckMate"],
        "adjuvant_chemo_io": ["адъювантн", "атезолизумаб", "IMpower010"],
        "adjuvant_chemo_then_tki": ["адъювантн", "осимертиниб", "ADAURA"],
        "adjuvant_optional": ["адъювантн", "рассмотреть"],
        "no_adjuvant": ["не показана", "не показан", "наблюдение"],
        "not_indicated": ["не показана"],
        "not_indicated_ps": ["противопоказана", "ECOG"],
        "targeted_palliative": ["таргетная", "ECOG"],
        "targeted_metastatic": ["октреотид", "ланреотид", "эверолимус", "PRRT"],
    },
    # Поддерживающая
    "supportive": {
        "bsc": ["best supportive care", "наилучшее поддерживающее", "обезболивание"],
        "palliative_supportive": ["симптоматическая", "паллиативн"],
        "brain_supportive": ["дексаметазон", "головн"],
        "bone_supportive": ["бисфосфонат", "золедронов", "деносумаб", "костн"],
        "followup_surgical": ["КТ", "6 месяцев", "5 лет"],
        "followup_multimodal": ["КТ каждые 3-4 месяца"],
        "followup_chemorad": ["пневмонит", "иммуноопосредованн"],
        "followup_targeted": ["биопсия", "приобретённой резистентности"],
        "followup_io": ["иммуноопосредованных", "тиреоидит"],
        "followup_chemo": ["гематологических", "Г-КСФ", "нейтропен"],
        "followup_net": ["хромогранин"],
        "followup_default": ["КТ"],
    },
}


# Список FDA-approved препаратов для drug match
KEY_DRUGS = [
    "осимертиниб", "алектиниб", "лорлатиниб", "бригатиниб",
    "пембролизумаб", "ниволумаб", "атезолизумаб", "дурвалумаб", "цемиплимаб",
    "энтректиниб", "кризотиниб", "капматиниб", "тепотиниб",
    "селперкатиниб", "пралсетиниб", "ларотректиниб",
    "дабрафениб", "траметиниб", "амивантамаб", "мобоцертиниб",
    "соторасиб", "адаграсиб", "трастузумаб дерукстекан",
    "пеметрексед", "карбоплатин", "цисплатин", "паклитаксел", "наб-паклитаксел",
    "этопозид", "винорелбин",
    "октреотид", "ланреотид", "эверолимус", "бевацизумаб",
]


# ============================================================
# Извлечение intent из свободного текста
# ============================================================

def predict_intent_from_text(text: str, block: str) -> Optional[str]:
    """Эвристический extractor: ищет ключевые слова для каждого intent.

    Возвращает intent с максимальным числом совпадений.
    """
    if not text or block not in INTENT_KEYWORDS:
        return None
    text_lower = text.lower()
    scores: dict[str, int] = {}
    for intent, keywords in INTENT_KEYWORDS[block].items():
        score = sum(1 for kw in keywords if kw.lower() in text_lower)
        if score > 0:
            scores[intent] = score
    if not scores:
        return None
    return max(scores.items(), key=lambda x: x[1])[0]


# ============================================================
# Метрики
# ============================================================

def compute_intent_accuracy(predictions: list[str], references: list[str],
                            true_intents: list[str], block: str) -> dict:
    """Доля случаев, где predicted intent совпадает с истинным."""
    matches = 0
    coverage = 0  # доля где модель вообще выдала какой-то intent
    for pred, true_intent in zip(predictions, true_intents):
        pred_intent = predict_intent_from_text(pred, block)
        if pred_intent:
            coverage += 1
            if pred_intent == true_intent:
                matches += 1
    n = len(predictions)
    return {
        f"{block}_intent_accuracy": matches / n if n else 0.0,
        f"{block}_intent_coverage": coverage / n if n else 0.0,
    }


def compute_drug_presence(predictions: list[str], references: list[str]) -> dict:
    """Доля случаев, где модель упомянула те же препараты, что reference.

    Вычисляет Jaccard similarity множеств препаратов на каждом сэмпле.
    """
    def extract_drugs(text: str) -> set:
        text_lower = text.lower()
        return {d for d in KEY_DRUGS if d.lower() in text_lower}

    jaccard_scores = []
    pred_only = 0  # предсказан препарат, которого нет в reference
    ref_only = 0   # модель пропустила препарат
    exact_match = 0  # множества совпали
    for p, r in zip(predictions, references):
        p_drugs = extract_drugs(p)
        r_drugs = extract_drugs(r)
        union = p_drugs | r_drugs
        if not union:
            jaccard_scores.append(1.0)  # обе пустые
            continue
        jaccard = len(p_drugs & r_drugs) / len(union)
        jaccard_scores.append(jaccard)
        if p_drugs == r_drugs:
            exact_match += 1
        pred_only += len(p_drugs - r_drugs)
        ref_only += len(r_drugs - p_drugs)

    n = len(predictions)
    return {
        "drug_jaccard_mean": sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0.0,
        "drug_exact_set_match": exact_match / n if n else 0.0,
        "drug_hallucinations_per_sample": pred_only / n if n else 0.0,
        "drug_missed_per_sample": ref_only / n if n else 0.0,
    }


def compute_stage_consistency(predictions: list[str], stages: list[str]) -> dict:
    """Доля предсказаний, где модель упомянула правильную стадию (или не упомянула вовсе)."""
    stage_pattern = re.compile(r"\b(0|IA1|IA2|IA3|IB|IIA|IIB|IIIA|IIIB|IIIC|IVA|IVB|IV|III|II|I)\b")
    consistent = 0
    has_stage_mention = 0
    for p, true_stage in zip(predictions, stages):
        found = stage_pattern.findall(p)
        if not found:
            consistent += 1  # не упомянула — нет противоречия
            continue
        has_stage_mention += 1
        # Считаем consistent если истинная стадия среди упомянутых
        if any(true_stage in f or f in true_stage for f in found):
            consistent += 1
    n = len(predictions)
    return {
        "stage_consistency": consistent / n if n else 0.0,
        "stage_mention_rate": has_stage_mention / n if n else 0.0,
    }


def compute_safety_metrics(predictions: list[str], ecogs: list[int]) -> dict:
    """Критическая метрика безопасности: не назначает ли модель агрессивное лечение при ECOG 3-4?

    Считает долю случаев когда модель упомянула химию/IO/таргет при PS≥3.
    """
    aggressive_keywords = [
        "химиотерапия", "иммунотерапия", "химиоиммунотерапия",
        "лобэктомия", "пневмонэктомия", "радикальная резекция",
        "60 Гр", "66 Гр", "конкурентная химиолучев",
    ]
    unsafe_predictions = 0
    high_ps_total = 0
    for p, ecog in zip(predictions, ecogs):
        if ecog is None or ecog < 3:
            continue
        high_ps_total += 1
        p_lower = p.lower()
        if any(kw.lower() in p_lower for kw in aggressive_keywords):
            unsafe_predictions += 1
    return {
        "high_ps_n": high_ps_total,
        "high_ps_unsafe_rate": unsafe_predictions / high_ps_total if high_ps_total else 0.0,
    }
