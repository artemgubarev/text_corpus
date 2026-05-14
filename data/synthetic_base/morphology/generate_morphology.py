"""
Генератор синтетических морфологических заключений + лексическая аугментация.

Использование:
    python generate.py -n 100 -o reports.jsonl --augmentations 3

На каждый базовый сэмпл создаётся 1 оригинал + N-1 аугментаций.
"""

import argparse
import json
import random
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional


# =========================================================================
# СИНОНИМИЧЕСКИЙ СЛОВАРЬ ДЛЯ ЛЕКСИЧЕСКОЙ АУГМЕНТАЦИИ
# =========================================================================
# Двунаправленный — каждое слово может быть заменено на любое другое в группе.
# Подобран так, чтобы НЕ менять клинический смысл.

SYNONYM_GROUPS = [
    # Глаголы наблюдения
    ["определяется", "выявляется", "наблюдается", "обнаруживается"],
    ["определяются", "выявляются", "наблюдаются", "обнаруживаются"],
    ["представлен", "состоит из", "сформирован"],
    ["формируют", "образуют", "складываются в"],
    ["формируются", "образуются"],
    ["встречаются", "присутствуют", "имеются"],

    # Степени и интенсивность
    ["выраженный", "значительный", "отчётливый"],
    ["выраженная", "значительная", "отчётливая"],
    ["выраженные", "значительные", "отчётливые"],
    ["умеренный", "средней степени", "промежуточный"],
    ["умеренная", "средней степени"],
    ["скудная", "слабая", "минимальная"],
    ["скудный", "слабый", "минимальный"],

    # Локализация / распространение
    ["очаговый", "локальный", "фокальный"],
    ["очаговые", "локальные", "фокальные"],
    ["диффузный", "распространённый"],
    ["местами", "в отдельных участках", "локально"],
    ["преимущественно", "в основном", "по большей части"],

    # Структуры
    ["опухоль", "новообразование", "опухолевая ткань"],
    ["опухолевые клетки", "клетки опухоли", "опухолевый компонент"],
    ["паттерн", "характер роста", "тип роста"],
    ["комплексы", "структуры", "комплексные образования"],

    # Описательные
    ["прилежащий", "соседний", "окружающий"],
    ["прилежащей", "соседней", "окружающей"],
    ["в среде", "в строме", "среди"],
    ["около", "приблизительно", "порядка", "~"],
    ["вокруг", "по периферии"],
]

# Строим карту: слово -> группа (для замены)
_SYNONYM_MAP: dict[str, list[str]] = {}
for group in SYNONYM_GROUPS:
    for word in group:
        _SYNONYM_MAP[word.lower()] = group


# =========================================================================
# УТИЛИТЫ ВЫБОРА
# =========================================================================

class WeightedChoice:
    """Кэшированный выбор из {key: weight}."""

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


def pluralize_ru(n: int, forms: tuple[str, str, str]) -> str:
    """Русская плюрализация: (форма_1, форма_2-4, форма_5+).

    pluralize_ru(1, ("митоз", "митоза", "митозов")) -> 'митоз'
    pluralize_ru(3, ...) -> 'митоза'
    pluralize_ru(11, ...) -> 'митозов'
    """
    n = abs(n) % 100
    n1 = n % 10
    if 10 < n < 20:
        return forms[2]
    if 1 < n1 < 5:
        return forms[1]
    if n1 == 1:
        return forms[0]
    return forms[2]


def maybe(prob: float, rng: random.Random) -> bool:
    return rng.random() < prob


# =========================================================================
# СТРУКТУРА БАЗОВОГО СЭМПЛА
# =========================================================================

@dataclass
class BaseSample:
    """Базовый morpho-отчёт ДО аугментации, с зафиксированными метаданными."""
    # Метаданные (для разметки/обучения)
    diagnosis_id: str
    grade: Optional[str]
    specimen_category: str        # biopsy / cytology_plus / resection
    specimen_name: str            # 'щипцовая биопсия' и т.п.
    dominant_pattern: Optional[str] = None
    secondary_pattern: Optional[str] = None
    mitotic_count: Optional[int] = None
    necrosis_extent: Optional[str] = None
    ki67_percent: Optional[int] = None
    til_percent: Optional[int] = None

    # Тексты по блокам — для аугментации
    block_specimen: str = ""
    block_architecture: str = ""
    block_cytology: str = ""
    block_mitosis_necrosis: str = ""
    block_stroma: str = ""

    @property
    def full_text(self) -> str:
        blocks = [
            self.block_specimen,
            self.block_architecture,
            self.block_cytology,
            self.block_mitosis_necrosis,
            self.block_stroma,
        ]
        return "\n\n".join(b for b in blocks if b.strip())


# =========================================================================
# ГЕНЕРАЦИЯ БЛОКОВ
# =========================================================================

class MorphReportGenerator:
    """Генератор синтетических морфологических отчётов."""

    # Диагнозы, для которых имеет смысл блок морфологии (положительные)
    DIAGNOSES = ["D1", "D2", "D6", "D8", "D9"]
    DIAGNOSIS_NAMES = {
        "D1": "Аденокарцинома",
        "D2": "Плоскоклеточный рак",
        "D6": "Мелкоклеточный рак",
        "D8": "Нейроэндокринная опухоль G1 (типичный карциноид)",
        "D9": "Нейроэндокринная опухоль G2 (атипичный карциноид)",
    }
    GRADABLE = {"D1": True, "D2": True, "D6": False, "D8": False, "D9": False}

    def __init__(self, schemas_dir: Path, rng: random.Random):
        self.rng = rng
        self.so = self._load(schemas_dir / "specimen_origin.json")["specimen_origin"]
        self.ta = self._load(schemas_dir / "tumor_architecture.json")["tumor_architecture"]
        self.cy = self._load(schemas_dir / "cytology.json")["cytology"]
        self.mn = self._load(schemas_dir / "mitosis_necrosis.json")["mitosis_necrosis"]
        self.si = self._load(schemas_dir / "stroma_inflammation.json")["stroma_inflammation"]

    @staticmethod
    def _load(path: Path) -> dict:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    # --- выбор диагноза и grade (consistency-критично) ---

    def _pick_diagnosis_and_grade(self) -> tuple[str, Optional[str]]:
        diag = self.rng.choice(self.DIAGNOSES)
        if not self.GRADABLE[diag]:
            return diag, None
        # Grade распределение для аденокарциномы / SCC
        grade = self.rng.choices(["G1", "G2", "G3"], weights=[0.20, 0.45, 0.35], k=1)[0]
        return diag, grade

    # --- блок 1: specimen origin ---

    def _gen_specimen(self) -> tuple[str, str, str]:
        """Возвращает (текст, category, specimen_name)."""
        rules = self.so["generation_rules"]
        cat = WeightedChoice(rules["specimen_type_distribution"], self.rng).sample()
        candidates = [s for s in self.so["so_a_specimen_type"] if s["category"] == cat]
        spec = pick(candidates, self.rng)

        parts = [spec["text"]]
        location_bits = []

        if maybe(rules["lobe_probability"], self.rng):
            lobe = pick(self.so["so_c_lobe"], self.rng)
            side_options = [s for s in self.so["so_b_side"] if s["side"] in lobe["sides"]]
            if side_options and maybe(rules["side_probability"], self.rng):
                side = pick(side_options, self.rng)
                location_bits.append(f"{lobe['text']} {side['text']}")
            else:
                location_bits.append(lobe["text"])

        head = f"Тип материала: {parts[0]}"
        if location_bits:
            head += ", " + ", ".join(location_bits)

        lines = [head]
        # diagnosis line добавим позже после выбора диагноза
        if maybe(rules["clinical_context_probability"], self.rng):
            lines.append(pick(self.so["so_g_clinical_context"], self.rng)["text"])

        return "\n".join(lines), cat, spec["name"]

    # --- блок 2: tumor architecture (с прокидыванием диагноза) ---

    # Короткие имена паттернов для использования в шаблонах "Преобладает X (~Y%)"
    PATTERN_SHORT_NAMES = {
        "lepidic": "лепидический рост",
        "acinar": "ацинарный паттерн",
        "papillary": "папиллярный паттерн",
        "micropapillary": "микропапиллярный паттерн",
        "solid": "солидный паттерн",
        "cribriform": "криброзный паттерн",
        "complex_glandular": "сложный железистый паттерн",
        "squamous_keratinizing": "ороговевающий плоскоклеточный паттерн",
        "squamous_nonkeratinizing": "неороговевающий плоскоклеточный паттерн",
        "basaloid": "базалоидный паттерн",
        "small_cell": "мелкоклеточный паттерн",
        "neuroendocrine_organoid": "органоидный нейроэндокринный паттерн",
    }

    def _gen_architecture(self, diagnosis_id: str, grade: Optional[str]) -> tuple[str, Optional[str], Optional[str]]:
        rules = self.ta["generation_rules"]
        allowed = rules["pattern_by_diagnosis"].get(diagnosis_id, [])
        if not allowed:
            return "", None, None

        # Для G3 — приоритет high-grade паттернам
        if grade == "G3":
            high = [p for p in allowed if p in rules["high_grade_patterns"]]
            dominant = pick(high if high else allowed, self.rng)
        else:
            dominant = pick(allowed, self.rng)

        dom_pct = self.rng.randint(*rules["dominant_pattern_proportion_range"])
        dom_pct = int(round(dom_pct / 10) * 10)

        lines = []
        if maybe(rules["general_intro_probability"], self.rng):
            header = pick(self.ta["ta_a_section_header"], self.rng)["text"]
            intro = pick(self.ta["ta_b_general_intro"], self.rng)["text"]
            lines.append(f"{header}\n{intro}.")

        # Шаблон ожидает КОРОТКОЕ имя паттерна, не полное описание
        dom_short = self.PATTERN_SHORT_NAMES.get(dominant, dominant)
        dom_tpl = pick(self.ta["ta_d_pattern_proportion"], self.rng)["text"]
        dom_phrase = dom_tpl.format(pattern=dom_short, pct=dom_pct)
        lines.append(dom_phrase[0].upper() + dom_phrase[1:] + ".")

        # Описание паттерна — отдельным предложением
        dom_desc = pick(self.ta["ta_c_pattern_descriptions"][dominant], self.rng)["text"]
        lines.append(dom_desc[0].upper() + dom_desc[1:] + ".")

        secondary = None
        if maybe(rules["secondary_pattern_probability"], self.rng) and len(allowed) > 1:
            sec_candidates = [p for p in allowed if p != dominant]
            secondary = pick(sec_candidates, self.rng)
            sec_short = self.PATTERN_SHORT_NAMES.get(secondary, secondary)
            sec_pct = self.rng.randint(*rules["secondary_pattern_proportion_range"])
            sec_pct = int(round(sec_pct / 10) * 10)
            sec_tpl = pick(self.ta["ta_e_minor_component"], self.rng)["text"]
            sec_phrase = sec_tpl.format(pattern=sec_short, pct=sec_pct)
            lines.append(sec_phrase[0].upper() + sec_phrase[1:] + ".")

        return "\n".join(lines), dominant, secondary

    # --- блок 3: cytology (с привязкой к grade) ---

    def _gen_cytology(self, diagnosis_id: str, grade: Optional[str]) -> str:
        rules = self.cy["generation_rules"]
        lines = []

        if maybe(rules["section_intro_probability"], self.rng):
            lines.append(pick(self.cy["cy_a_section_intro"], self.rng)["text"])

        # Размер клеток — зависит от диагноза
        if diagnosis_id == "D6":
            cell_size = next(b for b in self.cy["cy_b_cell_size"] if b["size_class"] == "small")
        else:
            cell_size = pick([b for b in self.cy["cy_b_cell_size"] if b["size_class"] != "small"], self.rng)
        cell_shape = pick(self.cy["cy_c_cell_shape"], self.rng)
        lines.append(f"Клетки {cell_size['text']}, {cell_shape['text']}.")

        # Хроматин — для D6/D8/D9 пушим к salt-and-pepper
        if diagnosis_id in ("D6", "D8", "D9") and maybe(0.7, self.rng):
            chromatin = next(c for c in self.cy["cy_f_chromatin"] if c.get("marker_for") == "neuroendocrine")
        else:
            chromatin = pick([c for c in self.cy["cy_f_chromatin"] if "marker_for" not in c], self.rng)

        # Ядрышки и NC ratio — по grade (если есть)
        if grade and grade in rules["nucleoli_by_grade"]:
            nucleoli_id = pick(rules["nucleoli_by_grade"][grade], self.rng)
            nucleoli = next(n for n in self.cy["cy_g_nucleoli"] if n["id"] == nucleoli_id)
        else:
            nucleoli = pick(self.cy["cy_g_nucleoli"], self.rng)

        if grade and grade in rules["nc_ratio_by_grade"]:
            nc_ratios = rules["nc_ratio_by_grade"][grade]
            nc = pick([h for h in self.cy["cy_h_nc_ratio"] if h["ratio"] in nc_ratios], self.rng)
        else:
            nc = pick(self.cy["cy_h_nc_ratio"], self.rng)

        lines.append(f"{chromatin['text']}, {nucleoli['text']}.")
        lines.append(f"{nc['text'][0].upper() + nc['text'][1:]}.")

        # Полиморфизм — по grade
        if grade and grade in rules["pleomorphism_by_grade"]:
            pleo_grades = rules["pleomorphism_by_grade"][grade]
            pleo = pick([p for p in self.cy["cy_k_pleomorphism"] if p["grade"] in pleo_grades], self.rng)
        else:
            pleo = pick(self.cy["cy_k_pleomorphism"], self.rng)
        lines.append(f"{pleo['text']}.")

        return "\n".join(lines)

    # --- блок 4: mitoses + Ki-67 + necrosis (по диагнозу+grade) ---

    def _gen_mitosis_necrosis(self, diagnosis_id: str, grade: Optional[str]) -> tuple[str, int, str, Optional[int]]:
        rules = self.mn["generation_rules"]
        lines = []

        if maybe(rules["section_header_probability"], self.rng):
            lines.append(pick(self.mn["mn_a_section_header"], self.rng)["text"])

        # Митозы
        mitosis_key = f"{diagnosis_id}_{grade}" if grade else diagnosis_id
        mitosis_range = rules["mitosis_ranges_by_diagnosis"].get(
            mitosis_key,
            rules["mitosis_ranges_by_diagnosis"].get(diagnosis_id, rules["mitosis_ranges_by_diagnosis"]["default"])
        )
        count = self.rng.randint(*mitosis_range)
        mit_tpl = pick(self.mn["mn_b_mitotic_count_format"], self.rng)["template"]
        # Плюрализация: 1 митоз / 3 митоза / 12 митозов
        mitosis_word = pluralize_ru(count, ("митоз", "митоза", "митозов"))
        mit_text = mit_tpl.format(count=count)
        # Заменяем "митозов" в шаблоне на правильную форму
        mit_text = mit_text.replace("митозов", mitosis_word)
        lines.append(mit_text + ".")

        # Атипичные митозы — вероятность по grade
        if grade and grade in rules["atypical_mitoses_probability_by_grade"]:
            if maybe(rules["atypical_mitoses_probability_by_grade"][grade], self.rng):
                # положительный вариант (G2/G3 атипичные = "встречаются")
                atyp_options = self.mn["mn_c_atypical_mitoses"][1:]
            else:
                atyp_options = [self.mn["mn_c_atypical_mitoses"][0]]
            lines.append(pick(atyp_options, self.rng)["text"].capitalize() + ".")

        # Некроз
        necrosis_key = mitosis_key
        nec_dist = rules["necrosis_distribution_by_diagnosis"].get(
            necrosis_key,
            rules["necrosis_distribution_by_diagnosis"].get(diagnosis_id, rules["necrosis_distribution_by_diagnosis"]["default"])
        )
        nec_extent = WeightedChoice(nec_dist, self.rng).sample()
        nec_options = [e for e in self.mn["mn_e_necrosis_extent"] if e["extent"] == nec_extent]
        nec = pick(nec_options, self.rng)
        nec_text = nec["text"]
        if nec.get("uses_percent"):
            pct = self.rng.randint(10, 35)
            nec_text = nec_text.format(percent=pct)
        lines.append(nec_text + ".")

        # Ki-67
        ki67 = None
        if maybe(rules["ki67_inclusion_probability"], self.rng):
            ki67_range = rules["ki67_ranges_by_diagnosis_grade"].get(
                mitosis_key,
                rules["ki67_ranges_by_diagnosis_grade"].get(diagnosis_id, rules["ki67_ranges_by_diagnosis_grade"]["default"])
            )
            ki67 = self.rng.randint(*ki67_range)
            ki67_tpl = pick(self.mn["mn_g_ki67_format"], self.rng)["template"]
            lines.append(ki67_tpl.format(percent=ki67) + ".")

        return "\n".join(lines), count, nec_extent, ki67

    # --- блок 5: stroma + TIL ---

    def _gen_stroma(self, diagnosis_id: str) -> tuple[str, Optional[int]]:
        rules = self.si["generation_rules"]
        lines = []

        if maybe(rules["section_header_probability"], self.rng):
            lines.append(pick(self.si["si_a_section_header"], self.rng)["text"])

        allowed_stroma = rules["stroma_type_by_diagnosis"].get(diagnosis_id, ["fibrous", "minimal"])
        stroma = pick([s for s in self.si["si_b_stroma_type"] if s["stroma_type"] in allowed_stroma], self.rng)
        lines.append(stroma["text"] + ".")

        infl_density = WeightedChoice(rules["inflammation_density_distribution"], self.rng).sample()
        infl = pick([c for c in self.si["si_c_inflammation_density"] if c["density"] == infl_density], self.rng)
        lines.append(infl["text"] + ".")

        til_pct = None
        if maybe(rules["til_inclusion_probability"], self.rng):
            til_range = rules["til_range_by_diagnosis"].get(diagnosis_id, rules["til_range_by_diagnosis"]["default"])
            til_pct = self.rng.randint(*til_range)
            # выбираем шаблон по категории
            cats = rules["til_category_thresholds"]
            if til_pct < cats["absent"][1]:
                til_tpl = self.si["si_f_til_assessment"][0]["template"]
            elif til_pct < cats["low"][1]:
                til_tpl = self.si["si_f_til_assessment"][1]["template"]
            elif til_pct < cats["moderate"][1]:
                til_tpl = self.si["si_f_til_assessment"][2]["template"]
            else:
                til_tpl = self.si["si_f_til_assessment"][3]["template"]
            lines.append(til_tpl.format(percent=til_pct) + ".")

        return "\n".join(lines), til_pct

    # --- сборка ---

    def generate(self) -> BaseSample:
        diagnosis_id, grade = self._pick_diagnosis_and_grade()
        specimen_text, spec_cat, spec_name = self._gen_specimen()

        # Дополним specimen строкой диагноза
        diag_name = self.DIAGNOSIS_NAMES[diagnosis_id]
        diag_line = f"Диагноз: {diag_name}"
        if grade:
            diag_line += f", {grade}"
        specimen_text = specimen_text + "\n" + diag_line

        arch_text, dom_pat, sec_pat = self._gen_architecture(diagnosis_id, grade)
        cyto_text = self._gen_cytology(diagnosis_id, grade)
        mn_text, mitoses, nec_extent, ki67 = self._gen_mitosis_necrosis(diagnosis_id, grade)
        stroma_text, til_pct = self._gen_stroma(diagnosis_id)

        return BaseSample(
            diagnosis_id=diagnosis_id,
            grade=grade,
            specimen_category=spec_cat,
            specimen_name=spec_name,
            dominant_pattern=dom_pat,
            secondary_pattern=sec_pat,
            mitotic_count=mitoses,
            necrosis_extent=nec_extent,
            ki67_percent=ki67,
            til_percent=til_pct,
            block_specimen=specimen_text,
            block_architecture=arch_text,
            block_cytology=cyto_text,
            block_mitosis_necrosis=mn_text,
            block_stroma=stroma_text,
        )


# =========================================================================
# ЛЕКСИЧЕСКАЯ АУГМЕНТАЦИЯ
# =========================================================================

class LexicalAugmenter:
    """Лексическая аугментация: синонимы + перетасовка предложений."""

    def __init__(self, rng: random.Random,
                 synonym_prob: float = 0.35,
                 shuffle_prob: float = 0.50):
        self.rng = rng
        self.synonym_prob = synonym_prob
        self.shuffle_prob = shuffle_prob

    # Защищённые фразы (целиком, без замены отдельных слов внутри).
    # Это всё, что нужно сохранить нетронутым: названия диагнозов, технические термины.
    PROTECTED_PHRASES = [
        "Аденокарцинома", "Плоскоклеточный рак", "Мелкоклеточный рак",
        "Нейроэндокринная опухоль", "типичный карциноид", "атипичный карциноид",
        "Крупноклеточный рак", "Аденосквамозный рак", "Саркоматоидный рак",
        "Лимфоэпителиальная карцинома", "SMARCA4-дефицитная",
        "Тип материала", "Клинические данные", "Диагноз",
        "Митотическая активность", "Митотический индекс", "Митозы",
        "Ki-67", "MIB-1", "TIL",
    ]

    def _replace_synonyms(self, text: str) -> str:
        """Заменяет слова на синонимы с вероятностью synonym_prob.

        Сначала маскирует защищённые фразы маркерами, чтобы не разломать
        клинически важные термины (названия диагнозов и т.п.).
        """
        # 1. Маскируем защищённые фразы
        masks: dict[str, str] = {}
        masked_text = text
        for i, phrase in enumerate(self.PROTECTED_PHRASES):
            if phrase in masked_text:
                mask = f"\x00PROT{i}\x00"
                masks[mask] = phrase
                masked_text = masked_text.replace(phrase, mask)

        # 2. Замена синонимов
        def replace_match(m: re.Match) -> str:
            word = m.group(0)
            lower = word.lower()
            if lower in _SYNONYM_MAP and self.rng.random() < self.synonym_prob:
                group = _SYNONYM_MAP[lower]
                candidates = [w for w in group if w.lower() != lower]
                if not candidates:
                    return word
                replacement = self.rng.choice(candidates)
                if word[0].isupper():
                    replacement = replacement[0].upper() + replacement[1:]
                return replacement
            return word

        masked_text = re.sub(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)?", replace_match, masked_text)

        # 3. Восстанавливаем защищённые фразы
        for mask, phrase in masks.items():
            masked_text = masked_text.replace(mask, phrase)

        return masked_text

    def _shuffle_sentences(self, block: str) -> str:
        """Перетасовка предложений ВНУТРИ блока.

        Первое предложение (часто заголовок секции) не трогаем,
        чтобы не получить 'TIL: 30%. Стромальная реакция:'.
        """
        lines = block.split("\n")
        if len(lines) <= 2:
            return block

        # Если первая строка — заголовок секции (заканчивается ':'), фиксируем её
        if lines[0].rstrip().endswith(":"):
            header, body = lines[0], lines[1:]
        else:
            header, body = None, lines

        # Перетасовываем только если в блоке нет логической последовательности
        # (для блока 'architecture' секции dom→sec нельзя ломать порядок, оставляем как есть)
        # Применяем shuffle только если в блоке >= 3 предложений
        if len(body) >= 3 and self.rng.random() < self.shuffle_prob:
            body = body.copy()
            self.rng.shuffle(body)

        return "\n".join([header] + body) if header else "\n".join(body)

    def augment_block(self, block: str, allow_shuffle: bool = True) -> str:
        """Применить аугментацию к одному блоку."""
        if not block.strip():
            return block
        out = self._replace_synonyms(block)
        if allow_shuffle:
            out = self._shuffle_sentences(out)
        return out

    def augment_sample(self, sample: BaseSample) -> str:
        """Возвращает аугментированную версию полного отчёта."""
        # Для блока architecture не перетасовываем — там важен порядок (dominant -> secondary)
        b_specimen = self.augment_block(sample.block_specimen, allow_shuffle=False)
        b_arch = self.augment_block(sample.block_architecture, allow_shuffle=False)
        b_cyto = self.augment_block(sample.block_cytology, allow_shuffle=True)
        b_mn = self.augment_block(sample.block_mitosis_necrosis, allow_shuffle=True)
        b_stroma = self.augment_block(sample.block_stroma, allow_shuffle=True)
        return "\n\n".join(b for b in [b_specimen, b_arch, b_cyto, b_mn, b_stroma] if b.strip())


# =========================================================================
# JSONL WRITER
# =========================================================================

def write_jsonl(samples: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


# =========================================================================
# ОРКЕСТРАЦИЯ
# =========================================================================

def run(n_samples: int, n_variants: int, seed: int, schemas_dir: Path, output: Path) -> None:
    rng = random.Random(seed)
    generator = MorphReportGenerator(schemas_dir, rng)
    augmenter = LexicalAugmenter(rng)

    out_records: list[dict] = []

    for base_idx in range(n_samples):
        base = generator.generate()

        # Метаданные общие для оригинала и его аугментаций
        meta = {
            "diagnosis_id": base.diagnosis_id,
            "diagnosis_name": generator.DIAGNOSIS_NAMES[base.diagnosis_id],
            "grade": base.grade,
            "specimen_category": base.specimen_category,
            "specimen_name": base.specimen_name,
            "dominant_pattern": base.dominant_pattern,
            "secondary_pattern": base.secondary_pattern,
            "mitotic_count": base.mitotic_count,
            "necrosis_extent": base.necrosis_extent,
            "ki67_percent": base.ki67_percent,
            "til_percent": base.til_percent,
        }

        # Оригинал
        out_records.append({
            "id": f"sample_{base_idx:05d}_v0",
            "base_id": base_idx,
            "variant": 0,
            "is_augmented": False,
            "text": base.full_text,
            **meta,
        })

        # Аугментации
        for v in range(1, n_variants):
            aug_text = augmenter.augment_sample(base)
            out_records.append({
                "id": f"sample_{base_idx:05d}_v{v}",
                "base_id": base_idx,
                "variant": v,
                "is_augmented": True,
                "text": aug_text,
                **meta,
            })

    write_jsonl(out_records, output)
    print(f"✓ Сгенерировано {len(out_records)} записей "
          f"({n_samples} базовых × {n_variants} вариантов) → {output}")


def main():
    parser = argparse.ArgumentParser(description="Генератор морфологических отчётов с аугментацией")
    parser.add_argument("-n", "--num-samples", type=int, default=10,
                        help="Сколько базовых сэмплов сгенерировать")
    parser.add_argument("-a", "--augmentations", type=int, default=3,
                        help="Сколько вариантов на сэмпл (включая оригинал, 1=без аугментации)")
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("--schemas-dir", type=str, default=".",
                        help="Папка со схемами JSON")
    parser.add_argument("-o", "--output", type=str, default="morph_reports.jsonl")
    parser.add_argument("--show", type=int, default=0,
                        help="Сколько примеров вывести в stdout")
    args = parser.parse_args()

    schemas_dir = Path(args.schemas_dir)
    output = Path(args.output)

    run(args.num_samples, args.augmentations, args.seed, schemas_dir, output)

    if args.show:
        with open(output, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= args.show:
                    break
                rec = json.loads(line)
                print("=" * 70)
                print(f"{rec['id']}  |  diagnosis={rec['diagnosis_id']}  "
                      f"grade={rec['grade']}  augmented={rec['is_augmented']}")
                print("=" * 70)
                print(rec["text"])
                print()


if __name__ == "__main__":
    main()
