"""Лексическая аугментация: синонимы + защита клинических терминов от замены."""

import random
import re


SYNONYM_GROUPS = [
    ["обнаружена", "выявлена", "определена"],
    ["обнаружено", "выявлено", "определено"],
    ["не обнаружена", "не выявлена", "отсутствует"],
    ["не обнаружено", "не выявлено", "отсутствует"],
    ["показана", "рекомендована", "целесообразна"],
    ["показан", "рекомендован", "целесообразен"],
    ["умеренная", "средней степени"],
    ["умеренный", "промежуточный"],
    ["опухоль", "новообразование"],
    ["проведена", "выполнена"],
    ["около", "приблизительно", "порядка"],
]

_SYN_MAP: dict[str, list[str]] = {}
for group in SYNONYM_GROUPS:
    for w in group:
        _SYN_MAP[w.lower()] = group


PROTECTED_TERMS = [
    # Диагнозы
    "Аденокарцинома", "Плоскоклеточный рак", "Крупноклеточный рак", "Мелкоклеточный рак",
    "Аденосквамозный рак", "Немелкоклеточный рак", "Саркоматоидный рак",
    "карциноид", "Нейроэндокринная опухоль",
    # Препараты
    "осимертиниб", "соторасиб", "адаграсиб", "алектиниб", "лорлатиниб", "бригатиниб",
    "пембролизумаб", "энтректиниб", "кризотиниб", "капматиниб", "тепотиниб",
    "селперкатиниб", "пралсетиниб", "ларотректиниб", "афатиниб", "гефитиниб",
    "дабрафениб", "траметиниб", "атезолизумаб", "дурвалумаб", "ниволумаб",
    "цемиплимаб", "амивантамаб", "мобоцертиниб", "трастузумаб дерукстекан",
    "пеметрексед", "карбоплатин", "цисплатин", "паклитаксел", "наб-паклитаксел",
    "этопозид", "октреотид", "ланреотид", "эверолимус", "репотректиниб",
    "бевацизумаб",
    # Гены и термины
    "EGFR", "KRAS", "ALK", "ROS1", "BRAF", "MET", "RET", "NTRK", "HER2", "ERBB2",
    "TP53", "PD-L1", "Ki-67", "TPS", "VAF",
    # Шкалы и стандарты
    "TNM", "IASLC", "ECOG", "PS", "FFPE", "NGS", "FISH", "ИГХ",
    "Карновскому", "LUMYKRAS", "KRAZATI",
    "KEYNOTE-024", "KEYNOTE-189", "KEYNOTE-407", "CheckMate", "FLAURA",
    "PACIFIC", "IMpower", "ADAURA", "CASPIAN", "DESTINY", "PAPILLON",
]


def lex_augment(text: str, rng: random.Random, syn_prob: float = 0.30) -> str:
    """Лексическая аугментация: подмена слов на синонимы с защитой клинических терминов."""
    # 1. Маскируем защищённые фразы
    masks: dict[str, str] = {}
    masked = text
    for i, phrase in enumerate(PROTECTED_TERMS):
        if phrase in masked:
            mask = f"\x00P{i}\x00"
            masks[mask] = phrase
            masked = masked.replace(phrase, mask)

    # 2. Замена синонимов
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

    # 3. Восстанавливаем защищённые
    for mask, phrase in masks.items():
        masked = masked.replace(mask, phrase)
    return masked
