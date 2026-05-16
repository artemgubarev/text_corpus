"""Шаг 01: Подготовка датасета для SFT (4 specialized моделей).

Входные данные: JSONL от unified_v2/generate_unified.py
Выход: 4 датасета (surgery, radiotherapy, systemic, supportive),
       каждый с разбивкой train/val/test.

Использование:
    python 01_prepare_dataset.py
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.data_utils import load_config, load_jsonl, save_jsonl, set_seed, ensure_dir


# ============================================================
# Извлечение клинического контекста (input для модели)
# ============================================================

def extract_clinical_input(record: dict) -> str:
    """Всё что было до 'Рекомендация по лечению.' — input для модели."""
    text = record["text"]
    marker = "Рекомендация по лечению."
    idx = text.find(marker)
    if idx == -1:
        return text
    return text[:idx].rstrip()


def extract_block(record: dict, block_name: str) -> str:
    """Извлечь конкретный блок рекомендации."""
    field_map = {
        "surgery": "surgery_text",
        "radiotherapy": "radiotherapy_text",
        "systemic": "systemic_text",
        "supportive": "supportive_text",
    }
    return record.get(field_map[block_name], "")


# ============================================================
# Промпты (Qwen2.5 chat format через apply_chat_template)
# ============================================================

SYSTEM_PROMPT_BLOCK = (
    "Ты — клинический онколог-эксперт по раку лёгкого. На основании представленного "
    "клинико-морфологического заключения сформулируй раздел рекомендации по {block_label}. "
    "Будь конкретен — указывай препараты, дозы, режимы при необходимости. "
    "Если данный вид лечения не показан — обоснуй это."
)

BLOCK_LABELS = {
    "surgery": "хирургическому лечению",
    "radiotherapy": "лучевой терапии",
    "systemic": "системной терапии",
    "supportive": "поддерживающей терапии и наблюдению",
}


def build_block_sample(record: dict, block_name: str) -> dict:
    """Сэмпл для одного блока: clinical_input → конкретный блок рекомендации."""
    clinical_input = extract_clinical_input(record)
    target = extract_block(record, block_name)

    system = SYSTEM_PROMPT_BLOCK.format(block_label=BLOCK_LABELS[block_name])
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": clinical_input},
        {"role": "assistant", "content": target},
    ]
    return {
        "id": f"{record['id']}__{block_name}",
        "base_id": record["base_id"],
        "block": block_name,
        "messages": messages,
        "input": clinical_input,
        "target": target,
        # Метаданные для метрик
        "diagnosis_id": record.get("diagnosis_id"),
        "stage": record.get("stage"),
        "stage_group": record.get("stage_group"),
        "ecog": record.get("ecog"),
        "block_intent": record.get(f"{block_name}_intent"),
        "actionable_gene": record.get("actionable_gene"),
        "actionable_variant_short": record.get("actionable_variant_short"),
        "has_brain_metastases": record.get("has_brain_metastases"),
        "has_bone_metastases": record.get("has_bone_metastases"),
    }


# ============================================================
# Split (stratified по base_id чтобы аугментации не разделялись)
# ============================================================

def stratified_split(records: list[dict], cfg: dict) -> tuple[list, list, list]:
    """Разбивает по base_id (оригинал и аугментации не разделяются между split'ами).

    Стратификация по (stage_group, ecog_bin).
    """
    set_seed(cfg["data"]["seed"])
    by_base: dict[int, list[dict]] = {}
    for r in records:
        by_base.setdefault(r["base_id"], []).append(r)

    def strat_key(group: list[dict]) -> str:
        sample = group[0]
        ecog = sample.get("ecog", 0)
        ecog_bin = "low" if ecog <= 1 else ("mid" if ecog == 2 else "high")
        return f"{sample.get('stage_group', 'unknown')}__{ecog_bin}"

    by_strat: dict[str, list[int]] = {}
    for bid, grp in by_base.items():
        by_strat.setdefault(strat_key(grp), []).append(bid)

    rng = random.Random(cfg["data"]["seed"])
    train_bids, val_bids, test_bids = [], [], []
    tr, vr = cfg["data"]["train_ratio"], cfg["data"]["val_ratio"]
    for strat, bids in by_strat.items():
        rng.shuffle(bids)
        n = len(bids)
        n_train = int(n * tr)
        n_val = int(n * vr)
        train_bids.extend(bids[:n_train])
        val_bids.extend(bids[n_train:n_train + n_val])
        test_bids.extend(bids[n_train + n_val:])

    train_set, val_set, test_set = set(train_bids), set(val_bids), set(test_bids)
    train, val, test = [], [], []
    for r in records:
        bid = r["base_id"]
        if bid in train_set:
            train.append(r)
        elif bid in val_set:
            val.append(r)
        elif bid in test_set:
            test.append(r)
    return train, val, test


def filter_augmentations(records: list[dict], use_augs: bool) -> list[dict]:
    if use_augs:
        return records
    return [r for r in records if r.get("variant", 0) == 0]


# ============================================================
# Анализ
# ============================================================

def print_stats(name: str, records: list[dict]) -> None:
    print(f"\n=== {name} ({len(records)} записей) ===")
    print(f"  Уникальных base_id: {len(set(r['base_id'] for r in records))}")
    print(f"  Оригиналов (variant=0): {sum(1 for r in records if r.get('variant', 0) == 0)}")
    print(f"  Аугментаций: {sum(1 for r in records if r.get('variant', 0) > 0)}")

    stages = Counter(r.get("stage", "?") for r in records)
    print(f"  Top stages: {stages.most_common(8)}")
    ecog_dist = Counter(r.get("ecog", "?") for r in records)
    print(f"  ECOG: {sorted(ecog_dist.items())}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["data"]["seed"])

    src_path = Path(cfg["data"]["source_jsonl"])
    out_dir = ensure_dir(cfg["data"]["output_dir"])

    print(f"Загружаю {src_path}...")
    records = load_jsonl(src_path)
    print(f"Всего записей: {len(records)}")

    train, val, test = stratified_split(records, cfg)
    print(f"\nSplit (с аугментациями):")
    print(f"  train: {len(train)}, val: {len(val)}, test: {len(test)}")

    train_filtered = filter_augmentations(train, cfg["data"]["use_augmentations_train"])
    val_filtered = filter_augmentations(val, cfg["data"]["use_augmentations_val"])
    test_filtered = filter_augmentations(test, use_augs=False)

    print(f"\nПосле фильтрации:")
    print_stats("TRAIN", train_filtered)
    print_stats("VAL", val_filtered)
    print_stats("TEST", test_filtered)

    print(f"\n--- Сборка 4 specialized-датасетов ---")
    summary = {
        "source": str(src_path),
        "total_records": len(records),
        "blocks": {},
        "split_seed": cfg["data"]["seed"],
    }
    for block in cfg["models"]["blocks"]:
        sp_train = [build_block_sample(r, block) for r in train_filtered]
        sp_val = [build_block_sample(r, block) for r in val_filtered]
        sp_test = [build_block_sample(r, block) for r in test_filtered]
        save_jsonl(sp_train, out_dir / f"{block}_train.jsonl")
        save_jsonl(sp_val, out_dir / f"{block}_val.jsonl")
        save_jsonl(sp_test, out_dir / f"{block}_test.jsonl")
        print(f"  {block}: train={len(sp_train)}, val={len(sp_val)}, test={len(sp_test)}")
        summary["blocks"][block] = {
            "train": len(sp_train), "val": len(sp_val), "test": len(sp_test)
        }

    with open(out_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Готово. Файлы в {out_dir}/")


if __name__ == "__main__":
    main()
