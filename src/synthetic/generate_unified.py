"""Главная точка входа. Запускает генерацию и пишет JSONL.

Использование:
    python generate_unified.py -n 100 -a 3 -o reports.jsonl
    python generate_unified.py -n 5 --show 3
"""

import argparse
import json
import random
import sys
from pathlib import Path

# Добавляем текущую директорию в path, чтобы импорты из core/blocks/render работали
sys.path.insert(0, str(Path(__file__).parent))

from core import load_schemas, validate_case, StateBuilder, lex_augment
from render import TextRenderer


META_KEYS = [
    "diagnosis_id", "diagnosis_name", "diagnosis_category", "grade",
    "subtype_name", "icd_o_3",
    "t_code", "n_code", "m_code", "tumor_size_mm", "stage", "stage_group",
    "molecular_included", "actionable_driver", "actionable_gene",
    "actionable_variant_short", "tp53_result", "pdl1_tps", "pdl1_category",
    "ecog", "karnofsky",
    "matched_rule_id", "treatment_intent",
    "lvi_status", "pleural_status", "perineural_status", "margin_status",
]


def generate(n_samples: int, n_variants: int, seed: int,
             schemas_dir: Path, output: Path, show: int = 0) -> None:
    rng = random.Random(seed)
    schemas = load_schemas(schemas_dir)
    builder = StateBuilder(schemas, rng)
    renderer = TextRenderer(schemas)

    records = []
    n_validation_errors = 0

    for base_idx in range(n_samples):
        case = builder.build()
        case.full_text = renderer.render(case)

        errors = validate_case(case)
        if errors:
            n_validation_errors += 1

        meta = {k: getattr(case, k) for k in META_KEYS}
        base_record = {
            "id": f"case_{base_idx:05d}_v0",
            "base_id": base_idx,
            "variant": 0,
            "is_augmented": False,
            "text": case.full_text,
            "validation_errors": errors,
            **meta,
        }
        records.append(base_record)

        for v in range(1, n_variants):
            aug_text = lex_augment(case.full_text, rng)
            aug_record = dict(base_record)
            aug_record["id"] = f"case_{base_idx:05d}_v{v}"
            aug_record["variant"] = v
            aug_record["is_augmented"] = True
            aug_record["text"] = aug_text
            records.append(aug_record)

    with open(output, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✓ Записей: {len(records)} ({n_samples} базовых × {n_variants})")
    print(f"  Ошибок согласованности: {n_validation_errors}/{n_samples}")
    print(f"  → {output}")

    if show:
        for r in records[:show]:
            print("=" * 70)
            print(
                f"{r['id']} | dx={r['diagnosis_id']} stage={r['stage']} "
                f"ecog={r['ecog']} intent={r['treatment_intent']} aug={r['is_augmented']}"
            )
            if r["validation_errors"]:
                print(f"  ⚠ {r['validation_errors']}")
            print("=" * 70)
            print(r["text"])
            print()


def main():
    p = argparse.ArgumentParser(
        description="Модульный генератор клинических заключений по раку лёгкого"
    )
    p.add_argument("-n", "--num-samples", type=int, default=10,
                   help="Количество базовых сэмплов")
    p.add_argument("-a", "--augmentations", type=int, default=3,
                   help="Вариантов на сэмпл (1 = только оригинал)")
    p.add_argument("-s", "--seed", type=int, default=42)
    p.add_argument("--schemas-dir", type=str, default="schemas")
    p.add_argument("-o", "--output", type=str, default="unified_reports.jsonl")
    p.add_argument("--show", type=int, default=0,
                   help="Сколько примеров вывести в stdout")
    args = p.parse_args()

    generate(
        args.num_samples, args.augmentations, args.seed,
        Path(args.schemas_dir), Path(args.output), args.show,
    )


if __name__ == "__main__":
    main()
