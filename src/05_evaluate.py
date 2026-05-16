"""Шаг 05: Вычисление метрик качества.

Считает метрики для всех найденных predictions_{source}_{block}.jsonl,
где source ∈ {base, lora, full} и block ∈ {surgery, radiotherapy, systemic, supportive}.

Использование:
    python 05_evaluate.py
    python 05_evaluate.py --skip-bertscore   # быстрее, без BERTScore
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.data_utils import load_config, load_jsonl, ensure_dir
from utils.metrics import compute_all_metrics
from utils.clinical_metrics import (
    compute_intent_accuracy, compute_drug_presence,
    compute_stage_consistency, compute_safety_metrics,
)


def evaluate_predictions(predictions_path: Path, block: str,
                          embedding_model: str,
                          skip_bertscore: bool = False) -> dict:
    """Полное вычисление метрик для одного predictions.jsonl."""
    records = load_jsonl(predictions_path)
    if not records:
        return {}

    predictions = [r["prediction"] for r in records]
    references = [r["target"] for r in records]

    print(f"\n=== {predictions_path.name} (n={len(records)}, block={block}) ===")

    # Лексические + эмбеддинговые
    metrics = compute_all_metrics(
        predictions, references,
        embedding_model=embedding_model,
        do_bertscore=not skip_bertscore,
    )

    # Клинические метрики
    # 1. Intent accuracy для конкретного блока
    true_intents = [r.get("block_intent") for r in records]
    if all(true_intents):
        metrics.update(compute_intent_accuracy(predictions, references, true_intents, block))

    # 2. Drug presence (упоминание препаратов)
    metrics.update(compute_drug_presence(predictions, references))

    # 3. Stage consistency
    stages = [r.get("stage") for r in records if r.get("stage")]
    if len(stages) == len(records):
        metrics.update(compute_stage_consistency(predictions, stages))

    # 4. Safety: высокий ECOG vs агрессивные рекомендации
    ecogs = [r.get("ecog") for r in records]
    metrics.update(compute_safety_metrics(predictions, ecogs))

    metrics["n_samples"] = len(records)
    metrics["block"] = block
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--skip-bertscore", action="store_true",
                        help="Пропустить BERTScore (быстрее)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pred_dir = Path(cfg["inference"]["output_dir"])
    out_dir = ensure_dir(cfg["evaluation"]["output_dir"])
    emb_model = cfg["evaluation"]["embedding_model"]

    all_results = {}

    blocks = cfg["models"]["blocks"]
    sources = ["base", "lora", "full"]

    for source in sources:
        for block in blocks:
            fname = f"predictions_{source}_{block}.jsonl"
            path = pred_dir / fname
            if path.exists():
                key = f"{source}_{block}"
                all_results[key] = evaluate_predictions(
                    path, block=block, embedding_model=emb_model,
                    skip_bertscore=args.skip_bertscore,
                )
            else:
                print(f"  (пропуск, нет файла: {fname})")

    out_path = out_dir / "all_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 70}")
    print(f"✓ Метрики сохранены: {out_path}")
    print(f"{'=' * 70}\n")

    # Краткая сводка по ключевым метрикам
    keys_to_show = [
        "bleu", "rougeL", "meteor",
        "cosine_similarity", "bertscore_f1",
        "drug_jaccard_mean", "drug_hallucinations_per_sample",
        "stage_consistency", "high_ps_unsafe_rate",
    ]
    # Также вытащим intent_accuracy если есть
    for name, m in all_results.items():
        block = m.get("block", "?")
        print(f"\n--- {name} (n={m.get('n_samples', 0)}) ---")
        for k in keys_to_show:
            if k in m:
                print(f"  {k}: {m[k]:.4f}")
        intent_key = f"{block}_intent_accuracy"
        if intent_key in m:
            print(f"  {intent_key}: {m[intent_key]:.4f}")


if __name__ == "__main__":
    main()
