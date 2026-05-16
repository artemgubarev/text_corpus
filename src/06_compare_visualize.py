"""Шаг 06: Визуализация и сравнение результатов.

Что строит:
    1. Кривые обучения (loss train/val, LR) для каждой модели
    2. Тройное сравнение base/lora/full по основным метрикам (bar chart по блокам)
    3. Heatmap всех метрик: 4 блока × 3 модели
    4. Confusion matrices intent для lora и full
    5. Эмбеддинги: target vs prediction для lora и full
    6. Радар-диаграмма (spider) клинических метрик

Использование:
    python 06_compare_visualize.py
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from utils.data_utils import load_config, load_jsonl, ensure_dir
from utils.plotting import setup_plot_style, save_fig
from utils.clinical_metrics import predict_intent_from_text


SOURCE_COLORS = {
    "base": "tomato",
    "lora": "steelblue",
    "full": "seagreen",
}
SOURCE_LABELS = {
    "base": "Base (zero-shot)",
    "lora": "LoRA (Qwen2.5-3B)",
    "full": "Full FT (Qwen2.5-1.5B)",
}


# ============================================================
# 1. Кривые обучения
# ============================================================

def plot_training_curves(log_path: Path, out_dir: Path, model_label: str) -> None:
    if not log_path.exists():
        print(f"  ⚠ Не найдено: {log_path}")
        return
    with open(log_path, encoding="utf-8") as f:
        history = json.load(f)

    train_steps = [e["step"] for e in history if "loss" in e and "eval_loss" not in e]
    train_loss = [e["loss"] for e in history if "loss" in e and "eval_loss" not in e]
    eval_steps = [e["step"] for e in history if "eval_loss" in e]
    eval_loss = [e["eval_loss"] for e in history if "eval_loss" in e]
    lr_steps = [e["step"] for e in history if "learning_rate" in e]
    lr_vals = [e["learning_rate"] for e in history if "learning_rate" in e]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(train_steps, train_loss, label="train", alpha=0.7)
    if eval_loss:
        axes[0].plot(eval_steps, eval_loss, label="eval", marker="o", linewidth=2)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Loss — {model_label}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if lr_vals:
        axes[1].plot(lr_steps, lr_vals, color="orange")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Learning rate")
        axes[1].set_title("Learning rate schedule")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, out_dir / f"training_curves_{model_label}.png")


def plot_all_training_curves(cfg: dict, out_dir: Path) -> None:
    """Loss curves для всех обученных моделей."""
    ckpt_dir = Path(cfg["training"]["output_dir"])
    short_lora = cfg["models"]["short_name_lora"]
    short_full = cfg["models"]["short_name_full"]
    for block in cfg["models"]["blocks"]:
        for short_name, method in [(short_lora, "lora"), (short_full, "full")]:
            log_path = ckpt_dir / f"{short_name}-{block}" / "train_log_history.json"
            plot_training_curves(log_path, out_dir, f"{method}_{block}")


def plot_loss_compare_methods(cfg: dict, out_dir: Path) -> None:
    """Один график на блок — eval loss для LoRA и Full FT вместе."""
    ckpt_dir = Path(cfg["training"]["output_dir"])
    short_lora = cfg["models"]["short_name_lora"]
    short_full = cfg["models"]["short_name_full"]

    for block in cfg["models"]["blocks"]:
        fig, ax = plt.subplots(figsize=(10, 5))
        for short_name, method in [(short_lora, "lora"), (short_full, "full")]:
            log_path = ckpt_dir / f"{short_name}-{block}" / "train_log_history.json"
            if not log_path.exists():
                continue
            with open(log_path, encoding="utf-8") as f:
                history = json.load(f)
            eval_steps = [e["step"] for e in history if "eval_loss" in e]
            eval_loss = [e["eval_loss"] for e in history if "eval_loss" in e]
            if eval_loss:
                ax.plot(eval_steps, eval_loss, label=SOURCE_LABELS[method],
                        marker="o", color=SOURCE_COLORS[method], linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Eval loss")
        ax.set_title(f"Eval loss — {block}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_fig(fig, out_dir / f"loss_compare_{block}.png")


# ============================================================
# 2. Сравнение метрик: bar charts
# ============================================================

def plot_metrics_comparison_grouped(metrics: dict, out_dir: Path,
                                     blocks: list[str]) -> None:
    """Группированный bar plot — для каждого блока показывает base/lora/full
    по ключевым метрикам."""
    key_metrics = [
        "bleu", "rouge1", "rougeL", "meteor",
        "cosine_similarity", "bertscore_f1",
        "drug_jaccard_mean", "stage_consistency",
    ]

    for block in blocks:
        rows = []
        for source in ["base", "lora", "full"]:
            key = f"{source}_{block}"
            if key not in metrics:
                continue
            m = metrics[key]
            for mk in key_metrics:
                if mk in m:
                    rows.append({
                        "metric": mk,
                        "value": m[mk],
                        "model": SOURCE_LABELS[source],
                    })
        if not rows:
            continue
        df = pd.DataFrame(rows)

        fig, ax = plt.subplots(figsize=(14, 6))
        palette = {SOURCE_LABELS[k]: v for k, v in SOURCE_COLORS.items()}
        sns.barplot(data=df, x="metric", y="value", hue="model",
                    palette=palette, ax=ax)
        ax.set_title(f"Сравнение моделей: {block}")
        ax.set_ylabel("Score")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
        plt.tight_layout()
        save_fig(fig, out_dir / f"metrics_compare_{block}.png")


def plot_metrics_heatmap_full(metrics: dict, out_dir: Path, blocks: list[str]) -> None:
    """Полная heatmap: строки = модели, колонки = метрики. Все блоки в один файл."""
    key_metrics = [
        "bleu", "rouge1", "rougeL", "meteor",
        "cosine_similarity", "bertscore_f1",
        "drug_jaccard_mean", "stage_consistency",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    for ax, block in zip(axes.flat, blocks):
        rows = {}
        for source in ["base", "lora", "full"]:
            key = f"{source}_{block}"
            if key in metrics:
                rows[SOURCE_LABELS[source]] = {
                    mk: metrics[key].get(mk, np.nan) for mk in key_metrics
                }
        if not rows:
            ax.axis("off")
            continue
        df = pd.DataFrame(rows).T
        sns.heatmap(df, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax,
                    cbar_kws={"label": "score"})
        ax.set_title(f"{block}")
        ax.set_xlabel("")
        ax.set_ylabel("")
    plt.tight_layout()
    save_fig(fig, out_dir / "metrics_heatmap_all_blocks.png")


# ============================================================
# 3. Радар клинических метрик
# ============================================================

def plot_clinical_radar(metrics: dict, out_dir: Path, blocks: list[str]) -> None:
    """Радар-диаграмма для клинических метрик: для каждого блока сравнение моделей."""
    # Метрики где БОЛЬШЕ = ЛУЧШЕ
    clinical_metrics = {
        "intent_accuracy": "Intent accuracy",
        "drug_jaccard_mean": "Drug Jaccard",
        "stage_consistency": "Stage consistency",
        "drug_exact_set_match": "Drug exact match",
    }
    # И одна где меньше=лучше
    clinical_metrics_inv = {
        "drug_hallucinations_per_sample": "1 - Hallucinations (norm)",
        "high_ps_unsafe_rate": "1 - Safety violations",
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 14),
                              subplot_kw=dict(projection="polar"))
    for ax, block in zip(axes.flat, blocks):
        all_metrics_keys = list(clinical_metrics.keys()) + list(clinical_metrics_inv.keys())
        labels = list(clinical_metrics.values()) + list(clinical_metrics_inv.values())
        angles = np.linspace(0, 2 * np.pi, len(all_metrics_keys), endpoint=False).tolist()
        angles += [angles[0]]

        for source in ["base", "lora", "full"]:
            key = f"{source}_{block}"
            if key not in metrics:
                continue
            m = metrics[key]
            values = []
            for mk in clinical_metrics:
                # для intent_accuracy ключ выглядит как f"{block}_intent_accuracy"
                if mk == "intent_accuracy":
                    v = m.get(f"{block}_intent_accuracy", 0)
                else:
                    v = m.get(mk, 0)
                # Клипуем в [0, 1]
                v = max(0, min(1, v if v is not None else 0))
                values.append(v)
            for mk in clinical_metrics_inv:
                # Инвертируем: 1 - значение
                v = m.get(mk, 0) or 0
                # hallucinations может быть >1, нормируем
                if mk == "drug_hallucinations_per_sample":
                    v = min(v, 3) / 3  # шкала 0-3
                values.append(max(0, 1 - v))
            values += [values[0]]

            ax.plot(angles, values, label=SOURCE_LABELS[source],
                    color=SOURCE_COLORS[source], linewidth=2)
            ax.fill(angles, values, color=SOURCE_COLORS[source], alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=8)
        ax.set_ylim(0, 1)
        ax.set_title(f"Клинические метрики: {block}", y=1.08, size=12)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
        ax.grid(True)
    plt.tight_layout()
    save_fig(fig, out_dir / "clinical_radar_all_blocks.png")


# ============================================================
# 4. Confusion matrices intent
# ============================================================

def plot_intent_confusion(predictions_path: Path, block: str, source: str,
                           out_dir: Path) -> None:
    if not predictions_path.exists():
        return
    records = load_jsonl(predictions_path)
    true_intents, pred_intents = [], []
    for r in records:
        ti = r.get("block_intent")
        if not ti:
            continue
        pi = predict_intent_from_text(r["prediction"], block) or "_NO_MATCH_"
        true_intents.append(ti)
        pred_intents.append(pi)

    if not true_intents:
        return

    labels = sorted(set(true_intents + pred_intents))
    cm = pd.crosstab(
        pd.Series(true_intents, name="True"),
        pd.Series(pred_intents, name="Predicted"),
    ).reindex(index=labels, columns=labels, fill_value=0)

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                cbar_kws={"label": "count"})
    ax.set_title(f"Intent confusion — {source} / {block}")
    plt.tight_layout()
    save_fig(fig, out_dir / f"confusion_intent_{source}_{block}.png")


# ============================================================
# 5. Эмбеддинги target vs prediction
# ============================================================

def plot_embeddings_target_vs_pred(predictions_path: Path, label: str,
                                    out_dir: Path, embedding_model: str,
                                    max_samples: int = 300) -> None:
    if not predictions_path.exists():
        return
    records = load_jsonl(predictions_path)[:max_samples]
    if not records:
        return

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(embedding_model)
    texts = [r["target"] for r in records] + [r["prediction"] for r in records]
    embs = model.encode(texts, batch_size=32, show_progress_bar=True,
                         normalize_embeddings=True)

    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42, metric="cosine")
        coords = reducer.fit_transform(embs)
    except ImportError:
        from sklearn.manifold import TSNE
        coords = TSNE(n_components=2, random_state=42, metric="cosine",
                       init="pca").fit_transform(embs)

    n = len(records)
    df = pd.DataFrame({
        "x": coords[:, 0], "y": coords[:, 1],
        "type": ["target"] * n + ["prediction"] * n,
    })

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=df, x="x", y="y", hue="type",
                    palette={"target": "tab:green", "prediction": "tab:orange"},
                    alpha=0.6, s=25, ax=ax)
    ax.set_title(f"Эмбеддинги target vs prediction — {label}")
    plt.tight_layout()
    save_fig(fig, out_dir / f"embeddings_target_vs_pred_{label}.png")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--skip-embeddings", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_plot_style(cfg)

    out_dir = ensure_dir(Path(cfg["visualization"]["output_dir"]) / "results")
    blocks = cfg["models"]["blocks"]

    # === 1. Кривые обучения ===
    print("[1/5] Кривые обучения...")
    plot_all_training_curves(cfg, out_dir)
    plot_loss_compare_methods(cfg, out_dir)

    # === 2. Метрики ===
    print("[2/5] Сравнение метрик...")
    metrics_path = Path(cfg["evaluation"]["output_dir"]) / "all_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            all_metrics = json.load(f)
        plot_metrics_comparison_grouped(all_metrics, out_dir, blocks)
        plot_metrics_heatmap_full(all_metrics, out_dir, blocks)
        print("[3/5] Радар клинических метрик...")
        plot_clinical_radar(all_metrics, out_dir, blocks)
    else:
        print(f"  ⚠ Не найдено {metrics_path}, пропускаю метрики")

    # === 4. Confusion matrices intent ===
    print("[4/5] Confusion matrices intent...")
    pred_dir = Path(cfg["inference"]["output_dir"])
    for block in blocks:
        for source in ["lora", "full"]:
            plot_intent_confusion(
                pred_dir / f"predictions_{source}_{block}.jsonl",
                block, source, out_dir,
            )

    # === 5. Эмбеддинги ===
    if not args.skip_embeddings:
        print("[5/5] Эмбеддинги target vs prediction...")
        embedding_model = cfg["evaluation"]["embedding_model"]
        for block in blocks:
            for source in ["lora", "full"]:
                plot_embeddings_target_vs_pred(
                    pred_dir / f"predictions_{source}_{block}.jsonl",
                    f"{source}_{block}", out_dir, embedding_model,
                )

    print(f"\n✓ Графики сохранены в {out_dir}/")


if __name__ == "__main__":
    main()
