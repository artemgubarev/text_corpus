"""Шаг 02: Визуализация данных и эмбеддингов.

Что строит:
    1. Распределения: диагнозы, стадии, ECOG, intent по 4 блокам, длины текстов
    2. Confusion matrix сочетаний: stage vs intent блоков
    3. Эмбеддинги входов (UMAP/t-SNE) с разными окрасками
    4. Эмбеддинги таргетов по 4 блокам

Использование:
    python 02_visualize_data.py
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from utils.data_utils import load_config, load_jsonl, set_seed, ensure_dir
from utils.plotting import setup_plot_style, save_fig


# ============================================================
# Распределения
# ============================================================

def plot_diagnosis_distribution(records: list[dict], out_dir: Path) -> None:
    counts = Counter(r["diagnosis_id"] for r in records)
    fig, ax = plt.subplots(figsize=(10, 5))
    items = counts.most_common()
    ax.bar([k for k, _ in items], [v for _, v in items])
    ax.set_title("Распределение диагнозов в датасете")
    ax.set_xlabel("Diagnosis ID")
    ax.set_ylabel("Количество записей")
    for i, (k, v) in enumerate(items):
        ax.text(i, v, str(v), ha="center", va="bottom")
    save_fig(fig, out_dir / "01_diagnosis_distribution.png")


def plot_stage_distribution(records: list[dict], out_dir: Path) -> None:
    counts = Counter(r["stage"] for r in records)
    order = ["0", "IA1", "IA2", "IA3", "IB", "IIA", "IIB", "IIIA", "IIIB", "IIIC", "IVA", "IVB"]
    counts_ordered = [(s, counts.get(s, 0)) for s in order if s in counts]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar([k for k, _ in counts_ordered], [v for _, v in counts_ordered])
    ax.set_title("Распределение стадий (TNM-9)")
    ax.set_xlabel("Stage")
    ax.set_ylabel("Количество")
    save_fig(fig, out_dir / "02_stage_distribution.png")


def plot_ecog_distribution(records: list[dict], out_dir: Path) -> None:
    counts = Counter(r["ecog"] for r in records)
    fig, ax = plt.subplots(figsize=(8, 5))
    items = sorted(counts.items())
    ax.bar([str(k) for k, _ in items], [v for _, v in items], color="steelblue")
    ax.set_title("Распределение ECOG Performance Status")
    ax.set_xlabel("ECOG PS")
    ax.set_ylabel("Количество")
    save_fig(fig, out_dir / "03_ecog_distribution.png")


def plot_intents_by_block(records: list[dict], out_dir: Path) -> None:
    """4 subplot — по одному на каждый блок рекомендаций."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    blocks = ["surgery", "radiotherapy", "systemic", "supportive"]
    block_labels = {
        "surgery": "Хирургия",
        "radiotherapy": "Лучевая терапия",
        "systemic": "Системная терапия",
        "supportive": "Поддерживающая",
    }
    for ax, block in zip(axes.flat, blocks):
        counts = Counter(r[f"{block}_intent"] for r in records)
        items = counts.most_common()
        labels = [k for k, _ in items]
        values = [v for _, v in items]
        ax.barh(labels, values, color=sns.color_palette("husl", 1)[0])
        ax.set_title(f"Intent: {block_labels[block]}")
        ax.set_xlabel("Количество")
        ax.invert_yaxis()
    plt.tight_layout()
    save_fig(fig, out_dir / "04_intents_by_block.png")


def plot_text_length_distribution(records: list[dict], out_dir: Path) -> None:
    """Распределение длин входов и таргетов в символах и токенах (приблизительно)."""
    input_lens = []
    target_lens = []
    for r in records:
        text = r.get("text", "")
        marker = "Рекомендация по лечению."
        idx = text.find(marker)
        if idx != -1:
            input_lens.append(len(text[:idx]))
            target_lens.append(len(text[idx:]))
        else:
            input_lens.append(len(text))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(input_lens, bins=40, color="steelblue", edgecolor="black", alpha=0.7)
    axes[0].set_title("Длина клинического контекста (chars)")
    axes[0].set_xlabel("Символы")
    axes[0].axvline(np.mean(input_lens), color="red", linestyle="--",
                    label=f"mean={int(np.mean(input_lens))}")
    axes[0].legend()

    if target_lens:
        axes[1].hist(target_lens, bins=40, color="orange", edgecolor="black", alpha=0.7)
        axes[1].set_title("Длина блока рекомендаций (chars)")
        axes[1].set_xlabel("Символы")
        axes[1].axvline(np.mean(target_lens), color="red", linestyle="--",
                        label=f"mean={int(np.mean(target_lens))}")
        axes[1].legend()
    plt.tight_layout()
    save_fig(fig, out_dir / "05_text_lengths.png")


def plot_stage_vs_intent_heatmap(records: list[dict], out_dir: Path) -> None:
    """Heatmap: stage x surgery_intent — показывает что для каких стадий что назначается."""
    df = pd.DataFrame(records)
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    blocks = ["surgery", "radiotherapy", "systemic", "supportive"]
    block_labels = {
        "surgery": "Хирургия",
        "radiotherapy": "Лучевая терапия",
        "systemic": "Системная терапия",
        "supportive": "Поддерживающая",
    }
    for ax, block in zip(axes.flat, blocks):
        ct = pd.crosstab(df["stage"], df[f"{block}_intent"])
        # Сортировка стадий
        stage_order = ["0", "IA1", "IA2", "IA3", "IB", "IIA", "IIB", "IIIA", "IIIB", "IIIC", "IVA", "IVB"]
        ct = ct.reindex([s for s in stage_order if s in ct.index])
        sns.heatmap(ct, annot=True, fmt="d", cmap="YlGnBu", ax=ax, cbar_kws={"label": "count"})
        ax.set_title(f"Stage × intent: {block_labels[block]}")
        ax.set_xlabel("Intent")
        ax.set_ylabel("Stage")
    plt.tight_layout()
    save_fig(fig, out_dir / "06_stage_vs_intent.png")


# ============================================================
# Эмбеддинги: UMAP/t-SNE
# ============================================================

def compute_embeddings(texts: list[str], model_name: str, batch_size: int = 32) -> np.ndarray:
    """Считает эмбеддинги через sentence-transformers (multilingual)."""
    from sentence_transformers import SentenceTransformer
    print(f"Загружаю модель эмбеддингов: {model_name}")
    model = SentenceTransformer(model_name)
    # Для e5-моделей нужен префикс "query: " — но для документов это меньше критично
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    return embeddings


def reduce_dimensions(embeddings: np.ndarray, method: str = "umap",
                      n_components: int = 2, seed: int = 42) -> np.ndarray:
    """UMAP или t-SNE для понижения размерности."""
    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=n_components, random_state=seed,
                                n_neighbors=15, min_dist=0.1, metric="cosine")
            return reducer.fit_transform(embeddings)
        except ImportError:
            print("⚠ umap-learn не установлен, использую t-SNE")
            method = "tsne"
    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=n_components, random_state=seed,
                       perplexity=30, metric="cosine", init="pca")
        return reducer.fit_transform(embeddings)
    raise ValueError(f"Неизвестный метод: {method}")


def plot_embeddings(coords: np.ndarray, labels: list, label_name: str,
                    out_path: Path, sample_limit: int = 2000) -> None:
    """Точечный график embeddings, окрашенный по labels."""
    if len(coords) > sample_limit:
        idx = np.random.RandomState(42).choice(len(coords), sample_limit, replace=False)
        coords = coords[idx]
        labels = [labels[i] for i in idx]
    df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "label": labels})

    fig, ax = plt.subplots(figsize=(12, 8))
    palette = sns.color_palette("husl", n_colors=len(set(labels)))
    sns.scatterplot(data=df, x="x", y="y", hue="label", palette=palette,
                    s=15, alpha=0.7, ax=ax, edgecolor="none")
    ax.set_title(f"Эмбеддинги текстов (окраска: {label_name})")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0,
              fontsize=9, frameon=True)
    plt.tight_layout()
    save_fig(fig, out_path)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Пропустить вычисление эмбеддингов (быстро)")
    parser.add_argument("--max-samples", type=int, default=1500,
                        help="Сколько примеров использовать для эмбеддингов")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_plot_style(cfg)
    set_seed(cfg["data"]["seed"])

    out_dir = ensure_dir(Path(cfg["visualization"]["output_dir"]) / "data_eda")

    # Загружаем train split
    data_dir = Path(cfg["data"]["output_dir"])
    train_path = data_dir / "baseline_train.jsonl"
    if not train_path.exists():
        print(f"⚠ Не найден {train_path}. Сначала запустите 01_prepare_dataset.py")
        return

    # Для EDA — используем оригинальный JSONL (там все мета-поля)
    src_path = Path(cfg["data"]["source_jsonl"])
    records = load_jsonl(src_path)
    # Только оригиналы (variant=0) для EDA, чтобы не дублировать
    records = [r for r in records if r.get("variant", 0) == 0]
    print(f"Загружено для EDA: {len(records)} записей")

    # 1. Распределения
    print("\n[1/3] Распределения...")
    plot_diagnosis_distribution(records, out_dir)
    plot_stage_distribution(records, out_dir)
    plot_ecog_distribution(records, out_dir)
    plot_intents_by_block(records, out_dir)
    plot_text_length_distribution(records, out_dir)
    plot_stage_vs_intent_heatmap(records, out_dir)

    if args.skip_embeddings:
        print(f"\n✓ Базовые графики сохранены в {out_dir}/")
        return

    # 2. Эмбеддинги
    print(f"\n[2/3] Считаю эмбеддинги ({min(len(records), args.max_samples)} текстов)...")
    sample_records = records[:args.max_samples]
    texts = [r["text"] for r in sample_records]
    embeddings = compute_embeddings(texts, cfg["evaluation"]["embedding_model"])
    np.save(out_dir / "embeddings.npy", embeddings)

    print(f"\n[3/3] Понижаю размерность ({cfg['visualization']['reduce_method']})...")
    coords = reduce_dimensions(
        embeddings,
        method=cfg["visualization"]["reduce_method"],
        n_components=cfg["visualization"]["reduce_n_components"],
        seed=cfg["data"]["seed"],
    )
    np.save(out_dir / f"coords_{cfg['visualization']['reduce_method']}.npy", coords)

    # Окраска разными способами
    labels_by_diagnosis = [r["diagnosis_id"] for r in sample_records]
    plot_embeddings(coords, labels_by_diagnosis, "diagnosis",
                    out_dir / "07_embeddings_by_diagnosis.png")

    labels_by_stage_group = [r.get("stage_group", "?") for r in sample_records]
    plot_embeddings(coords, labels_by_stage_group, "stage_group",
                    out_dir / "08_embeddings_by_stage_group.png")

    labels_by_surgery = [r["surgery_intent"] for r in sample_records]
    plot_embeddings(coords, labels_by_surgery, "surgery_intent",
                    out_dir / "09_embeddings_by_surgery_intent.png")

    labels_by_systemic = [r["systemic_intent"] for r in sample_records]
    plot_embeddings(coords, labels_by_systemic, "systemic_intent",
                    out_dir / "10_embeddings_by_systemic_intent.png")

    print(f"\n✓ Графики сохранены в {out_dir}/")


if __name__ == "__main__":
    main()
