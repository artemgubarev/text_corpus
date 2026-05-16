"""Лексические и эмбеддинговые метрики для генеративных моделей."""

import numpy as np
from typing import Optional


# ============================================================
# Лексические метрики
# ============================================================

def _ensure_nltk():
    """Скачать минимальные nltk-данные."""
    try:
        import nltk
        for resource in ["punkt", "punkt_tab", "wordnet", "omw-1.4"]:
            try:
                nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                except Exception:
                    pass
    except ImportError:
        pass


def compute_bleu(predictions: list[str], references: list[str]) -> float:
    """Corpus BLEU (sacrebleu)."""
    import sacrebleu
    refs = [[r] for r in references]
    # sacrebleu принимает list[list[str]] для refs (multi-reference)
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return float(bleu.score) / 100.0


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """ROUGE-1, ROUGE-2, ROUGE-L (mean F1)."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=False
    )
    r1, r2, rL = [], [], []
    for p, r in zip(predictions, references):
        scores = scorer.score(r, p)
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rL.append(scores["rougeL"].fmeasure)
    return {
        "rouge1": float(np.mean(r1)),
        "rouge2": float(np.mean(r2)),
        "rougeL": float(np.mean(rL)),
    }


def compute_meteor(predictions: list[str], references: list[str]) -> float:
    """METEOR (mean across samples)."""
    _ensure_nltk()
    try:
        from nltk.translate.meteor_score import meteor_score
        from nltk.tokenize import word_tokenize
    except ImportError:
        return float("nan")
    scores = []
    for p, r in zip(predictions, references):
        try:
            scores.append(meteor_score([word_tokenize(r)], word_tokenize(p)))
        except Exception:
            scores.append(0.0)
    return float(np.mean(scores)) if scores else 0.0


def compute_exact_match(predictions: list[str], references: list[str]) -> float:
    """Доля точно совпавших ответов (после strip)."""
    matches = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
    return matches / len(predictions) if predictions else 0.0


# ============================================================
# Эмбеддинговые метрики
# ============================================================

def compute_cosine_similarity(predictions: list[str], references: list[str],
                               model_name: str = "intfloat/multilingual-e5-large") -> float:
    """Средняя косинусная похожесть эмбеддингов prediction/reference."""
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    model = SentenceTransformer(model_name)
    pred_emb = model.encode(predictions, show_progress_bar=False, normalize_embeddings=True)
    ref_emb = model.encode(references, show_progress_bar=False, normalize_embeddings=True)
    # Pairwise cosine — только диагональ
    sims = np.array([
        cosine_similarity(p.reshape(1, -1), r.reshape(1, -1))[0, 0]
        for p, r in zip(pred_emb, ref_emb)
    ])
    return float(np.mean(sims))


def compute_bertscore(predictions: list[str], references: list[str],
                      lang: str = "ru") -> dict:
    """BERTScore (Precision, Recall, F1) — на multilingual BERT."""
    try:
        from bert_score import score
    except ImportError:
        return {"bertscore_p": float("nan"), "bertscore_r": float("nan"), "bertscore_f1": float("nan")}
    P, R, F1 = score(predictions, references, lang=lang, verbose=False, rescale_with_baseline=False)
    return {
        "bertscore_p": float(P.mean()),
        "bertscore_r": float(R.mean()),
        "bertscore_f1": float(F1.mean()),
    }


# ============================================================
# Полный bundle
# ============================================================

def compute_all_metrics(
    predictions: list[str],
    references: list[str],
    embedding_model: str = "intfloat/multilingual-e5-large",
    do_bertscore: bool = True,
    do_cosine: bool = True,
) -> dict:
    """Считает все лексические + эмбеддинговые метрики."""
    metrics = {}
    print("  - BLEU...")
    metrics["bleu"] = compute_bleu(predictions, references)
    print("  - ROUGE...")
    metrics.update(compute_rouge(predictions, references))
    print("  - METEOR...")
    metrics["meteor"] = compute_meteor(predictions, references)
    print("  - Exact match...")
    metrics["exact_match"] = compute_exact_match(predictions, references)
    if do_cosine:
        print("  - Cosine similarity (embeddings)...")
        metrics["cosine_similarity"] = compute_cosine_similarity(predictions, references, embedding_model)
    if do_bertscore:
        print("  - BERTScore...")
        metrics.update(compute_bertscore(predictions, references))
    return metrics
