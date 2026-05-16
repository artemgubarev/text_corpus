"""Шаг 07: Сборка финального HTML-отчёта для дипломной работы.

Собирает:
    - таблицы метрик с тройным сравнением (base / LoRA / Full FT) по 4 блокам
    - все графики
    - примеры генераций (3 колонки: target, LoRA, Full FT)

Использование:
    python 07_make_report.py
    # затем открой outputs/report.html
"""

import argparse
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.data_utils import load_config, load_jsonl


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>Дипломная работа: fine-tuning LLM для клинических рекомендаций по раку лёгкого</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1300px; margin: 30px auto; padding: 0 20px; color: #222; line-height: 1.55; }
  h1, h2, h3 { color: #1a3d5c; }
  h1 { border-bottom: 3px solid #1a3d5c; padding-bottom: 10px; }
  h2 { border-bottom: 1px solid #ccc; padding-bottom: 5px; margin-top: 40px; }
  table { border-collapse: collapse; margin: 20px 0; width: 100%; font-size: 13px; }
  th, td { border: 1px solid #ddd; padding: 8px; text-align: right; }
  th { background: #1a3d5c; color: white; text-align: center; }
  td:first-child { text-align: left; font-weight: 600; }
  tr:nth-child(even) td { background: #f5f9fc; }
  .best { background: #c8e6c9 !important; font-weight: 600; }
  img { max-width: 100%; margin: 15px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-radius: 4px; }
  .example { background: #f8f9fa; border-left: 4px solid #1a3d5c; padding: 15px; margin: 15px 0; border-radius: 4px; }
  .example h4 { margin: 0 0 10px 0; color: #1a3d5c; }
  pre { white-space: pre-wrap; font-family: inherit; font-size: 13px; line-height: 1.5; margin: 4px 0; }
  .grid3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin: 10px 0; }
  .grid3 > div { background: #fafafa; padding: 10px; border-radius: 4px; border: 1px solid #e0e0e0; }
  .grid3 b { display: block; color: #1a3d5c; margin-bottom: 6px; }
  .meta-box { background: #fff8e1; padding: 4px 8px; border-radius: 3px; font-size: 11px; color: #555; display: inline-block; }
  details summary { cursor: pointer; padding: 5px; background: #eef1f4; border-radius: 3px; margin: 5px 0; }
  details[open] summary { background: #d4dde6; }
</style>
</head>
<body>
{content}
</body></html>
"""


def img_tag(path: Path) -> str:
    if not path.exists():
        return f'<p style="color:#888"><em>[не найдено: {path.name}]</em></p>'
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}" alt="{path.name}">'


def build_metrics_table_3way(metrics: dict, block: str) -> str:
    """Таблица с 3 моделями (base/lora/full) для одного блока. Лучшая ячейка подсвечивается."""
    sources = ["base", "lora", "full"]
    source_labels = {
        "base": "Base (zero-shot)",
        "lora": "LoRA (Qwen2.5-3B)",
        "full": "Full FT (Qwen2.5-1.5B)",
    }

    # Группы метрик
    metric_groups = [
        ("Лексические", ["bleu", "rouge1", "rouge2", "rougeL", "meteor", "exact_match"], True),
        ("Эмбеддинговые", ["cosine_similarity", "bertscore_p", "bertscore_r", "bertscore_f1"], True),
        ("Клинические (больше = лучше)", [
            f"{block}_intent_accuracy", f"{block}_intent_coverage",
            "drug_jaccard_mean", "drug_exact_set_match",
            "stage_consistency", "stage_mention_rate",
        ], True),
        ("Клинические (меньше = лучше)", [
            "drug_hallucinations_per_sample", "drug_missed_per_sample",
            "high_ps_unsafe_rate",
        ], False),
    ]

    html = [f"<h3>Блок: {block}</h3>", "<table>"]
    html.append("<tr><th>Метрика</th>")
    for s in sources:
        html.append(f"<th>{source_labels[s]}</th>")
    html.append("</tr>")

    for group_name, keys, higher_better in metric_groups:
        html.append(f'<tr><td colspan="4" style="background:#dde6ee;font-weight:bold;text-align:left">{group_name}</td></tr>')
        for k in keys:
            row_values = []
            for s in sources:
                m = metrics.get(f"{s}_{block}", {})
                row_values.append(m.get(k))
            # Определяем лучшее значение
            non_none = [v for v in row_values if v is not None]
            if not non_none:
                continue
            best_val = max(non_none) if higher_better else min(non_none)

            html.append(f"<tr><td>{k}</td>")
            for v in row_values:
                if v is None:
                    html.append("<td>—</td>")
                else:
                    cls = "best" if v == best_val else ""
                    html.append(f'<td class="{cls}">{v:.4f}</td>')
            html.append("</tr>")
    html.append("</table>")
    return "\n".join(html)


def build_examples_3way(pred_dir: Path, block: str, max_examples: int = 3) -> str:
    """Side-by-side примеры: target / LoRA / Full FT."""
    paths = {
        s: pred_dir / f"predictions_{s}_{block}.jsonl"
        for s in ["base", "lora", "full"]
    }
    if not all(p.exists() for p in paths.values()):
        return ""

    by_id = {}
    for source, path in paths.items():
        for r in load_jsonl(path):
            by_id.setdefault(r["id"], {})[source] = r

    common = [k for k, v in by_id.items() if len(v) == 3][:max_examples]
    if not common:
        return ""

    html = [f"<h3>Примеры — {block}</h3>"]
    for cid in common:
        recs = by_id[cid]
        b = recs["base"]
        meta = f"stage={b.get('stage','?')} ECOG={b.get('ecog','?')} dx={b.get('diagnosis_id','?')} intent={b.get('block_intent','?')}"
        html.append('<div class="example">')
        html.append(f'<h4>{cid} <span class="meta-box">{meta}</span></h4>')
        html.append(f'<details><summary>Клинический контекст (input)</summary><pre>{b["input"]}</pre></details>')
        html.append('<div style="background:#e8f5e9;padding:10px;border-radius:4px;margin:10px 0">')
        html.append(f'<b>Ground truth (target):</b><pre>{b["target"]}</pre></div>')
        html.append('<div class="grid3">')
        html.append(f'<div><b>Base (zero-shot)</b><pre>{recs["base"]["prediction"]}</pre></div>')
        html.append(f'<div><b>LoRA</b><pre>{recs["lora"]["prediction"]}</pre></div>')
        html.append(f'<div><b>Full FT</b><pre>{recs["full"]["prediction"]}</pre></div>')
        html.append('</div></div>')
    return "\n".join(html)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pred_dir = Path(cfg["inference"]["output_dir"])
    metrics_dir = Path(cfg["evaluation"]["output_dir"])
    plots_dir = Path(cfg["visualization"]["output_dir"])
    out_path = Path("outputs/report.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    blocks = cfg["models"]["blocks"]
    sections = []

    # === Заголовок ===
    sections.append(f"""
    <h1>Fine-tuning LLM для клинических рекомендаций по раку лёгкого</h1>
    <div class="meta-box" style="display:block;padding:15px;margin:10px 0">
      <p><b>Задача:</b> генерация структурированных рекомендаций по лечению из 4 блоков
         (хирургия / лучевая терапия / системная терапия / поддерживающая терапия)</p>
      <p><b>Архитектура эксперимента:</b> 4 специализированные модели, по одной на блок</p>
      <p><b>Сравниваемые методы:</b></p>
      <ul>
        <li><b>Base (zero-shot)</b> — {cfg['models']['base_model_id']} без обучения</li>
        <li><b>LoRA</b> — {cfg['models']['base_model_id']} + LoRA адаптер (r={cfg['models']['lora']['r']}, 4-bit QLoRA)</li>
        <li><b>Full FT</b> — {cfg['models']['base_model_id_full_ft']} полностью дообученная</li>
      </ul>
      <p><b>Датасет:</b> синтетические клинические заключения (unified-генератор)</p>
    </div>
    """)

    # === 1. EDA данных ===
    sections.append("<h2>1. Анализ датасета (EDA)</h2>")
    eda_dir = plots_dir / "data_eda"
    if eda_dir.exists():
        for f in sorted(eda_dir.glob("*.png")):
            sections.append(img_tag(f))

    # === 2. Кривые обучения ===
    sections.append("<h2>2. Процесс обучения</h2>")
    sections.append("<h3>Сравнение кривых eval loss (LoRA vs Full FT)</h3>")
    results_dir = plots_dir / "results"
    if results_dir.exists():
        for block in blocks:
            sections.append(img_tag(results_dir / f"loss_compare_{block}.png"))

    sections.append("<h3>Детальные кривые для каждой модели</h3>")
    if results_dir.exists():
        for f in sorted(results_dir.glob("training_curves_*.png")):
            sections.append(img_tag(f))

    # === 3. Метрики ===
    sections.append("<h2>3. Сравнение моделей по метрикам</h2>")
    metrics_path = metrics_dir / "all_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            all_metrics = json.load(f)
        sections.append("<p>Зелёным выделено лучшее значение в строке.</p>")
        for block in blocks:
            sections.append(build_metrics_table_3way(all_metrics, block))

        sections.append("<h3>Графики сравнения</h3>")
        if results_dir.exists():
            for f in sorted(results_dir.glob("metrics_compare_*.png")):
                sections.append(img_tag(f))
            sections.append(img_tag(results_dir / "metrics_heatmap_all_blocks.png"))

        sections.append("<h3>Радар клинических метрик</h3>")
        sections.append(img_tag(results_dir / "clinical_radar_all_blocks.png"))

    # === 4. Intent classification ===
    sections.append("<h2>4. Классификация intent (confusion matrices)</h2>")
    if results_dir.exists():
        for f in sorted(results_dir.glob("confusion_intent_*.png")):
            sections.append(img_tag(f))

    # === 5. Эмбеддинги ===
    sections.append("<h2>5. Эмбеддинги target vs prediction</h2>")
    if results_dir.exists():
        for f in sorted(results_dir.glob("embeddings_target_vs_pred_*.png")):
            sections.append(img_tag(f))

    # === 6. Примеры генераций ===
    sections.append("<h2>6. Примеры генераций</h2>")
    for block in blocks:
        sections.append(build_examples_3way(pred_dir, block, max_examples=2))

    html = HTML_TEMPLATE.replace("{content}", "\n".join(sections))
    out_path.write_text(html, encoding="utf-8")
    print(f"✓ Отчёт собран: {out_path}")
    print(f"  Открой в браузере: file://{out_path.absolute()}")


if __name__ == "__main__":
    main()
