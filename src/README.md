# Pipeline: fine-tuning LLM для клинических рекомендаций (дипломная работа)

Полный пайплайн для дипломной работы — обучение и сравнение двух методов
fine-tuning'а языковых моделей (LoRA + Full FT) на 4 специализированных задачах
генерации клинических рекомендаций по раку лёгкого.

## Что сравнивается

Для каждого из 4 блоков рекомендаций (хирургия / лучевая терапия / системная
терапия / поддерживающая терапия) обучается **2 модели**:

| Метод | Модель | Параметров | VRAM при обучении |
|---|---|---|---|
| **LoRA** (QLoRA 4-bit) | Qwen2.5-3B-Instruct + LoRA r=16 | ~3B (адаптируется ~30M) | ~8-10 GB |
| **Full fine-tune** | Qwen2.5-1.5B-Instruct | 1.5B полностью | ~22-24 GB |
| **Base** (контрольная) | Qwen2.5-3B-Instruct без обучения | — | inference: ~3 GB |

**Итого 8 обучений** (4 блока × 2 метода) + 4 inference от base модели.

## Структура

```
training/
├── config.yaml                  ← все гиперпараметры
├── requirements.txt
├── 01_prepare_dataset.py        ← JSONL → 4 SFT-датасета (train/val/test)
├── 02_visualize_data.py         ← EDA + UMAP эмбеддингов
├── 03_train.py                  ← обучение (LoRA / Full / both)
├── 04_inference.py              ← генерация (base + lora + full)
├── 05_evaluate.py               ← все метрики
├── 06_compare_visualize.py      ← графики сравнения
├── 07_make_report.py            ← финальный HTML-отчёт
├── utils/
│   ├── data_utils.py
│   ├── plotting.py
│   ├── metrics.py               ← BLEU, ROUGE, BERTScore, cosine
│   └── clinical_metrics.py      ← intent accuracy, drug match, safety
└── outputs/
    ├── datasets/                ← готовые train/val/test JSONL
    ├── checkpoints/             ← обученные модели
    │   ├── qwen2.5-3b-lora-surgery/
    │   ├── qwen2.5-3b-lora-radiotherapy/
    │   ├── qwen2.5-3b-lora-systemic/
    │   ├── qwen2.5-3b-lora-supportive/
    │   ├── qwen2.5-1.5b-full-surgery/
    │   └── ... (всего 8 папок)
    ├── predictions/             ← генерации моделей
    ├── metrics/                 ← all_metrics.json
    └── plots/                   ← все графики
        ├── data_eda/
        └── results/
```

## Установка

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
```

## Запуск полного пайплайна

### 1. Сначала сгенерируй данные (если ещё не сделано)
```bash
cd ../synthetic
python generate_unified.py -n 5000 -a 3 -o reports.jsonl
cd ../training
```

### 2. Подготовка датасета
```bash
python 01_prepare_dataset.py
```

### 3. EDA / визуализация данных
```bash
python 02_visualize_data.py
# Графики в outputs/plots/data_eda/
```

### 4. Обучение

**Все 8 моделей последовательно (LoRA + Full FT) — это много часов на 24GB GPU:**
```bash
python 03_train.py --method both
```

**По частям, чтобы быстрее увидеть результаты:**
```bash
# Сначала только LoRA (быстрее, более стабильно)
python 03_train.py --method lora

# Потом Full FT
python 03_train.py --method full

# Или отдельные блоки
python 03_train.py --method lora --blocks surgery
python 03_train.py --method full --blocks surgery
```

### 5. Inference
```bash
# Все три типа моделей × 4 блока = 12 файлов predictions
python 04_inference.py --mode all

# Только обученные модели (без zero-shot baseline)
python 04_inference.py --mode lora full

# Один блок для отладки
python 04_inference.py --blocks surgery
```

### 6. Метрики
```bash
python 05_evaluate.py
# Быстрее без BERTScore
python 05_evaluate.py --skip-bertscore
```

### 7. Графики и отчёт
```bash
python 06_compare_visualize.py
python 07_make_report.py
# Откройте outputs/report.html
```

## Метрики

**Лексические** (поверхностное сходство текстов):
- BLEU, ROUGE-1/2/L, METEOR, Exact Match

**Эмбеддинговые** (семантическое сходство):
- Cosine similarity (multilingual-e5)
- BERTScore (Precision/Recall/F1)

**Клинические** (специфика медицинской задачи):
- **`intent_accuracy`** — совпадает ли клинический intent в generation с истинным
- **`drug_jaccard_mean`** — пересечение упомянутых препаратов
- **`drug_hallucinations_per_sample`** — препараты, которых нет в reference (галлюцинации)
- **`stage_consistency`** — правильно ли упомянута стадия
- **`high_ps_unsafe_rate`** — **критичная safety-метрика**: назначает ли модель агрессивное лечение при ECOG≥3

## Что попадает в дипломную работу

| Глава дипломной | Откуда брать |
|---|---|
| Анализ датасета (распределения) | `outputs/plots/data_eda/*.png` |
| Процесс обучения (loss curves) | `outputs/plots/results/training_curves_*.png`, `loss_compare_*.png` |
| Сравнительная таблица метрик | `outputs/metrics/all_metrics.json` + таблица в `report.html` |
| Графики сравнения моделей | `metrics_compare_*.png`, `metrics_heatmap_all_blocks.png` |
| Радар клинических метрик | `clinical_radar_all_blocks.png` |
| Confusion matrices intent | `confusion_intent_*.png` |
| Эмбеддинговая визуализация | `embeddings_target_vs_pred_*.png` |
| Примеры генераций | Раздел "Примеры" в `report.html` |
| Главный итог | `report.html` целиком |

## Эксперименты для расширения работы

В config.yaml можно поменять и сравнить:
- `models.lora.r` — 8 / 16 / 32 (баланс качество/память)
- `training.num_epochs` — переобучается ли модель на синтетике?
- `training.lora_learning_rate` — 1e-4 / 2e-4 / 5e-4
- `inference.temperature` — 0.0 для воспроизводимости, 0.7 для разнообразия

## VRAM

| Что | Видеопамять |
|---|---|
| LoRA train (Qwen2.5-3B 4-bit, batch=1, grad_accum=16) | ~8-10 GB |
| Full FT train (Qwen2.5-1.5B bf16, batch=1) | ~22-24 GB |
| LoRA inference (Qwen2.5-3B 4-bit) | ~3 GB |
| Full FT inference (Qwen2.5-1.5B bf16) | ~4 GB |

Если памяти не хватает на Full FT:
- Уменьши `models.max_length` с 3072 до 2048
- Поставь `gradient_accumulation_steps: 32`, `per_device_train_batch_size: 1`
- Включи `optim: "adafactor"` вместо `adamw_torch`
