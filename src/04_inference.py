"""Шаг 04: Inference на test split.

Для каждого из 4 блоков генерирует ответы тремя версиями модели:
    1. base (zero-shot, Qwen2.5-3B-Instruct без обучения) — общий baseline
    2. lora (Qwen2.5-3B + LoRA адаптер этого блока)
    3. full (Qwen2.5-1.5B полностью дообученная под этот блок)

Результат: outputs/predictions/predictions_{source}_{block}.jsonl

Использование:
    python 04_inference.py --mode all           # base + lora + full для 4 блоков
    python 04_inference.py --mode lora full     # только обученные модели
    python 04_inference.py --blocks surgery     # только один блок
"""

import argparse
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils.data_utils import load_config, load_jsonl, save_jsonl, ensure_dir


def load_model_and_tokenizer(model_id: str, adapter_path: str | None = None,
                              use_4bit: bool = True):
    """Загрузить модель и опционально подключить LoRA-адаптер.

    use_4bit=True для inference QLoRA-моделей и для zero-shot baseline (экономит память).
    use_4bit=False для full FT моделей — они уже маленькие (1.5B) и были обучены в bf16.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }
    if use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()

    if adapter_path:
        from peft import PeftModel
        print(f"  Подключаю LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()

    return model, tokenizer


def generate_batch(records: list[dict], model, tokenizer, cfg: dict,
                    source_label: str) -> list[dict]:
    """Генерация ответов для всех записей."""
    import torch
    inf_cfg = cfg["inference"]
    results = []

    for r in tqdm(records, desc=f"Inference ({source_label})"):
        messages = r["messages"][:-1]  # убираем assistant (его генерируем)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=inf_cfg["max_new_tokens"],
                temperature=inf_cfg["temperature"],
                top_p=inf_cfg["top_p"],
                do_sample=inf_cfg["do_sample"],
                num_beams=inf_cfg["num_beams"],
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        ).strip()

        results.append({
            "id": r["id"],
            "base_id": r["base_id"],
            "source": source_label,
            "block": r["block"],
            "input": r["input"],
            "target": r["target"],
            "prediction": generated,
            # Метаданные для метрик
            "diagnosis_id": r.get("diagnosis_id"),
            "stage": r.get("stage"),
            "stage_group": r.get("stage_group"),
            "ecog": r.get("ecog"),
            "block_intent": r.get("block_intent"),
            "actionable_gene": r.get("actionable_gene"),
            "actionable_variant_short": r.get("actionable_variant_short"),
        })

    return results


def run_inference_for(source_label: str, model_id: str,
                       adapter_path: str | None, use_4bit: bool,
                       test_records: list[dict], cfg: dict,
                       output_path: Path) -> None:
    """Полный цикл одной модели на одном блоке."""
    print(f"\n=== {source_label} ===")
    print(f"  Base model: {model_id}")
    if adapter_path:
        print(f"  Adapter:    {adapter_path}")

    model, tokenizer = load_model_and_tokenizer(model_id, adapter_path, use_4bit)

    n_limit = cfg["inference"].get("num_test_samples")
    test_subset = test_records[:n_limit] if n_limit else test_records
    print(f"  Test samples: {len(test_subset)}")

    results = generate_batch(test_subset, model, tokenizer, cfg, source_label)
    save_jsonl(results, output_path)
    print(f"✓ Сохранено: {output_path}")

    del model, tokenizer
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--mode", nargs="+", default=["base", "lora", "full"],
        choices=["base", "lora", "full", "all"],
        help="Какие версии инферить",
    )
    parser.add_argument(
        "--blocks", nargs="+", default=None,
        choices=["surgery", "radiotherapy", "systemic", "supportive"],
    )
    args = parser.parse_args()

    if "all" in args.mode:
        args.mode = ["base", "lora", "full"]

    cfg = load_config(args.config)
    data_dir = Path(cfg["data"]["output_dir"])
    out_dir = ensure_dir(cfg["inference"]["output_dir"])
    ckpt_dir = Path(cfg["training"]["output_dir"])

    blocks = args.blocks or cfg["models"]["blocks"]

    for block in blocks:
        test_path = data_dir / f"{block}_test.jsonl"
        if not test_path.exists():
            print(f"⚠ Нет {test_path}. Сначала: python 01_prepare_dataset.py")
            continue
        test_records = load_jsonl(test_path)

        # === Base zero-shot (Qwen2.5-3B без обучения) ===
        if "base" in args.mode:
            run_inference_for(
                source_label=f"base_{block}",
                model_id=cfg["models"]["base_model_id"],
                adapter_path=None,
                use_4bit=True,
                test_records=test_records,
                cfg=cfg,
                output_path=out_dir / f"predictions_base_{block}.jsonl",
            )

        # === LoRA ===
        if "lora" in args.mode:
            short_name = cfg["models"]["short_name_lora"]
            adapter = ckpt_dir / f"{short_name}-{block}" / "final"
            if adapter.exists():
                run_inference_for(
                    source_label=f"lora_{block}",
                    model_id=cfg["models"]["base_model_id"],
                    adapter_path=str(adapter),
                    use_4bit=True,
                    test_records=test_records,
                    cfg=cfg,
                    output_path=out_dir / f"predictions_lora_{block}.jsonl",
                )
            else:
                print(f"⚠ LoRA adapter не найден: {adapter}")
                print(f"  Запустите: python 03_train.py --method lora --blocks {block}")

        # === Full fine-tune ===
        if "full" in args.mode:
            short_name = cfg["models"]["short_name_full"]
            ckpt = ckpt_dir / f"{short_name}-{block}" / "final"
            if ckpt.exists():
                run_inference_for(
                    source_label=f"full_{block}",
                    model_id=str(ckpt),  # full FT — это новая модель целиком
                    adapter_path=None,
                    use_4bit=False,
                    test_records=test_records,
                    cfg=cfg,
                    output_path=out_dir / f"predictions_full_{block}.jsonl",
                )
            else:
                print(f"⚠ Full FT checkpoint не найден: {ckpt}")
                print(f"  Запустите: python 03_train.py --method full --blocks {block}")


if __name__ == "__main__":
    main()
