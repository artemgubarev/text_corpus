"""Шаг 03: Обучение 4 specialized моделей.

Поддерживает два метода:
    --method lora  (default) — QLoRA 4-bit на Qwen2.5-3B-Instruct
    --method full           — Full fine-tune на Qwen2.5-1.5B-Instruct
    --method both           — последовательно оба, для сравнения

Использование:
    python 03_train.py --method lora                           # все 4 LoRA модели
    python 03_train.py --method full                           # все 4 full FT
    python 03_train.py --method both                           # 8 моделей всего
    python 03_train.py --method lora --blocks surgery systemic # выборочно
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.data_utils import load_config, load_jsonl, set_seed, ensure_dir


def train_one_block(block: str, method: str, cfg: dict) -> None:
    """Обучить одну модель для конкретного блока конкретным методом."""
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
    from trl import SFTTrainer, SFTConfig

    # ============================================================
    # Конфигурация в зависимости от метода
    # ============================================================
    if method == "lora":
        model_id = cfg["models"]["base_model_id"]
        short_name = cfg["models"]["short_name_lora"]
        learning_rate = cfg["training"]["lora_learning_rate"]
        use_quantization = True
    elif method == "full":
        model_id = cfg["models"]["base_model_id_full_ft"]
        short_name = cfg["models"]["short_name_full"]
        learning_rate = cfg["training"]["full_learning_rate"]
        use_quantization = False
    else:
        raise ValueError(f"Неизвестный метод: {method}")

    output_dir = ensure_dir(
        Path(cfg["training"]["output_dir"]) / f"{short_name}-{block}"
    )

    print(f"\n{'=' * 70}")
    print(f"=== Обучение: {block.upper()} / {method.upper()} ===")
    print(f"Model:        {model_id}")
    print(f"Output:       {output_dir}")
    print(f"Learning rate: {learning_rate}")
    print(f"{'=' * 70}")

    # ============================================================
    # Tokenizer
    # ============================================================
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ============================================================
    # Датасет
    # ============================================================
    data_dir = Path(cfg["data"]["output_dir"])
    train_records = load_jsonl(data_dir / f"{block}_train.jsonl")
    val_records = load_jsonl(data_dir / f"{block}_val.jsonl")
    print(f"\nДатасет: train={len(train_records)}, val={len(val_records)}")

    def to_text(records):
        return [{"text": tokenizer.apply_chat_template(
            r["messages"], tokenize=False, add_generation_prompt=False
        )} for r in records]

    train_ds = Dataset.from_list(to_text(train_records))
    val_ds = Dataset.from_list(to_text(val_records))

    # ============================================================
    # Модель
    # ============================================================
    print(f"\nЗагружаю модель...")
    if use_quantization:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)
    else:
        # Full fine-tune — без quantization, в bf16
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model.config.use_cache = False

    # ============================================================
    # PEFT / LoRA — только если method=lora
    # ============================================================
    peft_config = None
    if method == "lora":
        lora_cfg = cfg["models"]["lora"]
        peft_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Full fine-tune: {n_params:,} trainable params")

    # ============================================================
    # Training arguments
    # ============================================================
    train_cfg = cfg["training"]
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=learning_rate,
        warmup_ratio=train_cfg["warmup_ratio"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        weight_decay=train_cfg["weight_decay"],
        logging_steps=train_cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=train_cfg["eval_steps"],
        save_strategy="steps",
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        bf16=train_cfg["bf16"],
        optim=train_cfg["optim"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=train_cfg["seed"],
        report_to=["tensorboard"] + (["wandb"] if cfg["logging"]["use_wandb"] else []),
        run_name=f"{short_name}-{block}",
        max_seq_length=cfg["models"]["max_length"],
        dataset_text_field="text",
        packing=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=sft_config,
        peft_config=peft_config,  # None для full FT
    )

    trainer.train()

    # ============================================================
    # Сохранение
    # ============================================================
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\n✓ Сохранено: {final_dir}")

    # Сохраняем историю для визуализации обучения
    with open(output_dir / "train_log_history.json", "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)

    # Освобождаем память перед следующей моделью
    del model, trainer
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--method", type=str, default="lora",
        choices=["lora", "full", "both"],
        help="lora=QLoRA на 3B, full=Full FT на 1.5B, both=оба последовательно",
    )
    parser.add_argument(
        "--blocks", nargs="+", default=None,
        choices=["surgery", "radiotherapy", "systemic", "supportive"],
        help="Конкретные блоки (по умолчанию все 4)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])

    blocks = args.blocks or cfg["models"]["blocks"]
    methods = ["lora", "full"] if args.method == "both" else [args.method]

    total_trainings = len(blocks) * len(methods)
    print(f"\nЗапуск: {total_trainings} обучений = {len(blocks)} блоков × {len(methods)} методов\n")

    for method in methods:
        for block in blocks:
            train_one_block(block, method, cfg)

    print(f"\n{'=' * 70}")
    print(f"✓ Все запрошенные модели обучены")
    print(f"  Методы:  {methods}")
    print(f"  Блоки:   {blocks}")
    print(f"  Чекпойнты: {cfg['training']['output_dir']}/")


if __name__ == "__main__":
    main()
