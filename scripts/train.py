import os
import sys
import yaml
import logging
from pathlib import Path

import torch
import transformers
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent


def load_config():
    config_path = ROOT / "config" / "training_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_data():
    data_dir = ROOT / "data" / "processed"
    train_file = data_dir / "train_instructions.parquet"
    eval_file = data_dir / "eval_instructions.parquet"

    if not train_file.exists():
        logger.error(f"Training data not found at {train_file}")
        logger.error("Run `python data/collect_data.py` and `python data/generate_instructions.py` first.")
        sys.exit(1)

    train_ds = Dataset.from_parquet(str(train_file))
    eval_ds = Dataset.from_parquet(str(eval_file)) if eval_file.exists() else None

    logger.info(f"Loaded {len(train_ds)} training samples")
    if eval_ds:
        logger.info(f"Loaded {len(eval_ds)} evaluation samples")

    return train_ds, eval_ds


def setup_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.truncation_side = "left"
    return tokenizer


def setup_model(cfg):
    model_cfg = cfg["model"]
    qlora_cfg = cfg["qlora"]
    lora_cfg = cfg["lora"]

    logger.info(f"Loading base model: {model_cfg['base_model_id']}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=qlora_cfg["load_in_4bit"],
        bnb_4bit_quant_type=qlora_cfg["quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, qlora_cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=qlora_cfg["bnb_4bit_use_double_quant"],
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model_id"],
        quantization_config=bnb_config,
        torch_dtype=getattr(torch, model_cfg["torch_dtype"]),
        attn_implementation=model_cfg["attn_implementation"],
        use_cache=model_cfg["use_cache"],
        trust_remote_code=True,
        device_map="auto",
    )

    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer


def formatting_func(example):
    return example["text"]


def main():
    set_seed(42)
    cfg = load_config()

    logger.info("=== Loading Data ===")
    train_ds, eval_ds = load_data()

    logger.info("=== Setting up Tokenizer ===")
    tokenizer = setup_tokenizer(cfg["model"]["base_model_id"])

    logger.info("=== Setting up Model with QLoRA ===")
    model, tokenizer = setup_model(cfg)

    train_cfg = cfg["training"]
    output_dir = ROOT / train_cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["max_grad_norm"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        eval_steps=train_cfg["eval_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        metric_for_best_model=train_cfg["metric_for_best_model"],
        greater_is_better=train_cfg["greater_is_better"],
        bf16=train_cfg["bf16"],
        tf32=train_cfg["tf32"],
        dataloader_num_workers=train_cfg["dataloader_num_workers"],
        remove_unused_columns=train_cfg["remove_unused_columns"],
        report_to=train_cfg["report_to"] if os.environ.get("WANDB_API_KEY") else "none",
        run_name=train_cfg["run_name"],
        ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
        logging_strategy="steps",
        save_strategy="steps",
        eval_strategy="steps" if eval_ds is not None else "no",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        formatting_func=formatting_func,
        max_seq_length=train_cfg["max_seq_length"],
        packing=train_cfg["packing"],
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            pad_to_multiple_of=8,
            padding=True,
        ),
    )

    logger.info("=== Starting Training ===")
    trainer.train()

    logger.info("=== Saving Final Model ===")
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info(f"Model saved to {final_dir}")

    logger.info("=== Training Complete ===")


if __name__ == "__main__":
    main()
