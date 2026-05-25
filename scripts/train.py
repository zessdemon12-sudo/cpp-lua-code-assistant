import os
import sys
import yaml
import logging
from pathlib import Path

import torch
import transformers
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
HAS_CUDA = torch.cuda.is_available()


def load_config():
    with open(ROOT / "config" / "training_config.yaml") as f:
        return yaml.safe_load(f)


def load_data():
    data_dir = ROOT / "data" / "processed"
    train_file = data_dir / "train_instructions.parquet"
    eval_file = data_dir / "eval_instructions.parquet"

    if not train_file.exists():
        logger.error(f"Training data not found at {train_file}")
        sys.exit(1)

    train_ds = Dataset.from_parquet(str(train_file))
    eval_ds = Dataset.from_parquet(str(eval_file)) if eval_file.exists() else None

    logger.info(f"Loaded {len(train_ds)} training samples")
    if eval_ds:
        logger.info(f"Loaded {len(eval_ds)} evaluation samples")

    return train_ds, eval_ds


def tokenize_function(examples, tokenizer, max_length):
    texts = examples["text"]
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=max_length,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def main():
    set_seed(42)
    cfg = load_config()
    train_cfg = cfg["training"]

    logger.info("=== Loading Data ===")
    train_ds, eval_ds = load_data()

    logger.info("=== Setting up Tokenizer ===")
    model_id = cfg["model"]["base_model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info("=== Tokenizing Dataset ===")
    max_len = train_cfg["max_seq_length"]
    train_ds = train_ds.map(
        lambda x: tokenize_function(x, tokenizer, max_len),
        batched=True,
        remove_columns=train_ds.column_names,
    )
    if eval_ds:
        eval_ds = eval_ds.map(
            lambda x: tokenize_function(x, tokenizer, max_len),
            batched=True,
            remove_columns=eval_ds.column_names,
        )

    logger.info("=== Loading Model ===")
    dtype = getattr(torch, cfg["model"]["torch_dtype"])
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    logger.info("=== Applying LoRA ===")
    lora_cfg = cfg["lora"]
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

    output_dir = ROOT / train_cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["max_grad_norm"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        eval_steps=train_cfg["eval_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        logging_strategy="steps",
        save_strategy="steps",
        eval_strategy="steps" if eval_ds is not None else "no",
        bf16=False,
        tf32=False,
        fp16=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
        run_name=train_cfg["run_name"],
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        padding=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    logger.info("=== Starting Training ===")
    trainer.train()

    logger.info("=== Saving Model ===")
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info(f"Model saved to {final_dir}")
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
