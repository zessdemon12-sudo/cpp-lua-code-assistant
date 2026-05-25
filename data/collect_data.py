import os
import yaml
import logging
from pathlib import Path

from datasets import load_dataset, DatasetDict, concatenate_datasets, Dataset
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "processed"
CACHE_DIR = Path(__file__).parent / "cache"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_config():
    config_path = Path(__file__).parent.parent / "config" / "data_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def filter_code(examples, min_bytes=100, max_bytes=50000):
    lengths = [len(c.encode("utf-8")) for c in examples["content"]]
    mask = [
        min_bytes <= l <= max_bytes
        for l in lengths
    ]
    return mask


def deduplicate_by_hash(dataset, hash_col="sha256"):
    seen = set()
    dedup_indices = []
    for i, h in enumerate(dataset[hash_col]):
        if h not in seen:
            seen.add(h)
            dedup_indices.append(i)
    return dataset.select(dedup_indices)


def collect_the_stack_v2(cfg):
    lang_cfg = cfg["datasets"]["the_stack_v2"]
    if not lang_cfg["enabled"]:
        logger.info("The Stack v2 disabled, skipping.")
        return None

    all_ds = []
    for lang in lang_cfg["languages"]:
        logger.info(f"Loading The Stack v2 for language: {lang}")
        ds = load_dataset(
            "bigcode/the-stack-v2",
            data_dir=f"data/{lang}",
            split=lang_cfg["split"],
            streaming=False,
            cache_dir=str(CACHE_DIR),
            trust_remote_code=True,
        )
        logger.info(f"  Raw {lang} size: {len(ds)}")

        ds = ds.filter(
            lambda x: filter_code(x, lang_cfg["min_size_bytes"], lang_cfg["max_size_bytes"]),
            batched=True,
        )

        if lang_cfg["deduplicate"]:
            ds = deduplicate_by_hash(ds)

        max_n = lang_cfg["max_samples_per_language"]
        if len(ds) > max_n:
            ds = ds.select(range(max_n))
            logger.info(f"  Trimmed {lang} to {max_n} samples")

        ds = ds.add_column("language", [lang] * len(ds))
        all_ds.append(ds)
        logger.info(f"  Final {lang} size: {len(ds)}")

    if not all_ds:
        return None
    return concatenate_datasets(all_ds)


def collect_code_alpaca(cfg):
    ca_cfg = cfg["datasets"]["code_alpaca"]
    if not ca_cfg["enabled"]:
        return None

    logger.info("Loading CodeAlpaca dataset...")
    ds = load_dataset(ca_cfg["name"], split="train", cache_dir=str(CACHE_DIR))
    max_n = ca_cfg["max_samples"]
    if len(ds) > max_n:
        ds = ds.select(range(max_n))

    def format_alpaca(example):
        return {
            "instruction": example["instruction"],
            "response": example["output"],
            "language": "unknown",
        }

    ds = ds.map(format_alpaca, remove_columns=ds.column_names)
    return ds


def main():
    cfg = load_config()

    logger.info("=== Step 1: Collect The Stack v2 (C++ and Lua) ===")
    stack_ds = collect_the_stack_v2(cfg)

    logger.info("=== Step 2: Collect CodeAlpaca ===")
    alpaca_ds = collect_code_alpaca(cfg)

    logger.info("=== Step 3: Merge and split ===")
    train_parts = []
    eval_parts = []

    eval_ratio = cfg["output"]["eval_split_ratio"]

    if stack_ds is not None:
        stack_ds = stack_ds.train_test_split(test_size=eval_ratio, seed=42)
        train_parts.append(stack_ds["train"])
        eval_parts.append(stack_ds["test"])
        logger.info(f"Stack: {len(stack_ds['train'])} train, {len(stack_ds['test'])} eval")

    if alpaca_ds is not None:
        alpaca_ds = alpaca_ds.train_test_split(test_size=eval_ratio, seed=42)
        train_parts.append(alpaca_ds["train"])
        eval_parts.append(alpaca_ds["test"])
        logger.info(f"Alpaca: {len(alpaca_ds['train'])} train, {len(alpaca_ds['test'])} eval")

    if not train_parts:
        logger.error("No datasets collected. Check configuration.")
        return

    train_dataset = concatenate_datasets(train_parts)
    eval_dataset = concatenate_datasets(eval_parts)

    train_file = str(DATA_DIR / "train_dataset.parquet")
    eval_file = str(DATA_DIR / "eval_dataset.parquet")

    train_dataset.to_parquet(train_file)
    eval_dataset.to_parquet(eval_file)

    logger.info(f"Train dataset saved: {train_file} ({len(train_dataset)} samples)")
    logger.info(f"Eval dataset saved: {eval_file} ({len(eval_dataset)} samples)")


if __name__ == "__main__":
    main()
