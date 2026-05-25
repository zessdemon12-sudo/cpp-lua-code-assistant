import os
import re
import yaml
import random
import logging
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset, concatenate_datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "processed"
CACHE_DIR = Path(__file__).parent / "cache"
DATA_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)

SYNTHESIS_TEMPLATES = {
    "C++": [
        "Write a C++ function that {action}",
        "Implement a C++ class for {action}",
        "Create a C++ program that {action}",
        "Write a C++ snippet to {action}",
        "Implement the following in C++: {action}",
    ],
    "Lua": [
        "Write a Lua function that {action}",
        "Create a Lua module for {action}",
        "Write a Lua script that {action}",
        "Implement the following in Lua: {action}",
        "Write a Lua function to {action}",
    ],
}

TASK_EXTRACTORS = {
    "C++": [
        r"(?:function|method|class)\s+\w+\s*(?:\([^)]*\))?\s*(?:{|\n)",
        r"(?:bool|int|float|double|void|char|string|auto|size_t)\s+\w+\s*\([^)]*\)\s*(?:{|\n)",
        r"(?:class|struct)\s+\w+",
    ],
    "Lua": [
        r"function\s+\w+[\.:]\w+\s*\([^)]*\)",
        r"function\s+\w+\s*\([^)]*\)",
        r"local\s+function\s+\w+\s*\([^)]*\)",
    ],
}

SIGNATURE_ACTIONS = {
    "sort": "sort an array of elements",
    "find": "find a specific element in a container",
    "search": "search through data",
    "merge": "merge two data structures",
    "parse": "parse input data",
    "validate": "validate input",
    "convert": "convert between formats",
    "calculate": "calculate a result",
    "compute": "compute a value",
    "process": "process data",
    "handle": "handle an event or input",
    "load": "load data from source",
    "save": "save data to destination",
    "read": "read data from a file or stream",
    "write": "write data to a file or stream",
    "open": "open a file or connection",
    "close": "close a file or connection",
    "init": "initialize a component",
    "cleanup": "clean up resources",
    "reset": "reset state",
    "update": "update state or data",
    "get": "get or retrieve a value",
    "set": "set a value",
    "add": "add elements together",
    "remove": "remove elements",
    "insert": "insert elements",
    "delete": "delete elements",
    "create": "create a new instance",
    "destroy": "destroy an instance",
    "build": "build a structure",
    "render": "render output",
    "draw": "draw graphics",
    "move": "move objects",
    "copy": "copy data",
    "compare": "compare values",
    "check": "check a condition",
    "test": "test functionality",
}


def extract_task_from_code(code, language):
    for pattern in TASK_EXTRACTORS.get(language, []):
        m = re.search(pattern, code, re.MULTILINE)
        if m:
            sig = m.group(0)
            for keyword, action in SIGNATURE_ACTIONS.items():
                if keyword in sig.lower():
                    return action
    return None


def generate_instruction_templates(cfg):
    yield from _load_existing_instructions()
    yield from _synthesize_from_code(cfg)
    yield from _load_code_alpaca()


def _load_existing_instructions():
    logger.info("Loading existing instruction datasets...")
    try:
        ds = load_dataset(
            "bigcode/self-instruct",
            split="train",
            cache_dir=str(CACHE_DIR),
        )
        ds = ds.filter(lambda x: x["lang"] in {"cpp", "lua"})

        for example in ds:
            yield {
                "instruction": example["instruction"],
                "response": example["response"],
                "language": "C++" if example["lang"] == "cpp" else "Lua",
            }
    except Exception as e:
        logger.warning(f"Could not load existing instructions: {e}")


def _synthesize_from_code(cfg):
    ss_cfg = cfg["datasets"]["self_instruct"]
    if not ss_cfg["enabled"]:
        return

    train_path = DATA_DIR / "train_dataset.parquet"
    if not train_path.exists():
        logger.warning("No raw train dataset found, skipping synthesis.")
        return

    logger.info("Synthesizing instruction pairs from raw code...")
    df = pd.read_parquet(train_path)

    if "language" not in df.columns:
        logger.warning("No 'language' column in dataset.")
        return

    templates = ss_cfg["templates"]
    lang_dfs = []
    for lang in ["C++", "Lua"]:
        lang_df = df[df["language"] == lang]
        if lang_df.empty:
            continue

        instructions = []
        for _, row in lang_df.iterrows():
            code = row.get("content", "")
            if isinstance(code, bytes):
                code = code.decode("utf-8", errors="replace")

            task = extract_task_from_code(code, lang)
            if task is None:
                funcs = re.findall(r"(?:fn|def|function)\s+(\w+)", code[:500])
                if funcs:
                    task = f"implement the {funcs[0]} function"
                else:
                    task = "perform a common operation"

            template = random.choice(templates + tuple(SYNTHESIS_TEMPLATES.get(lang, templates)))
            instruction = template.format(language=lang, task=task)

            instructions.append({
                "instruction": instruction,
                "response": code,
                "language": lang,
            })

        synth_df = pd.DataFrame(instructions)
        max_n = int(len(synth_df) * ss_cfg["synthesis_ratio"])
        synth_df = synth_df.sample(n=min(max_n, len(synth_df)), random_state=42)
        lang_dfs.append(synth_df)
        logger.info(f"Synthesized {len(synth_df)} pairs for {lang}")

    for d in lang_dfs:
        for _, row in d.iterrows():
            yield row.to_dict()


def _load_code_alpaca():
    try:
        ds = load_dataset("yahma/alpaca-cleaned", split="train")
        for example in ds:
            yield {
                "instruction": example["instruction"],
                "response": example["output"],
                "language": "General",
            }
    except Exception as e:
        logger.warning(f"Could not load alpaca instructions: {e}")


def format_instruction(row, fmt_template):
    lang = row.get("language", "unknown")
    code = row.get("response", row.get("content", ""))
    if isinstance(code, bytes):
        code = code.decode("utf-8", errors="replace")

    instruction = row.get("instruction", "")

    return fmt_template.format(
        instruction=instruction,
        language=lang.lower(),
        code=code,
    )


def main():
    config_path = Path(__file__).parent.parent / "config" / "data_config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    fmt = cfg["preprocessing"]["instruction_format"]
    max_len = cfg["preprocessing"]["max_length"]

    logger.info("Generating instruction pairs...")
    records = []
    for rec in generate_instruction_templates(cfg):
        text = format_instruction(rec, fmt)
        if len(text) > max_len * 4:
            text = text[:max_len * 4]
        records.append({
            "text": text,
            "language": rec["language"],
            "instruction": rec["instruction"],
        })

    random.shuffle(records)
    logger.info(f"Generated {len(records)} instruction pairs total")

    train_file = DATA_DIR / "train_instructions.parquet"
    eval_file = DATA_DIR / "eval_instructions.parquet"

    df = pd.DataFrame(records)
    eval_ratio = cfg["output"]["eval_split_ratio"]
    n_eval = max(1, int(len(df) * eval_ratio))
    eval_df = df.iloc[:n_eval]
    train_df = df.iloc[n_eval:]

    train_df.to_parquet(train_file, index=False)
    eval_df.to_parquet(eval_file, index=False)

    lang_counts = df["language"].value_counts().to_dict()
    logger.info(f"Final instruction dataset: {len(train_df)} train, {len(eval_df)} eval")
    logger.info(f"Language distribution: {lang_counts}")


if __name__ == "__main__":
    main()
