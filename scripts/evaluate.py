import os
import re
import json
import logging
import warnings
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore", message="torch.dtype is deprecated")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
HAS_CUDA = torch.cuda.is_available()

EVAL_PROMPTS = {
    "C++": [
        "Write a C++ function that reverses a linked list.",
        "Implement a C++ class for a thread-safe queue.",
        "Write a C++ program that reads a CSV file and prints the sum of each column.",
        "Implement a C++ template function to find the maximum element in an array.",
        "Write a C++ lambda that sorts a vector of strings by length.",
        "Implement a C++ RAII wrapper for a FILE* handle.",
        "Write a C++ constexpr function to compute factorial at compile time.",
        "Implement a C++ move constructor and move assignment operator for a String class.",
        "Write a C++ function that performs binary search on a sorted vector.",
        "Implement a C++ variadic template function that prints all arguments.",
    ],
    "Lua": [
        "Write a Lua function that deep-copies a table.",
        "Implement a Lua class using metatables for a 2D vector.",
        "Write a Lua function that reads a file line by line.",
        "Implement a Lua coroutine-based producer-consumer pattern.",
        "Write a Lua module that provides math utility functions.",
        "Implement a Lua function to merge two tables recursively.",
        "Write a Lua iterator that yields Fibonacci numbers.",
        "Implement a Lua function that serializes a table to a string.",
        "Write a Lua script that implements a simple event dispatcher.",
        "Implement a Lua function to detect if a value is a callable function.",
    ],
}


def load_model(model_id, adapter_path=None):
    logger.info(f"Loading model: {model_id} on {'CUDA' if HAS_CUDA else 'CPU'}")

    if HAS_CUDA:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter_path and Path(adapter_path).exists():
        from peft import PeftModel
        logger.info(f"Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()

    return model, tokenizer


def generate(model, tokenizer, prompt, max_new_tokens=96):
    full_prompt = f"### Instruction\n{prompt}\n\n### Response\n"

    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full.split("### Response\n")[-1] if "### Response\n" in full else full
    return response.strip()


def extract_code(text, language):
    pattern = rf"```{language.lower()}\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0]
    code_lines = [l for l in text.split("\n") if l.strip() and not l.startswith("###")]
    return "\n".join(code_lines)


def evaluate_model(model, tokenizer, label="model"):
    results = {}
    for lang, prompts in EVAL_PROMPTS.items():
        lang_results = []
        n = min(len(prompts), 3) if not HAS_CUDA else len(prompts)
        for prompt in prompts[:n]:
            try:
                output = generate(model, tokenizer, prompt)
                code = extract_code(output, lang)
                lang_results.append({
                    "prompt": prompt,
                    "response": output,
                    "extracted_code": code,
                    "code_length": len(code),
                    "has_code_block": bool(re.search(r"```", output)),
                })
            except Exception as e:
                logger.error(f"Error on prompt '{prompt[:50]}...': {e}")
                lang_results.append({"prompt": prompt, "error": str(e)})
        results[lang] = lang_results

    for lang, res in results.items():
        successes = [r for r in res if "error" not in r]
        s = {
            "total": len(res),
            "success": len(successes),
            "avg_code_length": sum(r.get("code_length", 0) for r in successes) / max(len(successes), 1),
            "with_code_block": sum(1 for r in successes if r.get("has_code_block")),
        }
        logger.info(f"  {lang}: {s['success']}/{s['total']} success, "
                     f"avg {s['avg_code_length']:.0f} chars, "
                     f"{s['with_code_block']} with code blocks")

    return results


def main():
    adapter_path = ROOT / "outputs" / "smollm2-360m-cpp-lua" / "final"
    base_model_id = "HuggingFaceTB/SmolLM2-360M"

    logger.info("Loading base model...")
    model, tokenizer = load_model(base_model_id)
    evaluate_model(model, tokenizer, label="Base Model")


if __name__ == "__main__":
    main()
