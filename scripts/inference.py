import os
import sys
import argparse
import logging
import warnings
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore", message="torch.dtype is deprecated")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent

HAS_CUDA = torch.cuda.is_available()


def load_model(adapter_path=None, base_model_id="HuggingFaceTB/SmolLM2-360M", device="auto"):
    logger.info(f"Loading base model: {base_model_id}")
    logger.info(f"Device: {'CUDA' if HAS_CUDA else 'CPU'}")

    if device == "auto":
        device = "cuda" if HAS_CUDA else "cpu"
    use_4bit = HAS_CUDA and device == "cuda"

    dtype = torch.float16 if use_4bit else torch.float32

    if use_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs = dict(quantization_config=bnb_config, device_map="auto")
    else:
        model_kwargs = dict(device_map=device)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        **model_kwargs,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter_path and Path(adapter_path).exists():
        from peft import PeftModel
        logger.info(f"Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
        logger.info("Fine-tuned model loaded successfully")
    else:
        logger.info("Using base model (no LoRA adapter loaded)")

    return model, tokenizer


def generate_code(
    model, tokenizer, prompt, language="auto",
    max_new_tokens=128, temperature=0.3,
    top_p=0.95, top_k=50, repetition_penalty=1.1,
):
    lang_hint = "" if language == "auto" else f" ({language})"

    full_prompt = (
        f"### Instruction\n"
        f"Write{lang_hint} code for: {prompt}\n\n"
        f"### Response\n"
    )

    inputs = tokenizer(
        full_prompt, return_tensors="pt", truncation=True, max_length=1024,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_output.split("### Response\n")[-1] if "### Response\n" in full_output else full_output
    return response.strip()


def interactive_mode(model, tokenizer):
    print("\n" + "="*60)
    print("C++/Lua Code Assistant (SmolLM2-135M)")
    if not HAS_CUDA:
        print("⚠️  CPU mode — responses will be slow (30-120s per generation)")
    print("Commands: /lang cpp | /lang lua | /lang auto | exit")
    print("="*60 + "\n")

    current_lang = "auto"

    while True:
        try:
            user_input = input(f"[{current_lang}] >>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if user_input.startswith("/lang"):
            parts = user_input.split()
            if len(parts) > 1:
                current_lang = parts[1].lower()
                print(f"Language set to: {current_lang}")
            continue

        output = generate_code(
            model, tokenizer, user_input,
            language=current_lang,
            max_new_tokens=64 if not HAS_CUDA else 256,
        )
        print(f"\n{output}\n")


def main():
    parser = argparse.ArgumentParser(description="C++/Lua Code Assistant Inference")
    parser.add_argument("--adapter", type=str,
        default=str(ROOT / "outputs" / "smollm2-135m-cpp-lua" / "final"),
        help="Path to LoRA adapter")
    parser.add_argument("--base-model", type=str,
        default="HuggingFaceTB/SmolLM2-135M", help="Base model ID")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt")
    parser.add_argument("--lang", type=str, default="auto",
        choices=["auto", "cpp", "lua"], help="Language hint")
    parser.add_argument("--max-tokens", type=int, default=128,
        help="Max tokens to generate (CPU default: 128, GPU default: 512)")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"], help="Device to run on")

    args = parser.parse_args()

    if HAS_CUDA and args.max_tokens == 128:
        args.max_tokens = 512

    adapter_path = Path(args.adapter) if args.adapter else None
    if adapter_path and not adapter_path.exists():
        logger.info(f"Adapter not found at {adapter_path}, using base model only")
        adapter_path = None

    lang_map = {"cpp": "C++", "lua": "Lua", "auto": "auto"}
    model, tokenizer = load_model(adapter_path, args.base_model, args.device)

    if args.prompt:
        output = generate_code(
            model, tokenizer, args.prompt,
            language=lang_map.get(args.lang, "auto"),
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(output)
    else:
        interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()
