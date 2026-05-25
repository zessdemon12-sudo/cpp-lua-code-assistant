import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).parent.parent
ADAPTER_PATH = ROOT / "outputs" / "smollm2-135m-cpp-lua" / "final"
OUTPUT_DIR = ROOT / "outputs" / "gguf"
BASE_MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID, torch_dtype=torch.float32, trust_remote_code=True,
)

if ADAPTER_PATH.exists():
    print("Merging LoRA adapter...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model = model.merge_and_unload()
    print("LoRA merged.")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

hf_dir = OUTPUT_DIR / "hf_model"
hf_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(hf_dir))
tokenizer.save_pretrained(str(hf_dir))
print(f"Merged model saved to {hf_dir}")

from gguf import GGUFWriter, GGMLQuantizationType, MODEL_ARCH

cfg = model.config
arch = "llama"
gguf_path = OUTPUT_DIR / "smollm2-135m-cpp-lua-code.gguf"

print(f"Writing GGUF: {gguf_path}")
w = GGUFWriter(str(gguf_path), arch)

w.add_name("SmolLM2-135M-CPP-Lua-Code")
w.add_description("Fine-tuned for C++ and Lua code generation")
w.add_context_length(getattr(cfg, "max_position_embeddings", 1024))
w.add_block_count(cfg.num_hidden_layers)
w.add_embedding_length(cfg.hidden_size)
w.add_feed_forward_length(cfg.intermediate_size)
w.add_head_count(cfg.num_attention_heads)
if hasattr(cfg, "num_key_value_heads"):
    w.add_head_count_kv(cfg.num_key_value_heads)
w.add_layer_norm_rms_eps(cfg.rms_norm_eps)

if hasattr(cfg, "rope_theta"):
    w.add_rope_freq_base(cfg.rope_theta)

w.add_file_type(GGMLQuantizationType.F32)

print("Writing tensors...")
state_dict = model.state_dict()
for name, tensor in state_dict.items():
    data = tensor.contiguous().float().numpy()
    w.add_tensor(name, data)

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()

size_mb = Path(gguf_path).stat().st_size / 1e6
print(f"\nDone! GGUF saved ({size_mb:.1f} MB): {gguf_path}")
print(f"Use: py -m llama_cpp --model \"{gguf_path}\" --prompt \"Write a C++ function\"")
