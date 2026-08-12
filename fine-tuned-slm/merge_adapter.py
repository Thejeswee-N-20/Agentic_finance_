"""Merge the QLoRA adapter into the base model as fp16 weights (CPU, so it can
run alongside GPU eval). Output is a plain HF checkpoint ready for GGUF conversion."""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="./out/final_adapter")
    ap.add_argument("--out", default="./merged_fp16")
    args = ap.parse_args()

    print("loading base model on CPU in fp16 ...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map={"": "cpu"}, low_cpu_mem_usage=True
    )
    print("applying adapter ...", flush=True)
    model = PeftModel.from_pretrained(base, args.adapter, torch_dtype=torch.float16)
    print("merging ...", flush=True)
    model = model.merge_and_unload()
    model = model.half()

    print(f"saving to {args.out} ...", flush=True)
    model.save_pretrained(args.out, safe_serialization=True)
    # tokenizer files must sit next to the weights for the GGUF converter
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.save_pretrained(args.out)
    print("done", flush=True)


if __name__ == "__main__":
    main()
