import argparse
import re

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DATASET_ID = "FinGPT/fingpt-forecaster-dow30-202305-202405"


def parse_llama_prompt(p):
    m = re.search(r"<<SYS>>(.*?)<</SYS>>", p, re.S)
    system = m.group(1).strip() if m else ""
    user = p.split("<</SYS>>", 1)[1].replace("[/INST]", "").strip()
    return system, user


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="./out/final_adapter")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=600)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    ds = load_dataset(DATASET_ID, split="test")
    row = ds[args.index]
    system, user = parse_llama_prompt(row["prompt"])
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )
    gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print("=== SYMBOL:", row["symbol"], "PERIOD:", row["period"], "GOLD LABEL:", row["label"], "===")
    print("=== MODEL OUTPUT ===")
    print(gen)
    print("\n=== REFERENCE ANSWER (first 800 chars) ===")
    print(row["answer"][:800])


if __name__ == "__main__":
    main()
