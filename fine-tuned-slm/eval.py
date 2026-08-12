import argparse
import json
import re
import time

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DATASET_ID = "FinGPT/fingpt-forecaster-dow30-202305-202405"

BUCKETS = [
    "up by 0-1%", "up by 1-2%", "up by 2-3%", "up by 3-4%", "up by 4-5%", "up by more than 5%",
    "down by 0-1%", "down by 1-2%", "down by 2-3%", "down by 3-4%", "down by 4-5%",
    "down by more than 5%",
]
# ordinal index: negative = down, positive = up, used for bucket-distance metrics
ORDINAL = {
    "down by more than 5%": -6, "down by 4-5%": -5, "down by 3-4%": -4,
    "down by 2-3%": -3, "down by 1-2%": -2, "down by 0-1%": -1,
    "up by 0-1%": 1, "up by 1-2%": 2, "up by 2-3%": 3,
    "up by 3-4%": 4, "up by 4-5%": 5, "up by more than 5%": 6,
}


def parse_llama_prompt(p):
    m = re.search(r"<<SYS>>(.*?)<</SYS>>", p, re.S)
    system = m.group(1).strip() if m else ""
    user = p.split("<</SYS>>", 1)[1].replace("[/INST]", "").strip()
    return system, user


def _bucket_in(seg):
    """Find a forecast bucket inside a chunk of text, tolerating spacing/dash variants."""
    seg = seg.lower().replace("–", "-").replace("—", "-")
    for b in BUCKETS:
        if b in seg:
            return b
    m = re.search(r"(up|down)\s+by\s+(more\s+than\s+5|\d\s*-\s*\d)\s*%?", seg)
    if m:
        mag = re.sub(r"\s+", " ", m.group(2)).replace(" -", "-").replace("- ", "-")
        cand = f"{m.group(1)} by {mag}%"
        if cand in ORDINAL:
            return cand
    return None


def extract_prediction(text):
    """Pull the forecast bucket out of the generated answer.

    The answer contains a '[Prediction & Analysis]' section header as well as a
    'Prediction: <bucket>' line, so prefer the line that actually carries a bucket
    rather than the first thing that looks like a prediction label.
    """
    for line in text.splitlines():
        if re.match(r"\s*\**\s*prediction\s*\**\s*:", line, re.I):
            b = _bucket_in(line)
            if b:
                return b
    # fall back to scanning the whole generation
    return _bucket_in(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="./out/final_adapter")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=600)
    ap.add_argument("--base_only", action="store_true", help="skip adapter, eval base model")
    ap.add_argument("--out", default="eval_results.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.bfloat16
    )
    if not args.base_only:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    ds = load_dataset(DATASET_ID, split="test").shuffle(seed=args.seed).select(range(args.n))

    rows = []
    t0 = time.time()
    for i, row in enumerate(ds):
        system, user = parse_llama_prompt(row["prompt"])
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                top_p=None, top_k=None, temperature=None,
                pad_token_id=tok.eos_token_id,
            )
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = extract_prediction(gen)
        rows.append({
            "i": i, "symbol": row["symbol"], "period": row["period"],
            "gold": row["label"], "pred": pred, "gen": gen,
        })
        print(f"[{i+1}/{args.n}] {row['symbol']:5s} gold={row['label']:22s} "
              f"pred={pred}  ({time.time()-t0:.0f}s)", flush=True)

    scored = [r for r in rows if r["pred"] is not None]
    n_parse = len(scored)
    exact = sum(r["pred"] == r["gold"] for r in scored)
    direction = sum(
        (ORDINAL[r["pred"]] > 0) == (ORDINAL[r["gold"]] > 0) for r in scored
    )
    within1 = sum(abs(ORDINAL[r["pred"]] - ORDINAL[r["gold"]]) <= 1 for r in scored)
    mae = sum(abs(ORDINAL[r["pred"]] - ORDINAL[r["gold"]]) for r in scored) / max(n_parse, 1)

    summary = {
        "n": args.n,
        "parsed": n_parse,
        "parse_rate": n_parse / args.n,
        "exact_bucket_acc": exact / max(n_parse, 1),
        "direction_acc": direction / max(n_parse, 1),
        "within_1_bucket_acc": within1 / max(n_parse, 1),
        "mean_abs_bucket_error": mae,
        "base_only": args.base_only,
        "seconds": time.time() - t0,
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
