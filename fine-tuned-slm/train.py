import argparse
import os
import re

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DATASET_ID = "FinGPT/fingpt-forecaster-dow30-202305-202405"
MAX_LEN = 2560


def parse_llama_prompt(p):
    m = re.search(r"<<SYS>>(.*?)<</SYS>>", p, re.S)
    system = m.group(1).strip() if m else ""
    user = p.split("<</SYS>>", 1)[1].replace("[/INST]", "").strip()
    return system, user


def build_dataset(tokenizer):
    ds = load_dataset(DATASET_ID)

    def to_text(row):
        system, user = parse_llama_prompt(row["prompt"])
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt_text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        full_text = prompt_text + row["answer"] + tokenizer.eos_token
        return {"prompt_text": prompt_text, "full_text": full_text}

    ds = ds.map(to_text, remove_columns=ds["train"].column_names)

    def tokenize(row):
        full = tokenizer(row["full_text"], truncation=True, max_length=MAX_LEN)
        prompt_ids = tokenizer(row["prompt_text"], truncation=True, max_length=MAX_LEN)["input_ids"]
        labels = list(full["input_ids"])
        # mask prompt tokens so loss is only on the assistant response
        n = min(len(prompt_ids), len(labels))
        for i in range(n):
            labels[i] = -100
        full["labels"] = labels
        return full

    ds = ds.map(tokenize, remove_columns=["prompt_text", "full_text"])
    return ds


class CausalCollator:
    def __init__(self, tokenizer):
        self.tok = tokenizer

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tok.pad_token_id
        input_ids, attn, labels = [], [], []
        for f in features:
            ids = f["input_ids"]
            lab = f["labels"]
            pad_n = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_n)
            attn.append([1] * len(ids) + [0] * pad_n)
            labels.append(lab + [-100] * pad_n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="./out")
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--per_device_bs", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = build_dataset(tokenizer)
    print("train size:", len(ds["train"]), "test size:", len(ds["test"]))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        data_collator=CausalCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final_adapter"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_adapter"))


if __name__ == "__main__":
    main()
