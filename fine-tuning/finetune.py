#!/usr/bin/env python3
"""
QLoRA Fine-Tuning Script for facebook/opt-350m
- Works with RTX 4070 (8GB VRAM) using 4-bit quantization
- Trains on your local JSON dataset of instruction-response pairs
"""

import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType

# ----------------------------
# Step 1: Configurations
# ----------------------------
MODEL_NAME = "facebook/opt-350m"
DATA_FILE = "dataset.json"  # Your local JSON dataset
OUTPUT_DIR = "./qlora-opt350m"
MAX_LENGTH = 512

# LoRA configuration
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# Training parameters
EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 2e-4

# ----------------------------
# Step 2: Load Tokenizer
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ----------------------------
# Step 3: Load Dataset
# ----------------------------
dataset = load_dataset("json", data_files=DATA_FILE)

# Function to merge instruction + input + output into one string
def format_example(example):
    if example.get("input"):
        return f"Instruction: {example['instruction']}\nInput: {example['input']}\nResponse: {example['output']}"
    else:
        return f"Instruction: {example['instruction']}\nResponse: {example['output']}"

# Tokenization
def tokenize_function(batch):
    instructions = batch["instruction"]
    inputs = batch.get("input", [None] * len(instructions))
    outputs = batch["output"]

    texts = []
    for instr, inp, out in zip(instructions, inputs, outputs):
        if inp and str(inp).strip():
            texts.append(f"Instruction: {instr}\nInput: {inp}\nResponse: {out}")
        else:
            texts.append(f"Instruction: {instr}\nResponse: {out}")

    # Do NOT pad here; let the collator handle it
    tokenized = tokenizer(
        texts,
        max_length=MAX_LENGTH,
        truncation=True,
        padding=False,
        return_tensors=None,
    )
    # No manual labels here
    return tokenized

tokenized_dataset = dataset["train"].map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# ----------------------------
# Step 4: Load Model in 4-bit
# ----------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16,
    use_safetensors=True
)

# ----------------------------
# Step 5: Apply LoRA
# ----------------------------
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj"],  # specific to OPT
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ----------------------------
# Step 6: Data Collator
# ----------------------------
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class CausalCollator:
    tokenizer: AutoTokenizer
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(features, return_tensors="pt")
        labels = batch["input_ids"].clone()
        pad_id = self.tokenizer.pad_token_id
        labels[labels == pad_id] = -100
        batch["labels"] = labels
        return batch

data_collator = CausalCollator(tokenizer)

# ----------------------------
# Step 7: Training Arguments
# ----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    optim="paged_adamw_8bit",
    remove_unused_columns=False,
    report_to="none",
)

# ----------------------------
# Step 8: Trainer
# ----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# ----------------------------
# Step 9: Train the model
# ----------------------------
print("Starting fine-tuning...")
trainer.train()

# ----------------------------
# Step 10: Save LoRA Adapter
# ----------------------------
adapter_dir = os.path.join(OUTPUT_DIR, "adapter")
model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)
print(f"LoRA adapter saved to: {adapter_dir}")

# ----------------------------
# Step 11: Test after training (optional)
# ----------------------------
def build_prompt(instruction: str, inp: str = None) -> str:
    if inp and inp.strip():
        return f"Instruction: {instruction}\nInput: {inp}\nResponse:"
    return f"Instruction: {instruction}\nResponse:"

prompt = build_prompt("Describe a cursed coin")
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
model.eval()
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.15,
        eos_token_id=tokenizer.eos_token_id,
    )
print("\n--- Test Output After Fine-Tuning ---")
print(tokenizer.decode(output[0], skip_special_tokens=True))
