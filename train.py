from transformers import AutoTokenizer, LlamaForCausalLM
import torch

# Path to fine-tuned model
model_path = "llama3-8b-output"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
#model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")

model = LlamaForCausalLM.from_pretrained(
    "llama3-8b-output",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

model.eval()

# Prompt
prompt = "Q: What does Cisco Secure Access provides?\nA:"

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.7,
        eos_token_id=tokenizer.eos_token_id
    )

# Decode and print
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("🧠 Model Output:\n", response)
(llama3) root@5962919b099f:~/PretrainingDatasetCreation# 
(llama3) root@5962919b099f:~/PretrainingDatasetCreation# 
(llama3) root@5962919b099f:~/PretrainingDatasetCreation# more train.py 
import os
from datasets import load_dataset, concatenate_datasets, interleave_datasets
from transformers import (
    AutoTokenizer, 
    LlamaForCausalLM, 
    Trainer, 
    TrainingArguments, 
    default_data_collator
)

# CONFIGURATION
model_name = "meta-llama/Meta-Llama-3-8B"
dataset_dir = "/root/PretrainingDatasetCreation"
output_dir = "llama3-8b-output"
max_seq_length = 2048

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
model.resize_token_embeddings(len(tokenizer))

# Load dataset with streaming
def load_jsonl_datasets_streaming(dataset_dir):
    dataset_files = sorted(
        os.path.join(dataset_dir, f)
        for f in os.listdir(dataset_dir)
        if f.endswith(".jsonl")
    )
    streams = [load_dataset("json", data_files=f, streaming=True)["train"] for f in dataset_files]
    return interleave_datasets(streams)

dataset = load_jsonl_datasets_streaming(dataset_dir)

# TrainingArguments
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    max_steps=20,
    learning_rate=2e-5,
    bf16=True,
    logging_steps=10,
    save_strategy="no",
    remove_unused_columns=False,
    report_to="none"
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
)

# Start training
print("🚀 Starting streaming training...")
trainer.train()
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print("✅ Streaming training complete. Model saved to:", output_dir)
