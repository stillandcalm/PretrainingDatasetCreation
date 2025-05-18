from transformers import AutoTokenizer, LlamaForCausalLM
import torch

# Path to fine-tuned model
model_path = "llama3-8b-output"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")

model.eval()

# Prompt
prompt = "Q: What are the benefits of Zero Trust security?\nA:"

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
