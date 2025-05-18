from transformers import LlamaForCausalLM, AutoTokenizer

model_dir = "/root/PretrainingDatasetCreation/llama3-8b-output"

model = LlamaForCausalLM.from_pretrained(model_dir, torch_dtype="auto", device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Save in PyTorch .bin format
model.save_pretrained(model_dir, safe_serialization=False)
tokenizer.save_pretrained(model_dir)

