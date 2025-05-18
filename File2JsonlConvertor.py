import os
import json
from pathlib import Path
from transformers import AutoTokenizer
from docx import Document
from pdfminer.high_level import extract_text as extract_pdf_text
import textract

# CONFIGURATION
input_dir = "/Users/prabhat7/Desktop/samples"
jsonl_output_dir = "jsonl_chunks"
text_output_dir = "text_chunks"
tokenizer_name = "meta-llama/Meta-Llama-3-8B"
max_seq_length = 2048
samples_per_file = 1000

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

os.makedirs(jsonl_output_dir, exist_ok=True)
os.makedirs(text_output_dir, exist_ok=True)

def extract_text(file_path):
    try:
        suffix = file_path.suffix.lower()
        if suffix == ".txt":
            return Path(file_path).read_text(encoding="utf-8", errors="ignore")
        elif suffix == ".docx":
            doc = Document(file_path)
            return "\n".join([p.text for p in doc.paragraphs])
        elif suffix == ".pdf":
            return extract_pdf_text(file_path)
        elif suffix == ".doc":
            return textract.process(str(file_path)).decode("utf-8", errors="ignore")
        else:
            return None
    except Exception as e:
        print(f" Error reading {file_path}: {e}")
        return None

sample_buffer = []
text_buffer = []
file_count = 0
sample_count = 0

# Traverse and process all supported files
for file_path in Path(input_dir).rglob("*"):
    if file_path.suffix.lower() not in [".txt", ".docx", ".doc", ".pdf"]:
        continue

    text = extract_text(file_path)
    if not text:
        continue

    input_ids = tokenizer.encode(text, add_special_tokens=True)
    chunks = [input_ids[i:i + max_seq_length] for i in range(0, len(input_ids), max_seq_length)]

    for chunk in chunks:
        sample = {"input_ids": chunk, "labels": chunk.copy()}
        sample_buffer.append(sample)
        decoded_text = tokenizer.decode(chunk, skip_special_tokens=True)
        text_buffer.append(decoded_text)
        sample_count += 1

        if len(sample_buffer) >= samples_per_file:
            jsonl_path = os.path.join(jsonl_output_dir, f"train_{file_count:05d}.jsonl")
            text_path = os.path.join(text_output_dir, f"text_{file_count:05d}.jsonl")

            with open(jsonl_path, "w", encoding="utf-8") as f_jsonl:
                for s in sample_buffer:
                    f_jsonl.write(json.dumps(s) + "\n")

            with open(text_path, "w", encoding="utf-8") as f_txt:
                for t in text_buffer:
                    json.dump({"text": t}, f_txt)
                    f_txt.write("\n")

            print(f" Wrote {len(sample_buffer)} samples to {jsonl_path} and {text_path}")
            file_count += 1
            sample_buffer = []
            text_buffer = []

# Final flush
if sample_buffer:
    jsonl_path = os.path.join(jsonl_output_dir, f"train_{file_count:05d}.jsonl")
    text_path = os.path.join(text_output_dir, f"text_{file_count:05d}.jsonl")

    with open(jsonl_path, "w", encoding="utf-8") as f_jsonl:
        for s in sample_buffer:
            f_jsonl.write(json.dumps(s) + "\n")

    with open(text_path, "w", encoding="utf-8") as f_txt:
        for t in text_buffer:
            json.dump({"text": t}, f_txt)
            f_txt.write("\n")

    print(f" Wrote final {len(sample_buffer)} samples to {jsonl_path} and {text_path}")

print(f" Done! Total samples: {sample_count}, total files: {file_count + 1}")
