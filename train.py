import os
import json
from pathlib import Path
from transformers import AutoTokenizer
from docx import Document
import textract
import fitz  # from pymupdf
import re

# CONFIGURATION
input_dir = "/Users/prabhat7/Desktop/samples"
jsonl_output_dir = "jsonl_chunks"
text_output_dir = "text_chunks"
tokenizer_name = "meta-llama/Meta-Llama-3-8B"
max_seq_length = 2048
samples_per_file = 50

# Check for tokenizer availability
try:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
except Exception as e:
    print(f"❌ Failed to load tokenizer: {e}")
    exit(1)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

os.makedirs(jsonl_output_dir, exist_ok=True)
os.makedirs(text_output_dir, exist_ok=True)

def is_sane_text(text):
    if not text:
        return False
    if len(text) < 100:
        return False
    ascii_ratio = sum(c.isascii() and c.isprintable() for c in text) / len(text)
    letter_ratio = sum(c.isalpha() for c in text) / len(text)
    return ascii_ratio > 0.75 and letter_ratio > 0.25

def extract_text(file_path):
    try:
        suffix = file_path.suffix.lower()
        if suffix == ".txt":
            return Path(file_path).read_text(encoding="utf-8", errors="ignore")
        elif suffix == ".docx":
            doc = Document(file_path)
            return "\n".join([p.text for p in doc.paragraphs])
        elif suffix == ".pdf":
            doc = fitz.open(file_path)
            text_pages = []
            for page in doc:
                text = page.get_text("text")
                if is_sane_text(text):
                    text_pages.append(text)
            return "\n".join(text_pages)
        elif suffix == ".doc":
            return textract.process(str(file_path)).decode("utf-8", errors="ignore")
        else:
            return None
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return None

sample_buffer = []
text_buffer = []
file_count = 0
sample_count = 0

# Accumulator for token packing
token_accumulator = []
doc_texts = []

# Traverse and process all supported files
for file_path in Path(input_dir).rglob("*"):
    if file_path.suffix.lower() not in [".txt", ".docx", ".doc", ".pdf"]:
        continue

    text = extract_text(file_path)
    if not text or len(text.strip()) == 0:
        continue

    try:
        tokens = tokenizer.encode(text, add_special_tokens=True)
    except Exception as e:
        print(f"❌ Tokenization failed for {file_path.name}: {e}")
        continue

    token_accumulator.extend(tokens)
    doc_texts.append(f"{file_path.name}: {text.strip()}")

    while len(token_accumulator) >= max_seq_length:
        chunk = token_accumulator[:max_seq_length]
        token_accumulator = token_accumulator[max_seq_length:]
        combined_text = "\n\n".join(doc_texts)

        sample = {"input_ids": chunk, "labels": chunk.copy()}
        sample_buffer.append(sample)
        formatted_text = json.dumps({"text": combined_text}, ensure_ascii=False)
        text_buffer.append(formatted_text)
        sample_count += 1
        doc_texts = []

        if len(sample_buffer) >= samples_per_file:
            jsonl_path = os.path.join(jsonl_output_dir, f"train_{file_count:05d}.jsonl")
            text_path = os.path.join(text_output_dir, f"text_{file_count:05d}.jsonl")

            with open(jsonl_path, "w", encoding="utf-8") as f_jsonl:
                for s in sample_buffer:
                    f_jsonl.write(json.dumps(s) + "\n")

            with open(text_path, "w", encoding="utf-8") as f_txt:
                for t in text_buffer:
                    f_txt.write(t + "\n")

            print(f"✅ Wrote {len(sample_buffer)} samples to {jsonl_path} and {text_path}")
            file_count += 1
            sample_buffer = []
            text_buffer = []

# Final flush
if token_accumulator:
    final_tokens = token_accumulator[:max_seq_length]
    if len(final_tokens) < max_seq_length:
        padding = [tokenizer.pad_token_id] * (max_seq_length - len(final_tokens))
        final_tokens.extend(padding)
    sample = {"input_ids": final_tokens, "labels": final_tokens.copy()}
    sample_buffer.append(sample)
    formatted_text = json.dumps({"text": "\n\n".join(doc_texts)}, ensure_ascii=False)
    text_buffer.append(formatted_text)

if sample_buffer:
    jsonl_path = os.path.join(jsonl_output_dir, f"train_{file_count:05d}.jsonl")
    text_path = os.path.join(text_output_dir, f"text_{file_count:05d}.jsonl")

    with open(jsonl_path, "w", encoding="utf-8") as f_jsonl:
        for s in sample_buffer:
            f_jsonl.write(json.dumps(s) + "\n")

    with open(text_path, "w", encoding="utf-8") as f_txt:
        for t in text_buffer:
            f_txt.write(t + "\n")

    print(f"✅ Wrote final {len(sample_buffer)} samples to {jsonl_path} and {text_path}")

print(f"🏁 Done! Total samples: {sample_count}, total files: {file_count + 1}")
