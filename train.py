import os
import logging
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer, 
    LlamaForCausalLM, 
    Trainer, 
    TrainingArguments, 
    default_data_collator
)

def load_multiple_jsonl_datasets(data_dir):
    dataset_files = [
        os.path.join(data_dir, fname)
        for fname in os.listdir(data_dir)
        if fname.endswith(".jsonl")
    ]
    dataset_list = [load_dataset("json", data_files=f)["train"] for f in dataset_files]
    return concatenate_datasets(dataset_list)

def main():
    model_name = "meta-llama/Meta-Llama-3-8B"
    dataset_dir = "jsonl_chunks"
    output_dir = "llama3-h200-output"

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

    # Load dataset
    print("📦 Loading datasets...")
    dataset = load_multiple_jsonl_datasets(dataset_dir)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-5,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    print("🚀 Starting training...")
    trainer.train()
    print("✅ Training complete. Model saved to:", output_dir)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
