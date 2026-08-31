import torch
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
import os

# 1. Define Paths
model_name = "t5-large"
output_dir = "./t5_crag_evaluator_final"
data_dir = "./t5_training_data"

print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Device: {torch.cuda.get_device_name(0)}")

print(f"Loading Tokenizer and Model: {model_name}...")
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# FIX: Enable Gradient Checkpointing to reduce VRAM footprint and bypass MIG bug
model.gradient_checkpointing_enable()
model.config.use_cache = False

# 2. Load the JSONL data we just curated
print("Loading datasets...")
raw_datasets = load_dataset("json", data_files={
    "train": os.path.join(data_dir, "train.jsonl"),
    "validation": os.path.join(data_dir, "val.jsonl"),
    "test": os.path.join(data_dir, "test.jsonl")
})

# 3. Tokenization Function
def preprocess_function(examples):
    inputs = examples["input"]
    targets = examples["target"]
    
    # Keeping max_length=512 as requested
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=2, truncation=True)
        
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Tokenizing datasets...")
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True, num_proc=4, remove_columns=raw_datasets["train"].column_names)

# 4. Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    learning_rate=1e-4,          
    per_device_train_batch_size=4,   
    gradient_accumulation_steps=4,   # Effective batch size = 16
    num_train_epochs=3,            
    weight_decay=0.01,
    save_total_limit=1,
    predict_with_generate=False,     
    bf16=True,                    
    logging_steps=100,            
    report_to="none"               
)

# 5. Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)

# 6. Train and Save
print("Starting training...")
trainer.train()

print("Saving final model...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Training complete. Model saved to {output_dir}")