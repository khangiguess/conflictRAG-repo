import torch
import json
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer

# 1. Load the fine-tuned model
MODEL_PATH = "./local_models/checkpoint-7059"
DATA_PATH = "./data/t5_training_data/test.jsonl"

print(f"Loading model from {MODEL_PATH}...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
model.eval()

# 2. CRITICAL: Find the token IDs for "1" and "-1"
# We use these to extract the continuous probability score.
id_1 = tokenizer.encode("1", add_special_tokens=False)[0]
id_neg1 = tokenizer.encode("-1", add_special_tokens=False)[0]
print(f"Token ID for '1': {id_1} | Token ID for '-1': {id_neg1}")

# 3. Load the test data
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    test_data = [json.loads(line) for line in f]

correct_predictions = 0
total_predictions = 0

print("Running inference and extracting logits...")

for i, item in enumerate(test_data):
    input_text = item["input"]
    true_target = item["target"]
    
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    
    with torch.no_grad():
        # Instead of generating, we do a single forward pass with the decoder start token
        # This gives us the logits for the FIRST token the model would generate.
        decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]])
        outputs = model(**inputs, decoder_input_ids=decoder_input_ids)
        
        # Get the logits for the first token position
        first_token_logits = outputs.logits[0, 0, :]
        
        # Apply softmax to get probabilities
        probs = torch.softmax(first_token_logits, dim=-1)
        
        # Extract probabilities for our specific tokens
        prob_1 = probs[id_1].item()
        prob_neg1 = probs[id_neg1].item()
        
        # Calculate continuous score from -1.0 to 1.0
        # (If prob_1 is high, score -> 1. If prob_neg1 is high, score -> -1)
        continuous_score = prob_1 - prob_neg1
        
        # For standalone accuracy, we threshold at 0.0
        predicted_target = "1" if continuous_score > 0 else "-1"
        
        if predicted_target == true_target:
            correct_predictions += 1
        total_predictions += 1
        
    if (i + 1) % 100 == 0:
        print(f"Processed {i+1}/{len(test_data)} examples...")

accuracy = (correct_predictions / total_predictions) * 100
print(f"\n====================================")
print(f"Standalone T5 Evaluator Accuracy: {accuracy:.2f}%")
print(f"====================================")