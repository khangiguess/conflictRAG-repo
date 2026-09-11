import os
import json
import sys
import requests

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from framework.feature_extractor import FeatureExtractor
from framework.policy import PolicyLayer
from framework.generator import ResponseGenerator
from framework import config

# ==============================================================================
# COMPONENT 3: LLAMA 3.1 API GENERATOR
# ==============================================================================
class ResponseGenerator:
    def __init__(self):
        self.api_url = config.LLM_API_URL
        
    def generate(self, system_prompt, user_prompt):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        payload = {
            "model": "llama3",  # Ignored by local server, but required by schema
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 256
        }
        
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            print(f"   [API Error] Llama generation failed: {e}")
            return "GENERATION_ERROR"

def main():
    # 1. Initialize components
    print("Initializing ConflictRAG Pipeline...")
    extractor = FeatureExtractor()
    policy = PolicyLayer()  # This will load logistic_policy.joblib
    generator = ResponseGenerator()
    
    results = []
    
    # 2. Load pilot data
    if not os.path.exists(config.DATA_PATH):
        print(f"CRITICAL: Data file not found at {config.DATA_PATH}.")
        return

    with open(config.DATA_PATH, "r") as f:
        data = [json.loads(line) for line in f]
        
    print(f"\nStarting ConflictRAG pipeline for {len(data)} examples...\n")
    
    # 3. Process each example
    for i, item in enumerate(data):
        question = item["question"]
        passages = item["passages"]
        gold_action = item["gold_action"]
        
        print(f"[{i+1}/{len(data)}] Question: {question[:60]}...")
        
        # Step A: Extract 6D Features
        features = extractor.extract(question, passages)
        
        # Step B: Decide Policy (Uses Logistic Regression + 0.65 threshold)
        predicted_action = policy.decide(features)
        
        # Step C: Generate Response
        # Prepare context string
        context = "\n\n".join(passages)
        generated_text = ""
        
        if predicted_action == "ANSWER":
            sys_prompt = "You are a factual question-answering assistant. Use ONLY the following passages to answer the question."
            user_prompt = f"Passages: {context}\n\nQuestion: {question}\n\nAnswer concisely:"
            generated_text = generator.generate(sys_prompt, user_prompt)
            
        elif predicted_action == "DISCLAIMER":
            sys_prompt = "You are a cautious assistant. The evidence is ambiguous. Provide a hedged answer based on the passages."
            user_prompt = f"Based on the available evidence, which may be incomplete:\nPassages: {context}\n\nQuestion: {question}\n\nAnswer:"
            generated_text = generator.generate(sys_prompt, user_prompt)
            
        elif predicted_action == "DISCLOSE":
            sys_prompt = "You are an analytical assistant. The retrieved sources disagree. Disclose the conflict to the user based on the passages."
            user_prompt = f"The retrieved sources disagree on this point.\nPassages: {context}\n\nQuestion: {question}\n\nAnswer:"
            generated_text = generator.generate(sys_prompt, user_prompt)
            
        elif predicted_action == "ABSTAIN":
            generated_text = "I cannot answer based on the provided passages."
        
        # Log output
        print(f"   Gold Action: {gold_action} | Predicted: {predicted_action}")
        print(f"   Response: {generated_text[:80]}...\n")
        
        results.append({
            "id": item["id"],
            "question": question,
            "gold_action": gold_action,
            "predicted_action": predicted_action,
            "features": features.tolist(), # Save as list for JSON compatibility
            "generated_text": generated_text,
            "gold_answer": item.get("gold_answer", "")
        })
        
    # 4. Save results
    os.makedirs(os.path.dirname(config.RESULTS_PATH), exist_ok=True)
    with open(config.RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"ConflictRAG Pilot complete! Results saved to {config.RESULTS_PATH}")

if __name__ == "__main__":
    main()
