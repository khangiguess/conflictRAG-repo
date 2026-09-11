import os
import sys
import json

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from framework.feature_extractor import FeatureExtractor
from framework.policy import PolicyLayer
from framework.generator import ResponseGenerator
from framework import config

def main():
    print("Initializing ConflictRAG Pipeline...")
    extractor = FeatureExtractor()
    policy = PolicyLayer()
    generator = ResponseGenerator()
    
    if not os.path.exists(config.DATA_PATH):
        print(f"CRITICAL: Data file not found at {config.DATA_PATH}.")
        return

    with open(config.DATA_PATH, "r") as f:
        data = [json.loads(line) for line in f]
        
    print(f"\nStarting ConflictRAG pipeline for {len(data)} examples...\n")
    
    results = []
    
    for i, item in enumerate(data):
        question = item["question"]
        passages = item["passages"]
        
        # Step A: Extract 6D Features
        features = extractor.extract(question, passages)
        
        # Step B: Decide Policy
        predicted_action = policy.decide(features)
        
        # Step C: Generate Response (Passes the action to the generator)
        generated_text = generator.generate(question, passages, predicted_action)
        
        print(f"[{i+1}/{len(data)}] Gold: {item['gold_action']} | Pred: {predicted_action}")
        
        results.append({
            "id": item["id"],
            "question": question,
            "gold_action": item["gold_action"],
            "predicted_action": predicted_action,
            "features": features.tolist(),
            "generated_text": generated_text,
            "gold_answer": item.get("gold_answer", "")
        })
        
    os.makedirs(os.path.dirname(config.RESULTS_PATH), exist_ok=True)
    with open(config.RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nConflictRAG complete! Results saved to {config.RESULTS_PATH}")

if __name__ == "__main__":
    main()