import os
import sys
import json
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from framework import config
from framework.feature_cache import load_raw_features
from framework.feature_extractor import FeatureExtractor, features_from_raw

TEST_PATH = os.path.join(ROOT_DIR, "data", "test_set_1.jsonl")
TEST_RAW_CACHE = os.path.join(ROOT_DIR, "data", "cache", "raw_test_set_1.json")

def main():
    # Load cached raw features
    test_records, test_raw = load_raw_features(TEST_PATH, TEST_RAW_CACHE, FeatureExtractor)
    
    # Build the 6D feature matrix using locked config
    X_test = np.array([features_from_raw(r) for r in test_raw])
    
    # Find the first ConflictQA example
    for i, record in enumerate(test_records):
        if "conflictqa" in record.get("id", ""):
            print(f"=== ConflictQA Example Found ===")
            print(f"ID: {record['id']}")
            print(f"Question: {record['question']}")
            print(f"Gold Action: {record['gold_action']}")
            print(f"Passage 1: {record['passages'][0][:150]}...")
            print(f"Passage 2: {record['passages'][1][:150]}...")
            print(f"\n=== 6D Feature Vector ===")
            print(f"Feature 0 (Dispersion):     {X_test[i][0]:.4f}")
            print(f"Feature 1 (NLI Condradict):  {X_test[i][1]:.4f}  <--- THE KEY VALUE")
            print(f"Feature 2 (Claim Support):   {X_test[i][2]:.4f}")
            print(f"Feature 3 (Mismatch):        {X_test[i][3]:.4f}")
            print(f"Feature 4 (k norm):          {X_test[i][4]:.4f}")
            print(f"Feature 5 (query len norm):  {X_test[i][5]:.4f}")
            
            print(f"\n=== DIAGNOSIS ===")
            if X_test[i][1] > 0.5:
                print("✅ FIX IS APPLIED: Feature 2 is HIGH on a contradiction pair.")
                print("   This means the NLI model is correctly reading P(contradiction).")
            else:
                print("🚨 BUG IS PRESENT: Feature 2 is LOW on a contradiction pair.")
                print("   This means the NLI model is reading P(neutral) instead of P(contradiction).")
                print("   You MUST rerun the grid search and retrain the policy.")
            return
            
    print("No ConflictQA examples found in test set.")

if __name__ == "__main__":
    main()