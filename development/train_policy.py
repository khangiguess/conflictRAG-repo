import os
import sys
import json
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from feature_extractor import FeatureExtractor
import config

DEV_DATA_PATH = os.path.join(ROOT_DIR, "data", "dev_policy.jsonl")
MODEL_SAVE_PATH = os.path.join(ROOT_DIR, "models", "logistic_policy.joblib")

def main():
    print("Initializing Feature Extractor...")
    extractor = FeatureExtractor()
    
    if not os.path.exists(DEV_DATA_PATH):
        print(f"ERROR: Dev data not found at {DEV_DATA_PATH}.")
        return

    print(f"Loading dev data from {DEV_DATA_PATH}...")
    with open(DEV_DATA_PATH, 'r', encoding='utf-8') as f:
        dev_data = [json.loads(line) for line in f]

    X = []
    y = []
    
    print(f"Extracting features for {len(dev_data)} examples...")
    for i, item in enumerate(dev_data):
        features = extractor.extract(item["question"], item["passages"])
        X.append(features)
        y.append(item["gold_action"])
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(dev_data)} examples...")

    X = np.array(X)
    y = np.array(y)
    
    print("\nTraining Logistic Regression Policy (One-vs-Rest)...")
    # The fix happens in the evaluation, not the training
    model = LogisticRegression(multi_class='ovr', max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X, y)
    
    train_acc = model.score(X, y) * 100
    print(f"Policy Training Accuracy (Standard argmax): {train_acc:.2f}%")
    
    # --- Threshold-Escalation Evaluation ---
    print("\nClassification Report (with Theta=0.50 Escalation):")
    probs = model.predict_proba(X)
    classes = list(model.classes_)
    answer_idx = classes.index("ANSWER")
    
    escalated_preds = []
    for i in range(len(probs)):
        prob_dist = probs[i].copy()
        prob_answer = prob_dist[answer_idx]
        
        # If not 50% confident in ANSWER, escalate
        if prob_answer < 0.50: # Abstention threshold
            # Mask ANSWER so it can't be chosen
            prob_dist[answer_idx] = -1.0
            best_idx = np.argmax(prob_dist)
            escalated_preds.append(classes[best_idx])
        else:
            escalated_preds.append("ANSWER")
            
    print(classification_report(y, escalated_preds, zero_division=0))
    
    # Save features for threshold tuning
    np.save(os.path.join(ROOT_DIR, "data", "X_dev.npy"), X)
    np.save(os.path.join(ROOT_DIR, "data", "y_dev.npy"), y)
    
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"\nSuccess! Trained policy saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()