"""
Finds the optimal abstention threshold (theta) for the trained policy layer.
Assumes train_policy.py has already been run.
"""
import os
import sys
import numpy as np
import joblib
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# Add root directory to Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

MODEL_PATH = os.path.join(ROOT_DIR, "models", "logistic_policy.joblib")
X_PATH = os.path.join(ROOT_DIR, "data", "X_dev.npy")
Y_PATH = os.path.join(ROOT_DIR, "data", "y_dev.npy")

def main():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(X_PATH):
        print("ERROR: Model or dev features not found. Please run train_policy.py first.")
        return

    print("Loading trained model and dev features...")
    model = joblib.load(MODEL_PATH)
    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    
    probs = model.predict_proba(X)
    classes = list(model.classes_)
    answer_idx = classes.index("ANSWER")
    
    thetas = np.arange(0.10, 0.95, 0.05)
    best_f1 = 0.0
    best_theta = 0.0
    results = []
    
    print("\nSweeping Theta (Threshold)...")
    print("{:<10} | {:<15} | {:<15}".format("Theta", "Macro F1", "Answer F1"))
    print("-" * 45)
    
    for theta in thetas:
        predictions = []
        for i in range(len(probs)):
            prob_dist = probs[i].copy()
            prob_answer = prob_dist[answer_idx]
            
            if prob_answer >= theta:
                predictions.append("ANSWER")
            else:
                prob_dist[answer_idx] = -1.0
                best_idx = np.argmax(prob_dist)
                predictions.append(classes[best_idx])
                
        macro_f1 = f1_score(y, predictions, average='macro', zero_division=0)
        y_answer = [1 if label == "ANSWER" else 0 for label in y]
        pred_answer = [1 if label == "ANSWER" else 0 for label in predictions]
        answer_f1 = f1_score(y_answer, pred_answer, zero_division=0)
        
        results.append({"theta": theta, "macro_f1": macro_f1, "answer_f1": answer_f1})
        
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_theta = theta
            
        print("{:<10.2f} | {:<15.4f} | {:<15.4f}".format(theta, macro_f1, answer_f1))
        
    print("\n" + "="*45)
    print(f"OPTIMAL THETA (θ*): {best_theta:.2f}")
    print(f"MAX MACRO F1: {best_f1:.4f}")
    print("="*45)
    print(f"\nACTION REQUIRED: Update self.theta in policy.py to {best_theta:.2f}")
    
    # Plot and save the Pareto frontier
    thetas_plot = [r["theta"] for r in results]
    macro_f1_plot = [r["macro_f1"] for r in results]
    
    plt.figure(figsize=(8, 5))
    plt.plot(thetas_plot, macro_f1_plot, marker='o', label="Macro F1")
    plt.axvline(x=best_theta, color='r', linestyle='--', label=f"Optimal θ* = {best_theta:.2f}")
    plt.title("Policy Threshold (θ) Sweep on Dev Set")
    plt.xlabel("Abstention Threshold (θ)")
    plt.ylabel("Macro F1 Score")
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(ROOT_DIR, "results", "theta_sweep.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    print(f"Sweep plot saved to {plot_path}")

if __name__ == "__main__":
    main()