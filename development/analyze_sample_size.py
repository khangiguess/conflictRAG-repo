import os
import sys
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score

# Add root directory to Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# Use the exact same features extracted by train_policy.py
X_PATH = os.path.join(ROOT_DIR, "data", "X_dev.npy")
Y_PATH = os.path.join(ROOT_DIR, "data", "y_dev.npy")

def main():
    if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
        print("ERROR: X_dev.npy or y_dev.npy not found.")
        print("Please run `python development/train_policy.py` first to extract the features.")
        return

    print("Loading pre-extracted dev features...")
    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    
    # Define the subset sizes to evaluate, up to 800
    sizes = [100, 200, 300, 400, 500, 600, 700, 800]
    mean_scores = []
    
    print("\nRunning Custom Learning Curve Analysis...")
    print("{:<15} | {:<15}".format("Total Samples", "Mean Macro F1"))
    print("-" * 35)
    
    for n in sizes:
        if n > len(X):
            print(f"Skipping {n} (exceeds dataset size of {len(X)})")
            continue
            
        # Take the first N examples
        X_subset = X[:n]
        y_subset = y[:n]
        
        # Use StratifiedShuffleSplit for robust CV on smaller subsets
        # 5 splits, 20% test size
        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        fold_scores = []
        
        for train_idx, val_idx in sss.split(X_subset, y_subset):
            X_train, X_val = X_subset[train_idx], X_subset[val_idx]
            y_train, y_val = y_subset[train_idx], y_subset[val_idx]
            
            # Train the model with the same hyperparameters as the final policy
            model = LogisticRegression(
                multi_class='ovr', 
                max_iter=1000, 
                random_state=42,
                class_weight='balanced'
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            
            # Calculate Macro F1
            score = f1_score(y_val, preds, average='macro', zero_division=0)
            fold_scores.append(score)
            
        mean_score = np.mean(fold_scores)
        mean_scores.append(mean_score)
        print(f"{n:<15} | {mean_score:<15.4f}")
        
    # Plot the learning curve
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, mean_scores, marker='o', color='b', label="Macro F1 Score (5-fold CV)")
    plt.title("Logistic Regression Learning Curve on Dev Set")
    plt.xlabel("Total Dev Set Samples Used")
    plt.ylabel("Macro F1 Score")
    plt.xticks(sizes)
    plt.ylim(0, 1.0)
    plt.legend(loc="best")
    plt.grid(True)
    
    # Save the plot
    plot_path = os.path.join(ROOT_DIR, "results", "learning_curve_800.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSuccess! Learning curve plot saved to {plot_path}")

if __name__ == "__main__":
    main()