import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score

# Add root directory to Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from feature_extractor import FeatureExtractor
import config

DATA_PATH = os.path.join(ROOT_DIR, "data", "dev_policy.jsonl")

def main():
    print("Initializing Feature Extractor for Tau_s Tuning...")
    extractor = FeatureExtractor()
    
    print(f"Loading dev data from {DATA_PATH}...")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        dev_data = [json.loads(line) for line in f]

    X_base = []
    similarities_list = []
    y = []
    
    print(f"Extracting base features and raw similarities for {len(dev_data)} examples...")
    print("(This will take a few minutes, but only needs to be done once)")
    
    for i, item in enumerate(dev_data):
        question = item["question"]
        passages = item["passages"]
        k = len(passages)
        
        # 1. Embed all passages
        passage_embeddings = extractor.embedder.encode(passages, convert_to_numpy=True)
        
        # 2. Extract dispersion (Feature 1)
        if k > 1:
            sim_matrix = np.inner(passage_embeddings, passage_embeddings)
            triu_indices = np.triu_indices(k, k=1)
            avg_sim = np.mean(sim_matrix[triu_indices])
            dispersion = 1.0 - avg_sim
        else:
            dispersion = 0.0
            
        # 3. Extract NLI (Feature 2)
        max_contradiction = 0.0
        if k > 1:
            pairs = [(passages[i], passages[j]) for i in range(k) for j in range(i+1, k)]
            scores = np.array([extractor.nli_model.predict([pair], apply_softmax=True)[0] for pair in pairs])
            contradiction_probs = scores[:, extractor.contra_idx] if len(scores.shape) > 1 else [scores[extractor.contra_idx]]
            max_contradiction = float(np.max(contradiction_probs))
            
        # 4. Extract Answer Span and similarities (For Feature 3)
        best_passage = passages[0]
        answer_span = question
        try:
            qa_result = extractor.qa_extractor(question=question, context=best_passage)
            answer_span = qa_result['answer']
        except:
            pass
            
        answer_embedding = extractor.embedder.encode([answer_span], convert_to_numpy=True)[0]
        similarities = np.inner(passage_embeddings, answer_embedding)
        
        # 5. Extract Mismatch (Feature 4)
        mismatch_flag = 0
        try:
            from langdetect import detect
            q_lang = detect(question)
            p_langs = [detect(p) for p in passages]
            if p_langs.count(q_lang) < (k / 2):
                mismatch_flag = 1
        except:
            pass
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([question] + passages)
            cos_sims = (tfidf_matrix[0:1] * tfidf_matrix[1:].T).toarray()[0]
            if np.max(cos_sims) < 0.1:
                mismatch_flag = 1
        except:
            pass

        # Save base features (1, 2, 4, 5, 6) and similarities
        X_base.append([dispersion, max_contradiction, mismatch_flag, k / 10.0, len(question.split()) / 20.0])
        similarities_list.append(similarities)
        y.append(item["gold_action"])
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(dev_data)} examples...")

    X_base = np.array(X_base)
    y = np.array(y)
    
    # Define the tau_s sweep range
    tau_values = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    mean_scores = []
    
    print("\nSweeping Tau_s (Claim Support Threshold)...")
    print("{:<10} | {:<15}".format("Tau_s", "Mean Macro F1"))
    print("-" * 30)
    
    for tau in tau_values:
        # Reconstruct Feature 3 (claim_support) for this specific tau
        claim_supports = []
        for sims in similarities_list:
            support_count = np.sum(sims >= tau)
            claim_support = support_count / len(sims)
            claim_supports.append(claim_support)
            
        # Insert the new Feature 3 into the base features (index 2)
        X_tau = np.insert(X_base, 2, claim_supports, axis=1)
        
        # Evaluate with 5-fold Stratified CV
        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        fold_scores = []
        
        for train_idx, val_idx in sss.split(X_tau, y):
            X_train, X_val = X_tau[train_idx], X_tau[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = LogisticRegression(
                multi_class='ovr', 
                max_iter=1000, 
                random_state=42,
                class_weight='balanced'
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            score = f1_score(y_val, preds, average='macro', zero_division=0)
            fold_scores.append(score)
            
        mean_score = np.mean(fold_scores)
        mean_scores.append(mean_score)
        print(f"{tau:<10.2f} | {mean_score:<15.4f}")
        
    # Plot the sensitivity curve
    plt.figure(figsize=(8, 5))
    plt.plot(tau_values, mean_scores, marker='o', color='b', label="Macro F1 Score (5-fold CV)")
    best_tau = tau_values[np.argmax(mean_scores)]
    plt.axvline(x=best_tau, color='r', linestyle='--', label=f"Optimal $\tau$$_s$ = {best_tau:.2f}")
    plt.title("Sensitivity Analysis: Claim Support Threshold ($\tau$$_s$)")
    plt.xlabel("Cosine Similarity Threshold ($\tau$$_s$)")
    plt.ylabel("Macro F1 Score")
    plt.xticks(tau_values)
    plt.legend(loc="best")
    plt.grid(True)
    
    plot_path = os.path.join(ROOT_DIR, "results", "tau_s_sensitivity.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSuccess! Sensitivity plot saved to {plot_path}")

if __name__ == "__main__":
    main()