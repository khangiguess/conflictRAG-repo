import os
import sys
import json
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from langdetect import detect

# Add root directory to Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from feature_extractor import FeatureExtractor
import config

DATA_PATH = os.path.join(ROOT_DIR, "data", "dev_policy.jsonl")

def main():
    print("Initializing Feature Extractor for TF-IDF Threshold Tuning...")
    extractor = FeatureExtractor()
    
    print(f"Loading dev data from {DATA_PATH}...")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        dev_data = [json.loads(line) for line in f]

    X_base = []
    max_cosine_sims_list = []
    y = []
    
    print(f"Extracting base features and raw TF-IDF for {len(dev_data)} examples...")
    print("(This will take a few minutes, but only needs to be done once)")
    
    for i, item in enumerate(dev_data):
        question = item["question"]
        passages = item["passages"]
        k = len(passages)
        
        # 1. Embed passages
        passage_embeddings = extractor.embedder.encode(passages, convert_to_numpy=True)
        
        # 2. Dispersion (Feature 1)
        if k > 1:
            sim_matrix = np.inner(passage_embeddings, passage_embeddings)
            triu_indices = np.triu_indices(k, k=1)
            avg_sim = np.mean(sim_matrix[triu_indices])
            dispersion = 1.0 - avg_sim
        else:
            dispersion = 0.0
            
        # 3. NLI (Feature 2)
        max_contradiction = 0.0
        if k > 1:
            pairs = [(passages[i], passages[j]) for i in range(k) for j in range(i+1, k)]
            scores = np.array([extractor.nli_model.predict([pair], apply_softmax=True)[0] for pair in pairs])
            contradiction_probs = scores[:, extractor.contra_idx] if len(scores.shape) > 1 else [scores[extractor.contra_idx]]
            max_contradiction = float(np.max(contradiction_probs))
                    
        # 4. Claim Support (Feature 3) - Using the optimal tau_s = 0.50
        best_passage = passages[0]
        answer_span = question
        try:
            qa_result = extractor.qa_extractor(question=question, context=best_passage)
            answer_span = qa_result['answer']
        except:
            pass
        answer_embedding = extractor.embedder.encode([answer_span], convert_to_numpy=True)[0]
        similarities = np.inner(passage_embeddings, answer_embedding)
        support_count = np.sum(similarities >= 0.50) # Hardcoded optimal tau_s
        claim_support = support_count / k
        
        # 5. Language Mismatch (Part of Feature 4)
        lang_mismatch = 0
        try:
            q_lang = detect(question)
            p_langs = [detect(p) for p in passages]
            if p_langs.count(q_lang) < (k / 2):
                lang_mismatch = 1
        except:
            pass
            
        # 6. Domain Mismatch TF-IDF (Part of Feature 4) - Save the raw cosine sims!
        max_cosine_sim = 0.0
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([question] + passages)
            cos_sims = (tfidf_matrix[0:1] * tfidf_matrix[1:].T).toarray()[0]
            max_cosine_sim = float(np.max(cos_sims)) # Save this!
        except:
            pass

        # Save base features (1, 2, 3, 5, 6) and max_cosine_sim
        X_base.append([dispersion, max_contradiction, claim_support, lang_mismatch, k / 10.0, len(question.split()) / 20.0])
        max_cosine_sims_list.append(max_cosine_sim)
        y.append(item["gold_action"])
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(dev_data)} examples...")

    X_base = np.array(X_base)
    y = np.array(y)
    
    # Define the TF-IDF threshold sweep range
    tfidf_thresholds = [0.01, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20]
    mean_scores = []
    
    print("\nSweeping TF-IDF Threshold (Domain Mismatch)...")
    print("{:<15} | {:<15}".format("Threshold", "Mean Macro F1"))
    print("-" * 35)
    
    for thresh in tfidf_thresholds:
        # Reconstruct Feature 4 (mismatch_flag) for this specific threshold
        mismatch_flags = []
        for i, max_sim in enumerate(max_cosine_sims_list):
            # Feature 4 is 1 if (lang_mismatch == 1) OR (max_sim < thresh)
            # X_base[i][3] is the lang_mismatch we saved earlier
            flag = 1 if (X_base[i][3] == 1 or max_sim < thresh) else 0
            mismatch_flags.append(flag)
            
        # Insert the new Feature 4 into the base features (index 3)
        X_thresh = np.insert(X_base, 3, mismatch_flags, axis=1)
        
        # Evaluate with 5-fold Stratified CV
        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        fold_scores = []
        
        for train_idx, val_idx in sss.split(X_thresh, y):
            X_train, X_val = X_thresh[train_idx], X_thresh[val_idx]
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
        print(f"{thresh:<15.2f} | {mean_score:<15.4f}")
        
    # Plot the sensitivity curve
    plt.figure(figsize=(8, 5))
    plt.plot(tfidf_thresholds, mean_scores, marker='o', color='g', label="Macro F1 Score (5-fold CV)")
    best_thresh = tfidf_thresholds[np.argmax(mean_scores)]
    plt.axvline(x=best_thresh, color='r', linestyle='--', label=f"Optimal TF-IDF = {best_thresh:.2f}")
    plt.title("Sensitivity Analysis: Domain Mismatch TF-IDF Threshold")
    plt.xlabel("TF-IDF Overlap Threshold")
    plt.ylabel("Macro F1 Score")
    plt.xticks(tfidf_thresholds)
    plt.legend(loc="best")
    plt.grid(True)
    
    plot_path = os.path.join(ROOT_DIR, "results", "tfidf_sensitivity.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSuccess! Sensitivity plot saved to {plot_path}")

if __name__ == "__main__":
    main()