import os
import sys
import json
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm

# Add root directory to Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from framework.experiment_logger import log_run
from framework.feature_extractor import FeatureExtractor
import framework.config

DATA_PATH = os.path.join(ROOT_DIR, "data", "dev_set_natural.jsonl")

def main():
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found. Please run dev_set_natural.py first.")
        return

    print("Initializing Feature Extractor for Joint Grid Search...")
    extractor = FeatureExtractor()
    
    print(f"Loading dev data from {DATA_PATH}...")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        dev_data = [json.loads(line) for line in f]

    # --- PHASE 1: Extract Base Features & Raw Similarities ONCE ---
    print(f"\nPhase 1: Extracting base features for {len(dev_data)} examples (Run once)...")
    
    X_base = []
    similarities_list = []
    tfidf_sims_list = []
    y = []
    
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
            pairs = [(passages[a], passages[b]) for a in range(k) for b in range(a+1, k)]
            scores = np.array([extractor.nli_model.predict([pair], apply_softmax=True)[0] for pair in pairs])
            contradiction_probs = scores[:, extractor.contra_idx] if len(scores.shape) > 1 else [scores[extractor.contra_idx]]
            max_contradiction = float(np.max(contradiction_probs))
            
        # 4. QA Answer Span & Similarities (For Feature 3)
        best_passage = passages[0]
        answer_span = question
        try:
            inputs = extractor.qa_tokenizer(question, best_passage, return_tensors='pt', truncation=True, max_length=512).to(config.DEVICE)
            with torch.no_grad():
                outputs = extractor.qa_model(**inputs)
            answer_start = torch.argmax(outputs.start_logits)
            answer_end = torch.argmax(outputs.end_logits) + 1
            inputs_ids = inputs['input_ids'][0]
            answer_span = extractor.qa_tokenizer.decode(inputs_ids[answer_start:answer_end])
        except:
            pass
            
        answer_embedding = extractor.embedder.encode([answer_span], convert_to_numpy=True)[0]
        ans_sims = np.inner(passage_embeddings, answer_embedding)
        
        # 5. Language Mismatch (Base for Feature 4)
        lang_mismatch = 0
        try:
            from langdetect import detect
            q_lang = detect(question)
            p_langs = [detect(p) for p in passages]
            if p_langs.count(q_lang) < (k / 2):
                lang_mismatch = 1
        except:
            pass
            
        # 6. Domain Mismatch TF-IDF (Base for Feature 4)
        max_tfidf_sim = 0.0
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([question] + passages)
            cos_sims = (tfidf_matrix[0:1] * tfidf_matrix[1:].T).toarray()[0]
            max_tfidf_sim = float(np.max(cos_sims))
        except:
            pass

        # Save base features [Disp, NLI, LangMismatch, k_norm, len_norm]
        X_base.append([dispersion, max_contradiction, lang_mismatch, k / 10.0, len(question.split()) / 20.0])
        similarities_list.append(ans_sims)
        tfidf_sims_list.append(max_tfidf_sim)
        y.append(item["gold_action"])
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(dev_data)} base examples...")

    X_base = np.array(X_base)
    y = np.array(y)
    
    # --- PHASE 2: Joint Grid Search with 5-Fold CV ---
    print("\nPhase 2: Running Joint Grid Search (5-Fold CV)...")
    
    # Hyperparameter Grids
    tau_values = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    tfidf_values = [0.05, 0.08, 0.10, 0.12, 0.15]
    c_values = [0.01,0.1, 1.0, 10.0]
    theta_values = [0.30, 0.40, 0.50, 0.60, 0.70]
    
    best_f1 = 0.0
    best_params = {}
    
    # Total combinations for progress bar
    total_combos = len(tau_values) * len(tfidf_values) * len(c_values) * len(theta_values)
    pbar = tqdm(total=total_combos, desc="Grid Search Progress")
    
    for tau, tfidf_thresh in product(tau_values, tfidf_values):
        # 1. Reconstruct Features for this (tau, tfidf) pair
        features = []
        for i, sims in enumerate(similarities_list):
            support_count = np.sum(sims >= tau)
            claim_support = support_count / len(sims)
            
            mismatch_flag = 1 if (X_base[i][2] == 1 or tfidf_sims_list[i] < tfidf_thresh) else 0
            
            feat = [X_base[i][0], X_base[i][1], claim_support, mismatch_flag, X_base[i][3], X_base[i][4]]
            features.append(feat)
            
        X = np.array(features)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for c, theta in product(c_values, theta_values):
            fold_f1s = []
            
            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model = LogisticRegression(
                    C=c, multi_class='ovr', max_iter=1000, 
                    random_state=42, class_weight='balanced'
                )
                model.fit(X_train, y_train)
                
                # Evaluate with threshold logic
                probs = model.predict_proba(X_val)
                classes = list(model.classes_)
                ans_idx = classes.index("ANSWER")
                
                preds = []
                for prob_dist in probs:
                    if prob_dist[ans_idx] >= theta:
                        preds.append("ANSWER")
                    else:
                        prob_dist[ans_idx] = -1.0
                        preds.append(classes[np.argmax(prob_dist)])
                        
                f1 = f1_score(y_val, preds, average='macro', zero_division=0)
                fold_f1s.append(f1)
                
            mean_f1 = np.mean(fold_f1s)
            
            if mean_f1 > best_f1:
                best_f1 = mean_f1
                best_params = {
                    'tau_s': tau, 'tfidf': tfidf_thresh, 'C': c, 'theta': theta
                }
                
            pbar.update(1)
            pbar.set_postfix({'Cur Best F1': f"{best_f1:.4f}", 'Params': best_params})
            
    pbar.close()
    
    print("\n" + "="*50)
    print("JOINT GRID SEARCH COMPLETE")
    print("="*50)
    print(f"Best Macro F1: {best_f1:.4f}")
    print(f"Optimal Parameters:")
    print(f"  tau_s (Claim Support): {best_params['tau_s']}")
    print(f"  TF-IDF (Domain Mismatch): {best_params['tfidf']}")
    print(f"  C (Regularization): {best_params['C']}")
    print(f"  theta (Abstention): {best_params['theta']}")
    print("="*50)


    print("\n" + "="*50)
    print("JOINT GRID SEARCH COMPLETE")
    print("="*50)
    print(f"Best Macro F1: {best_f1:.4f}")
    
    # ---> ADD THIS <---
    log_run(
        run_name="Joint Grid Search",
        data_path=DATA_PATH,
        parameters=best_params,
        metrics={"macro_f1": best_f1}
    )

if __name__ == "__main__":
    main()