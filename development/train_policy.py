"""
Trains the final ConflictRAG policy on dev_set_natural with the locked hyperparameters in
framework/config.py, and saves it to models/policy_dev_set_natural.joblib (the path
PolicyLayer loads).

Uses the shared FeatureExtractor.extract_raw() + features_from_raw(), and the same raw-feature
cache as joint_grid_search.py / ablation_study.py, so all three see identical features.

History: the previous version trained on dev_set_1.jsonl, which shares 59 questions with
test_set_1.jsonl (all PubMedQA DISCLAIMER), and hardcoded C / theta.
"""
import os
import sys

import joblib
import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from framework import config
from framework.feature_cache import load_raw_features
from framework.feature_extractor import FeatureExtractor, features_from_raw
from framework.logger import log_run
from framework.policy import gated_predict, make_policy_model

DEV_DATA_PATH = os.path.join(ROOT_DIR, "data", "dev_set_natural.jsonl")
RAW_CACHE = os.path.join(ROOT_DIR, "data", "cache", "raw_dev_set_natural.json")
MODEL_SAVE_PATH = os.path.join(ROOT_DIR, "models", "policy_dev_set_natural.joblib")


def main():
    records, raw = load_raw_features(DEV_DATA_PATH, RAW_CACHE, FeatureExtractor)
    X = np.array([features_from_raw(r) for r in raw])
    y = np.array([r["gold_action"] for r in records])

    params = {"tau_s": config.TAU_S, "tfidf": config.TFIDF_THRESH, "C": config.C, "theta": config.THETA}
    print(f"\nTraining on {len(y)} dev examples with {params}")

    # Honest estimate: 5-fold out-of-fold predictions (same folds as the grid search)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    oof_probs = cross_val_predict(make_policy_model(), X, y, cv=skf, method="predict_proba")
    classes = sorted(set(y))  # cross_val_predict orders columns by sorted class labels
    oof_preds = gated_predict(oof_probs, classes, config.THETA)
    oof_f1 = f1_score(y, oof_preds, average="macro", zero_division=0)
    print(f"\n5-fold out-of-fold report (theta={config.THETA}):")
    print(classification_report(y, oof_preds, zero_division=0))

    # Final model: fit on the full dev set
    model = make_policy_model().fit(X, y)
    train_preds = gated_predict(model.predict_proba(X), model.classes_, config.THETA)
    train_f1 = f1_score(y, train_preds, average="macro", zero_division=0)
    print(f"In-sample macro F1 (optimistic, for reference only): {train_f1:.4f}")
    print(f"Out-of-fold macro F1: {oof_f1:.4f}")

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"\nTrained policy saved to {MODEL_SAVE_PATH}")

    log_run(
        run_name="Train Policy (dev_set_natural, shared extractor)",
        data_path=DEV_DATA_PATH,
        parameters=params,
        metrics={"oof_macro_f1": float(oof_f1), "train_macro_f1": float(train_f1)},
    )


if __name__ == "__main__":
    main()
