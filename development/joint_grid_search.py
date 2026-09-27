"""
Joint grid search over (tau_s, TF-IDF threshold, C, theta) with 5-fold stratified CV on
dev_set_natural. Uses the SAME FeatureExtractor.extract_raw() as test-time inference, so
dev tuning and test evaluation compute identical features.

History: the previous version reimplemented extraction inline and referenced `config` /
`torch` without importing them; a bare `except:` swallowed the NameError, so the QA answer
span was never extracted during tuning (claim support was computed from the question).
"""
import os
import sys
from itertools import product

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from framework import config
from framework.feature_cache import load_raw_features
from framework.feature_extractor import FeatureExtractor, features_from_raw
from framework.logger import log_run
from framework.policy import gated_predict, make_policy_model

DATA_PATH = os.path.join(ROOT_DIR, "data", "dev_set_natural.jsonl")
RAW_CACHE = os.path.join(ROOT_DIR, "data", "cache", "raw_dev_set_natural.json")

TAU_VALUES = [0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
TFIDF_VALUES = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15]
C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0]
THETA_VALUES = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]


def main():
    records, raw = load_raw_features(DATA_PATH, RAW_CACHE, FeatureExtractor)
    y = np.array([r["gold_action"] for r in records])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    folds = list(skf.split(np.zeros(len(y)), y))

    best_f1, best_params = -1.0, {}
    pbar = tqdm(total=len(TAU_VALUES) * len(TFIDF_VALUES) * len(C_VALUES), desc="Grid Search")
    for tau, tfidf in product(TAU_VALUES, TFIDF_VALUES):
        X = np.array([features_from_raw(r, tau_s=tau, tfidf_thresh=tfidf) for r in raw])
        for c in C_VALUES:
            # Out-of-fold probabilities once per (tau, tfidf, C); theta is applied afterwards.
            oof = np.zeros((len(y), 4))
            classes = None
            for tr, va in folds:
                model = make_policy_model(C=c).fit(X[tr], y[tr])
                classes = list(model.classes_)
                oof[va] = model.predict_proba(X[va])
            for theta in THETA_VALUES:
                fold_f1s = [
                    f1_score(y[va], gated_predict(oof[va], classes, theta), average="macro", zero_division=0)
                    for _, va in folds
                ]
                mean_f1 = float(np.mean(fold_f1s))
                if mean_f1 > best_f1:
                    best_f1 = mean_f1
                    best_params = {"tau_s": tau, "tfidf": tfidf, "C": c, "theta": theta}
            pbar.update(1)
            pbar.set_postfix({"best": f"{best_f1:.4f}"})
    pbar.close()

    print("\n" + "=" * 50)
    print("JOINT GRID SEARCH COMPLETE")
    print("=" * 50)
    print(f"Best mean 5-fold Macro F1: {best_f1:.4f}")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print("NOTE: config.py values are NOT updated automatically. Changing locked hyperparameters "
          "is a methodology decision.")

    log_run(
        run_name="Joint Grid Search (shared extractor, fixed QA span + NLI index)",
        data_path=DATA_PATH,
        parameters=best_params,
        metrics={"macro_f1": best_f1},
    )


if __name__ == "__main__":
    main()
