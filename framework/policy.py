import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from framework import config


def make_policy_model(C=None):
    """One-vs-Rest Logistic Regression. Requires scikit-learn < 1.8 (pinned in requirements.txt);
    `multi_class` was removed in 1.8."""
    return LogisticRegression(
        C=config.C if C is None else C, multi_class="ovr", max_iter=1000,
        random_state=config.RANDOM_STATE, class_weight="balanced",
    )


def gated_predict(probs, classes, theta):
    """ANSWER if P(ANSWER) >= theta, otherwise the most probable non-ANSWER action.
    theta=None means plain argmax. Shared by inference, tuning and ablation."""
    classes = list(classes)
    probs = np.array(probs, dtype=float, copy=True)
    if theta is None:
        return np.array(classes)[np.argmax(probs, axis=1)]
    ans_idx = classes.index("ANSWER")
    answer_ok = probs[:, ans_idx] >= theta
    probs[:, ans_idx] = -1.0
    fallback = np.array(classes)[np.argmax(probs, axis=1)]
    return np.where(answer_ok, "ANSWER", fallback)


class PolicyLayer:
    def __init__(self, model_path=None):
        ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = model_path or os.path.join(ROOT_DIR, "models", "policy_dev_set_natural.joblib")
        if not os.path.exists(model_path):
            # Fail loudly: the old silent rule-based fallback produced plausible-looking
            # results that were not the logistic policy.
            raise FileNotFoundError(f"Trained policy not found at {model_path}")
        self.model = joblib.load(model_path)
        self.theta = config.THETA
        print(f"Loaded trained policy from {model_path} (theta={self.theta}).")

    def decide(self, features):
        probs = self.model.predict_proba([features])
        return str(gated_predict(probs, self.model.classes_, self.theta)[0])
