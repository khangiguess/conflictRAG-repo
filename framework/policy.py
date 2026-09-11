import numpy as np
import joblib
import os

class PolicyLayer:
    def __init__(self):
        ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(ROOT_DIR, "models", "policy_dev_set_natural.joblib")
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            self.theta = 0.55 #Optimal threshold
            print("Loaded trained Logistic Regression policy.")
        else:
            self.model = None
            print("WARNING: No trained policy found. Falling back to rule-based routing.")
            
        self.thresholds = {
            "abstain_support": 0.3,
            "disclose_contradiction": 0.6,
            "disclaimer_dispersion": 0.5
        }

    def decide(self, features):
        if self.model is not None:
            probs = self.model.predict_proba([features])[0]
            classes = self.model.classes_
            
            # Find the probability of ANSWER
            answer_idx = list(classes).index("ANSWER")
            prob_answer = probs[answer_idx]
            
            # THRESHOLD LOGIC
            if prob_answer >= self.theta:
                return "ANSWER"
            else:
                # Escalate: Mask the ANSWER probability to -1.0 so it won't be picked
                probs[answer_idx] = -1.0
                best_idx = np.argmax(probs)
                return classes[best_idx]
                
        # Fallback rule-based routing
        else:
            dispersion = features[0]
            nli_contradiction = features[1]
            claim_support = features[2]
            
            if claim_support < self.thresholds["abstain_support"]:
                return "ABSTAIN"
            elif nli_contradiction > self.thresholds["disclose_contradiction"]:
                return "DISCLOSE"
            elif dispersion > self.thresholds["disclaimer_dispersion"]:
                return "DISCLAIMER"
            else:
                return "ANSWER"