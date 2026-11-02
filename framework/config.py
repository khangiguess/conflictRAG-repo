import os

# Run tag: every artifact of this run is named with it (results, model, logs)
RUN_TAG = "test_set_1_4D"

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "test_set_1.jsonl")
RESULTS_PATH = os.path.join(BASE_DIR, "results", f"{RUN_TAG}_results.json")
POLICY_MODEL_PATH = os.path.join(BASE_DIR, "models", "policy_dev_set_natural_4D.joblib")

# Model Names
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
QA_MODEL = "deepset/minilm-uncased-squad2"

# Hyperparameters.
# NOTE (4D): these values were tuned for the 6D model (2026-10-23 run, mean 5-fold macro
# F1 = 0.6956). They must be re-tuned with development/joint_grid_search.py for the 4D model
# before final results are reported.
TAU_S = 0.2          # Threshold for semantic claim support
TFIDF_THRESH = 0.10  # Domain-mismatch TF-IDF overlap threshold
C = 10.0             # Logistic Regression regularization
THETA = 0.50         # ANSWER confidence gate
RANDOM_STATE = 42

# Shared theta grid (joint_grid_search.py and ablation_study.py must use the same grid)
THETA_GRID = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

# Hardware Settings
DEVICE = "cuda" # Offloads embedding and NLI math to gpu

# Local API Connection
LLM_API_URL = "http://127.0.0.1:8081/v1/chat/completions"
