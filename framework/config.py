import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "test_set_1.jsonl")
RESULTS_PATH = os.path.join(BASE_DIR, "results", "test_set_1_results.json")

# Model Names
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
QA_MODEL = "deepset/minilm-uncased-squad2"

# Locked hyperparameters (Joint Grid Search, dev_set_natural, see results/log.jsonl)
TAU_S = 0.6          # Threshold for semantic claim support
TFIDF_THRESH = 0.09  # Domain-mismatch TF-IDF overlap threshold
C = 10.0             # Logistic Regression regularization
THETA = 0.50         # ANSWER confidence gate
RANDOM_STATE = 42

# Hardware Settings
DEVICE = "cuda" # Offloads embedding and NLI math to gpu

# Local API Connection
LLM_API_URL = "http://127.0.0.1:8081/v1/chat/completions"
