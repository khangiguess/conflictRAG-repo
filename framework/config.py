import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "test_set_1.jsonl")
RESULTS_PATH = os.path.join(BASE_DIR, "results", "test_set_1_suggested_results.json")

# Model Names
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
QA_MODEL = "deepset/minilm-uncased-squad2"

# Hyperparameters
TAU_S = 0.65  # Threshold for semantic claim support

# Routing Configuration
ROUTING_MODE = "SYSTEM_ONLY" # Options: "SYSTEM_ONLY", "LLM_ONLY", "HYBRID_OVERRIDE"

# Hardware Settings
DEVICE = "cuda" # Offloads embedding and NLI math to gpu

# Local API Connection
LLM_API_URL = "http://127.0.0.1:8081/v1/chat/completions"