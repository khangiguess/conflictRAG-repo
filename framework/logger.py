import os
import json
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(ROOT_DIR, "results", "log.jsonl")

def log_run(run_name, data_path, parameters, metrics):
    """
    Logs an experiment run to the master JSONL log file.
    
    :param run_name: str (e.g., 'Joint Grid Search - dev_set_natural')
    :param data_path: str (path to the data used)
    :param parameters: dict (e.g., {'tau_s': 0.6, 'theta': 0.5})
    :param metrics: dict (e.g., {'macro_f1': 0.6898, 'accuracy': 0.74})
    """
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_name": run_name,
        "data_path": os.path.basename(data_path),
        "parameters": parameters,
        "metrics": metrics
    }
    
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry) + "\n")
        
    print(f"Run logged to log.jsonl")