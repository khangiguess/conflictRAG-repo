import os
import sys
import json
import requests
import random
from datasets import load_dataset
from huggingface_hub import hf_hub_download, HfApi

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "dev_set_natural.jsonl")

def generate_dev_natural():
    print("Generating dev_set_natural (Aligned with Test Set Distribution)...")
    all_examples = []
    random.seed(42)

    # Target Distribution (Total 1000 examples)
    # ANSWER: 52%, DISCLOSE: 24%, ABSTAIN: 14%, DISCLAIMER: 10%
    TARGET_ABSTAIN = 140
    TARGET_DISCLOSE = 240
    TARGET_DISCLAIMER = 100
    TARGET_ANSWER = 520

    # 1. RAG Benchmark (RGB) -> ABSTAIN (Take indices 0-139)
    try:
        print("Loading RGB for ABSTAIN...")
        url = "https://raw.githubusercontent.com/chen700564/RGB/master/data/en.json"
        response = requests.get(url)
        response.raise_for_status()
        
        rgb_count = 0
        for line in response.text.strip().split('\n'):
            if not line.strip(): continue
            if rgb_count >= TARGET_ABSTAIN: break
            
            item = json.loads(line)
            question = item.get("query", "")
            passages = item.get("negative", []) 
            
            if question and len(passages) >= 1:
                all_examples.append({
                    "id": f"rgb-dev-{rgb_count}", "question": question, "passages": passages[:2],
                    "gold_action": "ABSTAIN", "gold_answer": item.get("answer", ""), "has_conflict": False
                })
                rgb_count += 1
        print(f"  Extracted {rgb_count} ABSTAIN examples from RGB.")
    except Exception as e:
        print(f"  [Warning] Could not load RGB: {e}")

    # 2. ConflictQA -> DISCLOSE (Take indices 0-239)
    try:
        print("Loading ConflictQA for DISCLOSE...")
        api = HfApi()
        files = api.list_repo_files("osunlp/ConflictQA", repo_type="dataset")
        target_file = next((f for f in files if "chatgpt" in f and f.endswith(".json")), None)
        path = hf_hub_download("osunlp/ConflictQA", target_file, repo_type="dataset")
        
        cq_count = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                if cq_count >= TARGET_DISCLOSE: break
                item = json.loads(line)
                passages = [item.get("parametric_memory", ""), item.get("counter_memory", "")]
                all_examples.append({
                    "id": f"conflictqa-dev-{cq_count}", "question": item.get("question", ""), "passages": passages,
                    "gold_action": "DISCLOSE", "gold_answer": str(item.get("ground_truth", "")), "has_conflict": True
                })
                cq_count += 1
        print(f"  Extracted {cq_count} DISCLOSE examples from ConflictQA.")
    except Exception as e:
        print(f"  [Warning] Could not load ConflictQA: {e}")

    # 3. FaithDial (valid split) -> ANSWER / DISCLAIMER
    try:
        print("Loading FaithDial (valid split) for ANSWER/DISCLAIMER...")
        path = hf_hub_download("McGill-NLP/FaithDial", "data/valid.json", repo_type="dataset")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        fd_ans = 0
        fd_disc = 0
        # Targeting 60 DISCLAIMERs and 300 ANSWERs from FaithDial
        FD_DISC_TARGET = 60
        FD_ANS_TARGET = 300
        for dialogue in data:
            if (fd_ans + fd_disc) >= (FD_DISC_TARGET + FD_ANS_TARGET): break
            for utt in dialogue.get("utterances", []):
                if (fd_ans + fd_disc) >= (FD_DISC_TARGET + FD_ANS_TARGET): break
                history = utt.get("history", [])
                q = history[-1] if isinstance(history, list) and history else ""
                p = utt.get("knowledge", "")
                if q and p:
                    begin = utt.get("BEGIN", [])
                    if "Generic" in begin or "Uncooperative" in begin:
                        if fd_disc < FD_DISC_TARGET:
                            all_examples.append({
                                "id": f"faithdial-dev-disc-{fd_disc}", "question": q, "passages": [p],
                                "gold_action": "DISCLAIMER", "gold_answer": str(utt.get("response", "")), "has_conflict": False
                            })
                            fd_disc += 1
                    else:
                        if fd_ans < FD_ANS_TARGET:
                            all_examples.append({
                                "id": f"faithdial-dev-ans-{fd_ans}", "question": q, "passages": [p],
                                "gold_action": "ANSWER", "gold_answer": str(utt.get("response", "")), "has_conflict": False
                            })
                            fd_ans += 1
        print(f"  Extracted {fd_ans} ANSWER and {fd_disc} DISCLAIMER examples from FaithDial.")
    except Exception as e:
        print(f"  [Warning] Could not load FaithDial: {e}")

    # 4. PubMedQA (train split) -> ANSWER / DISCLAIMER (Fill remainder)
    try:
        print("Loading PubMedQA (train split) for remainder of ANSWER/DISCLAIMER...")
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        pm_ans = 0
        pm_disc = 0
        # Need 40 more DISCLAIMERs (100 - 60) and 220 more ANSWERs (520 - 300)
        PM_DISC_TARGET = TARGET_DISCLAIMER - fd_disc
        PM_ANS_TARGET = TARGET_ANSWER - fd_ans
        
        for i, row in enumerate(ds):
            if (pm_ans + pm_disc) >= (PM_DISC_TARGET + PM_ANS_TARGET): break
            contexts = row.get("context", {}).get("contexts", [])
            p = " ".join(contexts) if contexts else ""
            if p:
                decision = str(row.get("final_decision", "")).lower()
                gold = row.get("long_answer", "")
                if decision == "maybe":
                    if pm_disc < PM_DISC_TARGET:
                        all_examples.append({
                            "id": f"pubmedqa-dev-disc-{i}", "question": row["question"], "passages": [p],
                            "gold_action": "DISCLAIMER", "gold_answer": str(gold), "has_conflict": False
                        })
                        pm_disc += 1
                else:
                    if pm_ans < PM_ANS_TARGET:
                        all_examples.append({
                            "id": f"pubmedqa-dev-ans-{i}", "question": row["question"], "passages": [p],
                            "gold_action": "ANSWER", "gold_answer": str(gold), "has_conflict": False
                        })
                        pm_ans += 1
        print(f"  Extracted {pm_ans} ANSWER and {pm_disc} DISCLAIMER examples from PubMedQA.")
    except Exception as e:
        print(f"  [Warning] Could not load PubMedQA: {e}")

    # Shuffle and Save
    random.shuffle(all_examples)
    dev_data = all_examples[:1000]

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for item in dev_data:
            f.write(json.dumps(item) + "\n")
    
    from collections import Counter
    dist = Counter(item["gold_action"] for item in dev_data)
    print(f"\nSuccess! {len(dev_data)} natural dev samples saved to {OUTPUT_PATH}")
    print(f"Distribution: {dict(dist)}")

if __name__ == "__main__":
    generate_dev_natural()