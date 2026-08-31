#!/usr/bin/env python3
"""
Prepares ~16k training data for the CRAG T5 Retrieval Evaluator.
STRICT FIREWALL: NoMIRACL and test sets are completely excluded from training 
to prevent data leakage and ensure A/A* venue reproducibility.
"""
import json
import random
import os
from datasets import load_dataset
from huggingface_hub import hf_hub_download, HfApi

# Set a global seed for reproducibility
random.seed(42)
MAX_WORDS = 400 

def truncate(text):
    if not text: return ""
    words = text.split()
    return " ".join(words[:MAX_WORDS]) if len(words) > MAX_WORDS else text

def format_pair(question, passage):
    return f"Question: {question} Document: {truncate(passage)}"

def load_pubmedqa():
    """Loads PubMedQA Artificial for a stable positive base."""
    print("Loading PubMedQA (Artificial)...")
    pairs = []
    try:
        ds = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")
        for row in ds.select(range(4000)):
            q = row.get("question", "")
            contexts = row.get("context", {}).get("contexts", [])
            p = " ".join(contexts) if contexts else ""
            if q and p:
                pairs.append({"input": format_pair(q, p), "target": "1"})
    except Exception as e:
        print(f"  [Warning] Could not load PubMedQA: {e}")
    return pairs

def load_msmarco():
    """Loads MS MARCO for real-world web search distribution."""
    print("Loading MS MARCO...")
    pairs = []
    try:
        ds = load_dataset("microsoft/ms_marco", "v2.1", split="train")
        for row in ds.select(range(6000)):
            q = row.get("query", "")
            passages = row.get("passages", {}).get("passage_text", [])
            is_selected = row.get("passages", {}).get("is_selected", [])
            for p, sel in zip(passages, is_selected):
                if q and p and sel == 1:
                    pairs.append({"input": format_pair(q, p), "target": "1"})
                    break 
    except Exception as e:
        print(f"  [Warning] Could not load MS MARCO: {e}")
    return pairs

def load_conflictqa():
    """Manually downloads ConflictQA JSON to bypass deprecated script."""
    print("Loading ConflictQA (Manual Download)...")
    pairs = []
    try:
        api = HfApi()
        files = api.list_repo_files("osunlp/ConflictQA", repo_type="dataset")
        target_file = next((f for f in files if "chatgpt" in f and f.endswith(".json")), None)
        if not target_file:
            raise RuntimeError("ConflictQA JSON file not found.")
        
        path = hf_hub_download("osunlp/ConflictQA", target_file, repo_type="dataset")
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                q = item.get("question", "")
                mem = item.get("parametric_memory", "")
                cnt = item.get("counter_memory", "")
                # Both are topically relevant, so both are "1"
                if q and mem:
                    pairs.append({"input": format_pair(q, mem), "target": "1"})
                if q and cnt:
                    pairs.append({"input": format_pair(q, cnt), "target": "1"})
    except Exception as e:
        print(f"  [Warning] Could not load ConflictQA: {e}")
    return pairs

def generate_hard_negatives(positives, num_negatives):
    """Generates cross-dataset hard negatives by mismatching questions and passages."""
    print(f"Generating {num_negatives} cross-dataset hard negatives...")
    negatives = []
    questions = [p["input"].split(" Document: ")[0].replace("Question: ", "") for p in positives]
    passages = [p["input"].split(" Document: ")[1] for p in positives]
    
    if not questions or not passages: return []
    
    while len(negatives) < num_negatives:
        q = random.choice(questions)
        p = random.choice(passages)
        if q not in p:
            negatives.append({"input": format_pair(q, p), "target": "-1"})
            
    return negatives

def main():
    output_dir = "t5_training_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Gather all positive data (No NoMIRACL, No FaithDial to prevent errors/leakage)
    pubmed_pairs = load_pubmedqa()
    msmarco_pairs = load_msmarco()
    conflict_pairs = load_conflictqa()
    
    all_positives = pubmed_pairs + msmarco_pairs + conflict_pairs
    print(f"\nTotal Positives collected: {len(all_positives)}")
    
    if len(all_positives) == 0:
        print("ERROR: No positive examples loaded."); return

    # 2. Generate Hard Negatives to exactly match the number of positives
    num_negatives = len(all_positives)
    generated_negatives = generate_hard_negatives(all_positives, num_negatives)
    all_negatives = generated_negatives
    
    print(f"Total Negatives generated: {len(all_negatives)}")
    
    # 3. Combine and Balance (50/50 split)
    min_len = min(len(all_positives), len(all_negatives))
    random.shuffle(all_positives)
    random.shuffle(all_negatives)
    
    balanced_data = all_positives[:min_len] + all_negatives[:min_len]
    random.shuffle(balanced_data)
    
    print(f"Balanced Dataset Size: {len(balanced_data)}")
    
    # 4. Split into Train (80%), Val (10%), Test (10%)
    train_split = int(0.8 * len(balanced_data))
    val_split = int(0.9 * len(balanced_data))
    
    splits = {
        "train.jsonl": balanced_data[:train_split],
        "val.jsonl": balanced_data[train_split:val_split],
        "test.jsonl": balanced_data[val_split:]
    }
    
    # 5. Save to disk
    for filename, data in splits.items():
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"Saved {len(data)} examples to {filepath}")

if __name__ == "__main__":
    main()