#!/usr/bin/env python3
"""
Prepares ~16k training data for the CRAG T5 Retrieval Evaluator.
"""
import json
import random
import os
import gzip
import ast
import pandas as pd
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
    print("Loading PubMedQA (Artificial)...")
    pairs = []
    try:
        ds = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")
        for row in ds.select(range(4000)):  # Increased to 4000
            q = row.get("question", "")
            contexts = row.get("context", {}).get("contexts", [])
            p = " ".join(contexts) if contexts else ""
            if q and p:
                pairs.append({"input": format_pair(q, p), "target": "1"})
    except Exception as e:
        print(f"  [Warning] Could not load PubMedQA: {e}")
    return pairs

def load_msmarco():
    print("Loading MS MARCO...")
    pairs = []
    try:
        ds = load_dataset("microsoft/ms_marco", "v2.1", split="train")
        for row in ds.select(range(6000)):  # Increased to 6000
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
    print("Loading ConflictQA (Manual Download)...")
    pairs = []
    try:
        api = HfApi()
        files = api.list_repo_files("osunlp/ConflictQA", repo_type="dataset")
        target_file = next((f for f in files if "chatgpt" in f and f.endswith(".json")), None)
        if not target_file:
            raise RuntimeError("ConflictQA JSON file not found.")
        
        path = hf_hub_download("osunlp/ConflictQA", target_file, repo_type="dataset")
        
        # FIX: The file is JSON Lines, not a single JSON array. Read line by line.
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                q = item.get("question", "")
                mem = item.get("parametric_memory", "")
                cnt = item.get("counter_memory", "")
                if q and mem:
                    pairs.append({"input": format_pair(q, mem), "target": "1"})
                if q and cnt:
                    pairs.append({"input": format_pair(q, cnt), "target": "1"})
    except Exception as e:
        print(f"  [Warning] Could not load ConflictQA: {e}")
    return pairs

def load_faithdial():
    print("Loading FaithDial (Manual Download)...")
    pairs = []
    try:
        api = HfApi()
        files = api.list_repo_files("McGill-NLP/FaithDial", repo_type="dataset")
        # FIX: Dynamically find the validation/test CSV file regardless of folder
        target_file = next((f for f in files if f.endswith(".csv") and ("valid" in f or "test" in f)), None)
        if not target_file:
            raise RuntimeError("FaithDial CSV not found in repo.")
            
        path = hf_hub_download("McGill-NLP/FaithDial", target_file, repo_type="dataset")
        df = pd.read_csv(path)
        
        for _, row in df.iterrows():
            q = ""
            try:
                hist = ast.literal_eval(str(row.get("history", "[]")))
                if isinstance(hist, list) and hist:
                    q = str(hist[-1])
            except:
                pass
            
            p = str(row.get("knowledge", ""))
            if q and p and q != "nan" and p != "nan":
                pairs.append({"input": format_pair(q, p), "target": "1"})
    except Exception as e:
        print(f"  [Warning] Could not load FaithDial: {e}")
    return pairs

def load_nomiracl():
    print("Loading NoMIRACL (Manual Download)...")
    negatives = []
    try:
        repo = "miracl/nomiracl"
        lang = "english"
        split = "test"
        subset = "non_relevant"
        
        topics_p = hf_hub_download(repo, f"data/{lang}/topics/{split}.{subset}.tsv", repo_type="dataset")
        qrels_p = hf_hub_download(repo, f"data/{lang}/qrels/{split}.{subset}.tsv", repo_type="dataset")
        corpus_p = hf_hub_download(repo, f"data/{lang}/corpus.jsonl.gz", repo_type="dataset")
        
        q2query = {}
        with open(topics_p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split("\t")
                    if len(parts) >= 2: q2query[parts[0]] = parts[1]
                        
        q2docs = {}
        with open(qrels_p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        q2docs.setdefault(parts[0], []).append(parts[2])
                        
        doc2text = {}
        needed_docs = set(d for docs in q2docs.values() for d in docs)
        with gzip.open(corpus_p, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    o = json.loads(line)
                    d = o.get("docid")
                    if d in needed_docs:
                        title, text = o.get("title", ""), o.get("text", "")
                        doc2text[d] = (f"{title}. {text}" if title else text).strip()
                        if len(doc2text) == len(needed_docs):
                            break
                except:
                    continue
                    
        for qid, query in q2query.items():
            docs = q2docs.get(qid, [])[:2]
            for d in docs:
                text = doc2text.get(d, "")
                if query and text:
                    negatives.append({"input": format_pair(query, text), "target": "-1"})
                    
    except Exception as e:
        print(f"  [Warning] Could not load NoMIRACL: {e}")
    return negatives

def generate_hard_negatives(positives, num_negatives):
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
    
    pubmed_pairs = load_pubmedqa()
    msmarco_pairs = load_msmarco()
    conflict_pairs = load_conflictqa()
    faithdial_pairs = load_faithdial()
    
    all_positives = pubmed_pairs + msmarco_pairs + conflict_pairs + faithdial_pairs
    print(f"\nTotal Positives collected: {len(all_positives)}")
    
    if len(all_positives) == 0:
        print("ERROR: No positive examples loaded."); return

    nomiracl_negatives = load_nomiracl()
    print(f"Total Real Negatives collected: {len(nomiracl_negatives)}")
    
    # Target 8000 total negatives
    num_generated_needed = max(0, 8000 - len(nomiracl_negatives))
    generated_negatives = generate_hard_negatives(all_positives, num_generated_needed)
    all_negatives = nomiracl_negatives + generated_negatives
    
    min_len = min(len(all_positives), len(all_negatives))
    random.shuffle(all_positives)
    random.shuffle(all_negatives)
    
    balanced_data = all_positives[:min_len] + all_negatives[:min_len]
    random.shuffle(balanced_data)
    
    print(f"Balanced Dataset Size: {len(balanced_data)}")
    
    train_split = int(0.8 * len(balanced_data))
    val_split = int(0.9 * len(balanced_data))
    
    splits = {
        "train.jsonl": balanced_data[:train_split],
        "val.jsonl": balanced_data[train_split:val_split],
        "test.jsonl": balanced_data[val_split:]
    }
    
    for filename, data in splits.items():
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"Saved {len(data)} examples to {filepath}")

if __name__ == "__main__":
    main()