import os
import json
import gzip
import ast
import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download, HfApi

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(BASE_DIR) == "data": DATA_DIR = BASE_DIR
else: DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "dev_policy_leak.jsonl")

def generate_dev_data():
    print("Extracting 800 DISJOINT dev samples for policy training...")
    all_examples = []

    # 1. NoMIRACL (Skip first 50, take next 200)
    try:
        print("Loading NoMIRACL...")
        repo = "miracl/nomiracl"
        topics_p = hf_hub_download(repo, "data/english/topics/test.non_relevant.tsv", repo_type="dataset")
        qrels_p = hf_hub_download(repo, "data/english/qrels/test.non_relevant.tsv", repo_type="dataset")
        corpus_p = hf_hub_download(repo, "data/english/corpus.jsonl.gz", repo_type="dataset")
        
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
                    if len(parts) >= 4: q2docs.setdefault(parts[0], []).append(parts[2])
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
                        if len(doc2text) == len(needed_docs): break
                except: continue
        
        count = 0
        for qid, query in q2query.items():
            docs = q2docs.get(qid, [])[:3]
            passages = [doc2text.get(d, "") for d in docs if doc2text.get(d, "")]
            if query and passages:
                if count >= 50 and count < 250:  # TAKE 200 DISJOINT EXAMPLES
                    all_examples.append({
                        "id": f"nomiracl-dev-{count}", "question": query, "passages": passages,
                        "gold_action": "ABSTAIN", "gold_answer": "", "has_conflict": False
                    })
                count += 1
    except Exception as e:
        print(f"  [Warning] Could not load NoMIRACL: {e}")

    # 2. ConflictQA (Skip first 50, take next 200)
    try:
        print("Loading ConflictQA...")
        api = HfApi()
        files = api.list_repo_files("osunlp/ConflictQA", repo_type="dataset")
        target_file = next((f for f in files if "chatgpt" in f and f.endswith(".json")), None)
        path = hf_hub_download("osunlp/ConflictQA", target_file, repo_type="dataset")
        
        count = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                if count >= 50 and count < 250:  # TAKE 200 DISJOINT EXAMPLES
                    passages = [item.get("parametric_memory", ""), item.get("counter_memory", "")]
                    all_examples.append({
                        "id": f"conflictqa-dev-{count}", "question": item.get("question", ""), "passages": passages,
                        "gold_action": "DISCLOSE", "gold_answer": str(item.get("ground_truth", "")), "has_conflict": True
                    })
                count += 1
    except Exception as e:
        print(f"  [Warning] Could not load ConflictQA: {e}")

    # 3. FaithDial (Skip first 50, take next 200)
    try:
        print("Loading FaithDial...")
        path = hf_hub_download("McGill-NLP/FaithDial", "data/valid.json", repo_type="dataset")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        count = 0
        for dialogue in data:
            for utt in dialogue.get("utterances", []):
                history = utt.get("history", [])
                q = history[-1] if isinstance(history, list) and history else ""
                p = utt.get("knowledge", "")
                if q and p:
                    if count >= 50 and count < 250:  # TAKE 200 DISJOINT EXAMPLES
                        action = "DISCLAIMER" if count % 2 == 0 else "ANSWER"
                        all_examples.append({
                            "id": f"faithdial-dev-{count}", "question": q, "passages": [p],
                            "gold_action": action, "gold_answer": str(utt.get("response", "")), "has_conflict": False
                        })
                    count += 1
    except Exception as e:
        print(f"  [Warning] Could not load FaithDial: {e}")

    # 4. PubMedQA (Skip first 50, take next 200)
    try:
        print("Loading PubMedQA...")
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        for i, row in enumerate(ds):
            if i >= 50 and i < 250:  # TAKE 200 DISJOINT EXAMPLES
                contexts = row.get("context", {}).get("contexts", [])
                p = " ".join(contexts) if contexts else ""
                if p:
                    all_examples.append({
                        "id": f"pubmedqa-dev-{i}", "question": row["question"], "passages": [p],
                        "gold_action": "ANSWER", "gold_answer": row.get("long_answer", ""), "has_conflict": False
                    })
    except Exception as e:
        print(f"  [Warning] Could not load PubMedQA: {e}")

    import random
    random.seed(42)
    random.shuffle(all_examples)
    dev_data = all_examples[:800]

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for item in dev_data:
            f.write(json.dumps(item) + "\n")
    print(f"\nSuccess! {len(dev_data)} disjoint dev samples saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_dev_data()