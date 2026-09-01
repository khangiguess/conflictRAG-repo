import os
import sys
import json
import gzip
import ast
import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download, HfApi

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "final_testset.jsonl")

def generate_final_testset():
    print("Extracting FINAL TEST SET (Targeting DISCLAIMER representation)...")
    all_examples = []

    # 1. NoMIRACL (Take 438 from native TEST split -> ABSTAIN)
    try:
        print("Loading NoMIRACL (Test Split)...")
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
            if count >= 438: break
            docs = q2docs.get(qid, [])[:3]
            passages = [doc2text.get(d, "") for d in docs if doc2text.get(d, "")]
            if query and passages:
                all_examples.append({
                    "id": f"nomiracl-test-{count}", "question": query, "passages": passages,
                    "gold_action": "ABSTAIN", "gold_answer": "", "has_conflict": False
                })
                count += 1
    except Exception as e:
        print(f"  [Warning] Could not load NoMIRACL: {e}")

    # 2. ConflictQA (Take 408, skipping first 200 -> DISCLOSE)
    try:
        print("Loading ConflictQA...")
        api = HfApi()
        files = api.list_repo_files("osunlp/ConflictQA", repo_type="dataset")
        target_file = next((f for f in files if "chatgpt" in f and f.endswith(".json")), None)
        path = hf_hub_download("osunlp/ConflictQA", target_file, repo_type="dataset")
        
        count = 0
        extracted = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                if count >= 200 and extracted < 408:
                    item = json.loads(line)
                    passages = [item.get("parametric_memory", ""), item.get("counter_memory", "")]
                    all_examples.append({
                        "id": f"conflictqa-test-{count}", "question": item.get("question", ""), "passages": passages,
                        "gold_action": "DISCLOSE", "gold_answer": str(item.get("ground_truth", "")), "has_conflict": True
                    })
                    extracted += 1
                count += 1
    except Exception as e:
        print(f"  [Warning] Could not load ConflictQA: {e}")

    # 3. FaithDial (Target DISCLAIMER, fill rest with ANSWER, train split)
    try:
        print("Loading FaithDial (Train Split) - Targeting DISCLAIMER...")
        path = hf_hub_download("McGill-NLP/FaithDial", "data/train.json", repo_type="dataset")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        fd_disclaimers = []
        fd_answers = []
        for dialogue in data:
            for utt in dialogue.get("utterances", []):
                history = utt.get("history", [])
                q = history[-1] if isinstance(history, list) and history else ""
                p = utt.get("knowledge", "")
                if q and p:
                    begin = utt.get("BEGIN", [])
                    if "Generic" in begin or "Uncooperative" in begin:
                        fd_disclaimers.append({
                            "id": f"faithdial-disc-{len(fd_disclaimers)}", "question": q, "passages": [p],
                            "gold_action": "DISCLAIMER", "gold_answer": str(utt.get("response", "")), "has_conflict": False
                        })
                    else:
                        fd_answers.append({
                            "id": f"faithdial-ans-{len(fd_answers)}", "question": q, "passages": [p],
                            "gold_action": "ANSWER", "gold_answer": str(utt.get("response", "")), "has_conflict": False
                        })
        
        # Prioritize all disclaimers (up to 200), fill the rest with answers
        all_examples.extend(fd_disclaimers[:200])
        all_examples.extend(fd_answers[:533 - len(fd_disclaimers)])
    except Exception as e:
        print(f"  [Warning] Could not load FaithDial: {e}")

    # 4. PubMedQA (Target DISCLAIMER, fill rest with ANSWER, up to 428)
    try:
        print("Loading PubMedQA - Targeting DISCLAIMER...")
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        pm_disclaimers = []
        pm_answers = []
        for i, row in enumerate(ds):
            if i < 200: continue # Skip dev set
            contexts = row.get("context", {}).get("contexts", [])
            p = " ".join(contexts) if contexts else ""
            if p:
                decision = str(row.get("final_decision", "")).lower()
                gold = row.get("long_answer", "")
                if decision == "maybe":
                    pm_disclaimers.append({
                        "id": f"pubmedqa-disc-{i}", "question": row["question"], "passages": [p],
                        "gold_action": "DISCLAIMER", "gold_answer": str(gold), "has_conflict": False
                    })
                else:
                    pm_answers.append({
                        "id": f"pubmedqa-ans-{i}", "question": row["question"], "passages": [p],
                        "gold_action": "ANSWER", "gold_answer": str(gold), "has_conflict": False
                    })
                    
        # Prioritize all disclaimers (up to 150), fill the rest with answers
        all_examples.extend(pm_disclaimers[:150])
        all_examples.extend(pm_answers[:428 - len(pm_disclaimers)])
    except Exception as e:
        print(f"  [Warning] Could not load PubMedQA: {e}")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for item in all_examples:
            f.write(json.dumps(item) + "\n")
    
    from collections import Counter
    dist = Counter(item["gold_action"] for item in all_examples)
    print(f"\nSuccess! {len(all_examples)} final test examples saved to {OUTPUT_PATH}")
    print(f"Targeted Distribution: {dict(dist)}")

if __name__ == "__main__":
    generate_final_testset()