import os
import sys
import json
import gzip
import requests
import random
from datasets import load_dataset
from huggingface_hub import hf_hub_download, HfApi

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "test_set_1.jsonl")

def generate_test_set_1():
    print("Generating test_set_1 (Strictly Disjoint from dev_set_1)...")
    all_examples = []
    random.seed(42)
    ans_total = 0  # Global cap for ANSWER class
    ANS_LIMIT = 1600

    # 1. RGB -> ABSTAIN
    try:
        print("Loading RGB for ABSTAIN (Indices 200+)...")
        url = "https://raw.githubusercontent.com/chen700564/RGB/master/data/en.json"
        response = requests.get(url)
        response.raise_for_status()
        
        rgb_count = 0
        lines = response.text.strip().split('\n')
        for i, line in enumerate(lines):
            if not line.strip(): continue
            if i < 200: continue  # SKIP DEV SET INDICES
            if rgb_count >= 200: break
            
            item = json.loads(line)
            question = item.get("query", "")
            passages = item.get("negative", []) 
            
            if question and len(passages) >= 1:
                all_examples.append({
                    "id": f"rgb-test-{rgb_count}", "question": question, "passages": passages[:2],
                    "gold_action": "ABSTAIN", "gold_answer": item.get("answer", ""), "has_conflict": False
                })
                rgb_count += 1
        print(f"  Extracted {rgb_count} ABSTAIN examples from RGB.")
    except Exception as e:
        print(f"  [Warning] Could not load RGB: {e}")

    # 2. NoMIRACL -> ABSTAIN
    try:
        print("Loading NoMIRACL (Test Split) for ABSTAIN...")
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
        
        nm_count = 0
        for qid, query in q2query.items():
            if nm_count >= 500: break
            docs = q2docs.get(qid, [])[:3]
            passages = [doc2text.get(d, "") for d in docs if doc2text.get(d, "")]
            if query and passages:
                all_examples.append({
                    "id": f"nomiracl-test-{nm_count}", "question": query, "passages": passages,
                    "gold_action": "ABSTAIN", "gold_answer": "", "has_conflict": False
                })
                nm_count += 1
        print(f"  Extracted {nm_count} ABSTAIN examples from NoMIRACL.")
    except Exception as e:
        print(f"  [Warning] Could not load NoMIRACL: {e}")

    # 3. ConflictQA -> DISCLOSE
    try:
        print("Loading ConflictQA for DISCLOSE (Indices 300+)...")
        api = HfApi()
        files = api.list_repo_files("osunlp/ConflictQA", repo_type="dataset")
        target_file = next((f for f in files if "chatgpt" in f and f.endswith(".json")), None)
        path = hf_hub_download("osunlp/ConflictQA", target_file, repo_type="dataset")
        
        cq_count = 0
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if not line.strip(): continue
                if i < 300: continue 
                if cq_count >= 500: break
                
                item = json.loads(line)
                passages = [item.get("parametric_memory", ""), item.get("counter_memory", "")]
                all_examples.append({
                    "id": f"conflictqa-test-{cq_count}", "question": item.get("question", ""), "passages": passages,
                    "gold_action": "DISCLOSE", "gold_answer": str(item.get("ground_truth", "")), "has_conflict": True
                })
                cq_count += 1
        print(f"  Extracted {cq_count} DISCLOSE examples from ConflictQA.")
    except Exception as e:
        print(f"  [Warning] Could not load ConflictQA: {e}")

    # 4. WhoQA -> DISCLOSE / ANSWER
    try:
        print("Loading WhoQA (Local JSON)...")
        whoqa_path = os.path.join(DATA_DIR, "WhoQA.json")
        if not os.path.exists(whoqa_path):
            print("  [Warning] WhoQA.json not found. Skipping.")
        else:
            with open(whoqa_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            wq_ans = 0
            wq_disc = 0
            
            for i, x in enumerate(data):
                if (wq_ans + wq_disc) >= 300: break
                qs = x.get("questions") or []
                question = qs[0] if qs else ""
                ctxs = x.get("contexts") or []
                abc = x.get("answer_by_context") or {}
                
                texts = {j: (ctxs[j].get("candidate_texts") or "").strip() for j in range(len(ctxs)) if j < len(ctxs)}
                ans = {j: (abc.get(str(j), [[]])[0][-1].strip() if abc.get(str(j)) and abc.get(str(j))[0] else "") for j in range(len(ctxs))}
                
                agree, seen = None, {}
                for j in range(len(ctxs)):
                    a = ans.get(j, "").lower()
                    if not a: continue
                    if a in seen:
                        agree = (seen[a], j)
                        break
                    seen[a] = j
                
                if agree:
                    a, b = agree
                    passages = [t for t in (texts.get(a), texts.get(b)) if t]
                    if len(passages) == 2:
                        if ans_total < ANS_LIMIT:
                            all_examples.append({
                                "id": f"whoqa-test-ans-{wq_ans}", "question": question, "passages": passages,
                                "gold_action": "ANSWER", "gold_answer": ans.get(a, ""), "has_conflict": False
                            })
                            wq_ans += 1
                            ans_total += 1
                else:
                    idxs = [j for j in range(len(ctxs)) if texts.get(j)][:2]
                    if len(idxs) == 2:
                        a, b = idxs
                        passages = [texts[a], texts[b]]
                        all_examples.append({
                            "id": f"whoqa-test-disc-{wq_disc}", "question": question, "passages": passages,
                            "gold_action": "DISCLOSE", "gold_answer": "", "has_conflict": True
                        })
                        wq_disc += 1
            print(f"  Extracted from WhoQA: {wq_ans} ANSWER and {wq_disc} DISCLOSE.")
    except Exception as e:
        print(f"  [Warning] Could not load WhoQA: {e}")

    # 5. PubMedQA (Train Split, Indices 300+) -> ANSWER (Take EXACTLY 500) / DISCLAIMER (Take ALL)
    try:
        print("Loading PubMedQA (Train Split, Indices 300+) - Targeting 500 ANSWERs...")
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        pm_ans = 0
        pm_disc = 0
        for i, row in enumerate(ds):
            if i < 300: continue 
            contexts = row.get("context", {}).get("contexts", [])
            p = " ".join(contexts) if contexts else ""
            if p:
                decision = str(row.get("final_decision", "")).lower()
                gold = row.get("long_answer", "")
                if decision == "maybe":
                    all_examples.append({
                        "id": f"pubmedqa-test-disc-{i}", "question": row["question"], "passages": [p],
                        "gold_action": "DISCLAIMER", "gold_answer": str(gold), "has_conflict": False
                    })
                    pm_disc += 1
                else:
                    if pm_ans < 500:  # EXACT QUOTA FOR PUBMEDQA ANSWERS
                        all_examples.append({
                            "id": f"pubmedqa-test-ans-{i}", "question": row["question"], "passages": [p],
                            "gold_action": "ANSWER", "gold_answer": str(gold), "has_conflict": False
                        })
                        pm_ans += 1
                        ans_total += 1
        print(f"  Extracted {pm_ans} ANSWER and {pm_disc} DISCLAIMER examples from PubMedQA.")
    except Exception as e:
        print(f"  [Warning] Could not load PubMedQA: {e}")

    # 6. FaithDial (Test Split) -> ANSWER / DISCLAIMER (Take ALL DISCLAIMERs, fill remainder of ANSWER cap)
    try:
        print("Loading FaithDial (Test Split) - Extracting all DISCLAIMERs, filling ANSWER cap...")
        path = hf_hub_download("McGill-NLP/FaithDial", "data/test.json", repo_type="dataset")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        fd_ans = 0
        fd_disc = 0
        for dialogue in data:
            for utt in dialogue.get("utterances", []):
                history = utt.get("history", [])
                q = history[-1] if isinstance(history, list) and history else ""
                p = utt.get("knowledge", "")
                if q and p:
                    begin = utt.get("BEGIN", [])
                    if "Generic" in begin or "Uncooperative" in begin:
                        all_examples.append({
                            "id": f"faithdial-test-disc-{fd_disc}", "question": q, "passages": [p],
                            "gold_action": "DISCLAIMER", "gold_answer": str(utt.get("response", "")), "has_conflict": False
                        })
                        fd_disc += 1
                    else:
                        if ans_total < ANS_LIMIT:
                            all_examples.append({
                                "id": f"faithdial-test-ans-{fd_ans}", "question": q, "passages": [p],
                                "gold_action": "ANSWER", "gold_answer": str(utt.get("response", "")), "has_conflict": False
                            })
                            fd_ans += 1
                            ans_total += 1
        print(f"  Extracted {fd_ans} ANSWER and {fd_disc} DISCLAIMER examples from FaithDial.")
    except Exception as e:
        print(f"  [Warning] Could not load FaithDial: {e}")

    # Shuffle and Save
    random.shuffle(all_examples)
    test_data = all_examples

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")
    
    from collections import Counter
    dist = Counter(item["gold_action"] for item in test_data)
    print(f"\nSuccess! {len(test_data)} test samples saved to {OUTPUT_PATH}")
    print(f"Final Distribution: {dict(dist)}")

if __name__ == "__main__":
    generate_test_set_1()