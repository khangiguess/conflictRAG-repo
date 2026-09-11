import json
import os
from collections import Counter
import sys

# Add root directory to Python path to find the logger
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from framework.logger import log_run

def compute_metrics(filepath, is_crag=False):
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None
        
    with open(filepath, "r") as f:
        results = json.load(f)
        
    categories = ["ABSTAIN", "DISCLOSE", "DISCLAIMER", "ANSWER"]
    matrix = {gold: {pred: 0 for pred in categories} for gold in categories}
    correct_routing = 0
    
    for item in results:
        gold = item["gold_action"]
        
        # CRAG Taxonomy Mapping (Rigorous, No data leakage)
        if is_crag:
            pred = item.get("crag_internal_state", "FAILED_TO_ROUTE")
            if pred == "INCORRECT": 
                mapped_pred = "ABSTAIN"
            elif pred == "AMBIGUOUS": 
                mapped_pred = "DISCLAIMER"
                # LEAKAGE VERSION (Do not use for final paper):
                # mapped_pred = "DISCLOSE" if gold == "DISCLOSE" else "DISCLAIMER"
            elif pred == "CORRECT": 
                mapped_pred = "ANSWER"
            else: 
                mapped_pred = "FAILED_TO_ROUTE"
            pred = mapped_pred
        else:
            pred = item.get("predicted_action", "FAILED_TO_ROUTE")
            
        if gold in matrix and pred in matrix:
            matrix[gold][pred] += 1
        if gold == pred:
            correct_routing += 1
                
    class_metrics = {}
    for cat in categories:
        tp = matrix[cat][cat]
        fn = sum(matrix[cat][pred] for pred in categories if pred != cat)
        fp = sum(matrix[gold][cat] for gold in categories if gold != cat)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        class_metrics[cat] = {"P": precision, "R": recall, "F1": f1}
        
    # Calculate Macro F1
    macro_f1 = sum(class_metrics[cat]["F1"] for cat in categories) / len(categories)
        
    return {
        "accuracy": correct_routing / len(results) if len(results) > 0 else 0.0,
        "macro_f1": macro_f1,
        "class_metrics": class_metrics
    }

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Update these paths to match your actual result files
    pilot_path = os.path.join(ROOT_DIR, "results", "test_set_1_natural_results.json")
    baseline_path = os.path.join(ROOT_DIR, "results", "vanilla_test_set_1_results.json")
    crag_path = os.path.join(ROOT_DIR, "results", "crag_test_set_1_results.json")
    
    pilot = compute_metrics(pilot_path)
    baseline = compute_metrics(baseline_path)
    crag = compute_metrics(crag_path, is_crag=True)
    
    print("=" * 95)
    print("                CONFLICTRAG CONTEXT ROUTING PERFORMANCE MATRIX               ")
    print("=" * 95)
    
    print(f"{'Core Routing Metric':<28} | {'B1: Vanilla':<20} | {'B3: CRAG':<18} | {'ConflictRAG':<18}")
    print("-" * 95)
    
    b1_acc = f"{baseline['accuracy']*100:.1f}%" if baseline else "N/A"
    crag_acc = f"{crag['accuracy']*100:.1f}%" if crag else "N/A"
    p_acc = f"{pilot['accuracy']*100:.1f}%" if pilot else "N/A"
    print(f"{'Overall Routing Accuracy':<28} | {b1_acc:<20} | {crag_acc:<18} | {p_acc:<18}")
    
    b1_mf1 = f"{baseline['macro_f1']:.4f}" if baseline else "N/A"
    crag_mf1 = f"{crag['macro_f1']:.4f}" if crag else "N/A"
    p_mf1 = f"{pilot['macro_f1']:.4f}" if pilot else "N/A"
    print(f"{'Macro F1 Score':<28} | {b1_mf1:<20} | {crag_mf1:<18} | {p_mf1:<18}")
    print("=" * 95)
    
    print("PER-CLASS ROUTING STRATEGY COMPARISON (Precision / Recall / F1-Score)")
    print("-" * 95)
    
    categories = ["ABSTAIN", "DISCLOSE", "DISCLAIMER", "ANSWER"]
    for cat in categories:
        b1_str = f"{baseline['class_metrics'][cat]['P']:.2f}/{baseline['class_metrics'][cat]['R']:.2f}/{baseline['class_metrics'][cat]['F1']:.2f}" if baseline else "N/A"
        crag_str = f"{crag['class_metrics'][cat]['P']:.2f}/{crag['class_metrics'][cat]['R']:.2f}/{crag['class_metrics'][cat]['F1']:.2f}" if crag else "N/A"
        p_str = f"{pilot['class_metrics'][cat]['P']:.2f}/{pilot['class_metrics'][cat]['R']:.2f}/{pilot['class_metrics'][cat]['F1']:.2f}" if pilot else "N/A"
        print(f"{cat:<15} | {b1_str:<20} | {crag_str:<18} | {p_str:<18}")
    print("=" * 95)

    # ---> LOGGING <---
    try:
        if pilot:
            log_run(
                run_name="Test Set Evaluation (ConflictRAG)",
                data_path="test_set_1.jsonl",
                parameters={"tau_s": 0.60, "tfidf": 0.08, "C": 10.0, "theta": 0.50},
                metrics={
                    "overall_accuracy": pilot['accuracy'],
                    "macro_f1": pilot['macro_f1'],
                    "abstain_f1": pilot['class_metrics']['ABSTAIN']['F1'],
                    "disclose_f1": pilot['class_metrics']['DISCLOSE']['F1'],
                    "disclaimer_f1": pilot['class_metrics']['DISCLAIMER']['F1'],
                    "answer_f1": pilot['class_metrics']['ANSWER']['F1']
                }
            )
    except Exception as e:
        print(f"Could not log run: {e}")

if __name__ == "__main__":
    main()