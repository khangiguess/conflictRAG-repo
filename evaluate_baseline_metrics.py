import json
import os
import re
import string
import collections

from framework.logger import log_run

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(BASE_DIR, "results", "test_set_1_results.json")

def normalize_text(s):
    """Lower text and remove punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def token_f1(pred, gold):
    """Calculates token-level F1 score."""
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * (precision * recall) / (precision + recall)

def main():
    if not os.path.exists(RESULTS_PATH):
        print(f"Error: Could not find {RESULTS_PATH}")
        return

    with open(RESULTS_PATH, "r") as f:
        results = json.load(f)

    total_items = len(results)
    
    # Metric trackers
    action_match = 0
    answer_f1_sum = 0.0
    answerable_count = 0
    
    # Abstention Precision/Recall (treating non-ANSWER as positive)
    tp_abstain = 0  # Predicted non-answer, Gold non-answer
    fp_abstain = 0  # Predicted non-answer, Gold ANSWER
    fn_abstain = 0  # Predicted ANSWER, Gold non-answer
    total_gold_abstain = 0

    for item in results:
        gold_action = item.get("gold_action", "").upper()
        pred_action = item.get("predicted_action", "").upper()
        gen_text = item.get("generated_text", "")
        gold_answer = item.get("gold_answer", "")

        # 1. Action Accuracy
        if pred_action == gold_action:
            action_match += 1

        # 2. Answer F1 (only calculated on gold ANSWER examples)
        if gold_action == "ANSWER":
            answerable_count += 1
            if pred_action == "ANSWER":
                answer_f1_sum += token_f1(gen_text, gold_answer)
            else:
                answer_f1_sum += 0.0  # Failed to answer

        # 3. Abstention Precision / Recall
        is_gold_non_answer = gold_action in ["ABSTAIN", "DISCLAIMER", "DISCLOSE"]
        is_pred_non_answer = pred_action in ["ABSTAIN", "DISCLAIMER"]

        if is_gold_non_answer:
            total_gold_abstain += 1

        if is_pred_non_answer and is_gold_non_answer:
            tp_abstain += 1
        elif is_pred_non_answer and not is_gold_non_answer:
            fp_abstain += 1
        elif not is_pred_non_answer and is_gold_non_answer:
            fn_abstain += 1

    # Calculate Final Metrics
    action_accuracy = (action_match / total_items) * 100 if total_items > 0 else 0
    avg_answer_f1 = (answer_f1_sum / answerable_count) * 100 if answerable_count > 0 else 0
    
    abstain_precision = (tp_abstain / (tp_abstain + fp_abstain)) * 100 if (tp_abstain + fp_abstain) > 0 else 0
    abstain_recall = (tp_abstain / (tp_abstain + fn_abstain)) * 100 if (tp_abstain + fn_abstain) > 0 else 0

    print("\n" + "="*40)
    print("CRAG BASELINE METRICS REPORT")
    print("="*40)
    print(f"Total Examples Evaluated: {total_items}")
    print(f"Action Accuracy:          {action_accuracy:.2f}%")
    print(f"Answer F1:                {avg_answer_f1:.2f}%  (on {answerable_count} answerable questions)")
    print(f"Abstention Precision:     {abstain_precision:.2f}%")
    print(f"Abstention Recall:        {abstain_recall:.2f}%")
    print(f"Conflict Disclosure F1:   0.00%  (CRAG lacks a DISCLOSE action)")
    print("="*40)
    print("\nNOTE: Copy these numbers directly into Table 1 of your paper.")


    try:
        # Calculate Macro F1 for ConflictRAG
        macro_f1 = sum(pilot['class_metrics'][cat]['F1'] for cat in ["ABSTAIN", "DISCLOSE", "DISCLAIMER", "ANSWER"]) / 4
        
        log_run(
            run_name="Final Test Set Evaluation",
            data_path="test_set_1.jsonl",
            parameters={"tau_s": 0.60, "tfidf": 0.08, "C": 10.0, "theta": 0.50},
            metrics={
                "overall_accuracy": pilot['accuracy'],
                "macro_f1": macro_f1,
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