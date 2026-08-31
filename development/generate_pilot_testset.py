import os
import sys
import json
import random
from collections import defaultdict

# Add root directory to Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
INPUT_PATH = os.path.join(DATA_DIR, "final_testset.jsonl")
OUTPUT_PATH = os.path.join(DATA_DIR, "pilot_testset.jsonl")

def generate_pilot():
    if not os.path.exists(INPUT_PATH):
        print(f"Error: Could not find {INPUT_PATH}. Please run generate_final_testset.py first.")
        return

    print(f"Loading core test set from {INPUT_PATH}...")
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    # Group examples by gold_action
    examples_by_class = defaultdict(list)
    for item in data:
        examples_by_class[item["gold_action"]].append(item)

    pilot_data = []
    target_per_class = 50

    print("Extracting 50 examples per class (25% distribution)...")
    for action in ["ANSWER", "ABSTAIN", "DISCLOSE", "DISCLAIMER"]:
        examples = examples_by_class.get(action, [])
        
        if len(examples) < target_per_class:
            print(f"  [Warning] Only {len(examples)} examples available for {action}. Taking all.")
            pilot_data.extend(examples)
        else:
            # Take the first 50 (or you could use random.sample(examples, target_per_class))
            pilot_data.extend(examples[:target_per_class])

    # Shuffle the final pilot set for random distribution during inference
    random.seed(42)
    random.shuffle(pilot_data)

    # Save to disk
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for item in pilot_data:
            f.write(json.dumps(item) + "\n")

    # Print verification stats
    from collections import Counter
    dist = Counter(item["gold_action"] for item in pilot_data)
    print(f"\nSuccess! {len(pilot_data)} pilot examples saved to {OUTPUT_PATH}")
    print(f"Pilot Distribution: {dict(dist)}")

if __name__ == "__main__":
    generate_pilot()