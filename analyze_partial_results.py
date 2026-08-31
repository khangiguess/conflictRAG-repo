import os
import json
import re
from collections import Counter
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Update this path if your results are saved somewhere else
RESULTS_PATH = os.path.join(BASE_DIR, "results", "newSet_0.8_results.json")
PLOT_PATH = os.path.join(BASE_DIR, "results", "partial_0.8_results_distribution.png")

def main():
    if not os.path.exists(RESULTS_PATH):
        print(f"Error: Could not find {RESULTS_PATH}")
        return

    gold_actions = []
    content = ""

    print(f"Reading {RESULTS_PATH}...")
    with open(RESULTS_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    # Attempt 1: Standard JSON parsing
    try:
        data = json.loads(content)
        if isinstance(data, list):
            valid_samples = []
            for item in data:
                # STOP counting as soon as we hit the first generation error
                if item.get("generated_text") == "GENERATION_ERROR":
                    break
                valid_samples.append(item)
            
            gold_actions = [item.get("gold_action", "UNKNOWN") for item in valid_samples]
            print(f"Valid JSON parsed. Found {len(data)} total items, {len(valid_samples)} valid before first error.")
            
    except json.JSONDecodeError:
        print("File is also truncated/corrupted. Falling back to Regex parser...")
        # Find the index of the first "GENERATION_ERROR"
        error_match = re.search(r'"generated_text"\s*:\s*"GENERATION_ERROR"', content)
        if error_match:
            # Truncate the content to only include text BEFORE the error
            truncated_content = content[:error_match.start()]
        else:
            truncated_content = content
            
        # Find all gold_actions in the truncated content
        matches = re.findall(r'"gold_action"\s*:\s*"([A-Z]+)"', truncated_content)
        gold_actions = matches

    if not gold_actions:
        print("Error: Could not extract any valid gold_actions before the crash.")
        return

    total_processed = len(gold_actions)
    dist = Counter(gold_actions)

    # Print stats
    print("\n" + "="*50)
    print("VALID RESULTS DISTRIBUTION (Before LLM Crash)")
    print("="*50)
    print(f"Total Valid Samples Processed Before Crash: {total_processed}\n")
    
    class_order = ["ANSWER", "DISCLOSE", "ABSTAIN", "DISCLAIMER"]
    for action in class_order:
        count = dist.get(action, 0)
        percentage = (count / total_processed) * 100 if total_processed > 0 else 0
        print(f"{action:<12}: {count:>4} samples ({percentage:.2f}%)")
    print("="*50)

    # Generate Pie Chart
    labels = []
    sizes = []
    for action in class_order:
        if dist.get(action, 0) > 0:
            labels.append(f"{action}\n({dist[action]})")
            sizes.append(dist[action])

    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
    explode = [0.1 if label.startswith("ANSWER") else 0.0 for label in labels]

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, 
        explode=explode, 
        labels=labels, 
        colors=colors, 
        autopct='%1.1f%%',
        shadow=False, 
        startangle=140,
        textprops={'fontsize': 12, 'fontweight': 'bold'}
    )
    
    for autotext in autotexts:
        autotext.set_color('white')

    ax.axis('equal')
    plt.title(f"Valid Test Set Distribution (n={total_processed})", fontsize=14, pad=20)
    
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    plt.savefig(PLOT_PATH, dpi=300, bbox_inches='tight')
    print(f"\nSuccess! Pie chart saved to: {PLOT_PATH}")

if __name__ == "__main__":
    main()