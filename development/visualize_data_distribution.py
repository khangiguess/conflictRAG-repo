import os
import json
import sys
from collections import Counter
import matplotlib.pyplot as plt

# Add root directory to Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

DATA_DIR = os.path.join(ROOT_DIR, "data")
RESULTS_DIR = os.path.join(ROOT_DIR, "data")
INPUT_PATH = os.path.join(DATA_DIR, "dev_set_natural.jsonl")
output_path = os.path.join(RESULTS_DIR, "dev_set_natural_distribution_pie.png")

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Error: Could not find {INPUT_PATH}")
        return

    # 1. Load the dev data
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        dev_data = [json.loads(line) for line in f]

    # 2. Count the distribution of gold actions
    actions = [item["gold_action"] for item in dev_data]
    dist = Counter(actions)
    
    # 3. Print the raw numbers and percentages
    total = len(dev_data)
    print("\n" + "="*50)
    print("DEVELOPMENT SET DATA DISTRIBUTION")
    print("="*50)
    print(f"Total Examples: {total}\n")
    
    # Define the order of classes for consistent display
    class_order = ["ANSWER", "DISCLOSE", "ABSTAIN", "DISCLAIMER"]
    
    for action in class_order:
        count = dist.get(action, 0)
        percentage = (count / total) * 100
        print(f"{action:<12}: {count:>4} examples ({percentage:.2f}%)")
    print("="*50)

    # 4. Generate a Pie Chart
    labels = []
    sizes = []
    for action in class_order:
        if dist.get(action, 0) > 0:
            labels.append(f"{action}\n({dist[action]})")
            sizes.append(dist[action])

    # Academic color palette (Blue, Orange, Green, Red)
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
    
    # Explode the 1st slice (ANSWER) slightly for emphasis
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

    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title("ConflictRAG Development Set Natural Distribution", fontsize=14, pad=20)
    
    # Save the plot
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSuccess! Pie chart saved to: {output_path}")

if __name__ == "__main__":
    main()