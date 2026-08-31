import os
import json
import sys

# Add root directory to Python path to find data folder
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")

DEV_PATH = os.path.join(DATA_DIR, "dev_policy_wPilot.jsonl")
PILOT_PATH = os.path.join(DATA_DIR, "academic_pilot_data.jsonl")

def load_questions(filepath):
    """Loads a JSONL file and returns a set of questions."""
    questions = set()
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return questions
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                # Use the question as the unique identifier
                questions.add(item.get("question", "").strip())
    return questions

def main():
    print("Loading data to check for leakage...")
    dev_questions = load_questions(DEV_PATH)
    pilot_questions = load_questions(PILOT_PATH)
    
    print(f"Total questions in dev_policy.jsonl: {len(dev_questions)}")
    print(f"Total questions in academic_pilot_data.jsonl: {len(pilot_questions)}")
    
    # Find intersection (questions that appear in both sets)
    leakage = dev_questions.intersection(pilot_questions)
    
    print("\n" + "="*50)
    print("DATA LEAKAGE REPORT")
    print("="*50)
    
    if not leakage:
        print("✅ SUCCESS: Zero data leakage detected!")
        print("The dev set and pilot test set are strictly disjoint.")
    else:
        print(f"🚨 CRITICAL WARNING: {len(leakage)} overlapping questions found!")
        print("The following questions appear in BOTH files:")
        print("-" * 50)
        for i, q in enumerate(list(leakage)[:10]): # Print up to 10 examples
            print(f"{i+1}. {q[:100]}...")
        if len(leakage) > 10:
            print(f"...and {len(leakage) - 10} more.")
        print("-" * 50)
        print("You must regenerate the pilot set to ensure disjoint slices.")
    print("="*50)

if __name__ == "__main__":
    main()