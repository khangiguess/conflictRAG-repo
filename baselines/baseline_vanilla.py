import os
import json
import requests

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "test_set_1.jsonl")
RESULTS_PATH = os.path.join(BASE_DIR, "results", "vanilla_test_set_1_results.json")

# Llama.cpp Server API (OpenAI-compatible endpoint)
LLM_API_URL = "http://127.0.0.1:8080/v1/chat/completions"

# ==============================================================================
# COMPONENT 1: VANILLA LLM ZERO-SHOT ROUTER
# ==============================================================================
class VanillaLLMRouter:
    def route(self, question, passages):
        context = "\n".join(passages)
        
        system_prompt = "You are a strict Retrieval Evaluator for a RAG system."
        user_prompt = (
            "Classify the following question and context into exactly one of four categories:\n"
            "1. DISCLOSE: The passages contain direct factual contradictions about the answer.\n"
            "2. ABSTAIN: The passages are completely irrelevant or empty.\n"
            "3. DISCLAIMER: The passages are topically related but incomplete or ambiguous.\n"
            "4. ANSWER: The passages clearly contain sufficient and consistent information to answer.\n\n"
            "You must output ONLY the category name. Do not explain.\n\n"
            f"Passages:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Category:"
        )
        
        payload = {
            "model": "llama3",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 10  # We only need the category name, cap generation early
        }
        
        try:
            response = requests.post(LLM_API_URL, json=payload)
            response.raise_for_status()
            # Clean up the output to extract just the category word
            text = response.json()["choices"][0]["message"]["content"].strip().upper().split('\n')[0]
            
            valid_categories = ["ABSTAIN", "DISCLOSE", "DISCLAIMER", "ANSWER"]
            for cat in valid_categories:
                if cat in text:
                    return cat
            return "FAILED_TO_ROUTE"
            
        except requests.exceptions.RequestException as e:
            print(f"API Error during routing: {e}")
            return "ERROR"

# ==============================================================================
# COMPONENT 2: ORCHESTRATOR
# ==============================================================================
class VanillaBaseline:
    def __init__(self):
        self.router = VanillaLLMRouter()
        
    def process_item(self, item):
        question = item["question"]
        passages = item["passages"]
        
        # Step 1: Route using purely zero-shot LLM prompt
        predicted_action = self.router.route(question, passages)
            
        return {
            "id": item.get("id"),
            "question": question,
            "gold_action": item["gold_action"],
            "predicted_action": predicted_action,
            "generated_text": "N/A",  # Explicitly bypassing generation for this routing evaluation
            "gold_answer": item.get("gold_answer", "")
        }

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    print("Starting Vanilla LLM Baseline (Zero-Shot Routing Only)...")
    
    if not os.path.exists(DATA_PATH):
        print(f"CRITICAL: Data file not found at {DATA_PATH}.")
        return

    with open(DATA_PATH, "r") as f:
        data = [json.loads(line) for line in f]
        
    baseline = VanillaBaseline()
    results = []
    
    for i, item in enumerate(data):
        print(f"[{i+1}/{len(data)}] Processing: {item['question'][:50]}...")
        
        result = baseline.process_item(item)
        results.append(result)
        
        print(f"   -> Gold: {item['gold_action']:<10} | Pred: {result['predicted_action']}")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nVanilla LLM Baseline complete! Results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    main()