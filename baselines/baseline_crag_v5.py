import sys
sys.modules['apex'] = None
sys.modules['apex.normalization'] = None

import os
import json
import requests
import torch
import nltk
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Ensure NLTK sentence tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T5_MODEL_PATH = os.path.join(BASE_DIR, "local_models", "checkpoint-7059")
DATA_PATH = os.path.join(BASE_DIR, "data", "test_set_1.jsonl")
RESULTS_PATH = os.path.join(BASE_DIR, "results", "crag_test_set_1_results.json")

# FIX: Change port back to 8080 to match the running Llama server
LLM_API_URL = "http://127.0.0.1:8080/v1/chat/completions"

# CRAG Thresholds (Static-CRAG adaptation)
UPPER_THRESHOLD = 0.5  # If score > 0.5 -> CORRECT
LOWER_THRESHOLD = -0.5 # If score < -0.5 -> INCORRECT

# ==============================================================================
# COMPONENT 1: FINE-TUNED T5 EVALUATOR & REFINER
# ==============================================================================
class FineTunedCRAGEvaluator:
    def __init__(self, model_path):
        print(f"Loading Fine-Tuned T5 CRAG Evaluator from: {model_path}...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        
        # Get token IDs for logit extraction
        self.id_1 = self.tokenizer.encode("1", add_special_tokens=False)[0]
        self.id_neg1 = self.tokenizer.encode("-1", add_special_tokens=False)[0]
        
    def score_passage(self, question, passage):
        """Predicts continuous relevance score from -1.0 to 1.0 using logits."""
        input_text = f"Question: {question} Document: {passage}"
        inputs = self.tokenizer(input_text, return_tensors='pt', truncation=True, max_length=512, padding="max_length")
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            decoder_input_ids = torch.tensor([[self.model.config.decoder_start_token_id]]).to(self.device)
            outputs = self.model(**inputs, decoder_input_ids=decoder_input_ids)
            
            first_token_logits = outputs.logits[0, 0, :]
            probs = torch.softmax(first_token_logits, dim=-1)
            
            prob_1 = probs[self.id_1].item()
            prob_neg1 = probs[self.id_neg1].item()
            
            score = prob_1 - prob_neg1
        return score

    # FIX: Restored the missing refine_knowledge method
    def refine_knowledge(self, question, passage):
        """CRAG's decompose-then-recompose algorithm."""
        sentences = nltk.sent_tokenize(passage)
        if len(sentences) <= 1:
            return passage 
            
        refined_sentences = []
        for sent in sentences:
            score = self.score_passage(question, sent)
            if score > 0.0:  # Keep relevant sentences
                refined_sentences.append(sent)
                
        return " ".join(refined_sentences) if refined_sentences else passage

# ==============================================================================
# COMPONENT 2: LLAMA 3.1 API GENERATOR
# ==============================================================================
class LlamaAPIClient:
    def generate(self, system_prompt, user_prompt):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        payload = {
            "model": "llama3",
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 256
        }
        
        try:
            response = requests.post(LLM_API_URL, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            print(f"API Error during Llama generation: {e}")
            return "GENERATION_ERROR"

# ==============================================================================
# COMPONENT 3: CRAG BASELINE ORCHESTRATOR
# ==============================================================================
class CRAGBaseline:
    def __init__(self):
        self.evaluator = FineTunedCRAGEvaluator(T5_MODEL_PATH)
        self.generator = LlamaAPIClient()
        
    def get_crag_state_and_action(self, scores):
        """Static-CRAG Action Trigger (No Web Search)."""
        if any(s >= UPPER_THRESHOLD for s in scores):
            return "CORRECT", "ANSWER"
        elif all(s <= LOWER_THRESHOLD for s in scores):
            return "INCORRECT", "ABSTAIN"
        else:
            return "AMBIGUOUS", "DISCLAIMER"
            
    def process_item(self, item):
        question = item["question"]
        passages = item["passages"]
        
        raw_scores = [self.evaluator.score_passage(question, p) for p in passages]
        crag_state, predicted_action = self.get_crag_state_and_action(raw_scores)
        
        if crag_state in ["CORRECT", "AMBIGUOUS"]:
            refined_passages = [self.evaluator.refine_knowledge(question, p) for p in passages]
            context = "\n\n".join(refined_passages)
        else:
            context = "No relevant context available."
            
        generated_text = ""
        if predicted_action == "ANSWER":
            sys_prompt = "You are a factual question-answering assistant. Use ONLY the following passages to answer the question."
            user_prompt = f"Passages: {context}\n\nQuestion: {question}\n\nAnswer concisely:"
            generated_text = self.generator.generate(sys_prompt, user_prompt)
            
        elif predicted_action == "DISCLAIMER":
            sys_prompt = "You are a cautious assistant. The evidence is ambiguous. Provide a hedged answer based on the passages."
            user_prompt = f"Based on the available evidence, which may be incomplete:\nPassages: {context}\n\nQuestion: {question}\n\nAnswer:"
            generated_text = self.generator.generate(sys_prompt, user_prompt)
            
        elif predicted_action == "ABSTAIN":
            generated_text = "I cannot answer based on the provided passages."
            
        return {
            "id": item.get("id"),
            "question": question,
            "gold_action": item["gold_action"],
            "crag_internal_state": crag_state,
            "predicted_action": predicted_action,
            "retrieval_scores": raw_scores,
            "refined_context_used": context,
            "generated_text": generated_text,
            "gold_answer": item.get("gold_answer", "")
        }

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    print("Starting Static-CRAG Baseline (T5 Evaluator + Llama 3.1 Generator)...")
    
    if not os.path.exists(DATA_PATH):
        print(f"CRITICAL: Data file not found at {DATA_PATH}.")
        return

    with open(DATA_PATH, "r") as f:
        data = [json.loads(line) for line in f]
        
    crag = CRAGBaseline()
    results = []
    
    for i, item in enumerate(data):
        print(f"[{i+1}/{len(data)}] Processing: {item['question'][:50]}...")
        
        result = crag.process_item(item)
        results.append(result)
        
        print(f"   -> State: {result['crag_internal_state']} | Action: {result['predicted_action']}")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nCRAG Baseline complete! Results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    main()