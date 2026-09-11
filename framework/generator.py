import requests
import config

class ResponseGenerator:
    def __init__(self):
        print(f"Connecting to local LLaMA API at {config.LLM_API_URL}...")
        self.api_url = config.LLM_API_URL

    def generate(self, question, passages, action):
        context = "\n".join(passages)
        
        if action == "ABSTAIN":
            return "I cannot answer based on the provided passages."
            
        elif action == "DISCLOSE":
            prompt = f"The retrieved sources disagree. Source A states: {passages[0]}\nSource B states: {passages[1]}\nQuestion: {question}"
            
        elif action == "DISCLAIMER":
            prompt = f"Based on incomplete evidence which may not fully answer the question, answer the following:\nPassages: {context}\nQuestion: {question}"
            
        else: # ANSWER
            prompt = f"You are a factual assistant. Use ONLY the following passages to answer.\nPassages: {context}\nQuestion: {question}"
            
        # Format prompt for LLaMA-3.1 Instruct
        formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        # Payload for the /completion endpoint
        payload = {
            "prompt": formatted_prompt,
            "n_predict": 100, 
            "temperature": 0.0, 
            "stop": ["<|eot_id|>"]
        }
        
        # Send the API request to the running server
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            return response.json()["content"].strip()
        except requests.exceptions.RequestException as e:
            return f"[Generation Failed - Ensure llama-server is running in the background] Error: {e}"