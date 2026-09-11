import requests
import config

class ResponseGenerator:
    def __init__(self):
        self.api_url = config.LLM_API_URL
        
    def generate(self, question, passages, action):
        # 1. If ABSTAIN, bypass LLM entirely to save compute
        if action == "ABSTAIN":
            return "I cannot answer based on the provided passages."
            
        context = "\n\n".join(passages)
        
        # 2. Format prompts based on the action
        if action == "DISCLOSE":
            sys_prompt = "You are an analytical assistant. The retrieved sources disagree. Disclose the conflict to the user based on the passages."
            user_prompt = f"The retrieved sources disagree on this point.\nPassages: {context}\n\nQuestion: {question}\n\nAnswer:"
            
        elif action == "DISCLAIMER":
            sys_prompt = "You are a cautious assistant. The evidence is ambiguous. Provide a hedged answer based on the passages."
            user_prompt = f"Based on the available evidence, which may be incomplete:\nPassages: {context}\n\nQuestion: {question}\n\nAnswer:"
            
        else: # ANSWER
            sys_prompt = "You are a factual question-answering assistant. Use ONLY the following passages to answer the question."
            user_prompt = f"Passages: {context}\n\nQuestion: {question}\n\nAnswer concisely:"

        # 3. Make the API call
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        payload = {
            "model": "llama3", 
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 256
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=30)
            response.raise_for_status() # Automatically catches 400/500 HTTP errors
            
            # Parse the JSON safely
            data = response.json()
            
            # Attempt to extract using the standard OpenAI/Llama format
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"].strip()
            else:
                # If the format is different, print it so you know exactly what the server returned
                print(f"   [API Format Error] Unexpected response structure: {data}")
                return "GENERATION_ERROR"
                
        except requests.exceptions.RequestException as e:
            print(f"   [API Error] Llama generation failed: {e}")
            # If there is a response body attached to the error, print it for debugging
            if e.response is not None:
                print(f"   [API Error Body] {e.response.text}")
            return "GENERATION_ERROR"
            
        except (KeyError, IndexError, TypeError) as e:
            # Catches the exact KeyError you were getting before
            print(f"   [Parsing Error] Could not parse LLM response: {e}")
            print(f"   [Raw Response] {response.text}")
            return "GENERATION_ERROR"