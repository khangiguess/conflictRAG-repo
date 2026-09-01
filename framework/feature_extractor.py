import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline
from langdetect import detect
import config

class FeatureExtractor:
    def __init__(self):
        print("Loading embedding, QA, and NLI models...")
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL, device=config.DEVICE)
        self.nli_model = CrossEncoder(config.NLI_MODEL, device=config.DEVICE)
        
        # FIX 1: Use actual HuggingFace QA pipeline for span extraction
        self.qa_extractor = pipeline("question-answering", model=config.QA_MODEL, device=0 if config.DEVICE=="cuda" else -1)
        
        # Dynamically find the contradiction index
        self.nli_labels = self.nli_model.model.config.id2label
        self.contra_idx = list(self.nli_labels.values()).index("CONTRADICTION") if "CONTRADICTION" in self.nli_labels.values() else 2
        
    def extract(self, question, passages):
        k = len(passages)
        passage_embeddings = self.embedder.encode(passages, convert_to_numpy=True)
        
        # --- Formula 1: Retrieval Dispersion D(P) ---
        if k > 1:
            sim_matrix = np.inner(passage_embeddings, passage_embeddings)
            triu_indices = np.triu_indices(k, k=1)
            avg_sim = np.mean(sim_matrix[triu_indices])
            dispersion = 1.0 - avg_sim
        else:
            dispersion = 0.0
            
        # --- Formula 2: Lexical Contradiction C-hat(P) ---
        max_contradiction = 0.0
        if k > 1:
            pairs = [(passages[i], passages[j]) for i in range(k) for j in range(i+1, k)]
            scores = np.array([self.nli_model.predict([pair], apply_softmax=True)[0] for pair in pairs])
            contradiction_probs = scores[:, self.contra_idx] if len(scores.shape) > 1 else [scores[self.contra_idx]]
            max_contradiction = float(np.max(contradiction_probs))
                    
        # --- Formula 3: Semantic Claim Support S(a, P) ---
        # Extract the actual answer span, then embed it
        best_passage = passages[0]
        answer_span = question # fallback
        try:
            qa_result = self.qa_extractor(question=question, context=best_passage)
            answer_span = qa_result['answer']
        except:
            pass # Fallback to question if QA model fails
            
        answer_embedding = self.embedder.encode([answer_span], convert_to_numpy=True)[0]
        similarities = np.inner(passage_embeddings, answer_embedding)
        support_count = np.sum(similarities >= config.TAU_S)
        claim_support = support_count / k
        
        # --- Formula 4: Language AND Domain Mismatch L(q, P) ---
        mismatch_flag = 0
        # Language check
        try:
            q_lang = detect(question)
            p_langs = [detect(p) for p in passages]
            if p_langs.count(q_lang) < (k / 2):
                mismatch_flag = 1
        except:
            pass
            
        # Domain check (TF-IDF vocabulary overlap)
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([question] + passages)
            # Cosine similarity between query (row 0) and passages (row 1:)
            cos_sims = (tfidf_matrix[0:1] * tfidf_matrix[1:].T).toarray()[0]
            # If the best passage has < TF-IDF overlap, it's a domain mismatch
            if np.max(cos_sims) < 0.1:
                mismatch_flag = 1
        except:
            pass

        # Features 5 & 6
        k_norm = k / 10.0
        query_len_norm = len(question.split()) / 20.0

        return np.array([
            float(dispersion), 
            float(max_contradiction), 
            float(claim_support), 
            int(mismatch_flag),
            float(k_norm),
            float(query_len_norm)
        ])