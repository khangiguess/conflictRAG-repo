import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from langdetect import detect, LangDetectException
from framework import config

# Bump whenever extract_raw() semantics change, so cached raw features are invalidated.
EXTRACTOR_VERSION = "2026-09-26.1"

FEATURE_NAMES = ["dispersion", "nli_contradiction", "claim_support", "mismatch", "k_norm", "query_len_norm"]


class FeatureExtractor:
    def __init__(self):
        print("Loading embedding, QA, and NLI models...")
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL, device=config.DEVICE)
        self.nli_model = CrossEncoder(config.NLI_MODEL, device=config.DEVICE)

        # Bypass transformers pipeline to avoid 'Unknown task' KeyError on server
        self.qa_tokenizer = AutoTokenizer.from_pretrained(config.QA_MODEL)
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(config.QA_MODEL).to(config.DEVICE)
        self.qa_model.eval()

        # Resolve the contradiction index case-insensitively. The model's labels are lowercase
        # ("contradiction", "entailment", "neutral"); the previous exact match on "CONTRADICTION"
        # silently fell back to index 2, which is "neutral".
        id2label = self.nli_model.model.config.id2label
        matches = [int(i) for i, lab in id2label.items() if lab.lower() == "contradiction"]
        if len(matches) != 1:
            raise ValueError(f"Cannot locate 'contradiction' label in NLI model id2label: {id2label}")
        self.contra_idx = matches[0]

    def extract_raw(self, question, passages):
        """Threshold-independent quantities. tau_s and the TF-IDF threshold are applied in
        features_from_raw(), so hyperparameter sweeps / ablations do not require re-extraction."""
        k = len(passages)
        passage_embeddings = self.embedder.encode(passages, convert_to_numpy=True)

        # --- Formula 1: Retrieval Dispersion D(P) ---
        if k > 1:
            sim_matrix = np.inner(passage_embeddings, passage_embeddings)
            triu_indices = np.triu_indices(k, k=1)
            dispersion = 1.0 - float(np.mean(sim_matrix[triu_indices]))
        else:
            dispersion = 0.0

        # --- Formula 2: Contradiction C-hat(P) ---
        max_contradiction = 0.0
        if k > 1:
            pairs = [(passages[i], passages[j]) for i in range(k) for j in range(i + 1, k)]
            scores = np.asarray(self.nli_model.predict(pairs, apply_softmax=True)).reshape(len(pairs), -1)
            max_contradiction = float(np.max(scores[:, self.contra_idx]))

        # --- Formula 3 (raw): answer-span similarity to each passage ---
        inputs = self.qa_tokenizer(question, passages[0], return_tensors='pt', truncation=True, max_length=512).to(config.DEVICE)
        with torch.no_grad():
            outputs = self.qa_model(**inputs)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        answer_span = self.qa_tokenizer.decode(inputs['input_ids'][0][answer_start:answer_end])
        answer_embedding = self.embedder.encode([answer_span], convert_to_numpy=True)[0]
        answer_sims = np.inner(passage_embeddings, answer_embedding).astype(float)

        # --- Formula 4 (raw): language mismatch + max TF-IDF overlap ---
        lang_mismatch = 0
        try:
            q_lang = detect(question)
            p_langs = [detect(p) for p in passages]
            if p_langs.count(q_lang) < (k / 2):
                lang_mismatch = 1
        except LangDetectException:
            pass  # undetectable text (e.g. empty/numeric): no language evidence

        try:
            tfidf_matrix = TfidfVectorizer().fit_transform([question] + passages)
            max_tfidf_sim = float(np.max((tfidf_matrix[0:1] * tfidf_matrix[1:].T).toarray()[0]))
        except ValueError:
            max_tfidf_sim = float("nan")  # empty vocabulary: no domain evidence, never flags

        return {
            "dispersion": float(dispersion),
            "nli_contradiction": max_contradiction,
            "answer_sims": answer_sims,
            "lang_mismatch": int(lang_mismatch),
            "max_tfidf_sim": max_tfidf_sim,
            "k": k,
            "query_len": len(question.split()),
        }

    def extract(self, question, passages):
        return features_from_raw(self.extract_raw(question, passages))


def features_from_raw(raw, tau_s=None, tfidf_thresh=None):
    """Builds the 6D feature vector from extract_raw() output."""
    tau_s = config.TAU_S if tau_s is None else tau_s
    tfidf_thresh = config.TFIDF_THRESH if tfidf_thresh is None else tfidf_thresh

    sims = np.asarray(raw["answer_sims"], dtype=float)
    claim_support = float(np.sum(sims >= tau_s)) / raw["k"]
    domain_mismatch = raw["max_tfidf_sim"] < tfidf_thresh  # NaN compares False
    mismatch_flag = int(raw["lang_mismatch"] == 1 or domain_mismatch)

    return np.array([
        float(raw["dispersion"]),
        float(raw["nli_contradiction"]),
        claim_support,
        mismatch_flag,
        raw["k"] / 10.0,
        raw["query_len"] / 20.0,
    ])
