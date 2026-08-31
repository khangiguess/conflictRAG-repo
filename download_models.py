from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline

print("Downloading Retrieval Dispersion Encoder...")
# all-MiniLM-L6-v2: Used to measure semantic disagreement
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Downloading Lexical Contradiction Model...")
# nli-MiniLM2-L6-H768: Used to detect pairwise contradictions
nli_model = CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768")

print("Downloading Semantic Claim Support QA Model...")
# minilm-uncased-squad2: Used to extract candidate answer spans
qa_model = pipeline("question-answering", model="deepset/minilm-uncased-squad2")

print("All feature extraction models cached successfully!")