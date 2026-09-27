"""Raw-feature cache with a fingerprint, so stale features are never silently reused.

Raw features are threshold-independent (see FeatureExtractor.extract_raw); tau_s and the
TF-IDF threshold are applied afterwards via features_from_raw().
"""
import hashlib
import json
import os

from framework import config
from framework.feature_extractor import EXTRACTOR_VERSION


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(data_path):
    return {
        "extractor_version": EXTRACTOR_VERSION,
        "data_file": os.path.basename(data_path),
        "data_sha256": _sha256(data_path),
        "embedding_model": config.EMBEDDING_MODEL,
        "nli_model": config.NLI_MODEL,
        "qa_model": config.QA_MODEL,
    }


def load_raw_features(data_path, cache_path, extractor_factory):
    """Returns (records, raw_list). Rebuilds the cache if the fingerprint does not match."""
    with open(data_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    fp = fingerprint(data_path)
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("fingerprint") == fp and len(cached.get("raw", [])) == len(records):
            print(f"Loaded raw features from {cache_path} (fingerprint match).")
            return records, cached["raw"]
        print(f"Cache {cache_path} is stale (fingerprint mismatch). Re-extracting.")

    extractor = extractor_factory()
    raw = []
    for i, item in enumerate(records):
        r = extractor.extract_raw(item["question"], item["passages"])
        r["answer_sims"] = [float(s) for s in r["answer_sims"]]
        raw.append(r)
        if (i + 1) % 250 == 0:
            print(f"  Extracted {i + 1}/{len(records)} from {os.path.basename(data_path)}")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"fingerprint": fp, "raw": raw}, f)
    print(f"Raw features cached to {cache_path}")
    return records, raw
