# embed_cases.py — Offline Legal Corpus & Hybrid BM25 Indexer
import os
import re
import json
import pandas as pd
import chromadb
from rank_bm25 import BM25Okapi

# Paths & Collection
DATA_PATH = "data/legal_corpus.csv"
CHROMA_PATH = "data/chroma_db"
BM25_PATH = "data/bm25_index.json"
COLLECTION_NAME = "indian_cases"

def tokenize_legal_text(text: str):
    """Tokenize legal text for BM25 keyword search: lowercase, strip punctuation, preserve legal terms"""
    clean_text = re.sub(r"[^\w\s-]", " ", text.lower())
    tokens = [t.strip() for t in clean_text.split() if len(t.strip()) > 1]
    return tokens

def main():
    print(f"[INFO] Loading expanded legal corpus from '{DATA_PATH}'...")
    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Loaded {len(df)} total legal records:")
    print(df["doc_type"].value_counts().to_string())

    texts = df["content"].fillna("").tolist()
    ids = df["doc_id"].astype(str).tolist()

    metadatas = []
    for _, row in df.iterrows():
        metadatas.append({
            "doc_id": str(row.get("doc_id", "")),
            "doc_type": str(row.get("doc_type", "")),
            "case_name": str(row.get("title", "")),
            "citation": str(row.get("citation", "")),
            "year": int(row["year"]) if pd.notna(row.get("year")) else 0,
            "primary_article": str(row.get("primary_article", "")),
            "secondary_articles": str(row.get("secondary_articles", "")),
            "legal_topics": str(row.get("legal_topics", "")),
            "court": str(row.get("court", "")),
            "bench": str(row.get("bench", "")),
            "url": str(row.get("url", "")),
        })

    # 1. Build Persistent ChromaDB Collection
    print(f"\n[INFO] Initializing Persistent ChromaDB client at '{CHROMA_PATH}'...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    existing_collections = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing_collections:
        print(f"[INFO] Removing existing collection '{COLLECTION_NAME}' for clean rebuild...")
        client.delete_collection(COLLECTION_NAME)

    print(f"[INFO] Creating collection '{COLLECTION_NAME}' (384-dim ONNX all-MiniLM-L6-v2)...")
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "model": "all-MiniLM-L6-v2"}
    )

    print(f"[INFO] Generating 384-dim ONNX embeddings and indexing {len(texts)} legal documents...")
    collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=ids,
    )
    print(f"[SUCCESS] Successfully indexed {len(texts)} documents into ChromaDB!")

    # 2. Build and Serialize Local BM25 Index
    print(f"\n[INFO] Building local BM25 tokenized corpus for hybrid retrieval...")
    tokenized_corpus = [tokenize_legal_text(f"{m['case_name']} {m['citation']} {m['primary_article']} {t}") for m, t in zip(metadatas, texts)]
    
    # Verify BM25 build
    bm25_test = BM25Okapi(tokenized_corpus)
    test_scores = bm25_test.get_scores(tokenize_legal_text("substantive due process Article 21"))
    top_idx = int(pd.Series(test_scores).idxmax())
    print(f"[INFO] BM25 Sanity Check: Top match for 'substantive due process Article 21' -> {metadatas[top_idx]['case_name']}")

    bm25_data = {
        "doc_ids": ids,
        "tokenized_corpus": tokenized_corpus,
        "documents": texts,
        "metadatas": metadatas,
    }
    with open(BM25_PATH, "w", encoding="utf-8") as f:
        json.dump(bm25_data, f, ensure_ascii=False)
    print(f"[SUCCESS] Saved local BM25 index to '{BM25_PATH}' ({os.path.getsize(BM25_PATH) / 1024:.1f} KB)!")

if __name__ == "__main__":
    main()
