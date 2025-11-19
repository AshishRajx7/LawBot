# embed_cases.py
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import os

# Load cases
DATA_PATH = "data/cases.csv"
df = pd.read_csv(DATA_PATH)

print(f"⚖️ Loaded {len(df)} cases from {DATA_PATH}")

# 🔹 InLegalBERT model for Indian legal embeddings
model_name = "law-ai/InLegalBERT"
print(f"⏳ Loading model: {model_name}")
model = SentenceTransformer(model_name)

# Extract text list
texts = df["verdict_summary"].fillna("").tolist()

# Generate embeddings
print("🧠 Encoding case summaries into embeddings...")
embeddings = model.encode(texts, show_progress_bar=True)

# Initialize ChromaDB client
print("💾 Storing embeddings in Chroma vector database...")
import chromadb
client = chromadb.Client()
collection = client.create_collection("indian_cases")

# Add to collection
collection.add(
    documents=texts,
    embeddings=embeddings,
    metadatas=df.to_dict("records"),
    ids=[str(i) for i in range(len(texts))],
)

print(f"✅ Added {len(texts)} cases to Chroma collection 'indian_cases'")

# Optional: test query
query = "cases discussing reservation under Article 15"
query_embedding = model.encode([query])
results = collection.query(query_embeddings=query_embedding, n_results=3)

print("\n🔍 Sample retrieval results:")
for i, m in enumerate(results["metadatas"][0], 1):
    print(f"{i}. {m['case_name']} — {m['url']}")
