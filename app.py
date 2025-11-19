# app.py — Streamlit LawBot Chat (Final)
import os
import streamlit as st
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# ========== CONFIG ==========
DATA_PATH = "data/cases.csv"
MODEL_NAME = "law-ai/InLegalBERT"
CHAT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# Load API key
load_dotenv()
HF_TOKEN = os.getenv("HF_API_KEY")

# ========== INIT ==========
st.set_page_config(page_title="⚖️ LawBot – Indian Legal Assistant", layout="wide")
st.title("⚖️ LawBot — Your Indian Legal Chat Assistant")

# Cache models and database
@st.cache_resource(show_spinner=False)
def init_retriever():
    """Initialize InLegalBERT + Chroma retriever"""
    df = pd.read_csv(DATA_PATH)
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.Client()

    # Create or load collection
    collection_name = "indian_cases"
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        collection = client.get_collection(collection_name)
    else:
        texts = df["verdict_summary"].fillna("").tolist()
        embeddings = model.encode(texts, show_progress_bar=True)
        collection = client.create_collection(collection_name)
        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=df.to_dict("records"),
            ids=[str(i) for i in range(len(texts))],
        )
    return model, collection

@st.cache_resource(show_spinner=False)
def init_llm():
    """Initialize Mistral-7B model"""
    return InferenceClient(model=CHAT_MODEL, token=HF_TOKEN)

embed_model, vector_db = init_retriever()
llm = init_llm()

# ========== CHAT UI ==========
st.markdown("💬 Ask any legal question (e.g., *What is the interpretation of Article 21?*)")

if "history" not in st.session_state:
    st.session_state.history = []  # stores dicts of {'role': 'user'/'assistant', 'content': str}

query = st.text_input("Enter your question:")

if query:
    # Add user query to session
    st.session_state.history.append({"role": "user", "content": query})

    # 🔍 Step 1: Retrieve top relevant cases
    with st.spinner("🔍 Searching relevant cases..."):
        query_embedding = embed_model.encode([query])
        results = vector_db.query(query_embeddings=query_embedding, n_results=3)
        metadatas = results["metadatas"][0]
        context = "\n\n".join(
            [f"{m['case_name']}: {m['verdict_summary']}" for m in metadatas]
        )

    # 🧠 Step 2: Build contextual prompt using conversation memory
    recent_turns = st.session_state.history[-5:]  # last 5 turns
    conversation_context = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_turns]
    )

    full_prompt = f"""
You are LawBot, an AI legal assistant specialized in Indian constitutional law.
Refer to the case context and conversation below to generate an accurate, concise, and legally correct answer.
Always explain with reasoning and cite relevant cases when possible.

Case Context:
{context}

Conversation:
{conversation_context}

User Question: {query}

Answer:
"""

    # ⚖️ Step 3: Generate the answer
    with st.spinner("⚖️ Generating contextual answer..."):
        response = llm.chat_completion(
            messages=[
                {"role": "system", "content": "You are LawBot, an expert in Indian law."},
                {"role": "user", "content": full_prompt},
            ],
            max_tokens=400,
        )
        answer = response.choices[0].message["content"]

    st.session_state.history.append({"role": "assistant", "content": answer})

    # 🧾 Step 4: Display result
    st.subheader("🧠 LawBot’s Answer")
    st.write(answer)

    st.markdown("---")
    st.subheader("📚 Referenced Cases")
    for m in metadatas:
        st.markdown(
            f"**[{m['case_name']}]({m['url']})** — {m['verdict_summary'][:250]}…"
        )

# 🕘 Sidebar History
with st.sidebar:
    st.header("🕘 Chat History")
    for msg in st.session_state.history[::-1]:
        role = "👤 User" if msg["role"] == "user" else "⚖️ LawBot"
        st.markdown(f"**{role}:** {msg['content'][:180]}…")
