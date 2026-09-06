# app.py — Streamlit LawBot (Production Metadata-Aware Hybrid Retrieval + Knowledge Graph + Hallucination Auditing)
import os
import time
import json
import streamlit as st
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

from legal_retriever import LegalRetriever
from legal_hallucination_detector import LegalHallucinationDetector

# ========== CONFIG ==========
CHROMA_PATH = "data/chroma_db"
BM25_PATH = "data/bm25_index.json"
CHAT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
RERANK_MODEL = os.getenv("RERANK_MODEL", "ms-marco-TinyBERT-L-2-v2")

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_API_KEY")

# ========== PAGE CONFIG & INITIAL HEALTH CHECKS ==========
st.set_page_config(
    page_title="⚖️ LawBot – Indian Constitutional Law Assistant",
    page_icon="⚖️",
    layout="wide"
)
st.title("⚖️ LawBot — Indian Constitutional Law Intelligence Assistant")
st.caption("Next-Generation Constitutional Intelligence: Query Understanding • Knowledge Graph Traversal • Metadata-Aware Fusion • Hallucination Auditing")

# Validate environment variables
if not HF_TOKEN:
    st.error("🔑 **API Key Missing**: The `HF_API_KEY` environment variable is not configured. Please set `HF_API_KEY` in your environment or Render settings.")
    st.stop()

# Validate database files on disk
if not os.path.exists(CHROMA_PATH) or not os.path.exists(BM25_PATH):
    st.error(f"📁 **Database Files Missing**: Required knowledge-base indexes ('{CHROMA_PATH}', '{BM25_PATH}') were not found. Please run `python embed_cases.py` to build the hybrid index before launching.")
    st.stop()

# ========== INITIALIZE ENGINES (CACHED ONCE AT STARTUP) ==========
@st.cache_resource(show_spinner=False)
def init_retrieval_and_audit_engines():
    """Load persistent hybrid retriever, knowledge graph, and hallucination auditor"""
    try:
        retriever = LegalRetriever(rerank_model=RERANK_MODEL)
        detector = LegalHallucinationDetector()
        return retriever, detector, None
    except Exception as e:
        return None, None, f"Engine initialization failed: {e}"

@st.cache_resource(show_spinner=False)
def init_llm():
    """Initialize LLM model client"""
    return InferenceClient(model=CHAT_MODEL, token=HF_TOKEN)

retriever, detector, init_err = init_retrieval_and_audit_engines()
if init_err:
    st.error(f"❌ **Engine Initialization Error**: {init_err}")
    st.stop()

llm = init_llm()

# ========== CHAT UI & SESSION STATE ==========
st.markdown("💬 Ask any constitutional law question (e.g., *What is Article 21?*, *Which case established substantive due process?*, *What is the basic structure doctrine?*)")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Enter your constitutional question:")

if query:
    st.session_state.history.append({"role": "user", "content": query})

    # 🔍 Step 1: Execute End-to-End Metadata-Aware Hybrid Retrieval Pipeline
    with st.spinner("🔍 Executing query classification, graph traversal, hybrid retrieval & cross-encoder reranking..."):
        retrieval_res = retriever.retrieve(query, top_k=5)

    top_candidates = retrieval_res["top_results"]
    cls_res = retrieval_res["classification"]
    timings = retrieval_res["timings"]
    candidate_pool_size = retrieval_res["candidate_pool_size"]
    overlap_pct = retrieval_res["overlap_percentage"]
    graph_insights = retrieval_res.get("graph_insights", {})

    if not top_candidates:
        st.warning("⚠️ **Search Notice**: No relevant legal materials were found in the knowledge base.")
        st.stop()

    # 🏷️ Display Query Understanding Classification Pill
    col_cls1, col_cls2, col_cls3 = st.columns([2, 1, 1])
    with col_cls1:
        st.info(f"🏷️ **Query Intent:** `{cls_res['primary_class']}` (Confidence: `{cls_res['confidence']*100:.0f}%`)")
    with col_cls2:
        detected_arts = ", ".join(cls_res["detected_articles"]) if cls_res["detected_articles"] else "None"
        st.caption(f"📜 **Detected Articles:** `{detected_arts}`")
    with col_cls3:
        detected_cases = ", ".join(cls_res["detected_cases"]) if cls_res["detected_cases"] else "None"
        st.caption(f"⚖️ **Detected Cases:** `{detected_cases}`")

    # 📚 Step 2: Build Structured Legal Context Blocks (SOURCE 1, SOURCE 2, ...)
    context_blocks = []
    for i, c in enumerate(top_candidates, 1):
        doc_badge = (
            "Constitutional Article" if c["doc_type"] == "constitutional_article"
            else "Ratio Decidendi Chunk" if c["doc_type"] == "ratio_chunk"
            else "Landmark Precedent"
        )
        block = (
            f"SOURCE {i}\n"
            f"Title: {c['title']}\n"
            f"Citation: {c['citation']}\n"
            f"Court: {c['court']}\n"
            f"Year: {c['year']}\n"
            f"Document Type: {doc_badge}\n"
            f"Primary Provision: {c['primary_article']}\n\n"
            f"Content:\n{c['content']}"
        )
        context_blocks.append(block)

    separator = "\n\n" + ("=" * 40) + "\n\n"
    context = separator.join(context_blocks)

    # 🧠 Step 3: Structured Legal Analysis Prompt
    full_prompt = f"""You are an Indian constitutional law research assistant.

Answer only from the retrieved materials.

For every answer:
1. State the legal principle.
2. Cite the relevant constitutional provision.
3. Cite the relevant cases.
4. Explain the court's reasoning.
5. If retrieved materials are insufficient, explicitly say so.

Format your response strictly using the following four sections:

### Legal Principle
[State the exact legal and constitutional principle established by the authorities]

### Relevant Constitutional Provision
[Cite the constitutional article and explain its core constitutional mandate]

### Court Reasoning
[Detail the court's ratio decidendi and rationale from the retrieved sources]

### Authorities Relied Upon
[List every source cited with its full citation, formatted as:
1. Case / Article Name, Official Citation (Court, Year)
2. ...]

Retrieved Materials:
{context}

Question:
{query}"""

    # ⚖️ Step 4: Generate Grounded Answer via Hugging Face InferenceClient
    with st.spinner("⚖️ Synthesizing legal analysis from retrieved sources..."):
        try:
            response = llm.chat_completion(
                messages=[
                    {"role": "user", "content": full_prompt},
                ],
                max_tokens=750,
            )
            choice = response.choices[0]
            answer = (
                choice.message.content
                if hasattr(choice.message, "content")
                else choice.message.get("content", "")
            )
        except Exception as e:
            st.error(f"❌ **LLM API Error**: Failed to generate answer from Hugging Face Inference API: {e}")
            st.stop()

    st.session_state.history.append({"role": "assistant", "content": answer})

    # 🛡️ Step 5: Hallucination & Source Grounding Verification
    with st.spinner("🛡️ Auditing response for factual grounding and citation validity..."):
        audit_res = detector.audit(answer, top_candidates)

    # 🧾 Step 6: Render Answer
    st.subheader("🧠 LawBot’s Legal Analysis")
    st.markdown(answer)

    # 🛡️ Step 7: Render Hallucination & Grounding Audit Badge
    conf_score = audit_res["confidence_score"]
    badge_icon = "🟢" if conf_score >= 85 else "🟡" if conf_score >= 70 else "🔴"
    status_label = "High Confidence — Source Grounded" if conf_score >= 85 else "Moderate Confidence — Review Citations" if conf_score >= 70 else "Low Confidence — Potential Hallucinations Detected"

    st.markdown("---")
    st.subheader(f"🛡️ Source Grounding & Hallucination Audit {badge_icon}")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Grounding Confidence", f"{conf_score:.1f}%", help="Calculated based on citation validity, constitutional article existence, and sentence n-gram source overlap.")
    col_m2.metric("Supported Sources", ", ".join(audit_res["supported_sources"]) if audit_res["supported_sources"] else "None Identified")
    col_m3.metric("Verified Authorities", f"{len(audit_res['verified_authorities'])} Cited")

    # Display detected issues if any
    if audit_res["issues_detected"]:
        for issue in audit_res["issues_detected"]:
            st.warning(f"⚠️ {issue}")

    # Display potential unsupported statements if any
    if audit_res["potential_unsupported_statements"]:
        for unsupp in audit_res["potential_unsupported_statements"]:
            st.info(f"ℹ️ **Statement requiring verification:** *\"{unsupp}\"*")

    st.markdown("---")

    # 📚 Step 8: Display Structured Citation Authorities
    st.subheader("📚 Referenced Legal Authorities (Retrieved & Reranked)")
    for i, c in enumerate(top_candidates, 1):
        badge = (
            "📜 Constitutional Article" if c["doc_type"] == "constitutional_article"
            else "📌 Ratio Decidendi Chunk" if c["doc_type"] == "ratio_chunk"
            else "⚖️ Landmark Precedent"
        )
        cross_sc = c.get("cross_score", 0.0)
        meta_boost = c.get("meta_boost", 0.0)
        graph_boost = c.get("graph_boost", 0.0)
        final_sc = c.get("final_score", 0.0)
        
        st.markdown(
            f"**{i}. [{c['title']}]({c.get('url', '#')})** — `{c.get('citation', '')}` ({c.get('court', '')}, {c.get('year', '')})  \n"
            f"*{badge}* • **Final Score:** `{final_sc:.4f}` (Cross-Encoder: `{cross_sc:.3f}` | Meta Boost: `+{meta_boost:.2f}` | Graph Boost: `+{graph_boost:.2f}`)"
        )

    # 🛠️ Step 9: Advanced Diagnostics Expander
    with st.expander("🛠️ Advanced Retrieval & Observability Diagnostics", expanded=False):
        col_diag1, col_diag2 = st.columns(2)
        with col_diag1:
            st.write(f"**Query Class:** `{cls_res['primary_class']}`")
            st.write(f"**Candidate Pool Size:** `{candidate_pool_size}` unique documents")
            st.write(f"**Dense/BM25 Candidate Overlap:** `{overlap_pct:.1f}%`")
            st.write(f"**Total Retrieval Latency:** `{timings.get('total_pipeline_ms', 0):.2f} ms`")
        with col_diag2:
            st.write("**Stage Latency Breakdown (ms):**")
            st.json({k: round(v, 2) for k, v in timings.items()})

        if graph_insights and graph_insights.get("connected_nodes"):
            st.write("**1-Hop Knowledge Graph Traversals:**")
            for nid, node_meta in list(graph_insights["connected_nodes"].items())[:6]:
                st.write(f"- `[{node_meta['relation']}]` ➔ **{node_meta['name']}** ({node_meta['type']})")

        st.text_area("Exact Structured Prompt Sent to LLM", full_prompt, height=220, disabled=True)

# 🕘 Sidebar History
with st.sidebar:
    st.header("🕘 Query History")
    for msg in st.session_state.history[::-1]:
        role = "👤 User" if msg["role"] == "user" else "⚖️ LawBot"
        st.markdown(f"**{role}:** {msg['content'][:140]}…")
