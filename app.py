# app.py — Streamlit LawBot (Production Metadata-Aware Hybrid Retrieval + Knowledge Graph + Hallucination Auditing)
import os
import time
import json
import streamlit as st
from dotenv import load_dotenv

from legal_retriever import LegalRetriever
from legal_hallucination_detector import LegalHallucinationDetector
from legal_grounding_engine import build_grounded_generation_prompt, post_process_grounded_answer
from legal_llm_client import (
    PROVIDERS_CONFIG,
    detect_available_providers,
    detect_default_provider,
    get_friendly_error_message,
    generate_chat_completion,
    LLMException,
    LLMQuotaExhaustedError,
    LLMRateLimitError,
    LLMAuthError,
)

# ========== CONFIG ==========
CHROMA_PATH = "data/chroma_db"
BM25_PATH = "data/bm25_index.json"
RERANK_MODEL = os.getenv("RERANK_MODEL", "ms-marco-TinyBERT-L-2-v2")
MAX_SESSION_QUERIES = 25

# Load environment variables
load_dotenv()

# ========== PAGE CONFIG & INITIAL HEALTH CHECKS ==========
st.set_page_config(
    page_title="⚖️ LawBot – Indian Constitutional Law Assistant",
    page_icon="⚖️",
    layout="wide"
)
st.title("⚖️ LawBot — Indian Constitutional Law Intelligence Assistant")
st.caption("Next-Generation Constitutional Intelligence: Query Understanding • Knowledge Graph Traversal • Metadata-Aware Fusion • Hallucination Auditing")

# Sidebar for API Configuration & Session Management
with st.sidebar:
    st.header("⚙️ LLM Provider & Settings")

    provider_options = list(PROVIDERS_CONFIG.keys())
    provider_labels = {k: PROVIDERS_CONFIG[k]["display_name"] for k in provider_options}

    # Auto-detect default provider based on configured environment variables
    default_prov, default_key = detect_default_provider()
    default_idx = provider_options.index(default_prov) if default_prov in provider_options else 0

    selected_provider = st.selectbox(
        "Inference Provider",
        options=provider_options,
        index=default_idx,
        format_func=lambda x: provider_labels.get(x, x),
        help="Select your LLM inference provider. Groq is recommended for fast, 100% free synthesis."
    )

    provider_cfg = PROVIDERS_CONFIG[selected_provider]

    # Model selection
    supported_models = provider_cfg.get("supported_models", [provider_cfg["default_model"]])
    if selected_provider == "custom":
        active_model = st.text_input("Model Identifier", value=provider_cfg["default_model"])
        custom_endpoint = st.text_input("Endpoint URL", value=provider_cfg["base_url"])
        provider_cfg["base_url"] = custom_endpoint
    else:
        active_model = st.selectbox(
            "Model",
            options=supported_models,
            index=0,
            help=f"Default model for {provider_cfg['name']}: {provider_cfg['default_model']}"
        )

    # API Key resolution from env or user input (Never leak server env secrets to client DOM)
    env_keys = provider_cfg.get("env_keys", [])
    found_env_key = None
    for k in env_keys:
        val = os.getenv(k, "").strip()
        if val:
            found_env_key = val
            break

    if found_env_key:
        st.success(f"🔒 {provider_cfg['name']} key loaded securely from server environment")
        user_override = st.text_input(
            f"Override {provider_cfg['name']} Key (Optional)",
            value="",
            type="password",
            placeholder="Using server environment key",
            help="Leave blank to use the secure server environment key. Enter a custom key only if you wish to override it."
        )
        active_key = user_override.strip() if user_override.strip() else found_env_key
    else:
        user_key = st.text_input(
            f"{provider_cfg['name']} API Key",
            value="",
            type="password",
            placeholder=f"Enter {provider_cfg['name']} key...",
            help=f"Configure {provider_cfg['env_keys'][0]} in your environment or enter here."
        )
        active_key = user_key.strip() if user_key.strip() else None

    # Help links for obtaining API keys
    signup_url = provider_cfg.get("signup_url")
    if signup_url:
        if provider_cfg.get("is_free"):
            st.markdown(f"👉 **[Get Free {provider_cfg['name']} Key]({signup_url})** *(No credit card required)*")
        else:
            st.markdown(f"👉 **[{provider_cfg['name']} Billing & Keys]({signup_url})**")

    if provider_cfg.get("notes"):
        st.caption(f"💡 {provider_cfg['notes']}")

    st.divider()

    # Session Rate Limiter & Cache Observability
    if "session_queries" not in st.session_state:
        st.session_state.session_queries = 0

    st.markdown(f"📊 **Session Query Limit:** `{st.session_state.session_queries} / {MAX_SESSION_QUERIES}`")
    if st.button("🧹 Clear Response Cache", help="Clear cached responses to force fresh LLM synthesis"):
        st.cache_data.clear()
        st.success("Response cache cleared!")

# Validate database files on disk
if not os.path.exists(CHROMA_PATH) or not os.path.exists(BM25_PATH):
    st.error(
        f"📁 **Database Files Missing**: Required knowledge-base indexes ('{CHROMA_PATH}', '{BM25_PATH}') "
        f"were not found. Ensure `data/chroma_db` and `data/bm25_index.json` are committed to your repository."
    )
    st.stop()

# ========== INITIALIZE ENGINES (CACHED & LAZY-LOADED) ==========
@st.cache_resource(show_spinner=False)
def get_retriever(rerank_model: str):
    """Load persistent hybrid retriever, knowledge graph, and vector store."""
    return LegalRetriever(rerank_model=rerank_model)

@st.cache_resource(show_spinner=False)
def get_detector():
    """Load hallucination and source grounding detector."""
    return LegalHallucinationDetector()

@st.cache_data(show_spinner=False, ttl=86400)
def synthesize_cached_legal_answer(
    prompt: str,
    provider: str,
    model: str,
    api_key: str
) -> str:
    """Generate answer with query caching (24h TTL) to prevent repeated API credit consumption."""
    return generate_chat_completion(
        prompt=prompt,
        provider=provider,
        api_key=api_key,
        model=model,
        max_tokens=850,
        temperature=0.2,
    )

# Suppress Chroma telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Startup status wrapper allows Streamlit to respond to health probes while warming engines
with st.status("⚙️ Initializing LawBot Intelligence Engines...", expanded=False) as status:
    st.write("Loading hybrid vector store (ChromaDB) and BM25 index...")
    try:
        retriever = get_retriever(RERANK_MODEL)
        st.write("✓ Hybrid retriever ready")
    except Exception as e:
        if status is not None:
            status.update(label="❌ Failed to initialize retriever", state="error", expanded=True)
        st.error(f"Retriever initialization error: {e}")
        st.stop()

    st.write("Loading legal hallucination detection & grounding engine...")
    try:
        detector = get_detector()
        st.write("✓ Hallucination detector ready")
    except Exception as e:
        if status is not None:
            status.update(label="❌ Failed to initialize detector", state="error", expanded=True)
        st.error(f"Detector initialization error: {e}")
        st.stop()

    prov_title = provider_cfg.get("name", selected_provider)
    if active_key:
        st.write(f"✓ LLM client configured via {prov_title} (`{active_model}`)")
    else:
        st.write(f"ℹ️ {prov_title} API Key not configured. Retrieval & audit ready; synthesis requires key.")

    if status is not None:
        status.update(label="✅ LawBot Intelligence Engines Ready", state="complete", expanded=False)

# ========== CHAT UI & SESSION STATE ==========
st.markdown("💬 Ask any constitutional law question (e.g., *What is Article 21?*, *Which case established substantive due process?*, *What is the basic structure doctrine?*)")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Enter your constitutional question:")

if query:
    # Check session query cap
    if st.session_state.session_queries >= MAX_SESSION_QUERIES:
        st.warning(
            f"🛑 **Demo Usage Limit Reached**: You have reached the limit of {MAX_SESSION_QUERIES} queries for this session. "
            "Please restart the browser session to continue exploring."
        )
        st.stop()

    st.session_state.session_queries += 1
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

    # 🧠 Step 3: Structured Legal Analysis Prompt with Grounding & Contradiction Detection
    full_prompt, contradiction_warning = build_grounded_generation_prompt(
        query=query,
        context=context,
        top_candidates=top_candidates,
        cls_res=cls_res
    )

    # ⚖️ Step 4: Generate Grounded Answer via Multi-Provider LLM
    answer = None
    audit_res = None

    if not active_key:
        st.warning(
            f"🔑 **{provider_cfg['name']} API Key Required**: Please configure `{provider_cfg['env_keys'][0]}` "
            f"in your environment settings or enter an API key in the sidebar. "
            f"Grounded authorities and knowledge graph insights are retrieved below."
        )
    else:
        with st.spinner(f"⚖️ Synthesizing legal analysis with {provider_cfg['name']} ({active_model})..."):
            try:
                raw_answer = synthesize_cached_legal_answer(
                    prompt=full_prompt,
                    provider=selected_provider,
                    model=active_model,
                    api_key=active_key,
                )
                answer = post_process_grounded_answer(
                    raw_answer=raw_answer,
                    query=query,
                    top_candidates=top_candidates,
                    contradiction_warning=contradiction_warning
                )
                st.session_state.history.append({"role": "assistant", "content": answer})
            except LLMException as e:
                friendly = get_friendly_error_message(e, selected_provider)
                st.error(f"❌ **{friendly['title']}**\n\n{friendly['message']}\n\n👉 {friendly['action']}")
            except Exception as e:
                st.error(f"❌ **LLM Service Notice**: Unable to synthesize answer at this moment: {e}")

    # 🛡️ Step 5: Hallucination & Source Grounding Verification
    if answer:
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
    else:
        st.info("ℹ️ **Direct Authorities & Precedents**: Explore the retrieved constitutional provisions, ratios, and knowledge graph links below.")

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

        if contradiction_warning:
            st.warning(f"⚠️ **Pre-Generation Contradiction Detected:**\n\n{contradiction_warning}")

        st.text_area("Exact Structured Prompt Sent to LLM", full_prompt, height=220, disabled=True)

# 🕘 Sidebar History
with st.sidebar:
    st.header("🕘 Query History")
    for msg in st.session_state.history[::-1]:
        role = "👤 User" if msg["role"] == "user" else "⚖️ LawBot"
        st.markdown(f"**{role}:** {msg['content'][:140]}…")
