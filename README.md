# ⚖️ LawBot — Indian Constitutional Law Intelligence Assistant

LawBot is an AI-powered Constitutional Law Research Assistant designed to answer legal questions using a hybrid retrieval architecture that combines semantic search, keyword retrieval, knowledge graph traversal, and evidence-grounded legal reasoning.

Unlike traditional chatbots that rely solely on LLM generation, LawBot retrieves authoritative constitutional provisions and legal knowledge before generating responses, significantly reducing hallucinations and improving legal accuracy.

### Live Demo
🔗 https://lawbot-hehe.streamlit.app/

---

# 🚀 Key Features

### Constitutional Law Intelligence
- Specialized for Indian Constitutional Law
- Supports questions about Articles, Fundamental Rights, Directive Principles, and constitutional doctrines
- Retrieves legally relevant constitutional provisions before answer generation

### Hybrid Retrieval Engine
LawBot does not rely on a single retrieval strategy.

It combines:

- Semantic Retrieval using Vector Embeddings
- BM25 Keyword Search
- Metadata-Aware Retrieval
- Knowledge Graph Traversal
- Hybrid Result Fusion

This improves retrieval quality for both conceptual and fact-based legal queries.

### Query Understanding Layer
Automatically identifies:

- Constitutional Articles
- Legal Doctrines
- Legal Principles
- Case References
- User Intent

Example:

```
What is Article 21?
```

Detected as:

```
Constitutional Provision Query
Article: 21
Confidence: 95%
```

### Hallucination Detection
Every generated answer passes through a legal hallucination auditing pipeline.

LawBot verifies:

- Whether the retrieved evidence supports the answer
- Whether the user's assumptions are legally valid
- Whether constitutional provisions are accurately referenced

### Legal Reasoning Engine
Produces:

- Constitutional interpretation
- Legal principle extraction
- Supporting evidence
- Explanation grounded in retrieved authorities

### Retrieval Consistency Verification

LawBot explicitly checks:

- User premise supported?
- Retrieved evidence relevance
- Constitutional provision consistency

before generating the final answer.

### Explainable Responses

Responses include:

- Legal Principle
- Relevant Constitutional Provision
- Supporting Authority
- Evidence-Based Explanation

instead of returning black-box AI answers.

---

# 🏗️ System Architecture

```text
User Query
    │
    ▼
Intent Detection
    │
    ▼
Article / Case Extraction
    │
    ▼
Hybrid Retrieval Layer
 ┌─────────────────────┐
 │ Vector Search       │
 │ BM25 Search         │
 │ Metadata Retrieval  │
 │ Knowledge Graph     │
 └─────────────────────┘
    │
    ▼
Result Fusion Engine
    │
    ▼
Legal Hallucination Audit
    │
    ▼
Legal Reasoning Layer
    │
    ▼
Evidence-Grounded Response
```

---

# 🧠 Technical Architecture

## Retrieval Layer

### Vector Search

Uses Sentence Transformers to create semantic embeddings of constitutional legal content.

Purpose:

- Concept matching
- Semantic similarity search
- Legal context retrieval

---

### BM25 Retrieval

Traditional keyword-based retrieval used alongside vector search.

Purpose:

- Exact legal phrase matching
- Constitutional article lookups
- Legal terminology retrieval

---

### Metadata Fusion

Additional retrieval weighting using:

- Article numbers
- Legal categories
- Constitutional provisions
- Doctrine mappings

---

### Knowledge Graph Traversal

LawBot maintains relationships between:

- Constitutional Articles
- Legal Principles
- Doctrines
- Judicial Interpretations

This enables contextual retrieval beyond simple similarity search.

---

# 🧩 Hallucination Auditing Pipeline

Before generating a final answer, LawBot evaluates:

### Premise Validation

Example:

```
User: Article 21 protects personal liberty.
```

LawBot verifies whether this claim is supported by retrieved legal authorities.

---

### Evidence Consistency

Checks whether generated conclusions are supported by retrieved constitutional provisions.

---

### Retrieval Verification

Ensures:

- Relevant authorities were retrieved
- Legal reasoning aligns with evidence
- Unsupported claims are not introduced

---

# 📊 Knowledge Base

The legal corpus currently contains:

- Constitutional Articles
- Constitutional Principles
- Legal Doctrines
- Structured Metadata
- Knowledge Graph Relationships

Indexed through:

- ChromaDB
- BM25 Search Index
- Graph-Based Lookup Structures

---

# 🛠️ Tech Stack

| Layer | Technology |
|---------|-------------|
| Frontend | Streamlit |
| Backend | Python |
| Embeddings | Sentence Transformers |
| Vector Store | ChromaDB |
| Keyword Search | BM25 |
| Retrieval Framework | Custom Hybrid RAG |
| Knowledge Graph | NetworkX |
| LLM Integration | Hugging Face Inference API |
| Data Processing | Pandas |
| Deployment | Streamlit Community Cloud |

---

# 📂 Project Structure

```text
LawBot
│
├── app.py
├── legal_retriever.py
├── legal_reasoning_engine.py
├── legal_hallucination_detector.py
├── query_intent_classifier.py
├── hybrid_retrieval.py
│
├── data/
│   ├── chroma_db/
│   ├── bm25_index.json
│   ├── legal_knowledge_graph.json
│
├── requirements.txt
├── README.md
└── assets/
```

---

# ⚙️ Running Locally

## Clone Repository

```bash
git clone https://github.com/AshishRajx7/LawBot.git
cd LawBot
```

## Create Virtual Environment

```bash
python -m venv .venv
```

## Activate Environment

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Configure Hugging Face API

Create a `.env` file:

```env
HF_API_KEY=your_huggingface_token
```

## Run Application

```bash
streamlit run app.py
```

---

# 📈 Future Roadmap

### Short-Term

- Case law integration
- Constitutional precedent retrieval
- Improved legal citation support
- Expanded constitutional corpus

### Long-Term

- Multi-jurisdiction legal databases
- PDF ingestion and indexing
- Legal document analysis
- Case comparison engine
- Citation recommendation system
- Legal research workflow automation

---

# 👨‍💻 Author

### Ashish Raj

Backend Developer | AI Engineer | Full Stack Developer

GitHub:
https://github.com/AshishRajx7

LinkedIn:
https://www.linkedin.com/in/ashish-raj-71717b28a/

Email:
ashishrajcr7@gmail.com

---

# ⭐ Acknowledgements

Built using:

- Streamlit
- Hugging Face
- Sentence Transformers
- ChromaDB
- NetworkX
- Python Open Source Ecosystem

---

If you find this project useful, consider giving it a ⭐ on GitHub.
