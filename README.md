# ⚖️ LawBot — AI Legal Query Assistant

LawBot is an AI powered legal assistance tool that answers legal questions using Retrieval Augmented Generation combined with a vector based search pipeline. It retrieves relevant case summaries and legal information from an embedded knowledge base and generates natural language answers grounded on evidence rather than hallucination.

Deployed App  
🔗 https://lawbot-xbvl.onrender.com/

---

## ⭐ Key Features

• Retrieval Augmented Generation based legal answers  
• Vector database for storing case embeddings  
• LangChain question answering pipeline  
• Reduces hallucinations by grounding answers in legal corpus  
• Modular design making it easy to extend to new legal domains  
• Fully deployed on Render for public access

---

## ⚙️ Architecture

User Question
⬇
Query Embedding (SentenceTransformer)
⬇
Vector Database Search (FAISS / ChromaDB)
⬇
Top Relevant Case Results Retrieved
⬇
LLM Answer Generation via LangChain RAG
⬇
Response with citations and reasoning

---

## 🧠 Tech Stack

| Component | Technology |
|----------|------------|
| Embeddings | SentenceTransformer / HuggingFace |
| Vector Store | ChromaDB |
| LLM Pipeline | LangChain |
| API | FastAPI (or Flask depending on build) |
| Front End | HTML + JS |
| Deployment | Render.com |
| Data Storage | Local JSON / CSV knowledge base |

---

## 📂 Repository Structure
```
LawBot
│
├── app.py Main application server
├── test_hf.py HF embedding tests
├── test_vakil.py Local test script
├── embed_cases.py Embedding generator for case database
├── collect_cases.py Script to collect and format case data
├── requirements.txt Python dependencies
├── vercel.json Deployment config (if used earlier)
└── data/
└── cases.json Source legal cases and summaries

```

---

## 🚀 How It Works Internally

1. Case summaries are preprocessed and embedded into vectors using SentenceTransformer  
2. Vectors are indexed in ChromaDB for fast similarity search  
3. When a user asks a question  
   • The question is embedded  
   • Similar vectors are retrieved  
   • The retrieved cases are passed to the LLM prompt  
4. LangChain constructs a final contextual answer that cites retrieved evidence

This ensures answers are **supported by real legal documents** rather than invented by the model.

---

## ▶️ Running Locally

```bash
git clone https://github.com/<your-username>/LawBot.git
cd LawBot
pip install -r requirements.txt
python app.py



📌 Future Improvements
• Support for multi jurisdiction case databases
• Upload your own case PDFs for on device indexing
• Similar case recommendations
• Legal citation and reference extraction
• Timeline explanations for court rulings

👤 Author
Developed by Ashish Raj
📩 Email: ashishrajx7@gmail.com
🔗 GitHub: https github com AshishRajx7

If you find LawBot useful or interesting, please consider ⭐ starring the repository.
