# tests/test_retrieval_quality.py — Automated Retrieval Quality Evaluation Suite
import os
import re
import json
import chromadb
from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest

CHROMA_PATH = "data/chroma_db"
BM25_PATH = "data/bm25_index.json"
COLLECTION_NAME = "indian_cases"

KNOWN_ARTICLES = {
    "Article 12", "Article 13", "Article 14", "Article 15", "Article 16",
    "Article 17", "Article 18", "Article 19", "Article 20", "Article 21",
    "Article 21A", "Article 22", "Article 23", "Article 24", "Article 25",
    "Article 26", "Article 29", "Article 30", "Article 32", "Article 51A",
    "Article 124", "Article 141", "Article 142", "Article 226", "Article 300A",
    "Article 352", "Article 356", "Article 368"
}

def tokenize_legal_text(text: str):
    clean_text = re.sub(r"[^\w\s-]", " ", text.lower())
    return [t.strip() for t in clean_text.split() if len(t.strip()) > 1]

def detect_article(query: str):
    m = re.search(r"\bArticle\s*([0-9]{1,3}[A-Za-z]?)\b", query, re.IGNORECASE)
    if m:
        canonical = f"Article {m.group(1).upper()}"
        if canonical in KNOWN_ARTICLES:
            return canonical
    return None

class LegalRetrievalPipeline:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.vector_db = self.client.get_collection(COLLECTION_NAME)
        with open(BM25_PATH, "r", encoding="utf-8") as f:
            self.bm25_data = json.load(f)
        self.bm25 = BM25Okapi(self.bm25_data["tokenized_corpus"])
        self.ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")

    def retrieve_vector_only(self, query: str, top_k: int = 5):
        target_art = detect_article(query)
        kwargs = {"query_texts": [query], "n_results": top_k}
        if target_art:
            kwargs["where"] = {"primary_article": target_art}
        try:
            res = self.vector_db.query(**kwargs)
            if not res.get("documents", [[]])[0]:
                res = self.vector_db.query(query_texts=[query], n_results=top_k)
        except Exception:
            res = self.vector_db.query(query_texts=[query], n_results=top_k)
        
        return [
            {"doc_id": did, "title": m.get("case_name", ""), "doc_type": m.get("doc_type", "")}
            for did, m in zip(res.get("ids", [[]])[0], res.get("metadatas", [[]])[0])
        ]

    def retrieve_bm25_only(self, query: str, top_k: int = 5):
        tokens = tokenize_legal_text(query)
        scores = self.bm25.get_scores(tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {
                "doc_id": self.bm25_data["doc_ids"][i],
                "title": self.bm25_data["metadatas"][i].get("case_name", ""),
                "doc_type": self.bm25_data["metadatas"][i].get("doc_type", "")
            }
            for i in top_idx
        ]

    def retrieve_hybrid_reranked(self, query: str, top_k: int = 5):
        target_art = detect_article(query)
        
        # Vector top 20
        query_kwargs = {"query_texts": [query], "n_results": 20}
        if target_art:
            query_kwargs["where"] = {"primary_article": target_art}
        try:
            vec_res = self.vector_db.query(**query_kwargs)
            if not vec_res.get("documents", [[]])[0]:
                vec_res = self.vector_db.query(query_texts=[query], n_results=20)
        except Exception:
            vec_res = self.vector_db.query(query_texts=[query], n_results=20)

        vec_docs = vec_res.get("documents", [[]])[0]
        vec_metas = vec_res.get("metadatas", [[]])[0]
        vec_ids = vec_res.get("ids", [[]])[0]

        # BM25 top 20
        tokens = tokenize_legal_text(query)
        scores = self.bm25.get_scores(tokens)
        top_bm25_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:20]

        # Merge & deduplicate
        cand_map = {}
        for did, doc, m in zip(vec_ids, vec_docs, vec_metas):
            cand_map[did] = {
                "doc_id": did,
                "content": doc,
                "title": m.get("case_name", ""),
                "citation": m.get("citation", ""),
                "doc_type": m.get("doc_type", ""),
                "primary_article": m.get("primary_article", "")
            }
        for i in top_bm25_idx:
            did = self.bm25_data["doc_ids"][i]
            if did not in cand_map:
                cand_map[did] = {
                    "doc_id": did,
                    "content": self.bm25_data["documents"][i],
                    "title": self.bm25_data["metadatas"][i].get("case_name", ""),
                    "citation": self.bm25_data["metadatas"][i].get("citation", ""),
                    "doc_type": self.bm25_data["metadatas"][i].get("doc_type", ""),
                    "primary_article": self.bm25_data["metadatas"][i].get("primary_article", "")
                }

        candidates = list(cand_map.values())
        passages = [
            {"id": c["doc_id"], "text": f"[{c['title']} | Citation: {c['citation']} | Provision: {c['primary_article']}]\n{c['content']}"}
            for c in candidates
        ]
        
        rerank_req = RerankRequest(query=query, passages=passages)
        rerank_res = self.ranker.rerank(rerank_req)
        score_map = {r["id"]: r["score"] for r in rerank_res}
        ranked = sorted(candidates, key=lambda c: score_map.get(c["doc_id"], 0.0), reverse=True)
        return ranked[:top_k], len(candidates)

def evaluate():
    pipeline = LegalRetrievalPipeline()

    test_cases = [
        {
            "id": "TC_01",
            "query": "What is Article 21?",
            "expected_top1": ["Article 21: Protection of Life and Personal Liberty"],
            "expected_in_top5": ["Article 21", "Maneka Gandhi"]
        },
        {
            "id": "TC_02",
            "query": "Which case established substantive due process?",
            "expected_top1": ["Maneka Gandhi"],
            "expected_in_top5": ["Maneka Gandhi", "A.K. Gopalan"]
        },
        {
            "id": "TC_03",
            "query": "Which case established the right to privacy?",
            "expected_top1": ["Puttaswamy"],
            "expected_in_top5": ["Puttaswamy", "Kharak Singh", "PUCL"]
        },
        {
            "id": "TC_04",
            "query": "What is the basic structure doctrine?",
            "expected_top1": ["Kesavananda Bharati"],
            "expected_in_top5": ["Kesavananda Bharati", "Basic Structure"]
        },
        {
            "id": "TC_05",
            "query": "Can Parliament amend Fundamental Rights?",
            "expected_top1": ["Shankari Prasad", "Golaknath", "Kesavananda Bharati"],
            "expected_in_top5": ["Kesavananda", "Golaknath", "Article 368"]
        }
    ]

    print("=" * 80)
    print("PHASE 13 RETRIEVAL EVALUATION BENCHMARK")
    print("=" * 80)

    results_table = []

    for tc in test_cases:
        q = tc["query"]
        vec_results = pipeline.retrieve_vector_only(q, top_k=5)
        bm25_results = pipeline.retrieve_bm25_only(q, top_k=5)
        hybrid_results, pool_size = pipeline.retrieve_hybrid_reranked(q, top_k=5)

        # Check Top 1 for Hybrid
        top1_hybrid = hybrid_results[0]["title"]
        top1_vec = vec_results[0]["title"] if vec_results else "None"
        top1_bm25 = bm25_results[0]["title"] if bm25_results else "None"

        # Check match with expected
        pass_top1 = any(exp.lower() in top1_hybrid.lower() for exp in tc["expected_top1"])
        
        # Check Top 5
        top5_titles = [r["title"] for r in hybrid_results]
        pass_top5 = all(any(exp.lower() in t.lower() for t in top5_titles) for exp in tc["expected_in_top5"])

        status = "PASS" if (pass_top1 and pass_top5) else "FAIL"

        print(f"\n[{tc['id']}] Query: '{q}'")
        print(f"  Candidate Pool: {pool_size}")
        print(f"  Vector Only Rank 1:    {top1_vec}")
        print(f"  BM25 Only Rank 1:      {top1_bm25}")
        print(f"  Hybrid+Rerank Rank 1:  {top1_hybrid} [{'CORRECT' if pass_top1 else 'WRONG'}]")
        print("  Hybrid+Rerank Top 5:")
        for rank, r in enumerate(hybrid_results, 1):
            print(f"    {rank}. {r['title']} [{r['doc_type']}]")
        print(f"  Overall Status: [{status}]")

        results_table.append({
            "id": tc["id"],
            "query": q,
            "top1": top1_hybrid,
            "status": status
        })

    print("\n" + "=" * 80)
    all_passed = all(r["status"] == "PASS" for r in results_table)
    if all_passed:
        print("ALL 5 BENCHMARK TEST CASES PASSED WITH 100% ACCURACY!")
    else:
        print("SOME TEST CASES FAILED - REVIEW OUTPUT")
    print("=" * 80)

if __name__ == "__main__":
    evaluate()
