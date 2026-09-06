# legal_retriever.py — Production Metadata-Aware Hybrid Legal Retrieval Engine
import os
import re
import json
import time
from typing import List, Dict, Any, Tuple
import chromadb
from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest

from legal_query_classifier import LegalQueryClassifier
from legal_knowledge_graph import LegalKnowledgeGraph, normalize_article_to_graph_id

CHROMA_PATH = "data/chroma_db"
BM25_PATH = "data/bm25_index.json"
COLLECTION_NAME = "indian_cases"
LOG_DIR = "retrieval_logs"
LOG_FILE = os.path.join(LOG_DIR, "query_logs.jsonl")

class LegalRetriever:
    """
    Next-generation legal retriever featuring:
    - Query understanding & entity classification
    - Knowledge-graph 1-hop relevance expansion
    - Hybrid Dense + BM25 Candidate Generation
    - Cross-Encoder ONNX Reranking (TinyBERT / MiniLM)
    - Metadata-Aware Weighted Fusion
    - End-to-end stage telemetry and observability logging
    """

    def __init__(self, rerank_model: str = "ms-marco-TinyBERT-L-2-v2"):
        os.makedirs(LOG_DIR, exist_ok=True)
        
        # 1. Components
        self.classifier = LegalQueryClassifier()
        self.kg = LegalKnowledgeGraph()
        
        # 2. Vector Store
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.vector_db = self.client.get_collection(COLLECTION_NAME)

        # 3. BM25
        with open(BM25_PATH, "r", encoding="utf-8") as f:
            self.bm25_data = json.load(f)
        self.bm25 = BM25Okapi(self.bm25_data["tokenized_corpus"])

        # Corpus In-Memory Index for Fast Document Injection (Task 1)
        self.corpus_by_id = {}
        for did, doc, meta in zip(self.bm25_data["doc_ids"], self.bm25_data["documents"], self.bm25_data["metadatas"]):
            self.corpus_by_id[did] = {
                "doc_id": did,
                "content": doc,
                "title": meta.get("case_name", ""),
                "citation": meta.get("citation", ""),
                "court": meta.get("court", ""),
                "year": meta.get("year", 0),
                "doc_type": meta.get("doc_type", ""),
                "primary_article": meta.get("primary_article", ""),
                "secondary_articles": meta.get("secondary_articles", ""),
                "legal_topics": meta.get("legal_topics", ""),
                "url": meta.get("url", "#"),
            }

        # 4. FlashRank Cross-Encoder
        self.rerank_model = rerank_model
        self.ranker = Ranker(model_name=rerank_model, max_length=256)

    def _tokenize_text(self, text: str) -> List[str]:
        clean = re.sub(r"[^\w\s-]", " ", text.lower())
        return [t.strip() for t in clean.split() if len(t.strip()) > 1]

    def retrieve(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Executes complete metadata-aware hybrid retrieval pipeline.
        Returns top_k results and rich stage-by-stage observability diagnostics.
        """
        t_start = time.perf_counter()
        timings = {}

        # --- Stage 1: Query Understanding ---
        t0 = time.perf_counter()
        cls_res = self.classifier.classify(query)
        timings["classification_ms"] = (time.perf_counter() - t0) * 1000

        detected_articles = cls_res["detected_articles"]
        detected_cases = cls_res["detected_cases"]
        detected_doctrines = cls_res["detected_doctrines"]
        primary_class = cls_res["primary_class"]
        retrieval_bias = cls_res["retrieval_bias"]

        # --- Stage 2: Knowledge Graph Traversal & Expansion (Tasks 1, 3) ---
        t0 = time.perf_counter()
        all_entities = detected_cases + detected_doctrines + detected_articles
        graph_boosts = self.kg.compute_graph_boosts(all_entities, query=query, primary_class=primary_class)
        graph_rel = self.kg.get_related_entities(all_entities)

        # Bounded query expansion terms (Task 3)
        # Expand internal search queries for doctrine queries using canonical authorities
        query_expansion_terms = []
        if detected_doctrines:
            query_expansion_terms = self.kg.get_query_expansion_terms(
                detected_doctrines=detected_doctrines,
                max_terms=4
            )
        search_query = f"{query} {' '.join(query_expansion_terms)}".strip() if query_expansion_terms else query

        # Bounded candidate expansions (Task 1)
        graph_hops_used = 2
        kg_candidate_node_ids = self.kg.get_candidate_expansions(
            detected_articles=detected_articles,
            detected_cases=detected_cases,
            detected_doctrines=detected_doctrines,
            max_hops=graph_hops_used,
            max_expansions=10
        )

        doctrine_establishing_map = self.kg.get_doctrine_establishing_cases(detected_doctrines)
        doctrine_establishing_cases = set().union(*doctrine_establishing_map.values()) if doctrine_establishing_map else set()
        doctrine_linked_cases = self.kg.get_doctrine_linked_cases(detected_doctrines, max_hops=2)

        timings["graph_traversal_ms"] = (time.perf_counter() - t0) * 1000

        # --- Stage 3: Dense Vector Search (Top 20) with Query Expansion ---
        t0 = time.perf_counter()
        query_kwargs = {"query_texts": [search_query], "n_results": 20}
        if detected_articles and len(detected_articles) == 1:
            query_kwargs["where"] = {"primary_article": detected_articles[0]}
        
        try:
            vec_res = self.vector_db.query(**query_kwargs)
            if not vec_res.get("documents", [[]])[0]:
                vec_res = self.vector_db.query(query_texts=[search_query], n_results=20)
        except Exception:
            vec_res = self.vector_db.query(query_texts=[search_query], n_results=20)

        vec_docs = vec_res.get("documents", [[]])[0]
        vec_metas = vec_res.get("metadatas", [[]])[0]
        vec_ids = vec_res.get("ids", [[]])[0]
        vec_distances = vec_res.get("distances", [[]])[0] if "distances" in vec_res else [0.5]*len(vec_ids)
        timings["vector_search_ms"] = (time.perf_counter() - t0) * 1000

        # --- Stage 4: Lexical BM25 Search (Top 20) with Query Expansion ---
        t0 = time.perf_counter()
        tokens = self._tokenize_text(search_query)
        bm25_scores = self.bm25.get_scores(tokens)
        top_bm25_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:20]
        timings["bm25_search_ms"] = (time.perf_counter() - t0) * 1000

        # --- Stage 5: Candidate Merge & Knowledge Graph Candidate Injection (Task 1) ---
        t0 = time.perf_counter()
        candidate_map = {}
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0

        for rank, (did, doc, meta, dist) in enumerate(zip(vec_ids, vec_docs, vec_metas, vec_distances)):
            dense_sim = max(0.0, 1.0 - float(dist)) if dist is not None else 0.5
            candidate_map[did] = {
                "doc_id": did,
                "content": doc,
                "title": meta.get("case_name", ""),
                "citation": meta.get("citation", ""),
                "court": meta.get("court", ""),
                "year": meta.get("year", 0),
                "doc_type": meta.get("doc_type", ""),
                "primary_article": meta.get("primary_article", ""),
                "secondary_articles": meta.get("secondary_articles", ""),
                "legal_topics": meta.get("legal_topics", ""),
                "url": meta.get("url", "#"),
                "dense_sim": dense_sim,
                "dense_rank": rank + 1,
                "bm25_sim": 0.0,
                "bm25_rank": 999,
                "sources": ["vector"]
            }

        bm25_ids = []
        for rank, idx in enumerate(top_bm25_idx):
            did = self.bm25_data["doc_ids"][idx]
            bm25_ids.append(did)
            norm_bm25 = float(bm25_scores[idx]) / max_bm25
            if did in candidate_map:
                candidate_map[did]["bm25_sim"] = norm_bm25
                candidate_map[did]["bm25_rank"] = rank + 1
                candidate_map[did]["sources"].append("bm25")
            else:
                candidate_map[did] = {
                    "doc_id": did,
                    "content": self.bm25_data["documents"][idx],
                    "title": self.bm25_data["metadatas"][idx].get("case_name", ""),
                    "citation": self.bm25_data["metadatas"][idx].get("citation", ""),
                    "court": self.bm25_data["metadatas"][idx].get("court", ""),
                    "year": self.bm25_data["metadatas"][idx].get("year", 0),
                    "doc_type": self.bm25_data["metadatas"][idx].get("doc_type", ""),
                    "primary_article": self.bm25_data["metadatas"][idx].get("primary_article", ""),
                    "secondary_articles": self.bm25_data["metadatas"][idx].get("secondary_articles", ""),
                    "legal_topics": self.bm25_data["metadatas"][idx].get("legal_topics", ""),
                    "url": self.bm25_data["metadatas"][idx].get("url", "#"),
                    "dense_sim": 0.0,
                    "dense_rank": 999,
                    "bm25_sim": norm_bm25,
                    "bm25_rank": rank + 1,
                    "sources": ["bm25"]
                }

        # --- Knowledge Graph Candidate Injection (Task 1) ---
        kg_candidates_injected = []
        for exp_id in kg_candidate_node_ids:
            for corpus_did, corpus_item in self.corpus_by_id.items():
                is_match = (corpus_did == exp_id or corpus_did.startswith(f"{exp_id}_ratio_"))
                if is_match and corpus_did not in candidate_map:
                    candidate_map[corpus_did] = {
                        "doc_id": corpus_did,
                        "content": corpus_item["content"],
                        "title": corpus_item["title"],
                        "citation": corpus_item["citation"],
                        "court": corpus_item["court"],
                        "year": corpus_item["year"],
                        "doc_type": corpus_item["doc_type"],
                        "primary_article": corpus_item["primary_article"],
                        "secondary_articles": corpus_item["secondary_articles"],
                        "legal_topics": corpus_item["legal_topics"],
                        "url": corpus_item["url"],
                        "dense_sim": 0.0,
                        "dense_rank": 999,
                        "bm25_sim": 0.0,
                        "bm25_rank": 999,
                        "sources": ["kg_expansion"]
                    }
                    kg_candidates_injected.append(corpus_did)

        # Calculate candidate overlap
        overlap_ids = set(vec_ids).intersection(set(bm25_ids))
        overlap_pct = (len(overlap_ids) / max(1, len(vec_ids))) * 100
        timings["candidate_merge_ms"] = (time.perf_counter() - t0) * 1000

        candidates = list(candidate_map.values())

        # --- Stage 6: Cross-Encoder Reranking ---
        t0 = time.perf_counter()
        passages = []
        for c in candidates:
            passage_text = f"[{c['title']} | Citation: {c['citation']} | Provision: {c['primary_article']}]\n{c['content'][:1200]}"
            passages.append({"id": c["doc_id"], "text": passage_text})

        rerank_req = RerankRequest(query=query, passages=passages)
        rerank_results = self.ranker.rerank(rerank_req)
        score_map = {r["id"]: float(r["score"]) for r in rerank_results}
        timings["reranking_ms"] = (time.perf_counter() - t0) * 1000

        # --- Stage 7: Metadata-Aware Weighted Fusion & Doctrine-First Boosting (Tasks 2, 4) ---
        t0 = time.perf_counter()
        for c in candidates:
            did = c["doc_id"]
            cross_score = score_map.get(did, 0.0)
            
            # Metadata scoring boosts
            meta_boost = 0.0
            base_case_id = did.split("_ratio_")[0] if "_ratio_" in did else did
            
            # Article match
            if detected_articles and c["primary_article"] in detected_articles:
                meta_boost += 0.25
                if c["doc_type"] == "constitutional_article":
                    meta_boost += 0.15

            # Knowledge Graph boost
            graph_boost = 0.0
            if base_case_id in graph_boosts:
                graph_boost = graph_boosts[base_case_id]
            else:
                art_node_id = normalize_article_to_graph_id(c.get("primary_article", ""))
                if art_node_id in graph_boosts:
                    graph_boost = graph_boosts[art_node_id]

            # Doc type & domain alignment with query class (Metadata Boosts for all 9 classes)
            # --- Task 2: Doctrine-First & Explicit ESTABLISHES Relation Boosts ---
            if primary_class == "Doctrine Query":
                # Explicit ESTABLISHES relation boost
                if base_case_id in doctrine_establishing_cases:
                    meta_boost += 0.35
                    if c["doc_type"] == "ratio_chunk":
                        meta_boost += 0.15  # Total +0.50 for establishing ratio chunk
                elif base_case_id in doctrine_linked_cases or any(d.lower() in c.get("legal_topics", "").lower() for d in detected_doctrines):
                    meta_boost += 0.20
                    if c["doc_type"] == "ratio_chunk":
                        meta_boost += 0.10
                elif c["doc_type"] == "ratio_chunk":
                    meta_boost += 0.15 * retrieval_bias.get("ratio_boost", 1.0)

                # Unrelated detected case discount:
                # If query mentioned a case that does NOT establish and is NOT linked to the doctrine
                # (e.g. Maneka Gandhi in a Basic Structure query), suppress its case graph boost
                if detected_cases:
                    detected_case_node_ids = []
                    for cn in detected_cases:
                        matched_nids = self.kg.find_node_id(cn)
                        if matched_nids:
                            detected_case_node_ids.append(matched_nids[0])
                    if base_case_id in detected_case_node_ids and base_case_id not in doctrine_establishing_cases and base_case_id not in doctrine_linked_cases:
                        graph_boost = min(graph_boost, 0.0)

            elif primary_class == "Constitutional Provision Query" and c["doc_type"] == "constitutional_article":
                meta_boost += 0.25 * retrieval_bias.get("article_boost", 1.0)
            elif primary_class in ("Landmark Case Query", "Comparative Case Query") and c["doc_type"] in ("landmark_case", "ratio_chunk"):
                meta_boost += 0.15 * retrieval_bias.get("case_boost", 1.0)
            elif primary_class == "Amendment Query":
                if c["doc_type"] == "amendment" or "amend" in c.get("title", "").lower():
                    meta_boost += 0.15
                if "amendment" in c.get("title", "").lower() or c["primary_article"] == "Article 368":
                    meta_boost += 0.10
            elif primary_class == "Fundamental Rights Query":
                part_3_arts = {
                    "Article 12", "Article 13", "Article 14", "Article 15", "Article 16",
                    "Article 17", "Article 18", "Article 19", "Article 20", "Article 21",
                    "Article 21A", "Article 22", "Article 23", "Article 24", "Article 25",
                    "Article 26", "Article 27", "Article 28", "Article 29", "Article 30", "Article 32"
                }
                is_part_iii = (
                    "part iii" in c.get("secondary_articles", "").lower() or
                    "part iii" in c.get("legal_topics", "").lower() or
                    c["primary_article"] in part_3_arts
                )
                if is_part_iii:
                    meta_boost += 0.15
                if c["primary_article"] in part_3_arts:
                    meta_boost += 0.10
            elif primary_class == "Judicial Review Query":
                if "judicial review" in c.get("legal_topics", "").lower() or "judicial review" in c.get("title", "").lower():
                    meta_boost += 0.15
                if (c.get("doc_type") in ("landmark_case", "ratio_chunk") and 
                    any(t in c.get("legal_topics", "").lower() for t in ("judicial review", "basic structure", "writ", "habeas corpus")) or
                    c.get("doc_id") in ("case_041", "case_041_ratio_01", "art_032", "art_226")):
                    meta_boost += 0.10
            elif primary_class == "Procedural Law Query":
                is_procedural = any(w in c.get("legal_topics", "").lower() or w in c.get("title", "").lower() 
                                    for w in ("arrest", "bail", "detention", "procedure", "crpc", "guidelines"))
                if is_procedural or c.get("doc_id") in ("case_011", "case_012", "case_020"):
                    meta_boost += 0.10
                if c.get("doc_type") == "ratio_chunk" or any(w in c.get("legal_topics", "").lower() for w in ("procedure", "guidelines")):
                    meta_boost += 0.05
            elif primary_class == "General Legal Question":
                if c.get("doc_type") in ("constitutional_article", "landmark_case"):
                    meta_boost += 0.05

            # Topic keyword match
            if any(term in c["legal_topics"].lower() for term in tokens):
                meta_boost += 0.08

            # --- Task 4: Additive Class-Specific Bonuses ---
            class_bonus = 0.0
            if primary_class == "Constitutional Provision Query":
                class_bonus += 0.15 * c["bm25_sim"]
            elif primary_class == "Landmark Case Query":
                class_bonus += 0.15 * c["dense_sim"]
            elif primary_class == "Comparative Case Query":
                class_bonus += 0.10 * c["dense_sim"] + 0.10 * (1.0 if graph_boost > 0 else 0.0)

            # Preserved base fusion formula with additive class-specific bonuses
            final_score = (
                0.40 * cross_score +
                0.20 * c["dense_sim"] +
                0.15 * c["bm25_sim"] +
                meta_boost +
                graph_boost +
                class_bonus
            )

            c["cross_score"] = cross_score
            c["meta_boost"] = meta_boost
            c["graph_boost"] = graph_boost
            c["class_bonus"] = class_bonus
            c["final_score"] = final_score

        # Sort descending by final fused score
        ranked_candidates = sorted(candidates, key=lambda c: c["final_score"], reverse=True)
        top_results = ranked_candidates[:top_k]
        timings["fusion_ms"] = (time.perf_counter() - t0) * 1000

        total_latency_ms = (time.perf_counter() - t_start) * 1000
        timings["total_pipeline_ms"] = total_latency_ms

        # --- Stage 8: Observability Logging (Task 5) ---
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "query": query,
            "detected_class": primary_class,
            "detected_articles": detected_articles,
            "detected_cases": detected_cases,
            "detected_doctrines": detected_doctrines,
            "kg_candidates_injected": kg_candidates_injected,
            "query_expansion_terms": query_expansion_terms,
            "graph_hops_used": graph_hops_used,
            "graph_candidates_added": len(kg_candidates_injected),
            "classification": cls_res,
            "candidate_counts": {
                "vector_candidates": len(vec_ids),
                "bm25_candidates": len(bm25_ids),
                "kg_injected_candidates": len(kg_candidates_injected),
                "total_candidate_pool": len(candidates),
                "overlap_count": len(overlap_ids),
                "overlap_percentage": round(overlap_pct, 2)
            },
            "timings_ms": {k: round(v, 2) for k, v in timings.items()},
            "final_top_results": [
                {
                    "rank": i + 1,
                    "doc_id": r["doc_id"],
                    "title": r["title"],
                    "citation": r["citation"],
                    "doc_type": r["doc_type"],
                    "primary_article": r["primary_article"],
                    "final_score": round(r["final_score"], 4),
                    "cross_score": round(r["cross_score"], 4),
                    "meta_boost": round(r["meta_boost"], 4),
                    "graph_boost": round(r["graph_boost"], 4),
                    "class_bonus": round(r.get("class_bonus", 0.0), 4),
                    "sources": r["sources"]
                }
                for i, r in enumerate(top_results)
            ]
        }

        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

        return {
            "query": query,
            "classification": cls_res,
            "top_results": top_results,
            "candidate_pool_size": len(candidates),
            "overlap_percentage": overlap_pct,
            "kg_candidates_injected": kg_candidates_injected,
            "query_expansion_terms": query_expansion_terms,
            "graph_hops_used": graph_hops_used,
            "graph_candidates_added": len(kg_candidates_injected),
            "timings": timings,
            "graph_insights": graph_rel
        }

if __name__ == "__main__":
    retriever = LegalRetriever()
    print("=== LegalRetriever Production Test ===")
    res = retriever.retrieve("Which case established substantive due process in India?", top_k=5)
    print(f"Query: '{res['query']}'")
    print(f"Class: {res['classification']['primary_class']} (Conf: {res['classification']['confidence']:.2f})")
    print(f"Candidate Pool: {res['candidate_pool_size']} (Overlap: {res['overlap_percentage']:.1f}%)")
    print(f"Total Latency: {res['timings']['total_pipeline_ms']:.2f} ms")
    print("\nTop 5 Final Results:")
    for i, r in enumerate(res["top_results"], 1):
        print(f"  {i}. {r['title']} [{r['doc_type']}] — Score: {r['final_score']:.4f} (Cross: {r['cross_score']:.3f}, Meta: {r['meta_boost']:.2f}, Graph: {r['graph_boost']:.2f})")
