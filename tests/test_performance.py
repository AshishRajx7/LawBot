# tests/test_performance.py — Performance, Latency & Memory Profiling Suite (Phases 15–20)
import os
import sys
import time
import json
import psutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath("."))

from legal_retriever import LegalRetriever
from legal_hallucination_detector import LegalHallucinationDetector

CHROMA_PATH = "data/chroma_db"
BM25_PATH = "data/bm25_index.json"
CORPUS_PATH = "data/legal_corpus.csv"
KG_PATH = "data/legal_knowledge_graph.json"

def get_dir_size_mb(path: str) -> float:
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)

def profile():
    process = psutil.Process(os.getpid())
    mem_baseline = process.memory_info().rss / (1024 * 1024)

    t0_start = time.perf_counter()
    
    # 1. Load Complete Production Engines
    retriever = LegalRetriever(rerank_model="ms-marco-TinyBERT-L-2-v2")
    detector = LegalHallucinationDetector()
    
    t1_start = time.perf_counter()
    mem_after_load = process.memory_info().rss / (1024 * 1024)
    startup_time_ms = (t1_start - t0_start) * 1000

    # 2. Index Disk Sizes
    chroma_disk_mb = get_dir_size_mb(CHROMA_PATH)
    bm25_disk_mb = os.path.getsize(BM25_PATH) / (1024 * 1024)
    corpus_disk_mb = os.path.getsize(CORPUS_PATH) / (1024 * 1024)
    kg_disk_mb = os.path.getsize(KG_PATH) / (1024 * 1024)
    total_disk_mb = chroma_disk_mb + bm25_disk_mb + corpus_disk_mb + kg_disk_mb

    # 3. Test Queries across categories
    benchmark_queries = [
        "Which landmark case established substantive due process and the golden triangle in India?",
        "What is the basic structure doctrine under Article 368?",
        "Is the right to privacy a fundamental right under Article 21?",
        "What are the mandatory guidelines for arrest under D.K. Basu?",
        "Can high courts issue writs for non-fundamental rights under Article 226?"
    ]

    query_latencies = []
    audit_latencies = []
    
    for q in benchmark_queries:
        t_q0 = time.perf_counter()
        res = retriever.retrieve(q, top_k=5)
        dt_q = (time.perf_counter() - t_q0) * 1000
        query_latencies.append(dt_q)

        # Audit a sample answer
        sample_answer = f"""
        ### Legal Principle
        The Supreme Court established key legal principles regarding {q}.
        ### Relevant Constitutional Provision
        Relevant constitutional provisions apply.
        ### Authorities Relied Upon
        1. {res['top_results'][0]['title']}, {res['top_results'][0]['citation']}
        """
        t_a0 = time.perf_counter()
        detector.audit(sample_answer, res["top_results"])
        dt_a = (time.perf_counter() - t_a0) * 1000
        audit_latencies.append(dt_a)

    mem_peak = process.memory_info().rss / (1024 * 1024)
    avg_retrieval_latency = sum(query_latencies) / len(query_latencies)
    avg_audit_latency = sum(audit_latencies) / len(audit_latencies)

    print("=" * 80)
    print("PHASES 15–20 PRODUCTION PERFORMANCE & MEMORY AUDIT REPORT")
    print("=" * 80)
    print("1. Knowledge Base On-Disk Footprint:")
    print(f"  • Legal Corpus CSV:          {corpus_disk_mb:.2f} MB")
    print(f"  • ChromaDB ONNX Store:       {chroma_disk_mb:.2f} MB")
    print(f"  • BM25 Okapi Index:          {bm25_disk_mb:.2f} MB")
    print(f"  • Constitutional KG:         {kg_disk_mb:.2f} MB")
    print(f"  • Total On-Disk Assets:      {total_disk_mb:.2f} MB")
    print()
    print("2. Memory Profile (Render Free Tier Limit: 512 MB, Target: < 300 MB):")
    print(f"  • Python Runtime Baseline:   {mem_baseline:.1f} MB")
    print(f"  • Steady-State Loaded RAM:   {mem_after_load:.1f} MB")
    print(f"  • Peak Inference RAM:        {mem_peak:.1f} MB")
    print(f"  • Headroom on 300MB Target:  {300 - mem_peak:.1f} MB ({'PASS - Compliant' if mem_peak < 300 else 'FAIL'})")
    print(f"  • Headroom on Render 512MB:  {512 - mem_peak:.1f} MB ({((512 - mem_peak)/512)*100:.1f}% free)")
    print()
    print("3. Latency Profile (Target: < 750 ms):")
    print(f"  • Engine Cold-Start Startup: {startup_time_ms:.1f} ms")
    print(f"  • Mean Retrieval Latency:    {avg_retrieval_latency:.2f} ms ({'PASS' if avg_retrieval_latency < 750 else 'FAIL'})")
    print(f"  • Mean Hallucination Audit:  {avg_audit_latency:.2f} ms")
    print(f"  • End-to-End Processing:     {avg_retrieval_latency + avg_audit_latency:.2f} ms")
    print("=" * 80)

if __name__ == "__main__":
    profile()
