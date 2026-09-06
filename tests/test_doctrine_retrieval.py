# tests/test_doctrine_retrieval.py — Verification Suite for KG Candidate Injection, Query Expansion & Doctrine-First Retrieval
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath("."))

from legal_retriever import LegalRetriever
from legal_knowledge_graph import LegalKnowledgeGraph

class TestDoctrineRetrievalAndKGInjection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = LegalRetriever()
        cls.kg = LegalKnowledgeGraph()

    def test_failure_case_maneka_gandhi_basic_structure(self):
        """
        Critical Verification Test:
        Query: "Did Maneka Gandhi establish the Basic Structure Doctrine?"
        Failure Mode in Prior Pipeline:
          - Distractor token matching ranked Maneka Gandhi 1-4, missing Kesavananda Bharati.
        Target Behavior:
          - Kesavananda Bharati MUST return in the top results (Rank 1 / Top-3).
          - Unrelated detected case (Maneka Gandhi) graph boost MUST be suppressed.
          - Explicit ESTABLISHES relation boost MUST activate for Kesavananda Bharati.
        """
        query = "Did Maneka Gandhi establish the Basic Structure Doctrine?"
        res = self.retriever.retrieve(query, top_k=5)

        # Classification check
        self.assertEqual(res["classification"]["primary_class"], "Doctrine Query")
        self.assertIn("Basic Structure Doctrine", res["classification"]["detected_doctrines"])
        self.assertIn("Maneka Gandhi v. Union of India", res["classification"]["detected_cases"])

        # Top results check
        top_results = res["top_results"]
        self.assertGreaterEqual(len(top_results), 3)
        top1_title = top_results[0]["title"]
        self.assertIn("Kesavananda Bharati", top1_title, f"Top-1 result must be Kesavananda Bharati, got: {top1_title}")

        top3_titles = [d["title"] for d in top_results[:3]]
        kesavananda_in_top3 = any("Kesavananda Bharati" in t for t in top3_titles)
        self.assertTrue(kesavananda_in_top3, f"Kesavananda Bharati must be in Top 3, got: {top3_titles}")

        # Establishing relation boost verification
        kb_docs = [d for d in top_results if "Kesavananda Bharati" in d["title"]]
        self.assertTrue(len(kb_docs) > 0)
        # ESTABLISHES gives +0.35, and ratio chunk gives +0.15 => +0.50 (plus any topic match)
        self.assertGreaterEqual(kb_docs[0]["meta_boost"], 0.45)

        # Distractor case graph boost suppression verification
        mg_docs = [d for d in top_results if "Maneka Gandhi" in d["title"]]
        for mg in mg_docs:
            self.assertLessEqual(mg["graph_boost"], 0.0, "Distractor case graph boost must be suppressed in doctrine query")

    def test_kg_candidate_injection_into_candidate_pool(self):
        """
        Task 1 Verification:
        Verify that get_candidate_expansions injects actual corpus documents (including ratio chunks)
        rather than raw graph nodes into the candidate pool.
        """
        query = "Did Maneka Gandhi establish the Basic Structure Doctrine?"
        res = self.retriever.retrieve(query, top_k=5)
        
        injected = res["kg_candidates_injected"]
        self.assertIsInstance(injected, list)
        self.assertGreater(len(injected), 0, "Candidate pool must contain KG-injected candidates")
        
        # Verify injected items are valid corpus doc_ids and include ratio chunks
        for doc_id in injected:
            self.assertIn(doc_id, self.retriever.corpus_by_id, f"Injected candidate '{doc_id}' must exist in corpus")
        
        # Verify that ratio chunks are injected alongside parent cases
        has_ratio_chunk = any("_ratio_" in did for did in injected)
        self.assertTrue(has_ratio_chunk, "KG candidate injection must include ratio chunks for corpus cases")

    def test_bounded_query_expansion(self):
        """
        Task 3 Verification:
        Verify that query expansion terms are bounded (<= 4 terms) and do not mutate the user query.
        """
        query = "What is the basic structure doctrine?"
        res = self.retriever.retrieve(query, top_k=5)
        
        self.assertEqual(res["query"], query, "User-facing query must remain unchanged")
        exp_terms = res["query_expansion_terms"]
        self.assertIsInstance(exp_terms, list)
        self.assertLessEqual(len(exp_terms), 4, "Expansion terms must be bounded to <= 4")
        self.assertIn("Kesavananda Bharati", exp_terms, "Expansion terms must include establishing authority")

    def test_additive_class_bonus_preserves_base_fusion(self):
        """
        Task 4 Verification:
        Verify that base fusion formula weights (0.40 cross, 0.20 dense, 0.15 bm25) are preserved,
        and class bonuses are purely additive.
        """
        query = "What is Article 14 of the Indian Constitution?"
        res = self.retriever.retrieve(query, top_k=5)
        
        for doc in res["top_results"]:
            expected_base = (
                0.40 * doc["cross_score"] +
                0.20 * doc["dense_sim"] +
                0.15 * doc["bm25_sim"] +
                doc["meta_boost"] +
                doc["graph_boost"]
            )
            # final_score should equal expected_base + class_bonus
            self.assertAlmostEqual(doc["final_score"], expected_base + doc["class_bonus"], places=4)

    def test_retrieval_observability_logging(self):
        """
        Task 5 Verification:
        Verify that retrieval telemetry logs kg_candidates_injected, query_expansion_terms,
        graph_hops_used, and graph_candidates_added.
        """
        query = "What is the scope of substantive due process under Article 21?"
        res = self.retriever.retrieve(query, top_k=5)
        
        self.assertIn("kg_candidates_injected", res)
        self.assertIn("query_expansion_terms", res)
        self.assertIn("graph_hops_used", res)
        self.assertIn("graph_candidates_added", res)
        self.assertEqual(res["graph_hops_used"], 2)
        self.assertEqual(res["graph_candidates_added"], len(res["kg_candidates_injected"]))

if __name__ == "__main__":
    unittest.main()
