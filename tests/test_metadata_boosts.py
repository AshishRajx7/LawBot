# tests/test_metadata_boosts.py — Validation Suite for Metadata-Aware Weighted Fusion
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath("."))
from legal_retriever import LegalRetriever

class TestMetadataBoosts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = LegalRetriever()

    def test_constitutional_provision_boost(self):
        query = "What is Article 21 of the Indian Constitution?"
        res = self.retriever.retrieve(query, top_k=5)
        self.assertEqual(res["classification"]["primary_class"], "Constitutional Provision Query")
        # Constitutional articles should receive both article match (+0.25) and provision boost (+0.25)
        top_doc = res["top_results"][0]
        self.assertGreaterEqual(top_doc["meta_boost"], 0.25)
        self.assertEqual(top_doc["primary_article"], "Article 21")

    def test_landmark_case_boost(self):
        query = "What did the Supreme Court hold in Kesavananda Bharati v. State of Kerala?"
        res = self.retriever.retrieve(query, top_k=5)
        self.assertEqual(res["classification"]["primary_class"], "Landmark Case Query")
        top_doc = res["top_results"][0]
        self.assertGreaterEqual(top_doc["meta_boost"], 0.15)
        self.assertIn("Kesavananda Bharati", top_doc["title"])

    def test_doctrine_boost(self):
        query = "What is the basic structure doctrine?"
        res = self.retriever.retrieve(query, top_k=5)
        self.assertEqual(res["classification"]["primary_class"], "Doctrine Query")
        # Ratio chunks and doctrine cases should receive ratio boost
        ratio_found = any(d["doc_type"] == "ratio_chunk" and d["meta_boost"] >= 0.20 for d in res["top_results"])
        self.assertTrue(ratio_found, "Doctrine query should apply ratio_chunk metadata boost >= 0.20")

    def test_amendment_query_boost(self):
        # Task 3: Amendment Query metadata boost
        query = "Can Parliament amend fundamental rights under Article 368?"
        res = self.retriever.retrieve(query, top_k=5)
        self.assertEqual(res["classification"]["primary_class"], "Amendment Query")
        # Check that amendment or Article 368 documents receive amendment boost (+0.15 / +0.10)
        boosted_found = any(d["meta_boost"] >= 0.10 and (d["primary_article"] == "Article 368" or "amend" in d["title"].lower()) for d in res["top_results"])
        self.assertTrue(boosted_found, "Amendment query must apply amendment metadata boost")

    def test_fundamental_rights_boost(self):
        # Task 3: Fundamental Rights Query metadata boost
        query = "Is the right to privacy a fundamental right?"
        res = self.retriever.retrieve(query, top_k=5)
        self.assertEqual(res["classification"]["primary_class"], "Fundamental Rights Query")
        # Check that Part III / Fundamental Rights documents receive +0.15 / +0.10 boost
        boosted_found = any(d["meta_boost"] >= 0.10 for d in res["top_results"])
        self.assertTrue(boosted_found, "Fundamental rights query must apply Part III metadata boost")

    def test_judicial_review_boost(self):
        # Task 3: Judicial Review Query metadata boost
        query = "What is the scope of judicial review under Article 32 and Article 226?"
        res = self.retriever.retrieve(query, top_k=5)
        self.assertEqual(res["classification"]["primary_class"], "Judicial Review Query")
        boosted_found = any(d["meta_boost"] >= 0.10 for d in res["top_results"])
        self.assertTrue(boosted_found, "Judicial review query must apply judicial review metadata boost")

    def test_comparative_case_boost(self):
        query = "Compare A.K. Gopalan and Maneka Gandhi on personal liberty"
        res = self.retriever.retrieve(query, top_k=5)
        self.assertEqual(res["classification"]["primary_class"], "Comparative Case Query")
        top_doc = res["top_results"][0]
        self.assertGreaterEqual(top_doc["meta_boost"], 0.15)

    def test_procedural_law_boost(self):
        # Task 3: Procedural Law Query metadata boost
        query = "What are the procedural guidelines for arrest under D.K. Basu?"
        res = self.retriever.retrieve(query, top_k=5)
        self.assertIn(res["classification"]["primary_class"], ("Procedural Law Query", "Landmark Case Query"))
        boosted_found = any(d["meta_boost"] >= 0.05 for d in res["top_results"])
        self.assertTrue(boosted_found, "Procedural law query must apply procedural metadata boost")

    def test_general_legal_question_boost(self):
        # Task 3: General Legal Question metadata boost
        query = "What is the difference between an ordinance and an act?"
        res = self.retriever.retrieve(query, top_k=5)
        self.assertIn(res["classification"]["primary_class"], ("General Legal Question", "Doctrine Query"))
        boosted_found = any(d["meta_boost"] >= 0.05 for d in res["top_results"])
        self.assertTrue(boosted_found, "General legal question must apply broad constitutional relevance boost")

    def test_class_bonus_additive(self):
        # Task 4: Additive Class-Specific Bonuses
        # 1. Constitutional Provision Query: class_bonus = 0.15 * bm25_sim
        prov_res = self.retriever.retrieve("What is Article 21 of the Indian Constitution?", top_k=5)
        top_prov = prov_res["top_results"][0]
        self.assertAlmostEqual(top_prov["class_bonus"], 0.15 * top_prov["bm25_sim"], places=4)

        # 2. Landmark Case Query: class_bonus = 0.15 * dense_sim
        case_res = self.retriever.retrieve("What did the Supreme Court hold in Kesavananda Bharati?", top_k=5)
        top_case = case_res["top_results"][0]
        self.assertAlmostEqual(top_case["class_bonus"], 0.15 * top_case["dense_sim"], places=4)

    def test_doctrine_establishes_boost_and_distractor_suppression(self):
        # Tasks 1, 2, 4 Verification Query:
        # "Did Maneka Gandhi establish the Basic Structure Doctrine?"
        query = "Did Maneka Gandhi establish the Basic Structure Doctrine?"
        res = self.retriever.retrieve(query, top_k=5)
        
        self.assertEqual(res["classification"]["primary_class"], "Doctrine Query")
        self.assertIn("Basic Structure Doctrine", res["classification"]["detected_doctrines"])
        self.assertIn("Maneka Gandhi v. Union of India", res["classification"]["detected_cases"])

        # Check that Kesavananda Bharati is in the top results (specifically Top-1 or Top-2)
        top_titles = [d["title"] for d in res["top_results"][:2]]
        kesavananda_in_top = any("Kesavananda Bharati" in t for t in top_titles)
        self.assertTrue(kesavananda_in_top, f"Kesavananda Bharati must be in Top 2 results. Got: {top_titles}")

        # Check that establishing case received explicit ESTABLISHES boost (>= 0.35)
        kb_docs = [d for d in res["top_results"] if "Kesavananda Bharati" in d["title"]]
        self.assertTrue(len(kb_docs) > 0)
        self.assertGreaterEqual(kb_docs[0]["meta_boost"], 0.35, "Establishing case must receive >= 0.35 meta boost")

        # Check that distractor case (Maneka Gandhi) graph boost was suppressed to 0.0
        mg_docs = [d for d in res["top_results"] if "Maneka Gandhi" in d["title"]]
        if mg_docs:
            self.assertLessEqual(mg_docs[0]["graph_boost"], 0.0, "Unrelated case graph boost must be suppressed in doctrine query")

if __name__ == "__main__":
    unittest.main()
