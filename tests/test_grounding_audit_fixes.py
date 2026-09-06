# tests/test_grounding_audit_fixes.py — Tests for FM-1, AF-1, FM-2, FM-3, FM-5 Grounding Fixes
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath("."))

from legal_retriever import LegalRetriever
from legal_grounding_engine import (
    detect_query_premise_contradiction,
    post_process_grounded_answer,
    validate_article_domains,
    is_comparative_query,
    has_assumption_trigger,
    get_kg
)

class TestGroundingAuditFixes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = LegalRetriever()
        cls.kg = get_kg()

    # ==========================================================================
    # 1. FALSE ATTRIBUTION TESTS (Fix 1: FM-1)
    # ==========================================================================

    def test_false_attribution_golaknath_judicial_review(self):
        """Query: How did Golaknath establish Judicial Review? -> Must generate warning."""
        query = "How did Golaknath establish Judicial Review?"
        res = self.retriever.retrieve(query, top_k=5)
        warning = detect_query_premise_contradiction(
            query=query,
            detected_cases=res["classification"]["detected_cases"],
            detected_doctrines=res["classification"]["detected_doctrines"],
            top_candidates=res["top_results"],
            kg=self.kg
        )
        self.assertTrue(warning.startswith("WARNING:"), f"Warning should start with WARNING, got: {warning}")
        self.assertIn("User premise conflicts with retrieved authorities.", warning)
        self.assertIn("Golaknath", warning)
        self.assertTrue("L. Chandra Kumar" in warning or "Kesavananda Bharati" in warning)

    def test_false_attribution_shayara_bano_creamy_layer(self):
        """Query: Did Shayara Bano establish the Creamy Layer Principle? -> Must generate warning."""
        query = "Did Shayara Bano establish the Creamy Layer Principle?"
        res = self.retriever.retrieve(query, top_k=5)
        warning = detect_query_premise_contradiction(
            query=query,
            detected_cases=res["classification"]["detected_cases"],
            detected_doctrines=res["classification"]["detected_doctrines"],
            top_candidates=res["top_results"],
            kg=self.kg
        )
        self.assertTrue(warning.startswith("WARNING:"), f"Warning should start with WARNING, got: {warning}")
        self.assertIn("User premise conflicts with retrieved authorities.", warning)
        self.assertIn("Shayara Bano", warning)
        self.assertIn("Indra Sawhney", warning)

    def test_false_attribution_navtej_johar_colorable_legislation(self):
        """Query: Did Navtej Johar establish Colorable Legislation? -> Must generate warning."""
        query = "Did Navtej Johar establish Colorable Legislation?"
        res = self.retriever.retrieve(query, top_k=5)
        warning = detect_query_premise_contradiction(
            query=query,
            detected_cases=res["classification"]["detected_cases"],
            detected_doctrines=res["classification"]["detected_doctrines"],
            top_candidates=res["top_results"],
            kg=self.kg
        )
        self.assertTrue(warning.startswith("WARNING:"), f"Warning should start with WARNING, got: {warning}")
        self.assertIn("User premise conflicts with retrieved authorities.", warning)
        self.assertIn("Navtej", warning)
        self.assertIn("Kameshwar Singh", warning)

    def test_false_attribution_maneka_gandhi_basic_structure(self):
        """Query: Did Maneka Gandhi establish the Basic Structure Doctrine? -> Must generate warning."""
        query = "Did Maneka Gandhi establish the Basic Structure Doctrine?"
        res = self.retriever.retrieve(query, top_k=5)
        warning = detect_query_premise_contradiction(
            query=query,
            detected_cases=res["classification"]["detected_cases"],
            detected_doctrines=res["classification"]["detected_doctrines"],
            top_candidates=res["top_results"],
            kg=self.kg
        )
        self.assertTrue(warning.startswith("WARNING:"), f"Warning should start with WARNING, got: {warning}")
        self.assertIn("User premise conflicts with retrieved authorities.", warning)
        self.assertIn("Maneka Gandhi", warning)
        self.assertIn("Kesavananda Bharati", warning)

    # ==========================================================================
    # 2. CONSTITUTIONAL ARTICLE TESTS (Fix 3: FM-2)
    # ==========================================================================

    def test_article_misattribution_368_free_speech(self):
        """Query: How does Article 368 guarantee free speech? -> Must generate warning."""
        query = "How does Article 368 guarantee free speech?"
        warning = detect_query_premise_contradiction(
            query=query,
            detected_cases=[],
            detected_doctrines=[],
            kg=self.kg
        )
        self.assertTrue(warning.startswith("WARNING:"), f"Warning should start with WARNING, got: {warning}")
        self.assertIn("Article 19", warning)
        self.assertIn("Article 368", warning)

    def test_article_misattribution_32_preventive_detention(self):
        """Query: How does Article 32 create preventive detention? -> Must generate warning."""
        query = "How does Article 32 create preventive detention?"
        warning = detect_query_premise_contradiction(
            query=query,
            detected_cases=[],
            detected_doctrines=[],
            kg=self.kg
        )
        self.assertTrue(warning.startswith("WARNING:"), f"Warning should start with WARNING, got: {warning}")
        self.assertIn("Article 22", warning)
        self.assertIn("Article 32", warning)

    # ==========================================================================
    # 3. COMPARATIVE QUERY TESTS (Fix 2: AF-1)
    # ==========================================================================

    def test_comparative_query_gopalan_maneka(self):
        """Query: Compare Gopalan and Maneka Gandhi -> Must NOT generate warning."""
        query = "Compare Gopalan and Maneka Gandhi"
        res = self.retriever.retrieve(query, top_k=5)
        warning = detect_query_premise_contradiction(
            query=query,
            detected_cases=res["classification"]["detected_cases"],
            detected_doctrines=res["classification"]["detected_doctrines"],
            top_candidates=res["top_results"],
            kg=self.kg
        )
        self.assertEqual(warning, "", f"Comparative query should not trigger contradiction warning, got: {warning}")

    def test_comparative_query_trace_privacy(self):
        """Query: Trace privacy from Sharma to Puttaswamy -> Must NOT generate warning."""
        query = "Trace privacy from Sharma to Puttaswamy"
        res = self.retriever.retrieve(query, top_k=5)
        warning = detect_query_premise_contradiction(
            query=query,
            detected_cases=res["classification"]["detected_cases"],
            detected_doctrines=res["classification"]["detected_doctrines"],
            top_candidates=res["top_results"],
            kg=self.kg
        )
        self.assertEqual(warning, "", f"Evolutionary/trace query should not trigger contradiction warning, got: {warning}")

    def test_comparative_query_different_views(self):
        """Query: Did A.K. Gopalan and Maneka Gandhi hold different views on Article 21? -> Must NOT generate warning."""
        query = "Did A.K. Gopalan and Maneka Gandhi hold different views on Article 21?"
        res = self.retriever.retrieve(query, top_k=5)
        warning = detect_query_premise_contradiction(
            query=query,
            detected_cases=res["classification"]["detected_cases"],
            detected_doctrines=res["classification"]["detected_doctrines"],
            top_candidates=res["top_results"],
            kg=self.kg
        )
        self.assertEqual(warning, "", f"Contrast query should not trigger contradiction warning, got: {warning}")

    # ==========================================================================
    # 4. CONTROL TESTS (Aligned premises should NOT warn)
    # ==========================================================================

    def test_control_correct_doctrine_attribution(self):
        """Query: Did Kesavananda Bharati establish the Basic Structure Doctrine? -> NO warning."""
        query = "Did Kesavananda Bharati establish the Basic Structure Doctrine?"
        res = self.retriever.retrieve(query, top_k=5)
        warning = detect_query_premise_contradiction(
            query=query,
            detected_cases=res["classification"]["detected_cases"],
            detected_doctrines=res["classification"]["detected_doctrines"],
            top_candidates=res["top_results"],
            kg=self.kg
        )
        self.assertEqual(warning, "", f"Correct attribution should not warn, got: {warning}")

    def test_control_article_32_remedies(self):
        """Query: Explain Article 32 remedies. -> NO warning."""
        query = "Explain Article 32 remedies."
        warning = detect_query_premise_contradiction(
            query=query,
            detected_cases=[],
            detected_doctrines=[],
            kg=self.kg
        )
        self.assertEqual(warning, "", f"Valid article query should not warn, got: {warning}")

    def test_control_article_19_1_a(self):
        """Query: Explain Article 19(1)(a). -> NO warning."""
        query = "Explain Article 19(1)(a)."
        warning = detect_query_premise_contradiction(
            query=query,
            detected_cases=[],
            detected_doctrines=[],
            kg=self.kg
        )
        self.assertEqual(warning, "", f"Valid article query should not warn, got: {warning}")

    # ==========================================================================
    # 5. LEADING ASSUMPTION TESTS (Fix 4: FM-3)
    # ==========================================================================

    def test_leading_assumption_since_clause(self):
        """Query: Since Article 368 grants unlimited amending power... -> Warning generated."""
        query = "Since Article 368 grants unlimited amending power..."
        warning = detect_query_premise_contradiction(
            query=query,
            detected_cases=[],
            detected_doctrines=[],
            kg=self.kg
        )
        self.assertTrue(warning.startswith("WARNING:"), f"Warning should fire for unlimited amending power, got: {warning}")

    def test_leading_assumption_after_adm_jabalpur(self):
        """Query: After ADM Jabalpur recognized privacy... -> Warning generated."""
        query = "After ADM Jabalpur recognized privacy..."
        warning = detect_query_premise_contradiction(
            query=query,
            detected_cases=["ADM Jabalpur v. Shivkant Shukla"],
            detected_doctrines=["Right to Privacy"],
            kg=self.kg
        )
        self.assertTrue(warning.startswith("WARNING:"), f"Warning should fire for ADM Jabalpur + privacy, got: {warning}")
        self.assertIn("Puttaswamy", warning)

    # ==========================================================================
    # 6. GENERIC POST-GENERATION VALIDATOR TESTS (Fix 5: FM-5)
    # ==========================================================================

    def test_generic_post_processing_shayara_bano_creamy_layer(self):
        """Tests that post-processor dynamically rewrites Shayara Bano claiming Creamy Layer to Indra Sawhney."""
        query = "Did Shayara Bano establish the Creamy Layer Principle?"
        raw_answer = "The Creamy Layer Principle was established by Shayara Bano v. Union of India in 2017."
        processed = post_process_grounded_answer(
            raw_answer=raw_answer,
            query=query,
            contradiction_warning="WARNING: mismatch",
            kg=self.kg
        )
        self.assertIn("Indra Sawhney", processed)
        self.assertNotIn("established by Shayara Bano", processed)
        self.assertIn("### Retrieval Consistency Check", processed)
        self.assertIn("User premise supported: No", processed)

    def test_generic_post_processing_navtej_colorable_legislation(self):
        """Tests that post-processor dynamically rewrites Navtej Johar claiming Colorable Legislation to Kameshwar Singh."""
        query = "Did Navtej Johar establish Colorable Legislation?"
        raw_answer = "Navtej Singh Johar v. Union of India established the Colorable Legislation Doctrine."
        processed = post_process_grounded_answer(
            raw_answer=raw_answer,
            query=query,
            contradiction_warning="WARNING: mismatch",
            kg=self.kg
        )
        self.assertIn("Kameshwar Singh", processed)
        self.assertNotIn("Navtej Singh Johar v. Union of India established the Colorable Legislation", processed)
        self.assertIn("### Retrieval Consistency Check", processed)
        self.assertIn("User premise supported: No", processed)

if __name__ == "__main__":
    unittest.main()
