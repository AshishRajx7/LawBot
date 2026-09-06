# tests/test_generation_grounding.py — Unit Tests for Pre-Generation Grounding & Contradiction Control
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath("."))

from legal_retriever import LegalRetriever
from legal_grounding_engine import (
    clean_case_name,
    get_short_case_name,
    extract_grounding_facts,
    detect_query_premise_contradiction,
    format_grounding_block,
    build_grounded_generation_prompt,
    post_process_grounded_answer
)

class TestGenerationGrounding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = LegalRetriever()

    def test_grounding_facts_extraction(self):
        query = "Did Maneka Gandhi establish the Basic Structure Doctrine?"
        res = self.retriever.retrieve(query, top_k=5)
        top_candidates = res["top_results"]
        cls_res = res["classification"]

        facts = extract_grounding_facts(top_candidates, cls_res.get("detected_doctrines"))
        self.assertIn("Kesavananda Bharati", facts["highest_ranked_authority"])
        self.assertIn("Basic Structure Doctrine", facts["retrieved_doctrines"])
        self.assertTrue(any("Kesavananda Bharati" in a for a in facts["retrieved_supporting_authorities"]))

        block = format_grounding_block(facts)
        self.assertIn("Grounding Facts:", block)
        self.assertIn("Highest Ranked Authority: Kesavananda Bharati", block)
        self.assertIn("These facts override user assumptions.", block)

    def test_contradiction_detection_warning(self):
        query = "Did Maneka Gandhi establish the Basic Structure Doctrine?"
        res = self.retriever.retrieve(query, top_k=5)
        top_candidates = res["top_results"]
        cls_res = res["classification"]

        warning = detect_query_premise_contradiction(
            query,
            cls_res.get("detected_cases", []),
            cls_res.get("detected_doctrines", []),
            top_candidates
        )
        self.assertTrue(warning.startswith("WARNING:"))
        self.assertIn("User premise conflicts with retrieved authorities.", warning)
        self.assertIn("Kesavananda Bharati", warning)
        self.assertIn("Maneka Gandhi", warning)

    def test_prompt_construction_contains_mandatory_rules_and_consistency_check(self):
        query = "Did Maneka Gandhi establish the Basic Structure Doctrine?"
        res = self.retriever.retrieve(query, top_k=5)
        top_candidates = res["top_results"]
        cls_res = res["classification"]

        context = "SAMPLE RETRIEVED CONTEXT"
        full_prompt, warning = build_grounded_generation_prompt(query, context, top_candidates, cls_res)

        # Requirement 1: Mandatory contradiction check & source of truth rules
        self.assertIn("RETRIEVAL IS THE SOURCE OF TRUTH", full_prompt)
        self.assertIn("MANDATORY CONTRADICTION CHECK", full_prompt)
        self.assertIn("Never attribute a doctrine, constitutional principle, or precedent to a case unless the retrieved authorities support that attribution.", full_prompt)
        self.assertIn("Retrieved authorities are the only admissible legal evidence.", full_prompt)

        # Requirement 2: Retrieval Consistency Check section in format
        self.assertIn("### Retrieval Consistency Check", full_prompt)
        self.assertIn("* User premise supported: [Yes / No]", full_prompt)

        # Requirement 3 & 4: Grounding block and warning injected
        self.assertIn("Grounding Facts:", full_prompt)
        self.assertIn("Highest Ranked Authority: Kesavananda Bharati", full_prompt)
        self.assertIn("WARNING:\nUser premise conflicts with retrieved authorities.", full_prompt)

    def test_failure_query_generated_answer_compliance(self):
        """
        User Requirement 5 Test:
        Query: "Did Maneka Gandhi establish the Basic Structure Doctrine?"
        Expected generated answer MUST contain: "Kesavananda Bharati"
        Expected generated answer must NOT contain: "The Basic Structure Doctrine was established by Maneka Gandhi"
        """
        query = "Did Maneka Gandhi establish the Basic Structure Doctrine?"
        res = self.retriever.retrieve(query, top_k=5)
        top_candidates = res["top_results"]
        cls_res = res["classification"]

        _, warning = build_grounded_generation_prompt(query, "context", top_candidates, cls_res)

        # Simulate raw answer that attempts to echo the false user wording
        raw_hallucinated_answer = (
            "The Basic Structure Doctrine was established by Maneka Gandhi v. Union of India in 1978. "
            "However, judicial review is also important."
        )

        final_answer = post_process_grounded_answer(
            raw_answer=raw_hallucinated_answer,
            query=query,
            top_candidates=top_candidates,
            contradiction_warning=warning
        )

        # Expected generated answer must contain: "Kesavananda Bharati"
        self.assertIn("Kesavananda Bharati", final_answer)

        # Expected generated answer must NOT contain: "The Basic Structure Doctrine was established by Maneka Gandhi"
        self.assertNotIn("The Basic Structure Doctrine was established by Maneka Gandhi", final_answer)
        self.assertNotIn("established by Maneka Gandhi", final_answer)

        # Must contain Retrieval Consistency Check with No
        self.assertIn("### Retrieval Consistency Check", final_answer)
        self.assertIn("User premise supported: No", final_answer)

    def test_supported_premise_no_contradiction_warning(self):
        query = "Did Kesavananda Bharati establish the Basic Structure Doctrine?"
        res = self.retriever.retrieve(query, top_k=5)
        top_candidates = res["top_results"]
        cls_res = res["classification"]

        warning = detect_query_premise_contradiction(
            query,
            cls_res.get("detected_cases", []),
            cls_res.get("detected_doctrines", []),
            top_candidates
        )
        self.assertEqual(warning, "", "No contradiction warning should fire when premise is aligned with evidence")

if __name__ == "__main__":
    unittest.main()
