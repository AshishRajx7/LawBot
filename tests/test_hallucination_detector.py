# tests/test_hallucination_detector.py — Automated Validation Suite for Hallucination Detection
import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath("."))
from legal_hallucination_detector import LegalHallucinationDetector

class TestLegalHallucinationDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = LegalHallucinationDetector()
        cls.mock_sources = [
            {
                "title": "Maneka Gandhi v. Union of India",
                "citation": "AIR 1978 SC 597",
                "primary_article": "Article 21",
                "content": (
                    "The Supreme Court established substantive due process in India, holding that "
                    "any procedure established by law depriving a person of life or personal liberty "
                    "under Article 21 must be just, fair, and reasonable. The Court also articulated "
                    "the Golden Triangle of Articles 14, 19, and 21, establishing that fundamental rights "
                    "are not isolated islands but an intertwined protective code."
                )
            },
            {
                "title": "Article 21: Protection of Life and Personal Liberty",
                "citation": "Constitution of India, Part III",
                "primary_article": "Article 21",
                "content": "No person shall be deprived of his life or personal liberty except according to procedure established by law."
            }
        ]

    def test_grounded_answer(self):
        answer = """
        ### Legal Principle
        The Supreme Court established substantive due process under Article 21, requiring procedure established by law to be just, fair, and reasonable.
        ### Relevant Constitutional Provision
        Article 21 guarantees that no person shall be deprived of life or personal liberty except according to procedure established by law.
        ### Authorities Relied Upon
        1. Maneka Gandhi v. Union of India, AIR 1978 SC 597
        """
        res = self.detector.audit(answer, self.mock_sources)
        self.assertGreaterEqual(res["confidence_score"], 85.0)
        self.assertIn("SOURCE 1", res["supported_sources"])
        self.assertEqual(len(res["issues_detected"]), 0)

    def test_fabricated_article(self):
        answer = """
        Under Article 999 of the Constitution of India, the court held that personal liberty is absolute.
        """
        res = self.detector.audit(answer, self.mock_sources)
        self.assertLess(res["confidence_score"], 70.0)
        self.assertTrue(any("Article 999" in issue for issue in res["issues_detected"]))

    def test_unretrieved_case(self):
        answer = """
        In fictitious Sharma v. State of Narnia, the Supreme Court held that bail cannot be denied.
        """
        res = self.detector.audit(answer, self.mock_sources)
        self.assertLess(res["confidence_score"], 90.0)
        self.assertTrue(any("Sharma v. State of Narnia" in issue or "External/Unretrieved" in issue for issue in res["issues_detected"]))

    def test_unsupported_substantive_claim(self):
        answer = """
        The Supreme Court held in 2025 that private companies are completely exempt from fundamental rights under Article 21.
        """
        res = self.detector.audit(answer, self.mock_sources)
        # Should flag ungrounded statement or reduce confidence
        self.assertLess(res["confidence_score"], 100.0)

    def test_valid_amendment(self):
        # Task 4: Valid Constitutional Amendment
        sources_with_amendment = self.mock_sources + [
            {
                "title": "44th Constitutional Amendment Act, 1978",
                "citation": "Constitution (Forty-fourth Amendment) Act, 1978",
                "primary_article": "Article 368",
                "content": "The 44th Constitutional Amendment provided that Articles 20 and 21 cannot be suspended during Emergency."
            }
        ]
        answer = "The 44th Constitutional Amendment ensured that Article 21 cannot be suspended even during a proclamation of Emergency."
        res = self.detector.audit(answer, sources_with_amendment)
        self.assertGreaterEqual(res["confidence_score"], 80.0)
        self.assertFalse(any("Fabricated" in issue for issue in res["issues_detected"]))

    def test_fabricated_amendment(self):
        # Task 4: Fabricated Constitutional Amendment (e.g. 189th Amendment)
        answer = "Under the 189th Constitutional Amendment, the procedure established by law was permanently deleted."
        res = self.detector.audit(answer, self.mock_sources)
        self.assertLessEqual(res["confidence_score"], 80.0)
        self.assertTrue(
            any("189th Constitutional Amendment" in issue and "Fabricated" in issue for issue in res["issues_detected"]),
            f"Expected fabricated amendment issue, got: {res['issues_detected']}"
        )

    def test_false_attribution(self):
        # Task 5: NLI False Attribution Detection
        sources_doctrine = [
            {
                "title": "Kesavananda Bharati v. State of Kerala",
                "citation": "AIR 1973 SC 1461",
                "primary_article": "Article 368",
                "content": "The Supreme Court established the basic structure doctrine in Kesavananda Bharati v. State of Kerala, holding that Parliament cannot amend the basic structure of the Constitution."
            },
            {
                "title": "Maneka Gandhi v. Union of India",
                "citation": "AIR 1978 SC 597",
                "primary_article": "Article 21",
                "content": "The Supreme Court established substantive due process under Article 21, holding that procedure established by law must be just, fair, and reasonable."
            }
        ]
        answer = "The basic structure doctrine was established in Maneka Gandhi."
        res = self.detector.audit(answer, sources_doctrine)
        self.assertLessEqual(res["confidence_score"], 80.0)
        self.assertTrue(
            any("False Legal Attribution" in issue or "Contradiction" in issue for issue in res["issues_detected"]),
            f"Expected false attribution or contradiction, got: {res['issues_detected']}"
        )

    def test_semantic_inversion_and_contradiction(self):
        # Task 5: NLI Semantic Inversion / Contradiction Detection
        sources_doctrine = [
            {
                "title": "Kesavananda Bharati v. State of Kerala",
                "citation": "AIR 1973 SC 1461",
                "primary_article": "Article 368",
                "content": "The Supreme Court established the basic structure doctrine, ruling that Parliament cannot alter or destroy the basic structure of the Constitution under Article 368."
            }
        ]
        answer = "Parliament can amend the basic structure without restriction under Article 368."
        res = self.detector.audit(answer, sources_doctrine)
        self.assertLessEqual(res["confidence_score"], 75.0)
        self.assertTrue(
            any("Contradiction detected" in issue for issue in res["issues_detected"]),
            f"Expected contradiction issue, got: {res['issues_detected']}"
        )

if __name__ == "__main__":
    unittest.main()
