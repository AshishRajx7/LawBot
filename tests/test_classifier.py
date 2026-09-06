# tests/test_classifier.py — Unit Tests & Accuracy Benchmark for LegalQueryClassifier
import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath("."))
from legal_query_classifier import LegalQueryClassifier

class TestLegalQueryClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifier = LegalQueryClassifier()

    def test_constitutional_provision_queries(self):
        queries = [
            "What is Article 21 of the Indian Constitution?",
            "Explain the equality code under Article 14",
            "What rights are conferred by Article 19?",
            "What is Article 25 of the Constitution?"
        ]
        for q in queries:
            res = self.classifier.classify(q)
            self.assertEqual(res["primary_class"], "Constitutional Provision Query", f"Failed on: {q}")
            self.assertTrue(len(res["detected_articles"]) > 0, f"No article detected for: {q}")
            self.assertGreaterEqual(res["confidence"], 0.90)

    def test_landmark_case_queries(self):
        queries = [
            "What did the Supreme Court hold in Maneka Gandhi v. Union of India?",
            "Explain Kesavananda Bharati v. State of Kerala",
            "What was the ruling in Puttaswamy?",
            "Explain Indra Sawhney v. Union of India"
        ]
        for q in queries:
            res = self.classifier.classify(q)
            self.assertEqual(res["primary_class"], "Landmark Case Query", f"Failed on: {q}")
            self.assertTrue(len(res["detected_cases"]) > 0, f"No case detected for: {q}")

    def test_doctrine_queries(self):
        queries = [
            "What is the basic structure doctrine?",
            "Which case established substantive due process?",
            "Explain the golden triangle doctrine",
            "What is the manifest arbitrariness doctrine?"
        ]
        for q in queries:
            res = self.classifier.classify(q)
            self.assertEqual(res["primary_class"], "Doctrine Query", f"Failed on: {q}")
            self.assertTrue(len(res["detected_doctrines"]) > 0, f"No doctrine detected for: {q}")

    def test_fundamental_rights_queries(self):
        queries = [
            "Is the right to privacy a fundamental right?",
            "Does personal liberty include the right to livelihood?",
            "What are the reasonable restrictions on freedom of speech?"
        ]
        for q in queries:
            res = self.classifier.classify(q)
            self.assertEqual(res["primary_class"], "Fundamental Rights Query", f"Failed on: {q}")

    def test_amendment_queries(self):
        queries = [
            "Can Parliament amend fundamental rights under Article 368?",
            "What changes were made by the 42nd Amendment?",
            "Was the 99th Constitutional Amendment unconstitutional?"
        ]
        for q in queries:
            res = self.classifier.classify(q)
            self.assertEqual(res["primary_class"], "Amendment Query", f"Failed on: {q}")

    def test_judicial_review_queries(self):
        queries = [
            "What is the scope of judicial review under Article 32 and Article 226?",
            "Can a High Court issue a writ of habeas corpus under judicial review?",
            "Is judicial review an inviolable part of the Constitution?"
        ]
        for q in queries:
            res = self.classifier.classify(q)
            self.assertEqual(res["primary_class"], "Judicial Review Query", f"Failed on: {q}")

    def test_comparative_case_queries(self):
        queries = [
            "Compare A.K. Gopalan and Maneka Gandhi on personal liberty",
            "How does Puttaswamy differ from Kharak Singh?",
            "Shankari Prasad versus Golaknath on amending powers"
        ]
        for q in queries:
            res = self.classifier.classify(q)
            self.assertEqual(res["primary_class"], "Comparative Case Query", f"Failed on: {q}")

    def test_procedural_law_queries(self):
        queries = [
            "What are the mandatory guidelines for arrest under D.K. Basu?",
            "What principles govern anticipatory bail under section 438 crpc?",
            "What is the procedure for detention under preventive detention laws?"
        ]
        for q in queries:
            res = self.classifier.classify(q)
            self.assertIn(res["primary_class"], ("Procedural Law Query", "Landmark Case Query"), f"Failed on: {q}")

    def test_general_legal_questions(self):
        queries = [
            "What is the difference between an ordinance and an act?",
            "How does stare decisis apply to High Courts?"
        ]
        for q in queries:
            res = self.classifier.classify(q)
            self.assertIn(res["primary_class"], ("General Legal Question", "Doctrine Query"), f"Failed on: {q}")

    def test_retrieval_biases_configured(self):
        res = self.classifier.classify("What is the basic structure doctrine?")
        self.assertIn("ratio_boost", res["retrieval_bias"])
        self.assertGreater(res["retrieval_bias"]["ratio_boost"], 1.0)

if __name__ == "__main__":
    unittest.main()
