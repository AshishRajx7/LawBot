# tests/test_knowledge_graph.py — Unit Tests for Constitutional Knowledge Graph Engine
import unittest
import sys
import os
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath("."))
from legal_knowledge_graph import LegalKnowledgeGraph, KG_PATH, normalize_article_to_graph_id

class TestLegalKnowledgeGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kg = LegalKnowledgeGraph()
        with open(KG_PATH, "r", encoding="utf-8") as f:
            cls.raw_data = json.load(f)

    def test_normalize_article_to_graph_id(self):
        # Task 1: Article Zero-Padding Verification
        self.assertEqual(normalize_article_to_graph_id("Article 14"), "art_014")
        self.assertEqual(normalize_article_to_graph_id("Article 19"), "art_019")
        self.assertEqual(normalize_article_to_graph_id("Article 21"), "art_021")
        self.assertEqual(normalize_article_to_graph_id("Article 32"), "art_032")
        self.assertEqual(normalize_article_to_graph_id("Article 300A"), "art_300a")
        # Additional edge cases
        self.assertEqual(normalize_article_to_graph_id("Article 21A"), "art_021a")
        self.assertEqual(normalize_article_to_graph_id("art_14"), "art_014")
        self.assertEqual(normalize_article_to_graph_id("14"), "art_014")

    def test_graph_structure(self):
        nodes = self.raw_data.get("nodes", [])
        edges = self.raw_data.get("edges", [])
        self.assertGreaterEqual(len(nodes), 50, "Graph must contain at least 50 nodes")
        self.assertGreaterEqual(len(edges), 40, "Graph must contain at least 40 edges")

    def test_relation_types(self):
        valid_relations = {
            "ESTABLISHES", "OVERRULES", "RELIES_ON", "EXPANDS",
            "LIMITS", "INTERPRETS", "CONNECTED_TO"
        }
        for edge in self.raw_data.get("edges", []):
            rel = edge["relation"]
            self.assertIn(rel, valid_relations, f"Invalid relation '{rel}' found in edge: {edge}")

    def test_traversal_substantive_due_process(self):
        rel = self.kg.get_related_entities(["Substantive Due Process"])
        connected = rel["connected_nodes"]
        # Must connect to Maneka Gandhi (case_002)
        self.assertIn("case_002", connected)
        self.assertIn("REV_ESTABLISHES", connected["case_002"]["relation"])

    def test_traversal_basic_structure(self):
        rel = self.kg.get_related_entities(["Basic Structure Doctrine"])
        connected = rel["connected_nodes"]
        # Must connect to Kesavananda Bharati (case_004)
        self.assertIn("case_004", connected)

    def test_overruled_detection(self):
        # Maneka Gandhi overrules A.K. Gopalan (case_001)
        rel = self.kg.get_related_entities(["Maneka Gandhi"])
        self.assertIn("case_001", rel["overruled_nodes"])

    def test_overruled_case_penalty(self):
        # Task 2: Overruled Case Penalty Verification
        # Non-comparative query: overruled case (case_001, A.K. Gopalan) must receive -0.15 penalty
        boosts_standard = self.kg.compute_graph_boosts(
            ["Maneka Gandhi"],
            query="What is the ruling in Maneka Gandhi on personal liberty?",
            primary_class="Landmark Case Query"
        )
        self.assertIn("case_001", boosts_standard, "Overruled node case_001 must be in boosts")
        self.assertEqual(boosts_standard["case_001"], -0.15, "Overruled precedent must receive -0.15 penalty")

        # Comparative query class: penalty must be disabled
        boosts_comparative_class = self.kg.compute_graph_boosts(
            ["Maneka Gandhi"],
            query="Compare Gopalan and Maneka Gandhi",
            primary_class="Comparative Case Query"
        )
        self.assertNotIn("case_001", boosts_comparative_class, "Comparative Case Query class must disable penalty")

        # Comparative keywords in query: penalty must be disabled
        for kw in ["overruled", "reversed", "distinguish", "compare", "versus", " vs "]:
            boosts_kw = self.kg.compute_graph_boosts(
                ["Maneka Gandhi"],
                query=f"Was A.K. Gopalan {kw} by Maneka Gandhi?",
                primary_class="Landmark Case Query"
            )
            self.assertNotIn("case_001", boosts_kw, f"Comparative keyword '{kw}' must disable penalty")

    def test_graph_boosts(self):
        boosts = self.kg.compute_graph_boosts(["Substantive Due Process"])
        self.assertIn("case_002", boosts)
        self.assertGreater(boosts["case_002"], 0.0)

if __name__ == "__main__":
    unittest.main()
