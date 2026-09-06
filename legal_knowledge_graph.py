# legal_knowledge_graph.py — Constitutional Knowledge Graph Traversal Engine
import json
import os
from typing import List, Dict, Any, Set, Tuple

KG_PATH = "data/legal_knowledge_graph.json"

import re

def normalize_article_to_graph_id(article_str: str) -> str:
    """
    Extracts numeric and letter components of a constitutional article and
    converts it to a canonical 3-digit zero-padded graph node ID.
    Examples:
        "Article 21"   -> "art_021"
        "Article 14"   -> "art_014"
        "Article 300A" -> "art_300a"
        "Article 21A"  -> "art_021a"
        "Article 32"   -> "art_032"
    """
    if not article_str:
        return ""
    m = re.search(r"(\d+)([A-Za-z]*)", str(article_str).strip())
    if m:
        num = int(m.group(1))
        suffix = m.group(2).lower()
        return f"art_{num:03d}{suffix}"
    return str(article_str).lower().strip()

class LegalKnowledgeGraph:
    """
    In-memory graph traversal engine for Indian Constitutional Jurisprudence.
    Provides 1-hop traversal across ESTABLISHES, OVERRULES, RELIES_ON, EXPANDS,
    LIMITS, and INTERPRETS relationships to boost retrieval candidates.
    """

    def __init__(self, graph_path: str = KG_PATH):
        self.graph_path = graph_path
        self.nodes = {}
        self.edges = []
        self.adjacency = {}
        self.name_to_id = {}
        self._load_graph()

    def _load_graph(self):
        if not os.path.exists(self.graph_path):
            return
        with open(self.graph_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for n in data.get("nodes", []):
            nid = n["id"]
            self.nodes[nid] = n
            self.name_to_id[n["name"].lower()] = nid
            self.adjacency[nid] = []

        for e in data.get("edges", []):
            src, tgt, rel = e["source"], e["target"], e["relation"]
            self.edges.append(e)
            if src in self.adjacency:
                self.adjacency[src].append((rel, tgt))
            # Also maintain reverse links for bidirectional discovery
            if tgt in self.adjacency:
                self.adjacency[tgt].append((f"REV_{rel}", src))

    def find_node_id(self, query_term: str) -> List[str]:
        """Fuzzy match a case name, doctrine, or article to graph node ID"""
        q = query_term.lower().strip()
        matched = []
        
        # Check normalized article ID first
        norm_art = normalize_article_to_graph_id(query_term)
        if norm_art in self.nodes and norm_art not in matched:
            matched.append(norm_art)
            
        for name, nid in self.name_to_id.items():
            if q in name or name in q:
                if nid not in matched:
                    matched.append(nid)
        return matched

    def get_related_entities(self, query_entities: List[str]) -> Dict[str, Any]:
        """
        Given extracted entities (from LegalQueryClassifier), traverse 1-hop graph
        neighbors and return connected legal nodes and relationships.
        """
        connected_nodes = {}
        related_case_ids = set()
        related_article_ids = set()
        overruled_nodes = set()

        for entity in query_entities:
            node_ids = self.find_node_id(entity)
            for nid in node_ids:
                neighbors = self.adjacency.get(nid, [])
                for rel, target_id in neighbors:
                    target_node = self.nodes.get(target_id)
                    if not target_node:
                        continue
                    
                    connected_nodes[target_id] = {
                        "name": target_node["name"],
                        "type": target_node["type"],
                        "relation": rel
                    }
                    if target_node["type"] == "Case":
                        related_case_ids.add(target_id)
                    elif target_node["type"] == "Article":
                        related_article_ids.add(target_id)

                    if "OVERRULES" in rel:
                        overruled_nodes.add(target_id)

        return {
            "connected_nodes": connected_nodes,
            "related_case_ids": list(related_case_ids),
            "related_article_ids": list(related_article_ids),
            "overruled_nodes": list(overruled_nodes)
        }

    def compute_graph_boosts(
        self,
        query_entities: List[str],
        query: str = "",
        primary_class: str = ""
    ) -> Dict[str, float]:
        """
        Produces document boost multipliers based on graph connections.
        - Direct establishing/expanding cases: +0.20
        - Relies on, interprets, limits: +0.15
        - Connected to: +0.10
        - Overruled cases: -0.15 penalty (unless query specifically asks for comparison/overruling)
        """
        boosts = {}
        info = self.get_related_entities(query_entities)
        
        for nid, meta in info["connected_nodes"].items():
            rel = meta["relation"]
            if rel in ("ESTABLISHES", "REV_ESTABLISHES", "EXPANDS", "REV_EXPANDS"):
                boosts[nid] = 0.20
            elif rel in ("RELIES_ON", "INTERPRETS", "LIMITS"):
                boosts[nid] = 0.15
            elif "CONNECTED_TO" in rel:
                boosts[nid] = 0.10

        # Overruled case penalty (-0.15) unless comparative query or comparison keywords present
        q_lower = query.lower() if query else ""
        is_comparative = (
            primary_class == "Comparative Case Query" or
            any(w in q_lower for w in ("overruled", "reversed", "distinguish", "compare", "versus", " vs "))
        )

        if not is_comparative:
            for onode in info.get("overruled_nodes", []):
                # Apply -0.15 penalty only if the node hasn't earned a direct positive boost
                if onode not in boosts or boosts[onode] <= 0.0:
                    boosts[onode] = -0.15
        
        return boosts

if __name__ == "__main__":
    kg = LegalKnowledgeGraph()
    print("=== LegalKnowledgeGraph Sanity Test ===")
    test_entities = ["Substantive Due Process", "Maneka Gandhi", "Basic Structure Doctrine", "Right to Privacy"]
    for te in test_entities:
        boosts = kg.compute_graph_boosts([te])
        rel = kg.get_related_entities([te])
        print(f"\nQuery Entity: '{te}'")
        print(f"  Connected Nodes ({len(rel['connected_nodes'])}):")
        for k, v in list(rel["connected_nodes"].items())[:5]:
            print(f"    - [{v['relation']}] -> {v['name']} ({v['type']})")
        print(f"  Graph Boost Map: {boosts}")
