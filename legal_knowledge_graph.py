# legal_knowledge_graph.py — Constitutional Knowledge Graph Traversal Engine
import json
import os
from typing import List, Dict, Any, Set, Tuple, Optional

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
            
        q_norm = q.replace("colour", "color")
        for name, nid in self.name_to_id.items():
            name_norm = name.replace("colour", "color")
            if q in name or name in q or q_norm in name_norm or name_norm in q_norm:
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

    def get_candidate_expansions(
        self,
        detected_articles: List[str] = None,
        detected_cases: List[str] = None,
        detected_doctrines: List[str] = None,
        max_hops: int = 2,
        max_expansions: int = 10
    ) -> List[str]:
        """
        Performs bounded BFS (up to max_hops) from detected legal entities
        and returns connected case and article node IDs, prioritized by
        authoritative legal relationship (ESTABLISHES, EXPANDS, RELIES_ON, etc.).
        """
        seed_entities = (detected_doctrines or []) + (detected_cases or []) + (detected_articles or [])
        seed_nodes = []
        for ent in seed_entities:
            for nid in self.find_node_id(ent):
                if nid not in seed_nodes:
                    seed_nodes.append(nid)

        visited = set(seed_nodes)
        current_level = list(seed_nodes)
        ordered_expansions = []

        rel_priority = {
            "REV_ESTABLISHES": 1,
            "ESTABLISHES": 1,
            "REV_EXPANDS": 2,
            "EXPANDS": 2,
            "LIMITS": 2,
            "REV_LIMITS": 2,
            "REV_RELIES_ON": 3,
            "RELIES_ON": 3,
            "INTERPRETS": 3,
            "REV_INTERPRETS": 3,
            "CONNECTED_TO": 4,
            "REV_CONNECTED_TO": 4,
            "OVERRULES": 5,
            "REV_OVERRULES": 5,
        }

        for hop in range(1, max_hops + 1):
            next_candidates = []
            for nid in current_level:
                for rel, tgt in self.adjacency.get(nid, []):
                    if tgt not in visited:
                        visited.add(tgt)
                        prio = rel_priority.get(rel, 9)
                        next_candidates.append((prio, tgt))

            next_candidates.sort(key=lambda x: x[0])
            next_level = []
            for prio, tgt in next_candidates:
                next_level.append(tgt)
                node_type = self.nodes.get(tgt, {}).get("type", "")
                if node_type in ("Case", "Article", "Amendment"):
                    if tgt not in ordered_expansions:
                        ordered_expansions.append(tgt)
                        if len(ordered_expansions) >= max_expansions:
                            return ordered_expansions[:max_expansions]
            current_level = next_level

        return ordered_expansions[:max_expansions]

    def get_doctrine_establishing_cases(self, doctrines: List[str]) -> Dict[str, List[str]]:
        """
        Returns mapping from doctrine node IDs to the cases that establish them (ESTABLISHES relation).
        """
        establishing_map = {}
        for doc in (doctrines or []):
            for nid in self.find_node_id(doc):
                est_cases = []
                for rel, tgt in self.adjacency.get(nid, []):
                    if rel in ("REV_ESTABLISHES", "ESTABLISHES"):
                        tgt_node = self.nodes.get(tgt, {})
                        if tgt_node.get("type") == "Case":
                            est_cases.append(tgt)
                establishing_map[nid] = est_cases
        return establishing_map

    def get_canonical_authority_for_doctrine(self, doctrine_name: str) -> Optional[Dict[str, Any]]:
        """
        Returns the primary establishing case node for a doctrine, or None if unknown.
        Returns dict with {"id": case_id, "name": case_name, "citation": citation, ...}
        """
        matched_nids = self.find_node_id(doctrine_name)
        for nid in matched_nids:
            if self.nodes.get(nid, {}).get("type") == "Doctrine":
                for rel, tgt in self.adjacency.get(nid, []):
                    if rel in ("REV_ESTABLISHES", "ESTABLISHES"):
                        tgt_node = self.nodes.get(tgt)
                        if tgt_node and tgt_node.get("type") == "Case":
                            return tgt_node
        return None

    def get_canonical_authorities_for_doctrine(self, doctrine_name: str) -> List[Dict[str, Any]]:
        """
        Returns all establishing case nodes for a doctrine.
        """
        authorities = []
        matched_nids = self.find_node_id(doctrine_name)
        seen_ids = set()
        for nid in matched_nids:
            if self.nodes.get(nid, {}).get("type") == "Doctrine":
                for rel, tgt in self.adjacency.get(nid, []):
                    if rel in ("REV_ESTABLISHES", "ESTABLISHES"):
                        tgt_node = self.nodes.get(tgt)
                        if tgt_node and tgt_node.get("type") == "Case" and tgt not in seen_ids:
                            seen_ids.add(tgt)
                            authorities.append(tgt_node)
        return authorities

    def get_doctrine_linked_cases(self, doctrines: List[str], max_hops: int = 2) -> Set[str]:
        """
        Returns all case IDs connected to the specified doctrines within max_hops.
        """
        linked_cases = set()
        for doc in (doctrines or []):
            for nid in self.find_node_id(doc):
                curr = {nid}
                visited = {nid}
                for hop in range(1, max_hops + 1):
                    nxt = set()
                    for c_id in curr:
                        for rel, tgt in self.adjacency.get(c_id, []):
                            if tgt not in visited:
                                visited.add(tgt)
                                nxt.add(tgt)
                                if self.nodes.get(tgt, {}).get("type") == "Case":
                                    linked_cases.add(tgt)
                    curr = nxt
        return linked_cases

    def get_query_expansion_terms(
        self,
        detected_doctrines: List[str] = None,
        detected_cases: List[str] = None,
        detected_articles: List[str] = None,
        max_terms: int = 5
    ) -> List[str]:
        """
        Extracts concise canonical authority terms (case short names, article numbers)
        from 1- and 2-hop connected graph nodes to enrich dense and lexical query strings.
        """
        expansions = self.get_candidate_expansions(
            detected_articles=detected_articles,
            detected_cases=detected_cases,
            detected_doctrines=detected_doctrines,
            max_hops=2,
            max_expansions=max_terms * 2
        )
        terms = []
        for nid in expansions:
            node = self.nodes.get(nid, {})
            name = node.get("name", "")
            if not name:
                continue
            if node.get("type") == "Case":
                short_name = name.split(" v. ")[0].split(" versus ")[0].strip()
                if short_name and short_name not in terms:
                    terms.append(short_name)
            elif node.get("type") == "Article":
                if name not in terms:
                    terms.append(name)
            if len(terms) >= max_terms:
                break
        return terms[:max_terms]

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
