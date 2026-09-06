# legal_query_classifier.py — Lightweight, High-Precision Legal Query Classifier
import re
from typing import Dict, Any, List, Optional

class LegalQueryClassifier:
    """
    Zero-PyTorch, sub-millisecond legal query classifier.
    Categorizes queries into 9 constitutional law classes, extracts entities
    (articles, landmark cases, doctrines), and produces dynamic retrieval biases.
    """

    CLASSES = [
        "Constitutional Provision Query",
        "Landmark Case Query",
        "Doctrine Query",
        "Fundamental Rights Query",
        "Amendment Query",
        "Judicial Review Query",
        "Comparative Case Query",
        "Procedural Law Query",
        "General Legal Question"
    ]

    # Pre-compiled regex patterns for legal entities
    ARTICLE_PATTERN = re.compile(r"\bArticle\s*([0-9]{1,3}[A-Za-z]?)\b", re.IGNORECASE)
    PART_PATTERN = re.compile(r"\bPart\s*(III|IV|IVA|XII|XX)\b", re.IGNORECASE)
    AMENDMENT_PATTERN = re.compile(r"\b(\d{1,3})(?:st|nd|rd|th)?\s*(?:Constitutional\s*)?Amendment\b", re.IGNORECASE)

    DOCTRINES = {
        "basic structure": "Basic Structure Doctrine",
        "substantive due process": "Substantive Due Process",
        "procedure established by law": "Procedure Established by Law",
        "golden triangle": "Golden Triangle Doctrine",
        "non-arbitrariness": "Anti-Arbitrariness Doctrine",
        "anti-arbitrariness": "Anti-Arbitrariness Doctrine",
        "manifest arbitrariness": "Manifest Arbitrariness Doctrine",
        "essential religious practice": "Essential Religious Practices Doctrine",
        "pith and substance": "Pith and Substance Doctrine",
        "colorable legislation": "Colorable Legislation Doctrine",
        "severability": "Doctrine of Severability",
        "eclipse": "Doctrine of Eclipse",
        "waiver": "Doctrine of Waiver",
        "prospective overruling": "Doctrine of Prospective Overruling",
        "constitutional morality": "Constitutional Morality",
        "proportionality": "Proportionality Standard",
        "creamy layer": "Creamy Layer Principle",
        "locus standi": "Liberalized Locus Standi / PIL",
        "complete justice": "Plenary Power / Complete Justice (Art 142)",
        "stare decisis": "Doctrine of Precedent / Stare Decisis (Art 141)",
        "territorial nexus": "Doctrine of Territorial Nexus"
    }

    CASES = {
        "gopalan": "A.K. Gopalan v. State of Madras",
        "maneka": "Maneka Gandhi v. Union of India",
        "puttaswamy": "Justice K.S. Puttaswamy v. Union of India",
        "kesavananda": "Kesavananda Bharati v. State of Kerala",
        "minerva": "Minerva Mills Ltd. v. Union of India",
        "royappa": "E.P. Royappa v. State of Tamil Nadu",
        "indra sawhney": "Indra Sawhney v. Union of India",
        "mandal": "Indra Sawhney v. Union of India",
        "shayara bano": "Shayara Bano v. Union of India",
        "triple talaq": "Shayara Bano v. Union of India",
        "navtej": "Navtej Singh Johar v. Union of India",
        "shreya singhal": "Shreya Singhal v. Union of India",
        "section 66a": "Shreya Singhal v. Union of India",
        "mhetre": "Siddharam Satlingappa Mhetre v. State of Maharashtra",
        "d.k. basu": "D.K. Basu v. State of West Bengal",
        "dk basu": "D.K. Basu v. State of West Bengal",
        "vishaka": "Vishaka v. State of Rajasthan",
        "bommai": "S.R. Bommai v. Union of India",
        "umadevi": "Secretary, State of Karnataka v. Umadevi",
        "olga tellis": "Olga Tellis v. Bombay Municipal Corporation",
        "francis coralie": "Francis Coralie Mullin v. Administrator, UT of Delhi",
        "hussainara": "Hussainara Khatoon v. State of Bihar",
        "sunil batra": "Sunil Batra v. Delhi Administration",
        "selvi": "Selvi v. State of Karnataka",
        "common cause": "Common Cause v. Union of India",
        "pucl": "People's Union for Civil Liberties v. Union of India",
        "joseph shine": "Joseph Shine v. Union of India",
        "kharak singh": "Kharak Singh v. State of UP",
        "m.p. sharma": "M.P. Sharma v. Satish Chandra",
        "mp sharma": "M.P. Sharma v. Satish Chandra",
        "adm jabalpur": "ADM Jabalpur v. Shivkant Shukla",
        "habeas corpus case": "ADM Jabalpur v. Shivkant Shukla",
        "golaknath": "I.C. Golaknath v. State of Punjab",
        "waman rao": "Waman Rao v. Union of India",
        "i.r. coelho": "I.R. Coelho v. State of Tamil Nadu",
        "coelho": "I.R. Coelho v. State of Tamil Nadu",
        "r.c. cooper": "R.C. Cooper v. Union of India",
        "rc cooper": "R.C. Cooper v. Union of India",
        "bank nationalization": "R.C. Cooper v. Union of India",
        "champakam": "State of Madras v. Champakam Dorairajan",
        "shankari prasad": "Shankari Prasad v. Union of India",
        "sajjan singh": "Sajjan Singh v. State of Rajasthan",
        "romesh thappar": "Romesh Thappar v. State of Madras",
        "bennett coleman": "Bennett Coleman & Co. v. Union of India",
        "anuradha bhasin": "Anuradha Bhasin v. Union of India",
        "aruna shanbaug": "Aruna Ramchandra Shanbaug v. Union of India",
        "s.p. gupta": "S.P. Gupta v. Union of India",
        "first judges": "S.P. Gupta v. Union of India",
        "second judges": "Supreme Court Advocates-on-Record Association v. Union of India",
        "njac": "Supreme Court Advocates-on-Record Association v. Union of India (NJAC)",
        "fourth judges": "Supreme Court Advocates-on-Record Association v. Union of India (NJAC)",
        "chandra kumar": "L. Chandra Kumar v. Union of India",
        "t.m.a. pai": "T.M.A. Pai Foundation v. State of Karnataka",
        "tma pai": "T.M.A. Pai Foundation v. State of Karnataka",
        "inamdar": "P.A. Inamdar v. State of Maharashtra",
        "lily thomas": "Lily Thomas v. Union of India",
        "adr": "Union of India v. Association for Democratic Reforms",
        "nagaraj": "M. Nagaraj v. Union of India",
        "jarnail singh": "Jarnail Singh v. Lachhmi Narain Gupta",
        "sabarimala": "Indian Young Lawyers Association v. State of Kerala",
        "nalsa": "National Legal Services Authority v. Union of India",
        "kameshwar": "State of Bihar v. Kameshwar Singh",
        "kameshwar singh": "State of Bihar v. Kameshwar Singh"
    }

    def classify(self, query: str) -> Dict[str, Any]:
        """
        Classifies a legal query into one of 9 classes and returns rich contextual signals.
        """
        q_lower = query.lower()

        # 1. Entity Extraction
        detected_articles = []
        for m in self.ARTICLE_PATTERN.finditer(query):
            detected_articles.append(f"Article {m.group(1).upper()}")
        if "right to property" in q_lower or "property right" in q_lower:
            detected_articles.append("Article 300A")

        detected_doctrines = []
        for term, doctrine in self.DOCTRINES.items():
            if term in q_lower:
                detected_doctrines.append(doctrine)

        detected_cases = []
        for key, case_name in self.CASES.items():
            if re.search(r"\b" + re.escape(key) + r"\b", q_lower):
                if case_name not in detected_cases:
                    detected_cases.append(case_name)

        has_amendment = bool(self.AMENDMENT_PATTERN.search(query) or "amend" in q_lower or "368" in q_lower)
        has_case_comparison = (len(detected_cases) >= 2) or (bool(re.search(r"\b(v\.|vs|versus)\b", q_lower)) and len(detected_cases) >= 1 and bool(re.search(r"\b(compare|overruled|departed from|difference)\b", q_lower)))
        has_procedure = bool(re.search(r"\b(procedure|crpc|cpc|bail|arrest|warrant|anticipatory|section 438|memo of arrest|guidelines)\b", q_lower))
        has_judicial_review = bool(re.search(r"\b(judicial review|writ|habeas corpus|mandamus|certiorari|quo warranto|prohibition|article 32|article 226)\b", q_lower))
        has_rights = bool(re.search(r"\b(fundamental right|liberty|equality|speech|expression|religion|life|privacy|livelihood|part iii)\b", q_lower))

        # 2. Decision Logic for Classification
        primary_class = "General Legal Question"
        confidence = 0.70
        retrieval_bias = {"article_boost": 1.0, "case_boost": 1.0, "ratio_boost": 1.0}

        if has_case_comparison:
            primary_class = "Comparative Case Query"
            confidence = 0.92
            retrieval_bias = {"article_boost": 0.8, "case_boost": 1.4, "ratio_boost": 1.3}

        elif detected_doctrines:
            primary_class = "Doctrine Query"
            confidence = 0.95
            retrieval_bias = {"article_boost": 0.9, "case_boost": 1.2, "ratio_boost": 1.5}

        elif detected_cases:
            primary_class = "Landmark Case Query"
            confidence = 0.96
            retrieval_bias = {"article_boost": 0.7, "case_boost": 1.5, "ratio_boost": 1.3}

        elif has_amendment and ("amend" in q_lower or "basic structure" in q_lower or "368" in q_lower):
            primary_class = "Amendment Query"
            confidence = 0.94
            retrieval_bias = {"article_boost": 1.3, "case_boost": 1.3, "ratio_boost": 1.2}

        elif has_judicial_review and ("writ" in q_lower or "review" in q_lower or "32" in q_lower or "226" in q_lower):
            primary_class = "Judicial Review Query"
            confidence = 0.93
            retrieval_bias = {"article_boost": 1.3, "case_boost": 1.2, "ratio_boost": 1.2}

        elif has_procedure and ("arrest" in q_lower or "bail" in q_lower or "detention" in q_lower):
            primary_class = "Procedural Law Query"
            confidence = 0.91
            retrieval_bias = {"article_boost": 1.1, "case_boost": 1.3, "ratio_boost": 1.3}

        elif detected_articles:
            primary_class = "Constitutional Provision Query"
            confidence = 0.95
            retrieval_bias = {"article_boost": 1.6, "case_boost": 1.1, "ratio_boost": 1.2}

        elif has_rights:
            primary_class = "Fundamental Rights Query"
            confidence = 0.90
            retrieval_bias = {"article_boost": 1.2, "case_boost": 1.2, "ratio_boost": 1.3}

        return {
            "query": query,
            "primary_class": primary_class,
            "confidence": confidence,
            "detected_articles": list(set(detected_articles)),
            "detected_cases": detected_cases,
            "detected_doctrines": list(set(detected_doctrines)),
            "retrieval_bias": retrieval_bias
        }

if __name__ == "__main__":
    classifier = LegalQueryClassifier()
    test_queries = [
        "What is Article 21?",
        "Explain the facts and ratio of Maneka Gandhi v. Union of India",
        "What is the basic structure doctrine?",
        "Can Parliament amend Fundamental Rights under Article 368?",
        "Compare A.K. Gopalan and Maneka Gandhi on personal liberty",
        "What are the D.K. Basu guidelines for arrest?",
        "What writs can be issued under Article 32 and Article 226 for judicial review?",
        "Does the right to privacy include informational privacy?",
        "What is the legal difference between an ordinance and an act?"
    ]
    print("=== LegalQueryClassifier Sanity Test ===")
    for tq in test_queries:
        res = classifier.classify(tq)
        print(f"[{res['primary_class']}] (conf: {res['confidence']:.2f}) -> '{tq}'")
        if res["detected_articles"]: print(f"   Articles: {res['detected_articles']}")
        if res["detected_cases"]: print(f"   Cases: {res['detected_cases']}")
        if res["detected_doctrines"]: print(f"   Doctrines: {res['detected_doctrines']}")
