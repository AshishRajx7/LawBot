# legal_grounding_engine.py — Pre-Generation Grounding, Contradiction Detection & Prompt Synthesis
import re
from typing import List, Dict, Any, Tuple, Optional, Set

from legal_knowledge_graph import LegalKnowledgeGraph
from legal_query_classifier import LegalQueryClassifier

# Cached singleton
_KG_INSTANCE: Optional[LegalKnowledgeGraph] = None

def get_kg() -> LegalKnowledgeGraph:
    global _KG_INSTANCE
    if _KG_INSTANCE is None:
        _KG_INSTANCE = LegalKnowledgeGraph()
    return _KG_INSTANCE


# ==============================================================================
# 1. CASE & DOCTRINE CLEANING HELPERS
# ==============================================================================

def clean_case_name(title: str) -> str:
    """Extract clean case or article name without ratio chunk tags or metadata annotations."""
    if not title:
        return ""
    # Strip ratio chunk suffixes like " [Ratio 01: ...]" or " - Ratio 01"
    name = re.sub(r"\s*\[Ratio\s*\d+:[^\]]*\]", "", title, flags=re.IGNORECASE)
    name = re.sub(r"\s*:\s*Ratio\s*\d+.*", "", name, flags=re.IGNORECASE)
    return name.strip()

def get_short_case_name(name: str) -> str:
    """Extract short title (e.g., 'Kesavananda Bharati' from 'Kesavananda Bharati v. State of Kerala')."""
    clean = clean_case_name(name)
    if " v. " in clean:
        return clean.split(" v. ")[0].strip()
    if " versus " in clean.lower():
        return re.split(r"\s+versus\s+", clean, flags=re.IGNORECASE)[0].strip()
    return clean


# ==============================================================================
# 2. CONSTITUTIONAL ARTICLE DOMAIN REGISTRY (Fix 3: FM-2)
# ==============================================================================

ARTICLE_DOMAIN_REGISTRY: Dict[str, Dict[str, Any]] = {
    "14": {
        "label": "Article 14",
        "domain": "Equality before law and equal protection of the laws",
        "valid_concepts": ["equality", "equal protection", "equality before law", "non-arbitrariness", "arbitrariness", "reasonable classification"],
        "disallowed": {
            "constitutional amendment": ("Article 368", "Power of Parliament to amend the Constitution"),
            "amending power": ("Article 368", "Power of Parliament to amend the Constitution"),
            "preventive detention": ("Article 22", "Protection against Arrest and Preventive Detention"),
            "writs": ("Article 32", "Constitutional Remedies / Writs"),
        }
    },
    "19": {
        "label": "Article 19",
        "domain": "Protection of certain rights regarding freedom of speech, expression, assembly, movement",
        "valid_concepts": ["speech", "expression", "assembly", "association", "movement", "residence", "profession", "occupation", "trade", "press"],
        "disallowed": {
            "constitutional amendment": ("Article 368", "Power of Parliament to amend the Constitution"),
            "amending power": ("Article 368", "Power of Parliament to amend the Constitution"),
            "preventive detention": ("Article 22", "Protection against Arrest and Preventive Detention"),
            "constitutional remedies": ("Article 32", "Constitutional Remedies / Writs"),
            "writs": ("Article 32", "Constitutional Remedies / Writs"),
            "habeas corpus": ("Article 32", "Constitutional Remedies / Writs"),
        }
    },
    "21": {
        "label": "Article 21",
        "domain": "Protection of life and personal liberty",
        "valid_concepts": ["life", "personal liberty", "liberty", "privacy", "due process", "procedure established by law", "fair trial", "speedy trial", "livelihood", "dignity"],
        "disallowed": {
            "constitutional amendment": ("Article 368", "Power of Parliament to amend the Constitution"),
            "amending power": ("Article 368", "Power of Parliament to amend the Constitution"),
            "preventive detention": ("Article 22", "Protection against Arrest and Preventive Detention"),
        }
    },
    "22": {
        "label": "Article 22",
        "domain": "Protection against arrest and preventive detention in certain cases",
        "valid_concepts": ["preventive detention", "arrest", "detention", "grounds of arrest", "advisory board"],
        "disallowed": {
            "free speech": ("Article 19", "Freedom of Speech and Expression"),
            "freedom of speech": ("Article 19", "Freedom of Speech and Expression"),
            "constitutional remedies": ("Article 32", "Constitutional Remedies / Writs"),
            "constitutional amendment": ("Article 368", "Power of Parliament to amend the Constitution"),
        }
    },
    "32": {
        "label": "Article 32",
        "domain": "Remedies for enforcement of Fundamental Rights (Writs)",
        "valid_concepts": ["constitutional remedies", "remedies", "writs", "habeas corpus", "mandamus", "prohibition", "quo warranto", "certiorari", "enforcement of fundamental rights", "remedy"],
        "disallowed": {
            "preventive detention": ("Article 22", "Protection against arrest and preventive detention"),
            "free speech": ("Article 19", "Freedom of Speech and Expression"),
            "freedom of speech": ("Article 19", "Freedom of Speech and Expression"),
            "speech and expression": ("Article 19", "Freedom of Speech and Expression"),
            "constitutional amendment": ("Article 368", "Power of Parliament to amend the Constitution"),
            "amending power": ("Article 368", "Power of Parliament to amend the Constitution"),
        }
    },
    "368": {
        "label": "Article 368",
        "domain": "Power of Parliament to amend the Constitution and procedure therefor",
        "valid_concepts": ["amendment", "amending power", "constituent power", "procedure for amendment", "amend the constitution"],
        "disallowed": {
            "free speech": ("Article 19", "Freedom of Speech and Expression"),
            "freedom of speech": ("Article 19", "Freedom of Speech and Expression"),
            "speech and expression": ("Article 19", "Freedom of Speech and Expression"),
            "speech": ("Article 19", "Freedom of Speech and Expression"),
            "expression": ("Article 19", "Freedom of Speech and Expression"),
            "personal liberty": ("Article 21", "Right to Life and Personal Liberty"),
            "privacy": ("Article 21", "Right to Privacy"),
            "preventive detention": ("Article 22", "Protection against arrest and preventive detention"),
            "writs": ("Article 32", "Constitutional Remedies / Writs"),
            "constitutional remedies": ("Article 32", "Constitutional Remedies / Writs"),
            "habeas corpus": ("Article 32", "Constitutional Remedies / Writs"),
            "unlimited amending power": ("Basic Structure Doctrine (Kesavananda Bharati / Minerva Mills)", "Amending power under Article 368 is limited, not absolute"),
            "unlimited power to amend": ("Basic Structure Doctrine (Kesavananda Bharati / Minerva Mills)", "Amending power under Article 368 is limited, not absolute"),
        }
    }
}


# ==============================================================================
# 3. COMPARATIVE & EVOLUTION PATTERNS (Fix 2: AF-1)
# ==============================================================================

COMPARATIVE_PATTERNS = [
    r"\bcompare\b", r"\bcomparison\b", r"\bcontrasting?\b", r"\bcontrast\b",
    r"\bdifference\b", r"\bdifferences\b", r"\bdistinguish\b", r"\bdistinguishing\b",
    r"\bversus\b", r"\bvs\.?\b", r"\bevolution\b", r"\bdevelopment\b",
    r"\bjourney\b", r"\btrace\b", r"\btracing\b", r"\boverruled\b", r"\boverruling\b",
    r"\bdiffer(?:ent|ed|ing|s)?\s+views\b"
]

def is_comparative_query(query: str) -> bool:
    """Detects whether a query asks for comparison, evolution, or distinguishing between authorities."""
    q_lower = query.lower()
    return any(re.search(pat, q_lower) for pat in COMPARATIVE_PATTERNS)


# ==============================================================================
# 4. LEADING ASSUMPTION & ATTRIBUTION TRIGGERS (Fix 4: FM-3)
# ==============================================================================

ASSUMPTION_TRIGGERS = [
    # Explicit question verbs
    r"\bdid\b", r"\bhow\s+did\b", r"\bestablish(?:ed|es|ing)?\b",
    r"\bhold(?:s|ing)?\b", r"\bheld\b",
    r"\bcreate(?:d|s|ing)?\b",
    r"\bintroduce(?:d|s|ing)?\b",
    r"\brule(?:d|s|ing)?\b",
    r"\blaid\s+down\b", r"\blay\s+down\b",
    r"\bformulate(?:d|s|ing)?\b",
    
    # Expanded attribution verbs (FM-3)
    r"\brecogniz(?:ed|e|es|ing)\b",
    r"\bgrant(?:ed|s|ing)?\b",
    r"\bprotect(?:ed|s|ing)?\b",
    r"\baffirm(?:ed|s|ing)?\b",
    r"\bdeclar(?:ed|e|es|ing)\b",
    
    # Subordinate / leading clause triggers (FM-3)
    r"\bsince\b", r"\bafter\b", r"\bassuming\b", r"\bgiven\s+that\b", r"\bbecause\b"
]

def has_assumption_trigger(query: str) -> bool:
    """Checks whether the query contains leading assumption triggers or attribution verbs."""
    q_lower = query.lower()
    return any(re.search(pat, q_lower) for pat in ASSUMPTION_TRIGGERS)


# Extended doctrine aliases recognized for grounding/contradiction checks
GROUNDING_DOCTRINE_ALIASES = {
    "judicial review": "Judicial Review of Legislation",
    "judicial review as basic feature": "Judicial Review of Legislation",
    "privacy": "Right to Privacy",
    "right to privacy": "Right to Privacy",
    "colourable legislation": "Colorable Legislation Doctrine",
    "colorable legislation": "Colorable Legislation Doctrine",
    "creamy layer": "Creamy Layer Principle",
    "basic structure": "Basic Structure Doctrine",
    "substantive due process": "Substantive Due Process",
    "due process": "Substantive Due Process",
    "golden triangle": "Golden Triangle Doctrine",
    "speedy trial": "Right to Speedy Trial",
    "livelihood": "Right to Livelihood",
    "anti-arbitrariness": "Anti-Arbitrariness Doctrine",
    "arbitrariness": "Anti-Arbitrariness Doctrine",
    "manifest arbitrariness": "Manifest Arbitrariness Doctrine",
    "collegium": "Collegium System",
    "constitutional morality": "Constitutional Morality",
    "prospective overruling": "Prospective Overruling"
}


# ==============================================================================
# 5. GROUNDING FACTS EXTRACTION
# ==============================================================================

def extract_grounding_facts(
    top_candidates: List[Dict[str, Any]],
    detected_doctrines: List[str] = None
) -> Dict[str, Any]:
    """
    Extracts high-priority grounding facts from retrieved candidates:
    - Highest Ranked Authority
    - Retrieved Constitutional Provisions
    - Retrieved Doctrines
    - Retrieved Supporting Authorities
    """
    if not top_candidates:
        return {
            "highest_ranked_authority": "None",
            "retrieved_provisions": [],
            "retrieved_doctrines": detected_doctrines or [],
            "retrieved_supporting_authorities": []
        }

    top_1 = top_candidates[0]
    highest_ranked_authority = clean_case_name(top_1.get("title", ""))

    # Provisions from top candidates
    provisions = []
    for c in top_candidates[:5]:
        art = c.get("primary_article", "")
        if art and art not in provisions:
            provisions.append(art)

    # Doctrines from top candidates + detected doctrines
    doctrines = list(detected_doctrines or [])
    for c in top_candidates[:3]:
        topics = c.get("legal_topics", "")
        title = c.get("title", "")
        combined = f"{topics} {title}".lower()
        if "basic structure" in combined and "Basic Structure Doctrine" not in doctrines:
            doctrines.append("Basic Structure Doctrine")
        if "due process" in combined and "Substantive Due Process" not in doctrines:
            doctrines.append("Substantive Due Process")
        if "privacy" in combined and "Right to Privacy" not in doctrines:
            doctrines.append("Right to Privacy")
        if "golden triangle" in combined and "Golden Triangle Doctrine" not in doctrines:
            doctrines.append("Golden Triangle Doctrine")
        if "arbitrariness" in combined and "Anti-Arbitrariness Doctrine" not in doctrines:
            doctrines.append("Anti-Arbitrariness Doctrine")
        if "creamy layer" in combined and "Creamy Layer Principle" not in doctrines:
            doctrines.append("Creamy Layer Principle")
        if "judicial review" in combined and "Judicial Review as Basic Feature" not in doctrines:
            doctrines.append("Judicial Review as Basic Feature")

    # Supporting authorities (unique case titles from top 3)
    supporting_authorities = []
    for c in top_candidates[:3]:
        c_name = clean_case_name(c.get("title", ""))
        if c_name and c_name not in supporting_authorities:
            supporting_authorities.append(c_name)

    return {
        "highest_ranked_authority": highest_ranked_authority,
        "retrieved_provisions": provisions,
        "retrieved_doctrines": doctrines,
        "retrieved_supporting_authorities": supporting_authorities
    }


# ==============================================================================
# 6. PREMISE CONTRADICTION DETECTION (Fix 1: FM-1, Fix 2: AF-1, Fix 3: FM-2, Fix 4: FM-3)
# ==============================================================================

def validate_article_domains(query: str, detected_articles: Optional[List[str]] = None) -> str:
    """
    Validates that the query does not attribute unrelated rights/principles to a constitutional article.
    Returns a formatted WARNING string if contradiction exists, else empty string.
    """
    q_lower = query.lower()
    mentioned_nums = set()

    # Extract articles from query regex
    for art_match in re.findall(r"\b(?:article|art\.?)\s*([0-9]{1,3}[a-z]?)\b", q_lower):
        clean_num = re.sub(r"[^\d]", "", art_match)
        if clean_num:
            mentioned_nums.add(clean_num)

    # Add from detected_articles if provided
    if detected_articles:
        for da in detected_articles:
            clean_num = re.sub(r"[^\d]", "", da)
            if clean_num:
                mentioned_nums.add(clean_num)

    for art_num in mentioned_nums:
        if art_num in ARTICLE_DOMAIN_REGISTRY:
            reg = ARTICLE_DOMAIN_REGISTRY[art_num]
            for disallowed_concept, (correct_art, correct_subject) in reg["disallowed"].items():
                concept_pat = r"\b" + re.escape(disallowed_concept) + r"\b"
                if re.search(concept_pat, q_lower):
                    correct_num = re.sub(r"[^\d]", "", correct_art)
                    # If the correct article is also mentioned, query may be legitimately comparing both
                    if correct_num in mentioned_nums:
                        continue
                    return (
                        f"WARNING:\n"
                        f"User premise conflicts with retrieved authorities.\n"
                        f"{disallowed_concept.title()} is governed by {correct_art} ({correct_subject}), "
                        f"not {reg['label']} (which pertains to {reg['domain']})."
                    )
    return ""

def detect_query_premise_contradiction(
    query: str,
    detected_cases: List[str],
    detected_doctrines: List[str],
    top_candidates: List[Dict[str, Any]] = None,
    detected_articles: Optional[List[str]] = None,
    kg: Optional[LegalKnowledgeGraph] = None
) -> str:
    """
    Detects whether the user query makes a false assumption attributing a doctrine
    or constitutional principle to the wrong case or article.
    
    Fix 1 (FM-1): Canonical Knowledge Graph resolution. NEVER uses top_candidates[0] as truth.
    Fix 2 (AF-1): Suppresses warnings for comparative, evolutionary, or distinguishing queries.
    Fix 3 (FM-2): Validates constitutional article domains against ARTICLE_DOMAIN_REGISTRY.
    Fix 4 (FM-3): Detects leading assumptions and subordinate clauses ('since', 'after', 'given that', etc.).
    """
    # Fix 2 (AF-1): Return no warning if comparative/evolution intent is detected
    if is_comparative_query(query):
        return ""

    # Fix 3 (FM-2): Validate constitutional article domain attributions
    art_warning = validate_article_domains(query, detected_articles)
    if art_warning:
        return art_warning

    # Check for leading assumption triggers or attribution verbs (Fix 4: FM-3)
    trigger_active = has_assumption_trigger(query)

    q_lower = query.lower()

    # Determine candidate case mentions in query
    premise_cases = list(detected_cases or [])
    for kw, c_full in LegalQueryClassifier.CASES.items():
        if re.search(r"\b" + re.escape(kw) + r"\b", q_lower):
            if c_full not in premise_cases:
                premise_cases.append(c_full)

    # Determine candidate doctrine in query
    premise_doctrines = list(detected_doctrines or [])
    for kw, d_full in LegalQueryClassifier.DOCTRINES.items():
        if re.search(r"\b" + re.escape(kw) + r"\b", q_lower):
            if d_full not in premise_doctrines:
                premise_doctrines.append(d_full)
    for kw, d_full in GROUNDING_DOCTRINE_ALIASES.items():
        if re.search(r"\b" + re.escape(kw) + r"\b", q_lower):
            if d_full not in premise_doctrines:
                premise_doctrines.append(d_full)

    if not premise_doctrines or not premise_cases:
        return ""

    # If trigger is not active and there are multiple cases without attribution words, skip
    if not trigger_active and len(premise_cases) > 1:
        return ""

    knowledge_graph = kg or get_kg()

    # Fix 1 (FM-1): Resolve canonical establishing authority from Knowledge Graph
    for target_doctrine in premise_doctrines:
        canonical_authorities = knowledge_graph.get_canonical_authorities_for_doctrine(target_doctrine)
        if not canonical_authorities:
            # Fallback to single canonical authority if list is empty
            single_auth = knowledge_graph.get_canonical_authority_for_doctrine(target_doctrine)
            if single_auth:
                canonical_authorities = [single_auth]

        if not canonical_authorities:
            continue

        canonical_full_names = [clean_case_name(ca.get("name", "")) for ca in canonical_authorities]
        canonical_short_names = [get_short_case_name(name) for name in canonical_full_names]

        for case_a_full in premise_cases:
            case_a_short = get_short_case_name(case_a_full)

            # Check if Case A is one of the canonical establishing authorities
            is_canonical = any(
                case_a_short.lower() in c_full.lower() or c_short.lower() in case_a_full.lower()
                for c_full, c_short in zip(canonical_full_names, canonical_short_names)
            )

            if not is_canonical:
                primary_canonical_full = canonical_full_names[0]
                return (
                    f"WARNING:\n"
                    f"User premise conflicts with retrieved authorities.\n"
                    f"The {target_doctrine} was established by {primary_canonical_full}, not {case_a_full}."
                )

    return ""


# ==============================================================================
# 7. PROMPT SYNTHESIS
# ==============================================================================

def format_grounding_block(grounding_facts: Dict[str, Any]) -> str:
    """Formats the Grounding Facts block to inject into the system prompt."""
    highest_auth = grounding_facts.get("highest_ranked_authority", "None")
    doctrines = grounding_facts.get("retrieved_doctrines", [])
    doctrines_str = ", ".join(doctrines) if doctrines else "None specified"
    supporting = grounding_facts.get("retrieved_supporting_authorities", [])

    supporting_bullets = "\n".join(f"* {auth}" for auth in supporting) if supporting else "* (None)"

    return (
        f"Grounding Facts:\n"
        f"* Highest Ranked Authority: {highest_auth}\n"
        f"* Retrieved Doctrine: {doctrines_str}\n"
        f"* Retrieved Supporting Authorities:\n"
        f"{supporting_bullets}\n"
        f"These facts override user assumptions."
    )

def build_grounded_generation_prompt(
    query: str,
    context: str,
    top_candidates: List[Dict[str, Any]],
    cls_res: Dict[str, Any] = None
) -> Tuple[str, str]:
    """
    Constructs the grounded legal research prompt adhering strictly to all requirements:
    - Mandatory contradiction check
    - Pre-generation grounding summary block
    - Contradiction warning injection
    - Retrieval Consistency Check output section
    Returns: (full_prompt, contradiction_warning)
    """
    detected_doctrines = cls_res.get("detected_doctrines", []) if cls_res else []
    detected_cases = cls_res.get("detected_cases", []) if cls_res else []
    detected_articles = cls_res.get("detected_articles", []) if cls_res else []

    grounding_facts = extract_grounding_facts(top_candidates, detected_doctrines)
    grounding_block = format_grounding_block(grounding_facts)
    warning = detect_query_premise_contradiction(
        query=query,
        detected_cases=detected_cases,
        detected_doctrines=detected_doctrines,
        top_candidates=top_candidates,
        detected_articles=detected_articles
    )

    warning_section = f"\n\n{warning}\n" if warning else ""

    system_prompt = f"""You are LawBot, an expert Indian Constitutional Law research assistant.

Your task is to answer ONLY from the retrieved legal authorities provided in context.

{grounding_block}{warning_section}

CRITICAL RULES:

1. RETRIEVAL IS THE SOURCE OF TRUTH
   * Retrieved authorities are the only admissible legal evidence.
   * User wording, detected cases, detected doctrines, query classification, graph expansions, and metadata are routing signals only.
   * Never treat detected entities or user wording as evidence.
   * If the user premise conflicts with retrieved authorities, explicitly correct the premise.
   * Never attribute a doctrine, constitutional principle, or precedent to a case unless the retrieved authorities support that attribution.

2. MANDATORY CONTRADICTION CHECK
   * Identify the main legal proposition and check whether the user's premise is accurate.
   * If the user premise falsely assumes that Case A established or governed a doctrine when authoritative legal sources show Case B established it:
     YOU MUST EXPLICITLY REFUTE THE USER PREMISE IN BOTH THE 'Retrieval Consistency Check' AND 'Court Reasoning' SECTIONS.
   * Example: If the user asks "Did Maneka Gandhi establish the Basic Structure Doctrine?", state clearly:
     "No. The Basic Structure Doctrine was established by Kesavananda Bharati v. State of Kerala, not Maneka Gandhi v. Union of India."

3. DO NOT ASSUME RELATIONSHIPS
   * If the user mentions a case and a doctrine together, do not assume the case established, expanded, limited, or discussed that doctrine.
   * Verify every relationship using the retrieved authorities.
   * If the retrieved materials show another authority established the doctrine, state that explicitly.

4. PRIORITIZE HIGHEST-RANKED RELEVANT AUTHORITIES
   * When answering a doctrine question, first examine doctrine authorities and doctrine-establishing precedents.
   * When answering a constitutional provision question, prioritize constitutional articles.
   * When answering a case question, prioritize the relevant case ratios and holdings.

5. EVIDENCE BEFORE CONCLUSION
   * Determine which retrieved authority directly answers the question.
   * Base conclusions only on those authorities.
   * Ignore lower-ranked authorities when they conflict with higher-ranked authorities.

6. NO HALLUCINATED AUTHORITIES
   * Do not cite cases, articles, doctrines, amendments, judges, or constitutional provisions unless they appear in the retrieved materials.
   * If information is unavailable in retrieved sources, explicitly state:
     "The retrieved authorities do not provide sufficient information to answer this aspect of the question."

OUTPUT FORMAT STRICT COMPLIANCE:
Your response must strictly contain these six sections in this exact order:

### Retrieval Consistency Check
* User premise supported: [Yes / No]
* Explanation: [If No, explain the exact correction, identifying the false assumption and naming the authoritative retrieved precedent that established the doctrine/principle. If Yes, confirm alignment with retrieved evidence.]

### Legal Principle
[State the governing legal principle directly supported by the retrieved authorities.]

### Relevant Constitutional Provision
[List only provisions actually supported by retrieved materials.]

### Court Reasoning
[Summarize the ratio decidendi and reasoning from the most relevant retrieved authority. If the user question contained a false premise, re-clarify the true establishing authority here.]

### Authorities Relied Upon
[List only authorities used in the answer, with official citations and years.]

### Confidence Assessment
* [High / Medium / Low]: [Brief justification based on retrieved source completeness.]

Remember:
Retrieved authorities are evidence.
Detected entities are not evidence.
Ranking influences trust.
Grounding overrides assumptions."""

    user_message = f"""Retrieved Legal Authorities:
{context}

Question:
{query}"""

    full_prompt = f"{system_prompt}\n\n{user_message}"
    return full_prompt, warning


# ==============================================================================
# 8. GENERIC DOCTRINE-AGNOSTIC POST-GENERATION VALIDATOR (Fix 5: FM-5)
# ==============================================================================

def post_process_grounded_answer(
    raw_answer: str,
    query: str,
    top_candidates: List[Dict[str, Any]] = None,
    contradiction_warning: str = "",
    kg: Optional[LegalKnowledgeGraph] = None
) -> str:
    """
    Validates and enforces post-generation grounding compliance generically:
    1. Extracts doctrines and canonical establishing authorities dynamically from the Knowledge Graph.
    2. Dynamically scans for incorrect establishment claims (Pattern A, B, C) without doctrine-specific regexes.
    3. Corrects or annotates incorrect claims using the canonical authority.
    4. Enforces 'Retrieval Consistency Check' presence and accurate Yes/No flag.
    """
    answer = raw_answer.strip()
    knowledge_graph = kg or get_kg()

    # Step 1: Collect doctrines and their canonical establishing authorities from KG
    doctrine_canonical_map = []
    for nid, node in knowledge_graph.nodes.items():
        if node.get("type") == "Doctrine":
            d_name = node.get("name", "")
            canons = knowledge_graph.get_canonical_authorities_for_doctrine(d_name)
            if not canons:
                single = knowledge_graph.get_canonical_authority_for_doctrine(d_name)
                if single:
                    canons = [single]
            if canons:
                primary_full = clean_case_name(canons[0].get("name", ""))
                primary_short = get_short_case_name(primary_full)
                canon_fulls = [clean_case_name(c.get("name", "")).lower() for c in canons]
                canon_shorts = [get_short_case_name(clean_case_name(c.get("name", ""))).lower() for c in canons]
                doctrine_canonical_map.append({
                    "name": d_name,
                    "primary_full": primary_full,
                    "primary_short": primary_short,
                    "canon_fulls": canon_fulls,
                    "canon_shorts": canon_shorts
                })

    # Step 2: Build set of known cases
    known_cases = []
    for nid, node in knowledge_graph.nodes.items():
        if node.get("type") == "Case":
            c_name = clean_case_name(node.get("name", ""))
            known_cases.append((c_name, get_short_case_name(c_name)))
    for short_k, full_k in LegalQueryClassifier.CASES.items():
        known_cases.append((full_k, get_short_case_name(full_k)))

    unique_cases = {}
    for full_k, short_k in known_cases:
        if full_k and full_k not in unique_cases:
            unique_cases[full_k] = short_k

    # Step 3: Generic scan and replacement of false establishment attributions
    for doc_info in doctrine_canonical_map:
        d_name = doc_info["name"]
        primary_full = doc_info["primary_full"]
        canon_shorts = doc_info["canon_shorts"]
        canon_fulls = doc_info["canon_fulls"]

        # Build doctrine regex variants (e.g. "Basic Structure Doctrine" | "Basic Structure")
        d_terms = [re.escape(d_name)]
        core_d = re.sub(r"\s+(?:Doctrine|Principle|System)$", "", d_name, flags=re.IGNORECASE).strip()
        if core_d and core_d.lower() != d_name.lower():
            d_terms.append(re.escape(core_d))
        if "colorable" in core_d.lower():
            d_terms.append(re.escape(core_d.lower().replace("colorable", "colourable")))
        d_pattern = r"(?:" + "|".join(d_terms) + r")"

        for case_full, case_short in unique_cases.items():
            # Check if this case is canonical for this doctrine
            is_canon = any(
                case_short.lower() in c_full or c_short in case_full.lower()
                for c_full, c_short in zip(canon_fulls, canon_shorts)
            )
            if is_canon:
                continue

            case_terms = [re.escape(case_full), re.escape(case_short)]
            case_pat = r"(?:" + "|".join(case_terms) + r")"

            # Pattern A: "[Doctrine] [was/as] established/created by/in [Case]"
            pat_a = (
                rf"({d_pattern}\s*,?\s*(?:was|is)?\s*(?:as\s+)?(?:established|created|laid down|formulated|introduced)\s+(?:by|in)\s+)"
                rf"{case_pat}"
            )
            if re.search(pat_a, answer, flags=re.IGNORECASE):
                answer = re.sub(
                    pat_a,
                    rf"\1{primary_full} (not {case_short})",
                    answer,
                    flags=re.IGNORECASE
                )

            # Pattern B: "[Case] established/created [Doctrine]"
            pat_b = (
                rf"{case_pat}(\s+(?:established|created|laid down|formulated|introduced)\s+(?:the\s+)?{d_pattern})"
            )
            if re.search(pat_b, answer, flags=re.IGNORECASE):
                answer = re.sub(
                    pat_b,
                    rf"{primary_full} (not {case_short})\1",
                    answer,
                    flags=re.IGNORECASE
                )

            # Pattern C: "(by|in) [Case] ... creating/establishing [Doctrine]"
            pat_c = (
                rf"((?:by|in)\s+{case_pat}[^.\n]*?)(?:creating|establishing|laying down)\s+(?:the\s+)?{d_pattern}"
            )
            if re.search(pat_c, answer, flags=re.IGNORECASE):
                answer = re.sub(
                    pat_c,
                    rf"\1(whereas {d_name} was established by {primary_full})",
                    answer,
                    flags=re.IGNORECASE
                )

    # Step 4: Ensure Retrieval Consistency Check header exists and is truthful
    if contradiction_warning:
        # If contradiction was active, enforce that User premise supported is No
        if "* User premise supported: Yes" in answer:
            answer = answer.replace("* User premise supported: Yes", "* User premise supported: No")

    if "### Retrieval Consistency Check" not in answer:
        if contradiction_warning:
            header = (
                f"### Retrieval Consistency Check\n"
                f"* User premise supported: No\n"
                f"* Explanation: Authoritative constitutional precedents establish that the principle in question "
                f"governs under its canonical establishing authority, not the authority assumed in the user premise.\n\n"
            )
        else:
            header = (
                f"### Retrieval Consistency Check\n"
                f"* User premise supported: Yes\n"
                f"* Explanation: The legal principles discussed directly correspond with the authoritative constitutional precedents.\n\n"
            )
        answer = header + answer

    return answer
