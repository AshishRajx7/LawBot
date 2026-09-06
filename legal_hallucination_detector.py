# legal_hallucination_detector.py — Grounding & Hallucination Detection Engine with NLI
import re
import os
import numpy as np
from typing import List, Dict, Any, Set, Tuple, Optional

# Optional zero-PyTorch ONNX NLI imports
try:
    import onnxruntime as ort
    from tokenizers import Tokenizer
    from huggingface_hub import hf_hub_download
    HAS_ONNX_NLI = True
except ImportError:
    HAS_ONNX_NLI = False

class LegalHallucinationDetector:
    """
    Production Zero-PyTorch legal hallucination detector featuring:
    - Constitutional article validation (Articles 1-395 + lettered additions)
    - Constitutional amendment validation (1st to 106th Amendments)
    - Case citation grounding against retrieved materials
    - Pure-ONNX Natural Language Inference (NLI) for contradiction,
      false attribution, and unsupported claim detection
    - Sentence n-gram lexical overlap fallback
    - Source attribution mapping and confidence scoring (0-100%)
    """

    VALID_CONSTITUTIONAL_ARTICLES = {
        f"Article {i}" for i in range(1, 396)
    } | {
        "Article 21A", "Article 31A", "Article 31B", "Article 31C", "Article 51A",
        "Article 124A", "Article 124B", "Article 124C", "Article 243A", "Article 243B",
        "Article 243C", "Article 243D", "Article 243E", "Article 243F", "Article 243G",
        "Article 243H", "Article 243I", "Article 243J", "Article 243K", "Article 243L",
        "Article 243M", "Article 243N", "Article 243O", "Article 243P", "Article 243Q",
        "Article 243R", "Article 243S", "Article 243T", "Article 243U", "Article 243V",
        "Article 243W", "Article 243X", "Article 243Y", "Article 243Z", "Article 243ZA",
        "Article 243ZB", "Article 243ZC", "Article 243ZD", "Article 243ZE", "Article 243ZF",
        "Article 243ZG", "Article 300A", "Article 323A", "Article 323B", "Article 371A",
        "Article 371B", "Article 371C", "Article 371D", "Article 371E", "Article 371F",
        "Article 371G", "Article 371H", "Article 371I", "Article 371J"
    }

    # Constitutional Amendments passed in India (1st to 106th Amendment)
    VALID_AMENDMENTS = {str(i) for i in range(1, 107)}

    ARTICLE_REGEX = re.compile(r"\bArticle\s*([0-9]{1,3}[A-Za-z]?)\b", re.IGNORECASE)
    AMENDMENT_REGEX = re.compile(r"\b(\d{1,3})(?:st|nd|rd|th)?\s*(?:Constitutional\s*)?Amendment\b", re.IGNORECASE)
    CASE_REGEX = re.compile(r"\b([A-Z][A-Za-z\.\s\(\)]+?\s+(?:v\.|versus)\s+[A-Z][A-Za-z\.\s\(\)]+?)(?=[\,\;\:\n\.\(\)]|$)", re.IGNORECASE)

    def __init__(self, enable_nli: bool = True):
        self.enable_nli = enable_nli and HAS_ONNX_NLI
        self.nli_sess = None
        self.nli_tokenizer = None
        self._nli_initialized = False

    def _ensure_nli_loaded(self):
        """Lazily initializes NLI engine on first demand to minimize idle RAM"""
        if not self._nli_initialized:
            self._nli_initialized = True
            if self.enable_nli:
                self._init_nli_engine()

    def _init_nli_engine(self):
        """Initializes ultra-lightweight ONNX DeBERTa-v3 NLI engine (< 90MB, memory-optimized)"""
        try:
            repo_id = "onnx-community/nli-deberta-v3-xsmall-ONNX"
            model_path = hf_hub_download(repo_id, "onnx/model_quantized.onnx")
            tokenizer_path = hf_hub_download(repo_id, "tokenizer.json")
            
            self.nli_tokenizer = Tokenizer.from_file(tokenizer_path)
            self.nli_tokenizer.enable_truncation(max_length=96)

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            opts.enable_cpu_mem_arena = False  # Avoids large pre-allocated memory pool
            opts.enable_mem_pattern = False
            self.nli_sess = ort.InferenceSession(model_path, opts)
        except Exception:
            # Graceful fallback to deterministic lexical auditing
            self.nli_sess = None
            self.nli_tokenizer = None

    def compute_nli(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """
        Runs lightweight ONNX NLI on (premise, hypothesis).
        Returns probabilities for: contradiction, entailment, neutral.
        """
        self._ensure_nli_loaded()
        if not self.nli_sess or not self.nli_tokenizer:
            return {"contradiction": 0.0, "entailment": 0.5, "neutral": 0.5}

        try:
            enc = self.nli_tokenizer.encode(premise[:350], hypothesis[:180])
            input_ids = np.array([enc.ids], dtype=np.int64)
            att_mask = np.array([enc.attention_mask], dtype=np.int64)
            logits = self.nli_sess.run(["logits"], {"input_ids": input_ids, "attention_mask": att_mask})[0][0]
            exp = np.exp(logits - np.max(logits))
            probs = exp / np.sum(exp)
            # id2label: 0: contradiction, 1: entailment, 2: neutral
            return {
                "contradiction": float(probs[0]),
                "entailment": float(probs[1]),
                "neutral": float(probs[2])
            }
        except Exception:
            return {"contradiction": 0.0, "entailment": 0.5, "neutral": 0.5}

    def _clean_tokens(self, text: str) -> Set[str]:
        stop = {
            "the", "and", "that", "this", "with", "from", "for", "held", "court",
            "supreme", "india", "under", "which", "shall", "cannot", "also", "into",
            "have", "been", "where", "about", "other", "their", "these", "there",
            "article", "constitution", "constitutional", "right", "rights", "fundamental",
            "law", "case", "ruling", "ruled", "judgment", "per", "sec", "section",
            "are", "was", "were", "had", "has", "can", "may", "must", "any", "all"
        }
        tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        return {t for t in tokens if t not in stop}

    def audit(self, response_text: str, retrieved_sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Audits the generated LLM response against retrieved legal documents.
        """
        issues = []
        supported_sources = set()
        verified_authorities = []
        penalties = 0.0

        # Build concatenated source text and authority index
        retrieved_text_corpus = ""
        source_texts = []
        retrieved_titles = []
        retrieved_articles = set()

        for idx, src in enumerate(retrieved_sources, 1):
            s_name = f"SOURCE {idx}"
            content = src.get("content", "")
            title = src.get("title", "")
            primary_art = src.get("primary_article", "")
            
            retrieved_titles.append(title.lower())
            if primary_art:
                retrieved_articles.add(primary_art.lower())

            full_src_txt = f"{title} {src.get('citation', '')} {primary_art} {content}".lower()
            source_texts.append((s_name, full_src_txt))
            retrieved_text_corpus += " " + full_src_txt

        # 1. Constitutional Article Verification
        mentioned_articles = []
        for m in self.ARTICLE_REGEX.finditer(response_text):
            canonical = f"Article {m.group(1).upper()}"
            mentioned_articles.append(canonical)
            
            if canonical not in self.VALID_CONSTITUTIONAL_ARTICLES:
                issues.append(f"Fabricated constitutional article cited: '{canonical}' does not exist in the Constitution of India.")
                penalties += 35.0
            else:
                if canonical.lower() in retrieved_text_corpus:
                    verified_authorities.append(canonical)
                else:
                    issues.append(f"Unretrieved constitutional provision: '{canonical}' was cited in the answer but is not present in retrieved sources.")
                    penalties += 10.0

        # 2. Constitutional Amendment Verification (Task 4)
        for m in self.AMENDMENT_REGEX.finditer(response_text):
            num = m.group(1)
            raw_full = m.group(0).strip()
            
            if num not in self.VALID_AMENDMENTS:
                issues.append(f"Fabricated Constitutional Amendment: '{raw_full}' does not exist in the Constitution of India (valid amendments: 1st to 106th).")
                penalties += 20.0
            else:
                canon_amen = f"{num}th amendment"
                if num in retrieved_text_corpus or canon_amen in retrieved_text_corpus or "amend" in retrieved_text_corpus:
                    verified_authorities.append(raw_full)
                else:
                    issues.append(f"Unretrieved constitutional amendment: '{raw_full}' was cited but is not present in retrieved sources.")
                    penalties += 10.0

        # 3. Case Citation Verification
        mentioned_cases = []
        for m in self.CASE_REGEX.finditer(response_text):
            raw_case = m.group(1).strip()
            if len(raw_case) < 8 or "union of india" == raw_case.lower() or "state of" == raw_case.lower():
                continue
            mentioned_cases.append(raw_case)

            c_low = raw_case.lower()
            matched = False
            for t in retrieved_titles:
                if any(part in c_low for part in t.split(" v. ") if len(part) > 3):
                    matched = True
                    break

            if matched:
                verified_authorities.append(raw_case)
            else:
                issues.append(f"External/Unretrieved case citation: '{raw_case}' is referenced but was not present in the retrieved materials.")
                penalties += 15.0

        # 4. Sentence Grounding & NLI Verification (Task 5)
        clean_text = re.sub(r"^\s*#+.*$", "", response_text, flags=re.MULTILINE)
        clean_text = re.sub(r"^\s*\d+\..*$", "", clean_text, flags=re.MULTILINE)
        sentences = [s.strip() for s in re.split(r"(?<!\b[a-zA-Z]\.)(?<=[.!?\n])\s+", clean_text) if len(s.strip()) > 20]
        unsupported_statements = []

        for sent in sentences:
            sent_tokens = self._clean_tokens(sent)
            if not sent_tokens:
                continue

            # Lexical overlap check against all sources
            best_overlap = 0
            best_source = None
            best_source_txt = ""
            for s_name, s_txt in source_texts:
                overlap = sum(1 for tok in sent_tokens if tok in s_txt)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_source = s_name
                    best_source_txt = s_txt

            ratio = best_overlap / max(1, len(sent_tokens))
            is_assertion = any(w in sent.lower() for w in ("held", "ruled", "established", "declared", "concluded", "exempt", "unconstitutional", "struck down"))
            
            # --- NLI Deep Verification ---
            self._ensure_nli_loaded()
            nli_flagged = False
            if self.nli_sess and best_source_txt:
                # Compare against the best matching source passage
                nli_res = self.compute_nli(best_source_txt, sent)
                con_prob = nli_res["contradiction"]
                ent_prob = nli_res["entailment"]
                neu_prob = nli_res["neutral"]

                # Task 5 Rules:
                # Contradiction > 0.60
                if con_prob > 0.60:
                    nli_flagged = True
                    # Check if false attribution (mentions a case name while holding contradicts source)
                    case_short_names = [t.split(" v. ")[0].strip().lower() for t in retrieved_titles if " v. " in t]
                    sent_low = sent.lower()
                    has_case_mention = bool(
                        self.CASE_REGEX.search(sent) or
                        " v. " in sent or
                        " versus " in sent_low or
                        any(sn in sent_low for sn in case_short_names if len(sn) > 4)
                    )
                    if has_case_mention:
                        issues.append(f"False Legal Attribution detected: '{sent}'")
                        penalties += 20.0
                    else:
                        issues.append(f"Contradiction detected against retrieved authorities: '{sent}'")
                        penalties += 25.0
                    unsupported_statements.append(sent)

                # Neutral > 0.80 for assertions with low lexical grounding
                elif neu_prob > 0.80 and is_assertion and ratio < 0.35:
                    nli_flagged = True
                    issues.append(f"Unsupported claim lacking source grounding: '{sent}'")
                    unsupported_statements.append(sent)
                    penalties += 15.0

                # Entailment < 0.40 for assertive legal claims with low lexical overlap
                elif ent_prob < 0.40 and is_assertion and ratio < 0.35 and con_prob < 0.20:
                    penalties += 10.0

            # Lexical fallback check if not already flagged by NLI
            if not nli_flagged:
                threshold = 0.40 if is_assertion else 0.30
                if ratio >= threshold and best_source and best_overlap >= 3:
                    supported_sources.add(best_source)
                elif (ratio < 0.35 or best_overlap < 3) and is_assertion:
                    unsupported_statements.append(sent)
                    penalties += 15.0
                elif ratio < 0.20:
                    unsupported_statements.append(sent)
                    penalties += 8.0

        # 5. Final Confidence Score Calculation
        confidence_score = max(0.0, min(100.0, 100.0 - penalties))

        return {
            "confidence_score": round(confidence_score, 1),
            "supported_sources": sorted(list(supported_sources)),
            "potential_unsupported_statements": unsupported_statements[:3],
            "verified_authorities": list(set(verified_authorities)),
            "issues_detected": issues
        }

if __name__ == "__main__":
    detector = LegalHallucinationDetector()
    sources = [
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
            "content": "The Supreme Court established substantive due process under Article 21, requiring procedure established by law to be just, fair, and reasonable."
        }
    ]

    print("=== LegalHallucinationDetector Comprehensive Test ===")
    
    # 1. Grounded answer
    grounded = "The Supreme Court established the basic structure doctrine in Kesavananda Bharati v. State of Kerala, ruling that Article 368 does not grant unlimited power to destroy the Constitution."
    r1 = detector.audit(grounded, sources)
    print(f"\n1. Grounded -> Conf: {r1['confidence_score']}% | Issues: {r1['issues_detected']}")

    # 2. Contradiction
    contradiction = "Parliament can amend the basic structure of the Constitution without any judicial restrictions under Article 368."
    r2 = detector.audit(contradiction, sources)
    print(f"\n2. Contradiction -> Conf: {r2['confidence_score']}% | Issues: {r2['issues_detected']}")

    # 3. False Attribution
    false_attrib = "The basic structure doctrine was established in Maneka Gandhi v. Union of India."
    r3 = detector.audit(false_attrib, sources)
    print(f"\n3. False Attribution -> Conf: {r3['confidence_score']}% | Issues: {r3['issues_detected']}")

    # 4. Fabricated Amendment
    fab_amend = "Under the 189th Constitutional Amendment passed in 2024, the basic structure was abolished."
    r4 = detector.audit(fab_amend, sources)
    print(f"\n4. Fabricated Amendment -> Conf: {r4['confidence_score']}% | Issues: {r4['issues_detected']}")
