"""
nlp_engine.py — NLP pipeline for patient consent form analysis.

Implements the following techniques without external dependencies:
    - Rule-based sentence and word tokenisation
    - Stopword filtering and Porter stemming
    - TF-IDF scoring (with domain-specific background IDF weights)
    - Flesch-Kincaid grade-level readability
    - Lexicon-based named entity recognition for medical jargon
    - Regex pattern matching for risk and patient-rights clauses
    - TextRank extractive summarisation (Jaccard sentence similarity)
    - Rule-based lexical simplification via substitution map
    - Heuristic complexity and urgency classification
"""

import re
import math
from collections import Counter
from typing import List, Dict, Tuple

# Tokenisation

def tokenise_words(text: str) -> List[str]:
    """Return lowercase alphabetic tokens; strips punctuation and digits."""
    return re.findall(r"\b[a-z][a-z\-']*[a-z]\b|[a-z]", text.lower())


def tokenise_sentences(text: str) -> List[str]:
    """
    Split text into sentences using punctuation boundaries.
    Protects common title/abbreviation dots (Dr., Mr., etc.) from
    being treated as sentence terminals.
    """
    protected = re.sub(
        r"\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr|vs|etc|i\.e|e\.g|Fig|No|Vol)\.",
        lambda m: m.group().replace(".", "<DOT>"),
        text
    )
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", protected)
    return [s.replace("<DOT>", ".").strip() for s in parts if len(s) > 5]
 
# Stopword filtering
 
_STOPWORDS: set = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "that", "this", "these", "those", "i", "me", "my", "we", "our", "you",
    "your", "he", "she", "it", "they", "them", "their", "what", "which",
    "who", "whom", "when", "where", "why", "how", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such", "no", "not",
    "only", "same", "so", "than", "too", "very", "just", "also", "as",
    "if", "then", "because", "while", "although", "though", "however",
    # Legal boilerplate treated as stopwords for scoring purposes
    "hereby", "herein", "thereof", "thereto", "hereunder",
    "aforementioned", "pursuant", "notwithstanding", "whereas",
}

def filter_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]

# Porter stemmer

class PorterStemmer:
    """
    Lightweight Porter stemmer (steps 1a, 1b, 2).
    Reduces inflected forms to a common root for improved term matching.
    """

    def _contains_vowel(self, stem: str) -> bool:
        return bool(re.search(r"[aeiou]", stem))

    def _ends_double_consonant(self, stem: str) -> bool:
        return (len(stem) >= 2
                and stem[-1] == stem[-2]
                and stem[-1] not in "aeiou")

    def stem(self, word: str) -> str:
        if len(word) <= 3:
            return word

        # Step 1a — plural/third-person singular endings
        if word.endswith("sses"):
            word = word[:-2]
        elif word.endswith("ies"):
            word = word[:-2]
        elif word.endswith("ss"):
            pass
        elif word.endswith("s"):
            word = word[:-1]

        # Step 1b — past tense and progressive
        if word.endswith("eed"):
            if len(word) > 4:
                word = word[:-1]
        elif word.endswith("ing") and self._contains_vowel(word[:-3]):
            word = word[:-3]
            if word.endswith(("at", "bl", "iz")):
                word += "e"
            elif self._ends_double_consonant(word) and word[-1] not in "lsz":
                word = word[:-1]
        elif word.endswith("ed") and self._contains_vowel(word[:-2]):
            word = word[:-2]

        # Step 2 — derivational suffixes
        suffix_map = [
            ("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
            ("anci", "ance"), ("izer", "ize"),  ("alism", "al"),
            ("ation", "ate"), ("ator", "ate"),  ("ness", ""),
            ("ment", ""),     ("ful", ""),       ("ous", ""),
            ("ive", ""),      ("ize", ""),        ("ise", ""),
            ("ity", ""),      ("ly", ""),
        ]
        for suffix, replacement in suffix_map:
            if word.endswith(suffix) and len(word) - len(suffix) > 2:
                word = word[:-len(suffix)] + replacement
                break

        return word


_stemmer = PorterStemmer()

# TF-IDF scoring
 
# Approximate document-frequency values for common medical/consent vocabulary.
# Terms absent from this table are treated as rare (DF = 1), receiving a high IDF.
_DOMAIN_BACKGROUND_DF: Dict[str, float] = {
    "patient": 0.6, "consent": 0.3, "procedure": 0.4, "treatment": 0.5,
    "medical": 0.5, "doctor":  0.6, "hospital":  0.5, "health":    0.6,
    "form":    0.7, "sign":    0.7, "agree":     0.7, "understand": 0.7,
    "right":   0.6, "provide": 0.7, "include":   0.7, "perform":   0.5,
    "surgery": 0.3, "operation": 0.4, "risk":    0.4,
}
_BACKGROUND_CORPUS_SIZE = 10_000

def score_tfidf(tokens: List[str], top_n: int = 15) -> List[Tuple[str, float]]:
    """
    Compute TF-IDF scores for a token list.
    IDF is calculated against a small domain background corpus so that
    specialised medical/legal terms rank above everyday vocabulary.
    Returns up to top_n (term, score) pairs in descending order.
    """
    content_tokens = filter_stopwords(tokens)
    stemmed_tokens = [_stemmer.stem(t) for t in content_tokens]

    term_freq  = Counter(stemmed_tokens)
    total_terms = sum(term_freq.values()) or 1

    tfidf_scores: Dict[str, float] = {}
    for term, count in term_freq.items():
        tf  = count / total_terms
        df  = _DOMAIN_BACKGROUND_DF.get(term, 1)
        idf = math.log((_BACKGROUND_CORPUS_SIZE + 1) / (df + 1)) + 1
        tfidf_scores[term] = tf * idf

    return sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
 
# Readability — Flesch-Kincaid grade level

def _count_syllables(word: str) -> int:
    """
    Estimate syllable count via vowel-group heuristic.
    Applies a length-based bonus for long Latin/Greek medical terms.
    """
    word = word.lower().strip(".,!?;:")
    if len(word) <= 3:
        return 1

    syllable_count = len(re.findall(r"[aeiou]+", word))

    if word.endswith("e") and syllable_count > 1:
        syllable_count -= 1
    if word.endswith(("ia", "io", "ious", "eal", "iel")):
        syllable_count += 1
    if len(word) > 8:
        syllable_count += (len(word) - 8) // 4

    return max(1, syllable_count)


def flesch_kincaid_grade(text: str) -> float:
    """
    Flesch-Kincaid Grade Level:
        FK = 0.39 × (words / sentences) + 11.8 × (syllables / words) − 15.59
    Used internally to drive complexity classification; not shown to the user.
    """
    sentences = tokenise_sentences(text)
    words     = tokenise_words(text)
    if not sentences or not words:
        return 0.0

    total_syllables  = sum(_count_syllables(w) for w in words)
    avg_sent_length  = len(words) / len(sentences)
    avg_syllables_pw = total_syllables / len(words)

    grade = 0.39 * avg_sent_length + 11.8 * avg_syllables_pw - 15.59
    return round(max(0.0, grade), 1)

# Medical jargon detection — lexicon-based NER
 
# Maps medical/legal terms to plain-English definitions.
# Multi-word entries are matched before single-word entries at the call site.
MEDICAL_JARGON_LEXICON: Dict[str, str] = {
    # Procedures
    "laparoscopic":    "minimally invasive (keyhole) surgery",
    "cholecystectomy": "gallbladder removal",
    "appendectomy":    "appendix removal",
    "colonoscopy":     "camera examination of the large intestine",
    "endoscopy":       "camera examination of the digestive tract",
    "angioplasty":     "procedure to open blocked arteries",
    "biopsy":          "removal of a small tissue sample for testing",
    "intubation":      "insertion of a breathing tube into the airway",
    "catheterization": "insertion of a thin tube into a body cavity",

    # Conditions and anatomy
    "hemorrhage":  "serious bleeding",
    "thrombosis":  "blood clot forming inside a vessel",
    "embolism":    "blockage of a blood vessel by a clot or air bubble",
    "pulmonary":   "relating to the lungs",
    "cardiac":     "relating to the heart",
    "myocardial":  "relating to the heart muscle",
    "ischemia":    "insufficient blood supply to a tissue",
    "sepsis":      "life-threatening infection spread through the bloodstream",
    "edema":       "swelling caused by fluid buildup",
    "hematoma":    "localised collection of blood outside vessels",
    "necrosis":    "death of body tissue",
    "fibrosis":    "thickening and scarring of tissue",
    "stenosis":    "abnormal narrowing of a passage",
    "perforation": "a hole or tear in an organ",

    # Pharmacology
    "anesthesia":       "medication that causes loss of feeling or consciousness",
    "anesthetic":       "drug used to prevent pain during a procedure",
    "analgesic":        "pain-relieving medication",
    "anticoagulant":    "blood-thinning medication",
    "corticosteroid":   "anti-inflammatory steroid medication",
    "contraindication": "a reason not to use a particular treatment",
    "prophylaxis":      "preventive treatment or medication",
    "sedation":         "use of medication to calm or induce sleep",

    # Clinical and legal terms
    "informed consent": "your voluntary agreement after understanding the risks",
    "operative":        "relating to surgery",
    "postoperative":    "after surgery",
    "preoperative":     "before surgery",
    "intraoperative":   "during surgery",
    "morbidity":        "risk of illness or complications",
    "mortality":        "risk of death",
    "prognosis":        "expected outcome of a disease or treatment",
    "malignant":        "cancerous or life-threatening",
    "benign":           "not cancerous",
    "contraindicated":  "not recommended due to potential harm",
    "iatrogenic":       "caused unintentionally by medical treatment",
    "adverse":          "harmful or undesirable",

    # Multi-word anatomical and clinical phrases
    "bile duct":             "tube carrying digestive fluid from the liver to the intestine",
    "deep vein":             "vein located deep within a muscle",
    "pulmonary embolism":    "blood clot in the lungs",
    "deep vein thrombosis":  "blood clot in a deep vein",
    "general anesthesia":    "medication that renders you fully unconscious",
    "local anesthesia":      "medication that numbs only a specific area",
}

def detect_jargon(text: str) -> Dict[str, str]:
    """
    Scan text for entries in MEDICAL_JARGON_LEXICON using whole-word regex.
    Returns {detected_term: plain_english_definition} for all matches.
    """
    lowered = text.lower()
    return {
        term: definition
        for term, definition in MEDICAL_JARGON_LEXICON.items()
        if re.search(r"\b" + re.escape(term) + r"\b", lowered)
    }
 
# Risk and patient-rights clause extraction — pattern matching

_RISK_PATTERNS: List[str] = [
    r"risks?\s+(?:include|associated|of|include but are not limited to)[^.]*\.",
    r"complications?\s+(?:include|may include|such as)[^.]*\.",
    r"(?:may|might|could)\s+result\s+in[^.]*\.",
    r"(?:rare|serious|significant)\s+risk[^.]*\.",
    r"in\s+rare\s+cases[^.]*\.",
    r"(?:death|mortality|fatal)[^.]*\.",
    r"including\s+but\s+not\s+limited\s+to[^.]*\.",
    r"potential\s+(?:risks?|complications?|side\s+effects?)[^.]*\.",
]

_RIGHTS_PATTERNS: List[str] = [
    r"right\s+to\s+refuse[^.]*\.",
    r"right\s+to\s+withdraw[^.]*\.",
    r"may\s+revoke[^.]*\.",
    r"consent\s+may\s+be\s+(?:withdrawn|revoked)[^.]*\.",
    r"you\s+(?:have|retain)\s+the\s+right[^.]*\.",
    r"voluntary[^.]*\.",
    r"at\s+any\s+time[^.]*(?:refuse|withdraw|revoke)[^.]*\.",
    # Modern consent form patterns — acknowledge/confirm/verify phrasing
    r"(?:confirm|affirm|verify)\s+that[^.]*(?:explained|discussed|understood)[^.]*\.",
    r"(?:opportunity|chance)\s+to\s+(?:ask|query|question)[^.]*\.",
    r"(?:questions?|inquir)[^.]*(?:answered|addressed|satisfied)[^.]*\.",
    r"right\s+to\s+(?:seek|obtain)\s+(?:a\s+)?second\s+opinion[^.]*\.",
    r"informed\s+of[^.]*(?:risks?|benefits?|alternatives?)[^.]*\.",
]

_SIGNATURE_PATTERN = re.compile(
    r"_{4,}|signature|printed\s+name|relationship|witness|date:\s*_|time:\s*_",
    re.IGNORECASE
)

def _is_signature_line(sentence: str) -> bool:
    """Return True if the sentence is part of a signature/witness block."""
    return bool(_SIGNATURE_PATTERN.search(sentence))

def _match_sentences(text: str, patterns: List[str]) -> List[str]:
    """Return sentences matching any pattern, excluding signature-block lines."""
    matched: dict = {}
    for sentence in tokenise_sentences(text):
        if _is_signature_line(sentence):
            continue
        for pattern in patterns:
            if re.search(pattern, sentence.lower()):
                matched[sentence.strip()] = None
                break
    return list(matched)

def extract_risk_clauses(text: str) -> List[str]:
    """
    Extract risk-disclosure sentences. Also parses inline comma-separated
    risk lists that follow a 'risks include:' heading.
    """
    clauses = _match_sentences(text, _RISK_PATTERNS)

    inline_list = re.search(
        r"(?:risks?|complications?)[\s\w]*include[^:]*:\s*([^.]+\.)",
        text, re.IGNORECASE
    )
    if inline_list:
        items = [item.strip() for item in inline_list.group(1).split(",")]
        clauses.extend(item for item in items if len(item.split()) > 2)

    return list(dict.fromkeys(clauses))

def extract_rights_clauses(text: str) -> List[str]:
    """Extract sentences describing patient rights (refusal, withdrawal)."""
    return _match_sentences(text, _RIGHTS_PATTERNS)


# TextRank extractive summarisation
 
def _jaccard_similarity(tokens_a: List[str], tokens_b: List[str]) -> float:
    """
    Jaccard similarity over filtered token sets:
        J(A, B) = |A ∩ B| / |A ∪ B|
    """
    set_a = set(filter_stopwords(tokens_a))
    set_b = set(filter_stopwords(tokens_b))
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def extractive_summarise(text: str, num_sentences: int = 3) -> str:
    """
    TextRank-inspired extractive summarisation:
        1. Build an N×N Jaccard similarity matrix across all sentences.
        2. Score each sentence by mean similarity to all others (centrality).
        3. Apply a 1.3× boost to sentences matching risk or rights patterns.
        4. Return the top-k sentences in their original document order.
    """
    sentences = tokenise_sentences(text)
    if len(sentences) <= num_sentences:
        return text

    tokenised = [tokenise_words(s) for s in sentences]
    n = len(sentences)

    sim_matrix = [
        [_jaccard_similarity(tokenised[i], tokenised[j]) if i != j else 0.0
         for j in range(n)]
        for i in range(n)
    ]

    centrality_scores = [sum(sim_matrix[i]) / (n - 1) for i in range(n)]

    for i, sentence in enumerate(sentences):
        for pattern in _RISK_PATTERNS + _RIGHTS_PATTERNS:
            if re.search(pattern, sentence.lower()):
                centrality_scores[i] *= 1.3
                break

    top_indices = sorted(
        sorted(range(n), key=lambda i: centrality_scores[i], reverse=True)[:num_sentences]
    )
    return " ".join(sentences[i] for i in top_indices)

 
# Lexical simplification — rule-based substitution
 
# Substitution pairs applied longest-first to prevent partial-match collisions.
_SIMPLIFICATION_MAP: Dict[str, str] = {
    # Legal boilerplate
    "in witness whereof":           "by signing below",
    "including but not limited to": "such as",
    "administration of anesthesia": "giving you anesthesia",
    "qualified medical personnel":  "trained medical staff",
    "unforeseen conditions":        "unexpected problems",
    "professional judgment":        "medical opinion",
    "in accordance with":   "following",
    "for the purpose of":   "to",
    "in the event of":      "if",
    "with respect to":      "about",
    "at such time as":      "when",
    "subsequent to":        "after",
    "prior to":             "before",
    "pursuant to":          "following",
    "in lieu of":           "instead of",
    "in order to":          "to",
    "set forth":            "described",
    "duly authorized":      "approved",
    "notwithstanding":      "despite",
    "aforementioned":       "previously mentioned",
    "hereunder":            "under this agreement",
    "whereas":              "because",
    "herein":               "in this document",
    "thereof":              "of this",
    "thereto":              "to this",
    "hereby":               "",
    "necessitate":          "require",
    "acknowledge":          "agree",
    "authorize":            "allow",
    "consent to":           "agree to",

    # Medical terms
    "pulmonary embolism":   "a blood clot in the lungs",
    "deep vein thrombosis": "a blood clot in a deep vein",
    "adverse reaction":     "harmful reaction",
    "adverse effects":      "harmful side effects",
    "postoperative":        "after the operation",
    "preoperative":         "before the operation",
    "intraoperative":       "during the operation",
    "operative":            "surgical",
    "hemorrhage":           "serious bleeding",
    "anesthetic":           "numbing or sleep medicine",
    "anesthesia":           "medicine that prevents pain or consciousness",
    "in rare cases":        "rarely",
}

def simplify_text(text: str) -> str:
    """
    Apply _SIMPLIFICATION_MAP substitutions (longest match first),
    split semicolon-joined clauses into separate sentences,
    and remove signature/witness block lines before returning.
    """
    output = text

    for phrase, replacement in sorted(
        _SIMPLIFICATION_MAP.items(), key=lambda x: len(x[0]), reverse=True
    ):
        output = re.compile(re.escape(phrase), re.IGNORECASE).sub(replacement, output)

    output = re.sub(r";\s*", ". ", output)
    output = re.sub(r"  +", " ", output)
    output = re.sub(r"\.\s*\.", ".", output)

    # Remove signature/witness block lines
    cleaned_lines = [
        line for line in output.splitlines()
        if not _is_signature_line(line)
    ]
    return "\n".join(cleaned_lines).strip()
 
# Classification — complexity and urgency
 
def classify_complexity(text: str, fk_grade: float) -> str:
    """
    Classify form complexity as Simple / Moderate / Complex.
    Uses Flesch-Kincaid grade combined with jargon density per 100 words.
    """
    word_count         = len(tokenise_words(text)) or 1
    jargon_count       = len(detect_jargon(text))
    jargon_per_hundred = jargon_count / (word_count / 100)

    if fk_grade >= 16 or jargon_per_hundred >= 3:
        return "Complex"
    if fk_grade >= 12 or jargon_per_hundred >= 1.5:
        return "Moderate"
    return "Simple"

def classify_urgency(text: str) -> str:
    """
    Classify urgency as Routine / Time-sensitive / Emergency.
    Only sentences in procedural/indication sections are examined to avoid
    false positives from keywords appearing in benefits or alternatives text.
    """
    # Restrict keyword scanning to the first 40% of the document, where
    # indication and urgency language appears in standard form layouts.
    scan_window = text[: int(len(text) * 0.4)].lower()

    emergency_keywords      = ["emergency procedure", "life-threatening",
                                "urgent procedure", "emergent", "without delay",
                                "immediate intervention"]
    time_sensitive_keywords = ["within 24 hours", "same-day", "time-sensitive",
                                "urgent elective", "soon as possible"]

    if any(kw in scan_window for kw in emergency_keywords):
        return "Emergency"
    if any(kw in scan_window for kw in time_sensitive_keywords):
        return "Time-sensitive"
    return "Routine"

def detect_procedure_name(text: str) -> str:
    """
    Heuristic extraction of the procedure name from the document heading.
    Tries patterns in priority order; strips trailing section headings
    (e.g. 'DESCRIPTION', 'INFORMATION') before returning.
    Falls back to 'Medical Procedure' if no match is found.
    """
    heading_patterns = [
        r"(?:proposed\s+)?intervention\s+is\s+([A-Za-z\s\(\),\-]+?)(?:\n|\.|,\s+involving)",
        r"consent\s+for\s+([A-Za-z\s\(\)\-]+?)(?:\n|\.|,)",
        r"following\s+(?:operative\s+)?procedure[^:]*:\s*([A-Za-z\s\(\)\-]+?)(?:\n|\.|,)",
        r"oncology\s+protocol[:\s]+([A-Za-z\s\(\)\-\/]+?)(?:\n|\.|for\b)",
        # Sentence-level: 'Right heart catheterization via...' or 'Coronary angiography...'
        r"^(?:procedure\s+description\s+)?([A-Z][A-Za-z\s\(\)\-]+?(?:catheteriz|angiograph|ventriculograph|ectomy|oscopy|plasty|otomy|ostomy)[A-Za-z\s\(\)\-]*?)(?:\s+via|\s+under|\s+using|\s+with|\.|,)",
        r"procedure[:\s]+([A-Za-z\s\(\)\-]+?)(?:\n|\.|,)",
        r"treatment[:\s]+([A-Za-z\s\(\)\-]+?)(?:\n|\.|,)",
        r"operation[:\s]+([A-Za-z\s\(\)\-]+?)(?:\n|\.|,)",
    ]

    # Section headings that appear after a colon but are not procedure names
    _SECTION_WORDS = {"description", "information", "details", "section",
                      "summary", "overview", "consent", "acknowledgment"}

    for pattern in heading_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            name = match.group(1).strip().rstrip(",")
            # Strip a leading section-heading word left by the generic fallback pattern
            name = re.sub(r"^(?:description|information|details|section|summary|overview)\s+",
                          "", name, flags=re.IGNORECASE).strip()
            if name.lower() in _SECTION_WORDS:
                continue
            if 3 < len(name) < 100:
                return name

    return "Medical Procedure"