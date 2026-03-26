# Patient Consent Form Simplifier

A command-line NLP tool that analyses medical patient consent forms and produces plain-English output a patient can act on. Built entirely on classical NLP techniques — no external libraries, no API calls.

---

## Motivation

Standard consent forms are written at a post-graduate reading level and dense with legal and medical terminology. Patients are asked to sign documents they cannot readily understand. This tool processes the raw form text and surfaces the information that actually matters: what is happening, what the risks are, what rights the patient holds, and what every unfamiliar term means.

---

## NLP Pipeline

Each form is processed through the following stages, all implemented from scratch in `nlp_engine.py`:

| Stage | Technique | Purpose |
|---|---|---|
| Tokenisation | Rule-based regex with abbreviation protection | Split text into words and sentences |
| Stopword filtering | Domain-aware stopword set | Remove high-frequency, low-signal tokens |
| Stemming | Porter stemmer (steps 1a, 1b, 2) | Reduce inflected forms to a common root |
| Readability | Flesch-Kincaid Grade Level | Drive complexity classification |
| TF-IDF | Background domain corpus weighting | Rank content-bearing terms |
| Jargon detection | Lexicon-based NER (50+ entries) | Identify and define medical/legal terms |
| Clause extraction | Regex pattern matching | Isolate risk disclosures and patient-rights statements |
| Summarisation | TextRank / Jaccard sentence similarity | Extract the most central sentences |
| Simplification | Rule-based lexical substitution map | Replace complex phrases with plain equivalents |
| Classification | FK grade + jargon density heuristic | Label form complexity and urgency |

---

## Output

For any consent form the tool produces five sections:

- **Procedure** — detected procedure name, complexity label, urgency label.
- **Summary** — three key sentences selected by TextRank centrality.
- **Risks** — sentences containing risk disclosures, extracted by pattern matching.
- **Your Rights** — refusal and withdrawal clauses extracted by pattern matching.
- **Medical Terms Explained** — every detected jargon term with a plain-English definition.
- **Plain-English Version** — full text rewritten via the lexical substitution map.

Results can optionally be exported as a structured JSON file.

---

## Project Structure

```
consent_nlp/
├── nlp_engine.py   # All NLP algorithms
├── app.py          # Pipeline orchestration and CLI
└── README.md
```

---

## Requirements

Python 3.10 or later. No third-party packages required.

---

## Usage

```bash
python app.py
```

At the prompt, choose to paste the consent form text directly or provide a path to a `.txt` file. Enter `END` on its own line to finish a paste.

**Example input** — paste the raw text of a consent form and press `END`:

```
INFORMED CONSENT FOR LAPAROSCOPIC APPENDECTOMY

The patient hereby consents to the performance of the following
operative procedure: Laparoscopic Appendectomy.
...
END
```

**Example output** (truncated):

```
══════════════════════════════════════════════════════════
  PROCEDURE
══════════════════════════════════════════════════════════
  Laparoscopic Appendectomy
  Complexity: Complex   |   Urgency: Routine

──────────────────────────────────────────────────────────
  RISKS
──────────────────────────────────────────────────────────

  • The risks associated with this procedure include such as:
    serious bleeding, infection, and rarely, death.

──────────────────────────────────────────────────────────
  MEDICAL TERMS EXPLAINED
──────────────────────────────────────────────────────────

  LAPAROSCOPIC
    minimally invasive (keyhole) surgery

  HEMORRHAGE
    serious bleeding
```

---

## Limitations

- Jargon detection is lexicon-bound; terms not in `MEDICAL_JARGON_LEXICON` will not be explained.
- Lexical simplification is phrase-level; it does not restructure grammar or shorten sentences beyond semicolon splitting.
- Summarisation is extractive; it selects existing sentences rather than generating new ones.
- Procedure name detection relies on common heading patterns and may fail on non-standard form layouts.
