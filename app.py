"""
app.py — Patient Consent Form Simplifier entry point.

Orchestrates the NLP pipeline from nlp_engine and presents results
through a terminal interface. Accepts input via stdin (paste) or file path.
"""

import json
import sys
import traceback
from nlp_engine import (
    flesch_kincaid_grade,
    detect_jargon,
    extract_risk_clauses,
    extract_rights_clauses,
    extractive_summarise,
    simplify_text,
    classify_complexity,
    classify_urgency,
    detect_procedure_name,
)

_DIVIDER      = "─" * 58
_DIVIDER_BOLD = "═" * 58
_WRAP_WIDTH   = 68

# NLP pipeline

class ConsentFormAnalyzer:
    """Runs the full NLP pipeline on a consent form and returns structured output."""

    def analyse(self, raw_text: str) -> dict:
        if not raw_text or len(raw_text.strip()) < 30:
            raise ValueError("Input text is too short to analyse.")

        text = raw_text.strip()

        fk_grade   = flesch_kincaid_grade(text)
        procedure  = detect_procedure_name(text)
        complexity = classify_complexity(text, fk_grade)
        urgency    = classify_urgency(text)
        jargon     = detect_jargon(text)
        risks      = extract_risk_clauses(text)
        rights     = extract_rights_clauses(text)
        summary    = extractive_summarise(text, num_sentences=3)
        simplified = simplify_text(text)

        return {
            "procedure"  : procedure,
            "complexity" : complexity,
            "urgency"    : urgency,
            "jargon"     : jargon,
            "risks"      : risks,
            "rights"     : rights,
            "summary"    : summary,
            "simplified" : simplified,
        }

# Output formatting

def _wrap(text: str, indent: str = "  ") -> str:
    """Word-wrap text to _WRAP_WIDTH, preserving the given indent per line."""
    words = text.split()
    lines, current_line = [], []
    for word in words:
        current_line.append(word)
        if len(" ".join(current_line)) > _WRAP_WIDTH:
            lines.append(indent + " ".join(current_line[:-1]))
            current_line = [word]
    if current_line:
        lines.append(indent + " ".join(current_line))
    return "\n".join(lines)


def _section(heading: str, bold: bool = False) -> None:
    divider = _DIVIDER_BOLD if bold else _DIVIDER
    print(f"\n{divider}\n  {heading}\n{divider}")


def render_results(analysis: dict) -> None:
    _section("PROCEDURE", bold=True)
    print(f"  {analysis['procedure']}")
    print(f"  Complexity: {analysis['complexity']}   |   Urgency: {analysis['urgency']}")

    _section("SUMMARY")
    print()
    print(_wrap(analysis["summary"]))
    print()

    _section("RISKS")
    meaningful_risks = [r for r in analysis["risks"] if len(r.split()) > 4][:3]
    if meaningful_risks:
        for risk in meaningful_risks:
            print(f"\n  • {_wrap(risk, indent='    ')}")
    else:
        print("\n  No explicit risk clauses identified.")
    print()

    _section("YOUR RIGHTS")
    if analysis["rights"]:
        for clause in analysis["rights"]:
            print(f"\n  • {_wrap(clause, indent='    ')}")
    else:
        print("\n  No patient-rights clauses identified.")
    print()

    _section("MEDICAL TERMS EXPLAINED")
    if analysis["jargon"]:
        for term, definition in analysis["jargon"].items():
            print(f"\n  {term.upper()}")
            print(f"    {definition}")
    else:
        print("\n  No medical jargon detected.")
    print()

    _section("PLAIN-ENGLISH VERSION")
    print()
    for sentence in analysis["simplified"].split(". "):
        sentence = sentence.strip()
        if len(sentence.split()) > 3:
            if not sentence.endswith("."):
                sentence += "."
            print(_wrap(sentence))
    print()

    print(_DIVIDER_BOLD + "\n")

# Input handling

def _read_from_stdin() -> str:
    """Collect multi-line paste from stdin; terminated by a line containing only 'END'."""
    print("\n  Paste the consent form text below.")
    print("  Enter END on its own line when finished.\n")
    lines = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


def _read_from_file() -> str:
    file_path = input("\n  File path: ").strip()
    with open(file_path, "r", encoding="utf-8") as fh:
        content = fh.read()
    print(f"  Loaded {len(content):,} characters from {file_path}")
    return content


def prompt_for_input() -> str | None:
    """
    Present input options and return the consent form text,
    or None if the user chooses to quit.
    """
    print("  Input options:")
    print("  [1] Paste text")
    print("  [2] Load from file")
    print("  [q] Quit\n")

    choice = input("  Choice: ").strip().lower()

    if choice == "q":
        return None
    if choice == "1":
        return _read_from_stdin()
    if choice == "2":
        return _read_from_file()

    print("  Invalid choice.")
    return prompt_for_input()

# Entry point

def main() -> None:
    print(f"\n{_DIVIDER_BOLD}")
    print("  PATIENT CONSENT FORM SIMPLIFIER")
    print(f"{_DIVIDER_BOLD}\n")

    analyzer = ConsentFormAnalyzer()

    while True:
        raw_text = prompt_for_input()
        if raw_text is None:
            print("\n  Exiting.\n")
            sys.exit(0)

        print("\n  Analysing...\n")

        try:
            analysis = analyzer.analyse(raw_text)
            render_results(analysis)

            if input("  Save results to JSON? [y/N]: ").strip().lower() == "y":
                output_path = "consent_analysis.json"
                with open(output_path, "w", encoding="utf-8") as fh:
                    json.dump(analysis, fh, indent=2, ensure_ascii=False)
                print(f"  Saved to {output_path}\n")

        except ValueError as exc:
            print(f"\n  Error: {exc}\n")
        except FileNotFoundError as exc:
            print(f"\n  File not found: {exc}\n")
        except Exception:
            traceback.print_exc()

        if input("  Analyse another form? [y/N]: ").strip().lower() != "y":
            print("\n  Exiting.\n")
            break


if __name__ == "__main__":
    main()
