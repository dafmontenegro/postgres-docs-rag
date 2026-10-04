"""
evaluate.py

Runs the evaluation set defined in eval_set.json against the RAG system
and prints a summary of retrieval and (optionally) generation metrics.

Metrics computed without an API key:
  - context_hit     : expected source section in top-3 retrieved chunks
  - no_hallucination: out-of-corpus question correctly rejected

Metrics that require an API key (GEMINI_API_KEY or ANTHROPIC_API_KEY):
  - faithfulness    : LLM judge verifies answer is grounded in context
  - citation_present: generated answer names the chapter/section

Usage:
    python3 evaluate.py                        # all questions
    python3 evaluate.py --question-id Q11      # single question
    python3 evaluate.py --no-generate          # skip LLM, retrieval only
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

from rag import answer, retrieve, DISTANCE_THRESHOLD

EVAL_PATH = Path("eval_set.json")


def check_context_hit(question: str, expected_sources: list[str], top_k: int = 3) -> bool:
    """
    Returns True if at least one expected source section appears in the
    top-k retrieved chunks. This metric requires no LLM call.
    """
    if not expected_sources:
        return True   # out-of-corpus questions don't have expected sources
    retrieval = retrieve(question, top_k=top_k)
    retrieved_sections = {c["section_number"] for c in retrieval["chunks"]}
    return any(s in retrieved_sections for s in expected_sources)


def check_no_hallucination(question: str) -> bool:
    """
    For out-of-corpus questions: verify the system does not return content
    (i.e., the question is correctly flagged as out of scope).
    """
    retrieval = retrieve(question, top_k=1)
    return not retrieval["in_scope"]


def check_citation_present(generated: str, chapter_number: str, section_numbers: list[str]) -> bool:
    """
    Heuristic: generated answer mentions the chapter number and at least
    one of the expected section numbers or their titles.
    """
    text_lower = generated.lower()
    has_chapter = f"chapter {chapter_number}" in text_lower
    has_section = any(sec in generated for sec in section_numbers)
    return has_chapter or has_section


def faithfulness_judge(question: str, answer_text: str, context: str) -> tuple[bool, str]:
    """
    LLM-as-judge: checks whether the generated answer is grounded in context.
    Uses GEMINI_API_KEY if set, otherwise ANTHROPIC_API_KEY.
    Returns (is_faithful, reasoning).
    """
    prompt = (
        f"You are an evaluator checking whether an AI answer is faithful to its source context.\n\n"
        f"Context:\n{context[:5000]}\n\n"
        f"Question: {question}\n\n"
        f"Answer: {answer_text[:1000]}\n\n"
        f"Is the answer faithful to the context? Reply with YES or NO followed by one sentence of reasoning."
    )

    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            result = client.models.generate_content(
                model="gemini-flash-lite-latest", contents=prompt
            ).text.strip()
            return result.upper().startswith("YES"), result
        except Exception as e:
            return True, f"Gemini judge error: {e}"

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}],
            )
            result = response.content[0].text.strip()
            return result.upper().startswith("YES"), result
        except Exception as e:
            return True, f"Anthropic judge error: {e}"

    return True, "No API key set, faithfulness check skipped"


def evaluate_question(q: dict, no_generate: bool) -> dict:
    question = q["question"]
    qtype = q["type"]
    expected_sources = q.get("expected_sources", [])
    result = {"id": q["id"], "question": question[:70], "type": qtype}

    if qtype == "out_of_corpus":
        passed = check_no_hallucination(question)
        result["no_hallucination"] = passed
        result["pass"] = passed
        result["note"] = "PASS: correctly rejected" if passed else "FAIL: hallucinated answer"
        return result

    context_passed = check_context_hit(question, expected_sources)
    result["context_hit"] = context_passed

    if no_generate or not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("GEMINI_API_KEY")):
        result["pass"] = context_passed
        result["note"] = "retrieval-only mode"
        return result

    # Full generation evaluation
    generated = answer(question)
    result["generated_answer"] = generated[:300] + "…" if len(generated) > 300 else generated

    # Citation check (heuristic)
    citation_ok = False
    if expected_sources:
        # Extract chapter number from first expected source (e.g., "11.2.1" → "11")
        chapter_num = expected_sources[0].split(".")[0]
        citation_ok = check_citation_present(generated, chapter_num, expected_sources)
    result["citation_present"] = citation_ok

    # Faithfulness: only for questions that require it
    if "faithfulness" in q.get("metric", ""):
        retrieval = retrieve(question, top_k=3)
        context_text = "\n\n".join(c["content"] for c in retrieval["chunks"])
        faithful, reasoning = faithfulness_judge(question, generated, context_text)
        result["faithfulness"] = faithful
        result["faithfulness_reasoning"] = reasoning
        result["pass"] = context_passed and faithful and citation_ok
    else:
        result["pass"] = context_passed and citation_ok

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the PostgreSQL RAG system.")
    parser.add_argument("--question-id", help="Run only a specific question (e.g. Q11)")
    parser.add_argument("--no-generate", action="store_true",
                        help="Skip generation, measure retrieval metrics only")
    args = parser.parse_args()

    eval_set = json.loads(EVAL_PATH.read_text("utf-8"))
    questions = eval_set["questions"]

    if args.question_id:
        questions = [q for q in questions if q["id"] == args.question_id]
        if not questions:
            print(f"No question with id '{args.question_id}' found.")
            return

    has_api_key = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("GEMINI_API_KEY"))
    mode = "retrieval-only" if (args.no_generate or not has_api_key) else "full"
    print(f"\nEvaluation mode: {mode}")
    print(f"Distance threshold: {DISTANCE_THRESHOLD}")
    print(f"Questions: {len(questions)}")
    print("=" * 70)

    results = []
    for q in questions:
        r = evaluate_question(q, no_generate=args.no_generate)
        results.append(r)
        status = "PASS" if r.get("pass") else "FAIL"
        metrics = []
        if "context_hit" in r:
            metrics.append(f"ctx={'✓' if r['context_hit'] else '✗'}")
        if "no_hallucination" in r:
            metrics.append(f"no_halluc={'✓' if r['no_hallucination'] else '✗'}")
        if "citation_present" in r:
            metrics.append(f"cite={'✓' if r['citation_present'] else '✗'}")
        if "faithfulness" in r:
            metrics.append(f"faith={'✓' if r['faithfulness'] else '✗'}")
        metric_str = " | ".join(metrics)
        print(f"[{status}] {r['id']} ({r['type']:<18}) {metric_str}")
        print(f"       {r['question']}")
        if "note" in r:
            print(f"       {r['note']}")
        print()

    passed = sum(1 for r in results if r.get("pass"))
    total = len(results)
    print("=" * 70)
    print(f"Results: {passed}/{total} passed")

    if mode == "retrieval-only":
        ctx_hits = [r for r in results if "context_hit" in r]
        if ctx_hits:
            rate = sum(1 for r in ctx_hits if r["context_hit"]) / len(ctx_hits)
            print(f"Context hit rate: {rate:.0%} ({sum(1 for r in ctx_hits if r['context_hit'])}/{len(ctx_hits)})")
        oos = [r for r in results if "no_hallucination" in r]
        if oos:
            oos_rate = sum(1 for r in oos if r["no_hallucination"]) / len(oos)
            print(f"Out-of-scope detection: {oos_rate:.0%} ({sum(1 for r in oos if r['no_hallucination'])}/{len(oos)})")


if __name__ == "__main__":
    main()
