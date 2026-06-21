"""
Grade one or multiple model answers against a rubric using a judge.

Input:  --answer   answer.txt (or JSON array of answers)
        --rubric   rubric.json (or JSON array of rubrics)
        --model    "moonshotai/Kimi-K2-Instruct"
        --out      grades.json
"""

import asyncio
import json
import re
import argparse
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

MODEL   = os.environ["ENRICHMENT_MODEL"]
URL     = os.environ["ENRICHMENT_BASE_URL"]
API_KEY = os.environ["ENRICHMENT_API_KEY"]

MAX_TOKENS        = 2048
CONCURRENCY_LIMIT = 16
MAX_RETRIES       = 5


# ── Prompts ───────────────────────────────────────────────────────────────

JUDGE_SYSTEM = """\
You are a strict financial analysis grader. You will receive:
1. A question about an S-1 filing
2. A model answer
3. A single criterion (an atomic fact)

Decide whether the answer explicitly addresses the criterion.
MET   — the answer directly states or clearly implies the criterion.
UNMET — the answer does not address the criterion, addresses it incorrectly, \
        or is too vague.

Return ONLY a JSON object. No preamble, no fences:
{"verdict": "MET" | "UNMET", "reason": "one sentence explanation"}
"""

JUDGE_USER = """\
Question: {question}

Answer:
{answer}

Criterion: {criterion}

Is the criterion MET or UNMET?
"""

HALLUCINATION_SYSTEM = """\
You are a fact-checking assistant for financial filings.
Decide whether the answer makes ANY numerical claim that is NOT supported \
by the supplied ground-truth facts.

Rules:
- Arithmetic derived directly from the provided facts is acceptable.
  Example: if facts state Revenue = 100 and Cost = 40, a stated Margin of 60% is fine.
- Only flag figures that could not be computed from the supplied facts.

Return ONLY:
{"hallucination": true | false, "reason": "one sentence"}
"""

HALLUCINATION_USER = """\
Question: {question}

Answer:
{answer}

Ground-truth facts:
{facts_block}

Does the answer contain numerical claims NOT found in, or directly derivable from, \
the ground-truth facts?
"""


# ── JSON extraction ───────────────────────────────────────────────────────

def extract_json(raw: str) -> dict:
    """
    Strip markdown fences, then parse.
    Falls back to first { ... last } slice if direct parse fails.
    Uses rfind to avoid the greedy multi-object trap from re.search.
    """
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw)
        raw = raw.replace("```", "").strip()

    try:
        return json.loads(raw)
    except Exception:
        pass

    start = raw.find("{")
    end   = raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in response: {raw!r}")

    return json.loads(raw[start : end + 1])


# ── Helper loaders ────────────────────────────────────────────────────────

def load_json_or_text(path_or_text: str):
    p = Path(path_or_text)
    if p.exists():
        text = p.read_text(encoding="utf-8")
        try:
            return json.loads(text)
        except Exception:
            return text
    return path_or_text


def extract_answer_text(answer_record) -> str:
    """
    Flexibly extract the answer string from a record that may use
    different field names depending on the benchmark output format.
    """
    if isinstance(answer_record, dict):
        return (
            answer_record.get("answer")
            or answer_record.get("model_answer")
            or answer_record.get("response")
            or answer_record.get("output")
            or ""
        )
    return str(answer_record)


# ── HTTP call with exponential-backoff retry ──────────────────────────────

async def call_api(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    api_key: str,
    payload: dict,
) -> str:
    """
    Single rate-limited API call with retries.
    Returns the raw message content string.
    Raises RuntimeError after MAX_RETRIES exhausted.
    """
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            async with sem:
                resp = await client.post(
                    URL,
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=payload,
                    timeout=120,
                )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            last_exc = exc
            wait = 2 ** attempt
            print(f"  [retry {attempt + 1}/{MAX_RETRIES}] {type(exc).__name__}: {exc} — waiting {wait}s")
            await asyncio.sleep(wait)
    raise RuntimeError(f"API call failed after {MAX_RETRIES} retries") from last_exc


# ── Single criterion judge call ───────────────────────────────────────────

async def judge_criterion(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    api_key: str,
    question: str,
    answer: str,
    criterion: str,
) -> dict:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user",   "content": JUDGE_USER.format(
                question=question,
                answer=answer,
                criterion=criterion,
            )},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
    }
    raw    = await call_api(client, sem, api_key, payload)
    parsed = extract_json(raw)

    verdict = parsed.get("verdict", "UNMET")
    if verdict not in ("MET", "UNMET"):
        verdict = "UNMET"

    return {
        "met":    verdict == "MET",
        "reason": parsed.get("reason", ""),
    }


# ── Hallucination judge ───────────────────────────────────────────────────

async def judge_hallucination(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    api_key: str,
    question: str,
    answer: str,
    all_facts: list[str],
) -> bool:
    """Returns True if the judge detects unsupported numerical claims."""
    facts_block = "\n".join(f"- {f}" for f in all_facts)
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": HALLUCINATION_SYSTEM},
            {"role": "user",   "content": HALLUCINATION_USER.format(
                question=question,
                answer=answer,
                facts_block=facts_block,
            )},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
    }
    raw    = await call_api(client, sem, api_key, payload)
    parsed = extract_json(raw)
    return bool(parsed.get("hallucination", False))


# ── Score calculation ─────────────────────────────────────────────────────

def calculate_score(
    verdicts: dict,
    weights: dict,
    hallucination_detected: bool,
) -> dict:
    """
    verdicts: {"critical": [{"met": bool}, ...], "important": [...], "optional": [...]}
    weights:  {"critical": 12.0, "important": 6.0, "optional": 2.0, "hallucination": -10.0}

    Hallucination deducts from raw points before normalization.
    abs() ensures the stored sign (positive or negative) doesn't matter.
    """
    breakdown = {}
    raw_score = 0.0
    max_score = 0.0

    for tier in ("critical", "important", "optional"):
        w        = weights.get(tier, 0.0)
        criteria = verdicts.get(tier, [])
        earned   = sum(w for c in criteria if c["met"])
        possible = len(criteria) * w

        breakdown[f"{tier}_earned"]   = earned
        breakdown[f"{tier}_possible"] = possible
        raw_score += earned
        max_score += possible

    # Always treat as a deduction regardless of stored sign
    penalty = -abs(weights.get("hallucination", 0.0)) if hallucination_detected else 0.0
    breakdown["hallucination_penalty"] = penalty
    raw_score = max(0.0, raw_score + penalty)

    breakdown["raw_score"]  = raw_score
    breakdown["max_score"]  = max_score
    breakdown["normalized"] = round(raw_score / max_score, 4) if max_score > 0 else 0.0

    return breakdown


# ── Per-question grading ──────────────────────────────────────────────────

async def grade(
    question: str,
    answer: str,
    rubric_data: dict,
    sem: asyncio.Semaphore,
    client: httpx.AsyncClient,
    api_key: str,
) -> dict:
    rubric  = rubric_data["rubric"]
    weights = rubric_data["weights"]


    
    n_criteria = sum(len(rubric.get(t, [])) for t in ("critical", "important", "optional"))
    print(f"  → [{n_criteria} criteria] {question[:70]}", flush=True)
    # ↑ ADD THIS

    all_facts = (
        rubric.get("critical",  []) +
        rubric.get("important", []) +
        rubric.get("optional",  [])
    )

    # Launch all criterion calls concurrently across all tiers
    tier_criterion_tasks = {}
    for tier in ("critical", "important", "optional"):
        tier_criterion_tasks[tier] = [
            judge_criterion(client, sem, api_key, question, answer, criterion)
            for criterion in rubric.get(tier, [])
        ]

    hallucination_task = judge_hallucination(
        client, sem, api_key, question, answer, all_facts
    )

    # Gather per tier with return_exceptions=True so one bad call
    # doesn't kill the whole question
    verdicts = {}
    for tier, tasks in tier_criterion_tasks.items():
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        tier_verdicts = []
        for criterion, result in zip(rubric.get(tier, []), raw_results):
            if isinstance(result, Exception):
                tier_verdicts.append({
                    "fact":   criterion,
                    "met":    False,
                    "reason": f"judge failure: {result}",
                })
            else:
                tier_verdicts.append({
                    "fact":   criterion,
                    "met":    result["met"],
                    "reason": result["reason"],
                })
        verdicts[tier] = tier_verdicts

    try:
        hallucination_detected = await hallucination_task
    except Exception as exc:
        print(f"  [hallucination judge failed] {exc} — defaulting to False")
        hallucination_detected = False

    breakdown = calculate_score(verdicts, weights, hallucination_detected)

    print(f"  ✓ {breakdown['normalized']:.3f}  {question[:70]}", flush=True)
    # ↑ ADD THIS

    return {
        "question": question,
        "score":    breakdown["normalized"],
        "verdict": {
            **verdicts,
            "hallucination_detected": hallucination_detected,
        },
        "score_breakdown": breakdown,
    }


# ── Main ──────────────────────────────────────────────────────────────────

async def process_all(rubric_list, answer_list, api_key):
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    async with httpx.AsyncClient() as client:
        tasks = [
            grade(
                question=rubric_record.get("question", ""),
                answer=extract_answer_text(answer_record),
                rubric_data=rubric_record,
                sem=sem,
                client=client,
                api_key=api_key,
            )
            for rubric_record, answer_record in zip(rubric_list, answer_list)
        ]
        return list(await asyncio.gather(*tasks))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer", required=True,
                        help="Path to answer .txt/.json or inline string")
    parser.add_argument("--rubric", required=True,
                        help="Output of stage3/4 (rubric.json)")
    parser.add_argument("--model",  required=True,
                        help="Name of the model that produced the answer")
    parser.add_argument("--out",    required=True)
    args = parser.parse_args()

    with open(args.rubric, encoding="utf-8") as f:
        rubric_data = json.load(f)
    if isinstance(rubric_data, dict):
        rubric_data = [rubric_data]

    answers_data = load_json_or_text(args.answer)
    if not isinstance(answers_data, list):
        answers_data = [{
            "question": rubric_data[0].get("question", ""),
            "answer":   str(answers_data),
        }]

    if len(rubric_data) != len(answers_data):
        raise ValueError(
            f"Rubric count ({len(rubric_data)}) != answer count ({len(answers_data)}). "
            "Ensure both files cover the same set of questions."
        )

    results = asyncio.run(process_all(rubric_data, answers_data, API_KEY))

    scores  = [r["score"] for r in results]
    summary = {
        "count":         len(scores),
        "average_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "min_score":     round(min(scores), 4) if scores else 0.0,
        "max_score":     round(max(scores), 4) if scores else 0.0,
    }

    output = {
        "model":   args.model,
        "summary": summary,
        "results": results,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"✓ Graded {len(scores)} answers.")
    print(f"  Average Score: {summary['average_score']:.4f} → {args.out}")


if __name__ == "__main__":
    main()
