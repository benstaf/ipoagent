
"""
Grade one model answer against the rubric using an judge ensemble.

Input:  --answer   answer.txt (or inline string)
        --rubric   rubric.json (output of stage3/4)
        --model    "moonshotai/Kimi-K2-Instruct"
        --out      grade.json

Output:
{
    "question": "...",
    "model": "...",
    "score": 0.82,
    "verdict": {
        "critical":  [{"fact": "...", "met": true,  "reason": "..."}],
        "important": [{"fact": "...", "met": false, "reason": "..."}],
        "optional":  [{"fact": "...", "met": true,  "reason": "..."}],
        "hallucination_penalty": false
    },
    "score_breakdown": {
        "critical_earned":   24.0,
        "critical_possible": 24.0,
        "important_earned":  5.0,
        "important_possible":10.0,
        "optional_earned":   2.0,
        "optional_possible": 4.0,
        "hallucination_penalty": 0.0,
        "raw_score": 31.0,
        "max_score": 38.0,
        "normalized": 0.82
    }
}
"""

import asyncio
import json
import argparse
import os
from pathlib import Path

import httpx

from dotenv import load_dotenv

load_dotenv()

MODEL   = os.environ["ENRICHMENT_MODEL"]
URL     = os.environ["ENRICHMENT_BASE_URL"]
API_KEY = os.environ["ENRICHMENT_API_KEY"]



# Judge ensemble — majority vote across these three
JUDGE_MODELS = [
    MODEL,
    MODEL,
    MODEL,
]

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
by the criterion list provided. This includes invented figures, wrong totals, \
or calculations with fabricated inputs.

Return ONLY:
{"hallucination": true | false, "reason": "one sentence"}
"""

HALLUCINATION_USER = """\
Question: {question}

Answer:
{answer}

Ground-truth facts:
{facts_block}

Does the answer contain numerical claims NOT found in the ground-truth facts?
"""


# ── Single criterion judge call ───────────────────────────────────────────
async def judge_criterion(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    question: str,
    answer: str,
    criterion: str,
) -> dict:
    resp = await client.post(
        URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
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
        },
        timeout=2000,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
    return json.loads(raw)


async def judge_hallucination(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    question: str,
    answer: str,
    all_facts: list[str],
) -> dict:
    facts_block = "\n".join(f"- {f}" for f in all_facts)
    resp = await client.post(
        f"{DEEPINFRA_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
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
        },
        timeout=2000,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
    return json.loads(raw)


# ── Majority vote across ensemble ─────────────────────────────────────────
async def ensemble_judge_criterion(
    client: httpx.AsyncClient,
    api_key: str,
    question: str,
    answer: str,
    criterion: str,
) -> dict:
    """Call all judges in parallel, return majority verdict."""
    tasks = [
        judge_criterion(client, api_key, m, question, answer, criterion)
        for m in JUDGE_MODELS
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    verdicts = []
    reasons  = []
    for r in results:
        if isinstance(r, Exception):
            continue
        verdicts.append(r.get("verdict", "UNMET"))
        reasons.append(r.get("reason", ""))

    met_count = verdicts.count("MET")
    majority  = "MET" if met_count > len(JUDGE_MODELS) / 2 else "UNMET"

    return {
        "met": majority == "MET",
        "votes": verdicts,
        "reason": reasons[0] if reasons else "",
    }


async def ensemble_judge_hallucination(
    client: httpx.AsyncClient,
    api_key: str,
    question: str,
    answer: str,
    all_facts: list[str],
) -> bool:
    """Returns True if majority of judges detect hallucination."""
    tasks = [
        judge_hallucination(client, api_key, m, question, answer, all_facts)
        for m in JUDGE_MODELS
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    hallucination_votes = []
    for r in results:
        if isinstance(r, Exception):
            continue
        hallucination_votes.append(r.get("hallucination", False))

    return hallucination_votes.count(True) > len(JUDGE_MODELS) / 2


# ── Score calculation ─────────────────────────────────────────────────────
def calculate_score(
    verdicts: dict,
    weights: dict,
    hallucination_detected: bool,
) -> dict:
    """
    verdicts: {"critical": [{"met": bool}, ...], "important": [...], "optional": [...]}
    weights:  {"critical": 12.0, "important": 6.0, "optional": 2.0, "hallucination": -10.0}
    """
    breakdown = {}
    raw_score = 0.0
    max_score = 0.0

    for tier in ("critical", "important", "optional"):
        w         = weights[tier]
        criteria  = verdicts[tier]
        earned    = sum(w for c in criteria if c["met"])
        possible  = len(criteria) * w
        breakdown[f"{tier}_earned"]   = earned
        breakdown[f"{tier}_possible"] = possible
        raw_score += earned
        max_score += possible

    penalty = weights["hallucination"] if hallucination_detected else 0.0
    breakdown["hallucination_penalty"] = penalty
    raw_score = max(0.0, raw_score + penalty)

    breakdown["raw_score"]  = raw_score
    breakdown["max_score"]  = max_score
    breakdown["normalized"] = round(raw_score / max_score, 4) if max_score > 0 else 0.0

    return breakdown


# ── Main grading loop ─────────────────────────────────────────────────────
async def grade(
    question: str,
    answer: str,
    rubric_data: dict,
    model_name: str,
    api_key: str,
) -> dict:
    rubric  = rubric_data["rubric"]
    weights = rubric_data["weights"]

    all_facts = (
        rubric.get("critical",  []) +
        rubric.get("important", []) +
        rubric.get("optional",  [])
    )

    async with httpx.AsyncClient() as client:
        # Grade all criteria concurrently within each tier
        verdict_tasks = {}
        for tier in ("critical", "important", "optional"):
            tier_tasks = [
                ensemble_judge_criterion(client, api_key, question, answer, criterion)
                for criterion in rubric.get(tier, [])
            ]
            verdict_tasks[tier] = tier_tasks

        # Hallucination check runs in parallel with criterion grading
        hallucination_task = ensemble_judge_hallucination(
            client, api_key, question, answer, all_facts
        )

        # Await everything
        verdicts = {}
        for tier, tasks in verdict_tasks.items():
            results = await asyncio.gather(*tasks)
            verdicts[tier] = [
                {"fact": fact, "met": r["met"], "reason": r["reason"]}
                for fact, r in zip(rubric.get(tier, []), results)
            ]

        hallucination_detected = await hallucination_task

    breakdown = calculate_score(verdicts, weights, hallucination_detected)

    return {
        "question":  question,
        "model":     model_name,
        "score":     breakdown["normalized"],
        "verdict": {
            **verdicts,
            "hallucination_penalty": hallucination_detected,
        },
        "score_breakdown": breakdown,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer",  required=True,
                        help="Path to answer .txt or inline string")
    parser.add_argument("--rubric",  required=True,
                        help="Output of stage5 (rubric.json)")
    parser.add_argument("--model",   required=True,
                        help="Name of the model that produced the answer")
    parser.add_argument("--out",     required=True)
    args = parser.parse_args()

    api_key = JUDGE_API_KEY

    answer_path = Path(args.answer)
    answer = answer_path.read_text() if answer_path.exists() else args.answer

    with open(args.rubric) as f:
        rubric_data = json.load(f)

    result = asyncio.run(grade(
        question=rubric_data["question"],
        answer=answer,
        rubric_data=rubric_data,
        model_name=args.model,
        api_key=api_key,
    ))

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✓ Score: {result['score']:.4f}  → {args.out}")


if __name__ == "__main__":
    main()
