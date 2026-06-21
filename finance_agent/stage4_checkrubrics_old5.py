"""
Stage 4: Validate rubric quality before grading.

Input:  --rubric  rubric.json   (output of stage3 — single dict or list of dicts)
Output: prints a report per question.
        Writes --out  rubric_checked.json with "quality" block added to each record.

Exit codes:
    0 — all rubrics pass quality gate
    1 — one or more rubrics fail (use --no-gate to suppress)
"""

import asyncio
import json
import argparse
import os
import re
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()

from pipeline_shared import recommend_next_step

MODEL   = os.environ["ENRICHMENT_MODEL"]
URL     = os.environ["ENRICHMENT_BASE_URL"]
API_KEY = os.environ["ENRICHMENT_API_KEY"]

MAX_TOKENS        = 8000
QUALITY_THRESHOLD = 0.70


# ---------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a rubric quality reviewer for a financial analysis benchmark.

You will receive a grading rubric (critical / important / optional criteria)
for a question about an S-1 filing. Evaluate its quality across four dimensions:

1. Specificity (0-1): Are criteria precise and verifiable, or vague?
   Bad:  "discusses revenue growth"
   Good: "states FY2025 Connectivity revenue was $11,387M"

2. Atomicity (0-1): Is each criterion a single testable claim, or double-barreled?
   Bad:  "identifies both the revenue figure and the YoY growth rate"
   Good: separate criteria for each

3. Tier correctness (0-1): Are critical/important/optional classifications sensible?
   Final calculated answers → critical.
   Supporting inputs and intermediate values → important.
   Interpretation and context → optional.

4. Coverage (0-1): Do the criteria collectively answer the question,
   or are there obvious gaps?

Return ONLY a JSON object. No preamble, no fences:
{
  "scores": {
    "specificity": 0.0,
    "atomicity": 0.0,
    "tier_correctness": 0.0,
    "coverage": 0.0
  },
  "overall": 0.0,
  "issues": ["..."],
  "suggestions": ["..."]
}

overall = simple mean of the four scores.
"""

USER_TEMPLATE = """\
Question: {question}
Category: {category}

Rubric:

CRITICAL:
{critical}

IMPORTANT:
{important}

OPTIONAL:
{optional}

Evaluate rubric quality and return the JSON report.
"""


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def extract_json(raw: str):
    raw = raw.strip()

    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw)
        raw = raw.replace("```", "").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, flags=re.S)
    if not match:
        raise ValueError(f"Could not extract JSON from model output:\n{raw[:1000]}")

    return json.loads(match.group(0))


def fmt_tier(rubric: dict, tier: str) -> str:
    items = rubric.get(tier, [])
    return "\n".join(f"- {f}" for f in items) if items else "(none)"


# ---------------------------------------------------------------------
# Quality check
# ---------------------------------------------------------------------

async def check_rubric(record: dict) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            URL,
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": USER_TEMPLATE.format(
                            question=record["question"],
                            category=record["category"],
                            critical=fmt_tier(record["rubric"], "critical"),
                            important=fmt_tier(record["rubric"], "important"),
                            optional=fmt_tier(record["rubric"], "optional"),
                        ),
                    },
                ],
                "max_tokens": MAX_TOKENS,
                "temperature": 0.0,
                "thinking": {"type": "disabled"},
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()

    return extract_json(raw)


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def print_report(question: str, quality: dict, threshold: float) -> bool:
    scores  = quality["scores"]
    overall = quality["overall"]
    passed  = overall >= threshold

    q_short = question[:80] + "..." if len(question) > 80 else question

    print(f"\n{'='*60}")
    print(f"  {q_short}")
    print(f"{'='*60}")
    print(f"  Specificity:      {scores['specificity']:.2f}")
    print(f"  Atomicity:        {scores['atomicity']:.2f}")
    print(f"  Tier correctness: {scores['tier_correctness']:.2f}")
    print(f"  Coverage:         {scores['coverage']:.2f}")
    print(f"  {'─'*40}")
    print(f"  Overall:          {overall:.2f}  "
          f"{'✓ PASS' if passed else '✗ FAIL'} (threshold {threshold})")

    if quality.get("issues"):
        print(f"\n  Issues:")
        for issue in quality["issues"]:
            print(f"    • {issue}")

    if quality.get("suggestions"):
        print(f"\n  Suggestions:")
        for s in quality["suggestions"]:
            print(f"    → {s}")

    return passed


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rubric",
        required=True,
        help="Output of stage3 — single dict or list of dicts",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Write rubric + quality block to this path",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=QUALITY_THRESHOLD,
    )
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="Print report but always exit 0",
    )
    args = parser.parse_args()

    with open(args.rubric, encoding="utf-8") as f:
        raw_data = json.load(f)

    # Handle both single dict and list (mirrors stage 3 output shape)
    if isinstance(raw_data, dict):
        raw_data = [raw_data]

    all_passed = True
    results    = []

    for record in raw_data:
        quality        = asyncio.run(check_rubric(record))
        recommendation = asyncio.run(recommend_next_step(quality))
        passed         = print_report(record["question"], quality, args.threshold)
#        print(f"\n  Next step: {recommendation.upper()}")

        if not passed:
            all_passed = False

        results.append({
            **record,
            "quality":        quality,
            "recommendation": recommendation,
        })

    # Summary
    n        = len(results)
    n_passed = sum(1 for r in results if r["quality"]["overall"] >= args.threshold)

    recommendations = {"stop": [], "repair": [], "enrich": []}
    for r in results:
        rec = r.get("recommendation", "stop")
        recommendations[rec].append(r["question"][:60])

    print(f"\n{'='*60}")
    print(f"  {n_passed}/{n} rubrics passed (threshold {args.threshold})")
    print(f"{'─'*60}")
    for action, questions in recommendations.items():
        if questions:
            print(f"  {action.upper()} ({len(questions)}): "
                  + " | ".join(q + "..." for q in questions))
    print(f"{'='*60}\n")

    if args.out:
        output = results if len(results) != 1 else results[0]
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"✓ Rubrics + quality written → {args.out}")

    if not all_passed and not args.no_gate:
        print("One or more rubrics failed the quality gate. "
              "Fix issues or rerun with --no-gate to proceed anyway.")
        sys.exit(1)


if __name__ == "__main__":
    main()
