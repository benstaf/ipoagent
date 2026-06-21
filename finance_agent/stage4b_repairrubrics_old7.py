"""
Stage 4b: Auto-repair rubric structural issues flagged by stage 4.

Input:  --rubric  rubrics_checked.json   (output of stage 4 — must have "quality" block)
Output: --out     rubrics_repaired.json

Routing:
- Records with "recommendation": "repair"  → rubric is repaired and
  "recommendation" is cleared so Stage 4 re-evaluates them fresh.
- Records with "recommendation": "stop"    → passed through unchanged.
- All other records                        → passed through unchanged.

Safety:
- Repair may only restructure or split existing facts.
- Repair may NOT introduce new facts not present in the original rubric.
- If hallucination detected, original rubric is preserved and "recommendation"
  is left as "repair" so the record surfaces for manual review.

Exit codes:
  0 — all rubrics at or above threshold after repair
  1 — one or more rubrics still below threshold after repair
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

MODEL   = os.environ["ENRICHMENT_MODEL"]
URL     = os.environ["ENRICHMENT_BASE_URL"]
API_KEY = os.environ["ENRICHMENT_API_KEY"]

MAX_TOKENS        = 50000
QUALITY_THRESHOLD = 0.70

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

REPAIR_SYSTEM = """\
You are a rubric editor for a financial analysis benchmark.

You will receive a grading rubric with identified structural issues and
suggestions for fixing them. Your job is to repair the rubric by
restructuring existing facts only.

TIER SEMANTICS (strict):
- CRITICAL: final calculated numerical answers that directly answer the
  question — growth rates, ratios, intensity figures, ARPU values,
  calculated margins. Derived calculations explicitly stated as results
  belong here.
- IMPORTANT: input figures used to derive critical answers, intermediate
  values, definitions, exclusions, methodology, comparative statements
  ("X exceeded Y"), average calculations used as inputs.
- OPTIONAL: interpretive conclusions and qualitative assessments using
  words like: implies, indicates, reflects, exhibits, suggests, phase,
  stage, maturity, buildout, overstates, characterizes.

ATOMICITY RULES:
- Each criterion must contain exactly ONE verifiable claim.
- Split any criterion that combines a dollar figure with an exclusion
  statement into two separate criteria.
- Split any criterion that combines a calculated value with an
  interpretation into a factual criterion (critical/important) and an
  interpretive criterion (optional).
- Split any criterion that references two time periods into two separate
  criteria unless it is explicitly a change/expansion claim.

REDUNDANCY RULES:
- If the same dollar figure appears in multiple criteria, consolidate to one.
- If individual period values AND an expansion/change criterion all appear
  for the same metric, keep whichever is more informative and remove exact
  overlaps. Do not remove both — keep at least one.
- If two criteria assert the same claim with different phrasing, keep the
  more specific and precise version.

HARD CONSTRAINTS:
- You may only use facts already present in the input rubric.
- You may split existing facts into more atomic claims.
- You may reassign facts between tiers.
- You may remove genuine duplicates.
- You may NOT introduce any new fact, number, metric, or claim not
  already present in the input rubric.
- You may NOT merge two distinct facts into one if they assert different
  claims.
- Every tier must contain at least one criterion.

Return ONLY a JSON object. No preamble, no markdown fences:
{
  "critical": ["..."],
  "important": ["..."],
  "optional": ["..."]
}
"""

REPAIR_USER = """\
Question: {question}

Current rubric:
CRITICAL:
{critical}

IMPORTANT:
{important}

OPTIONAL:
{optional}

Issues identified by quality checker:
{issues}

Suggestions from quality checker:
{suggestions}

Repair the rubric following the rules. Return only the corrected JSON.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def all_facts(rubric: dict) -> set:
    return set(
        rubric.get("critical", [])
        + rubric.get("important", [])
        + rubric.get("optional", [])
    )


def detect_hallucinations(original: dict, repaired: dict) -> list[str]:
    """
    Return list of facts in repaired rubric that cannot be traced to any
    fact in the original rubric.

    A repaired fact is acceptable if:
    - It exactly matches an original fact, OR
    - It is a substring of an original fact (split from a compound claim), OR
    - An original fact is a substring of it (minor rewording of atomic claim)
    """
    original_facts = all_facts(original)
    repaired_facts = all_facts(repaired)

    hallucinated = []
    for rf in repaired_facts:
        if rf in original_facts:
            continue
        rf_lower = rf.lower()
        if any(
            rf_lower in of.lower() or of.lower() in rf_lower
            for of in original_facts
        ):
            continue
        hallucinated.append(rf)
    return hallucinated


def rubric_stats(rubric: dict) -> dict:
    return {
        "critical":  len(rubric.get("critical",  [])),
        "important": len(rubric.get("important", [])),
        "optional":  len(rubric.get("optional",  [])),
        "total":     sum([
            len(rubric.get("critical",  [])),
            len(rubric.get("important", [])),
            len(rubric.get("optional",  [])),
        ]),
    }


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------

async def repair_rubric(record: dict) -> dict:
    quality     = record.get("quality", {})
    issues      = quality.get("issues", [])
    suggestions = quality.get("suggestions", [])

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            URL,
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": REPAIR_SYSTEM},
                    {
                        "role": "user",
                        "content": REPAIR_USER.format(
                            question=record["question"],
                            critical=fmt_tier(record["rubric"], "critical"),
                            important=fmt_tier(record["rubric"], "important"),
                            optional=fmt_tier(record["rubric"], "optional"),
                            issues="\n".join(f"- {i}" for i in issues),
                            suggestions="\n".join(f"- {s}" for s in suggestions),
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


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_repair_report(
    question: str,
    original_score: float,
    repaired_score: float | None,
    skipped: bool,
    hallucinated: list[str],
    reverted: bool,
) -> None:
    q_short = question[:80] + "..." if len(question) > 80 else question
    print(f"\n{'='*60}")
    print(f"  {q_short}")
    print(f"{'='*60}")

    if skipped:
        print(f"  ⊘ Skipped (recommendation: stop, score {original_score:.2f})")
        return

    if reverted:
        print(f"  ✗ Repair reverted — hallucinated facts detected:")
        for h in hallucinated:
            print(f"      · {h}")
        print(f"  → Original rubric preserved (score {original_score:.2f})")
        return

    arrow = "↑" if repaired_score > original_score else ("↓" if repaired_score < original_score else "→")
    print(f"  Score: {original_score:.2f} {arrow} {repaired_score:.2f}")
    if repaired_score >= QUALITY_THRESHOLD:
        print(f"  ✓ PASS after repair")
    else:
        print(f"  ✗ Still below threshold after repair")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rubric",
        required=True,
        help="Output of stage 4 — must contain 'quality' block per record",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Write repaired rubrics to this path",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=QUALITY_THRESHOLD,
        help="Quality threshold for pass/fail (default 0.70)",
    )
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="Always exit 0 even if rubrics still fail after repair",
    )
    args = parser.parse_args()

    with open(args.rubric, encoding="utf-8") as f:
        raw_data = json.load(f)
    if isinstance(raw_data, dict):
        raw_data = [raw_data]

    # Validate input has quality blocks
    for record in raw_data:
        if "quality" not in record:
            print(
                f"ERROR: record missing 'quality' block. "
                f"Run stage4_checkrubrics.py first.\n"
                f"Question: {record.get('question', '')[:80]}"
            )
            sys.exit(1)

    all_passed = True
    results    = []

    for record in raw_data:
        quality             = record["quality"]
        original_score      = quality.get("overall", 0.0)
        prev_recommendation = record.get("recommendation")

        # Only repair records Stage 4 explicitly flagged for repair.
        if prev_recommendation != "repair":
            print_repair_report(
                record["question"],
                original_score=original_score,
                repaired_score=None,
                skipped=True,
                hallucinated=[],
                reverted=False,
            )
            results.append(record)
            continue

        # Run repair
        repaired_rubric = asyncio.run(repair_rubric(record))

        # Hallucination check — revert and leave "repair" label intact so
        # the record surfaces for manual review on the next Stage 4 pass.
        hallucinated = detect_hallucinations(record["rubric"], repaired_rubric)
        if hallucinated:
            print_repair_report(
                record["question"],
                original_score=original_score,
                repaired_score=None,
                skipped=False,
                hallucinated=hallucinated,
                reverted=True,
            )
            results.append(record)
            if original_score < args.threshold:
                all_passed = False
            continue

        # Accept repair and re-check quality inline.
        record_repaired = {
            **record,
            "rubric":          repaired_rubric,
            "rubric_repaired": True,
            "stats":           rubric_stats(repaired_rubric),
        }
        from stage4_checkrubrics import check_rubric
        quality2       = asyncio.run(check_rubric(record_repaired))
        repaired_score = quality2["overall"]

        print_repair_report(
            record["question"],
            original_score=original_score,
            repaired_score=repaired_score,
            skipped=False,
            hallucinated=[],
            reverted=False,
        )

        # Clear recommendation so Stage 4 re-evaluates this record fresh
        # rather than skipping it as "repair" again.
        record_repaired["quality"]        = quality2
        record_repaired["recommendation"] = None

        if repaired_score < args.threshold:
            all_passed = False

        results.append(record_repaired)

    # Summary
    n        = len(results)
    n_passed = sum(
        1 for r in results
        if r["quality"].get("overall", 0.0) >= args.threshold
    )
    print(f"\n{'='*60}")
    print(f"  {n_passed}/{n} rubrics at or above threshold after repair")
    print(f"{'='*60}\n")

    output = results if len(results) != 1 else results[0]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"✓ Repaired rubrics written → {args.out}")

    if not all_passed and not args.no_gate:
        sys.exit(1)


if __name__ == "__main__":
    main()
