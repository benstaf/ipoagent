"""
Stage 1: Extract atomic facts from a single model answer.

Input:
    --question  "Using FY2025 Connectivity segment figures..."
    --answer    answer.txt  (or inline string)
    --model     glm-test
    --out       facts.json

Output:
{
    "question": "...",
    "category": "quantitative",
    "model": "...",
    "facts": [...],
    "answer_file": "..."
}

Design goals:
- Facts must be atomic (exactly one claim per fact)
- Facts must be explicit (no inferred information)
- Facts must preserve uncertainty/hedging language
- Duplicate facts from the same answer are removed
- Category is detected automatically from the question text
"""

import asyncio
import json
import argparse
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

from pipeline_shared import extract_json

load_dotenv()

ENRICHMENT_BASE_URL = os.getenv("ENRICHMENT_BASE_URL")
EXTRACTION_MODEL    = os.getenv("ENRICHMENT_MODEL")
ENRICHMENT_API_KEY  = os.getenv("ENRICHMENT_API_KEY")

MAX_TOKENS = 8000

VALID_CATEGORIES = {
    "quantitative", "disclosure", "governance",
    "forensic", "modeling", "comparative",
}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
CLASSIFY_SYSTEM = """\
You are classifying a financial benchmark question into exactly one category.

Categories:
- quantitative: requires numerical calculation or ratio derived directly from
  disclosed figures (e.g. YoY growth, margins, ARPU, capex intensity).
  Use this even if the question also asks for interpretation of the results.
- disclosure: asks what is or isn't disclosed in the filing, without requiring
  calculation
- governance: asks about board, ownership, control, or corporate structure
- forensic: asks about accounting quality, restatements, or distortions
- modeling: asks about forward-looking projections, DCF assumptions, or
  scenario analysis NOT directly calculable from disclosed figures
- comparative: asks for comparison across peers or industries (not across
  segments or time periods of the same company, which is quantitative)

When in doubt between quantitative and any other category, choose quantitative
if the question requires computing a number from the filing.

Return ONLY a JSON object:
{"category": "<category>"}

No preamble. No markdown.
"""

CLASSIFY_USER = "Question: {question}"

EXTRACTION_SYSTEM = """\
You are a financial analyst assistant.

Extract atomic facts from an answer to a financial analysis question about
an S-1 filing.

Definition of an atomic fact:
- A single, self-contained, verifiable claim
- Exactly ONE claim per fact
- Specific: includes numbers, names, dates, metrics, entities, or defined
  terms when present
- Not a restatement of the question
- Only what the answer explicitly states as a result or finding

ATOMICITY RULES
If a sentence contains multiple claims, split them.

BAD:
- "Connectivity revenue was $8.241B and grew 14% YoY"

GOOD:
- "Connectivity revenue was $8.241B"
- "Connectivity revenue grew 14% YoY"

QUESTION RELEVANCE RULE
Extract ONLY facts that directly answer the benchmark question.

Include:
- Input figures required to compute the answer
- Final calculated results: growth rates, margins, ratios, intensity
  figures, multipliers, ARPU figures, leverage multiples
- Definitions or exclusions explicitly requested by the question
- Qualitative conclusions that directly answer the interpretation portion

Exclude:
- Supporting evidence and explanatory narrative
- Background filing context and accounting history
- Operational details and entity descriptions unless the question asks
- Side calculations not requested by the question
- Additional metrics beyond what the question asks for

PRIORITIZATION RULE
Before extracting a fact, ask:

"Would removing this fact make it harder to answer the benchmark question?"

If NO, do not extract it.

ASSUMPTIONS RULE
Extract assumptions only if BOTH are true:
1. The question explicitly asks for assumptions, and
2. The assumption materially affects the calculation or conclusion.

Do not extract generic modeling caveats, methodological disclaimers,
or implementation details.

BAD:
- "The calculation assumes smooth linear subscriber growth."
- "No material timing distortions are assumed."
- "The estimate does not adjust for churn."

GOOD:
- "The estimate assumes Enterprise & Government revenue is excluded
   from the subscriber count."
- "The estimate assumes Consumer revenue corresponds to the disclosed
   subscriber base."

WHAT TO EXTRACT
Extract a fact if it passes the relevance rule and is explicitly stated,
including:

- Raw input figures needed for the requested calculation
- Final calculated results:
  growth rates, margins, ratios, intensity figures,
  ARPU figures, leverage multiples, comparison metrics
- Definitions and exclusions explicitly requested by the question
- Assumptions that materially affect the result (when requested)
- Qualitative conclusions that directly answer the question

Examples:

GOOD:
- "Connectivity revenue was $11,387M in FY2025"
- "Connectivity revenue grew 49.85% YoY"
- "AI capex intensity was 944% in Q1 2026"
- "Managed enterprise customers are excluded from the subscriber count"
- "The Connectivity segment demonstrates strong positive operating leverage"

BAD:
- "Colossus II contains 110,000 GB300 processors"
- "Construction-in-progress increased by $9.4B"
- "The company filed its S-1 on May 20, 2026"
- "Historical results were recast to include xAI and X Holdings"

unless the benchmark question specifically asks about those facts.

DO NOT EXTRACT:
- Intermediate arithmetic steps not labeled as a result
- Methodological explanations that assert no specific claim
- Generic filing background
- Restatements of the question
- Facts that are true but not needed to answer this specific question
- Supporting evidence used only to justify a conclusion
- Generic caveats and modeling disclaimers

UNCERTAINTY PRESERVATION
Preserve hedging exactly as written.

BAD:
- Answer says "approximately 14% growth"
- Output: "Growth was 14%"

GOOD:
- Output: "Growth was approximately 14%"

BAD:
- Answer says "appears to indicate strong leverage"
- Output: "The segment exhibits strong leverage"

GOOD:
- Output: "The filing appears to indicate strong leverage"

DUPLICATE RULES
Do not output duplicate facts.
If two statements assert the same claim, output only one.

OUTPUT FORMAT
Return ONLY a JSON array of strings.

Example:
[
  "Connectivity revenue was $8.241B",
  "Connectivity revenue grew 14% YoY"
]

No preamble.
No markdown fences.

If the answer contains no verifiable claims, return [].
"""


EXTRACTION_USER = """\
Question:
{question}

Answer:
{answer}

Extract all atomic facts as a JSON array of strings.
"""


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

async def classify_question(question: str) -> str:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            ENRICHMENT_BASE_URL,
            headers={"Authorization": f"Bearer {ENRICHMENT_API_KEY}"},
            json={
                "model": EXTRACTION_MODEL,
                "messages": [
                    {"role": "system", "content": CLASSIFY_SYSTEM},
                    {"role": "user",   "content": CLASSIFY_USER.format(
                        question=question,
                    )},
                ],
                "max_tokens": 8000,
                "temperature": 0.0,
                "thinking": {"type": "disabled"},
            },
            timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()

    category = extract_json(raw)["category"]
    if category not in VALID_CATEGORIES:
        raise ValueError(
            f"Model returned unknown category '{category}'. "
            f"Valid: {sorted(VALID_CATEGORIES)}"
        )
    return category


async def extract(question: str, answer: str) -> list[str]:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            ENRICHMENT_BASE_URL,
            headers={"Authorization": f"Bearer {ENRICHMENT_API_KEY}"},
            json={
                "model": EXTRACTION_MODEL,
                "messages": [
                    {"role": "system", "content": EXTRACTION_SYSTEM},
                    {"role": "user",   "content": EXTRACTION_USER.format(
                        question=question,
                        answer=answer,
                    )},
                ],
                "max_tokens": MAX_TOKENS,
                "temperature": 0.0,
                "thinking": {"type": "disabled"},
            },
            timeout=2000,
        )
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        raw = choice["message"]["content"].strip()
        finish_reason = choice.get("finish_reason")

    try:
        facts = extract_json(raw)
    except ValueError:
        print(
            f"[extract_json FAILED] finish_reason={finish_reason!r} "
            f"len(raw)={len(raw)} head={raw[:300]!r} tail={raw[-300:]!r}"
        )
        raise

    assert isinstance(facts, list), f"Expected list, got {type(facts)}"

    seen    = set()
    cleaned = []
    for fact in facts:
        if not isinstance(fact, str):
            continue
        fact = fact.strip()
        if not fact or fact in seen:
            continue
        seen.add(fact)
        cleaned.append(fact)
    return cleaned






# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--answer",   required=True,
                        help="Path to .txt file or inline string")
    parser.add_argument("--model",    required=True,
                        help="Name of model that produced the answer")
    parser.add_argument("--out",      required=True,
                        help="Output JSON path")
    args = parser.parse_args()

    answer_path = Path(args.answer)
    if answer_path.exists():
        answer      = answer_path.read_text(encoding="utf-8")
        answer_file = str(answer_path.resolve())
    else:
        answer      = args.answer
        answer_file = None

    category, facts = asyncio.run(asyncio.gather(
        classify_question(args.question),
        extract(question=args.question, answer=answer),
    ))

    print(f"  category: {category}")

    result = {
        "question":    args.question,
        "category":    category,
        "model":       args.model,
        "facts":       facts,
        "answer_file": answer_file,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✓ {len(facts)} facts → {args.out}")


if __name__ == "__main__":
    main()
"""
Stage 2: Consolidate facts from N model fact files.
"""

import asyncio
import json
import argparse

from dotenv import load_dotenv

import httpx

from pipeline_shared import (
    MIN_AGREEMENT_DEFAULT,
    build_facts_block,
    cluster_facts,
    canonicalize_cluster,
)

load_dotenv()

from fact_normalizer import (
    split_facts_by_type,
    group_quant_facts,
    incremental_consolidate_qual,
)
from pipeline_shared import (
    MIN_AGREEMENT_DEFAULT,
    URL, API_KEY, MODEL, MAX_TOKENS_CLUSTER,
    canonicalize_cluster,
    extract_json,
)

async def consolidate_question(
    client: httpx.AsyncClient,
    question: str,
    model_facts: list[dict],
) -> list[dict]:
    quant_facts, qual_facts = split_facts_by_type(model_facts)
    results = []

    if quant_facts:
        quant_clusters = group_quant_facts(quant_facts)
        canonical_quants = await asyncio.gather(*[
            canonicalize_cluster(client, cluster)
            for cluster in quant_clusters
        ])
        for cluster, canon in zip(quant_clusters, canonical_quants):
            results.append({
                "fact":            canon,
                "agreed_by":       list({item["model"] for item in cluster}),
                "agreement_count": len({item["model"] for item in cluster}),
                "fact_type":       "quantitative",
            })

    if qual_facts:
        qual_canonical = await incremental_consolidate_qual(
            client=client,
            question=question,
            qual_facts=qual_facts,
            url=URL,
            api_key=API_KEY,
            model=MODEL,
            max_tokens=MAX_TOKENS_CLUSTER,
            extract_json_fn=extract_json,
        )
        for item in qual_canonical:
            results.append({
                "fact":            item["fact"],
                "agreed_by":       item["models"],
                "agreement_count": len(item["models"]),
                "fact_type":       "qualitative",
            })

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--facts", nargs="+", required=True,
                        help="One facts JSON file per model")
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-agreement", type=int, default=MIN_AGREEMENT_DEFAULT)
    args = parser.parse_args()

    all_model_data = []
    for path in args.facts:
        with open(path, encoding="utf-8") as f:
            all_model_data.append(json.load(f))

    n_questions = len(all_model_data[0])
    for path, data in zip(args.facts, all_model_data):
        assert len(data) == n_questions, (
            f"{path} has {len(data)} questions, expected {n_questions}"
        )

    async def run_all():
        async with httpx.AsyncClient() as client:
            results = []
            for i in range(n_questions):
                question = all_model_data[0][i]["question"]
                print(f"[{i+1}/{n_questions}] consolidating: {question[:60]}...")

                model_facts = [
                    {"model": data[i]["model"], "facts": data[i]["facts"]}
                    for data in all_model_data
                ]
                answer_files = [
                    data[i]["answer_file"]
                    for data in all_model_data
                    if data[i].get("answer_file")
                ]

                all_facts = await consolidate_question(client, question, model_facts)

                core      = [f for f in all_facts if f["agreement_count"] >= args.min_agreement]
                discarded = [f for f in all_facts if f["agreement_count"] <  args.min_agreement]


                results.append({
                     "question":        question,
                     "category":        all_model_data[0][i].get("category", "quantitative"),
                     "min_agreement":   args.min_agreement,
                     "core_facts":      core,
                     "discarded_count": len(discarded),
                     "answer_files":    answer_files,
                })

                print(f"  -> {len(core)} core facts, {len(discarded)} discarded")

            return results

    results = asyncio.run(run_all())

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total_core = sum(len(r["core_facts"]) for r in results)
    print(f"\n✓ {total_core} total core facts across {n_questions} questions -> {args.out}")


if __name__ == "__main__":
    main()
"""
Stage 3: Induce a grading rubric from consolidated facts + question.

Input:  --consolidated  consolidated_facts.json  (output of stage 2)
        --out           rubric.json

Output:
[
  {
    "question": "...",
    "category": "quantitative",
    "weights": {...},
    "rubric": {
        "critical":  [...],
        "important": [...],
        "optional":  [...]
    },
    "stats": {...},
    "answer_files": [...]
  },
  ...
]
"""

import asyncio
import json
import argparse

import httpx
from dotenv import load_dotenv

from pipeline_shared import (
    WEIGHT_PROFILES,
    repair_rubric,
    validate_rubric,
    rubric_stats,
    induce_rubric,
)

load_dotenv()

async def process_one(
    data: dict,
    client: httpx.AsyncClient,
    min_agreement_override: int | None,
) -> dict:
    question = data["question"]
    category = data["category"]

    min_agreement = (
        min_agreement_override
        if min_agreement_override is not None
        else data.get("min_agreement", 2)
    )

    facts = [
        item["fact"]
        for item in data["core_facts"]
        if item["agreement_count"] >= min_agreement
    ]

    if not facts:
        raise ValueError(
            f"No facts survive min_agreement={min_agreement}. "
            f"Available counts: "
            f"{sorted(set(f['agreement_count'] for f in data['core_facts']), reverse=True)}"
        )

    print(
        f"Inducing rubric for category={category}, "
        f"{len(facts)} facts (min_agreement={min_agreement})..."
    )

    rubric = await induce_rubric(client, question, category, facts)  # ← was asyncio.run(...)
    rubric = repair_rubric(rubric, facts)
    validate_rubric(rubric, facts)
    stats = rubric_stats(rubric)

    if stats["total"] > 0 and stats["critical"] > 0.8 * stats["total"]:
        print("  WARNING: >80% of rubric facts marked critical.")
    if stats["total"] > 0 and stats["optional"] == 0:
        print("  WARNING: rubric contains no optional criteria.")

    print(
        f"  ✓ {stats['total']} criteria "
        f"({stats['critical']} critical, "
        f"{stats['important']} important, "
        f"{stats['optional']} optional)"
    )

    return {
        "question":     question,
        "category":     category,
        "weights":      WEIGHT_PROFILES[category],
        "rubric":       rubric,
        "stats":        stats,
        "answer_files": data.get("answer_files", []),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--consolidated", required=True,
                        help="Output of stage 2 (consolidated_facts.json)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-agreement", type=int, default=None)
    args = parser.parse_args()

    with open(args.consolidated, encoding="utf-8") as f:
        raw_data = json.load(f)
    if isinstance(raw_data, dict):
        raw_data = [raw_data]

    all_results = []

    async def run_all():
      async with httpx.AsyncClient() as client:
        for data in raw_data:
            result = await process_one(          # ← add await
                data=data,
                client=client,
                min_agreement_override=args.min_agreement,
            )
            all_results.append(result)


    asyncio.run(run_all())

    output = all_results if len(all_results) != 1 else all_results[0]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()

"""
Stage 4: Validate rubric quality before grading.
Input:  --rubric  rubric.json   (output of stage3 â€” single dict or list of dicts)
Output: prints a report per question.
        Writes --out  rubric_checked.json with "quality" block added to each record.

Records whose previous "recommendation" was "stop" or "repair"
are carried forward unchanged instead of being re-checked
against the LLM judge. Only "enrich" records are re-evaluated.

Exit codes:
    0 â€” all rubrics pass quality gate
    1 â€” one or more rubrics fail (use --no-gate to suppress)
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
   Word choice does not determine tier. A criterion using "stage," "indicates,"
   "suggests," or similar language is still CRITICAL if it is the direct answer
   to what the question asked for.

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

Evaluate this rubric according to the instructions and return only the JSON object.
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
    payload = {
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
    }

    for attempt in range(5):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    URL,
                    headers={
                        "Authorization": f"Bearer {API_KEY}"
                    },
                    json=payload,
                    timeout=120.0,
                )

            resp.raise_for_status()

            raw = (
                resp.json()["choices"][0]
                ["message"]["content"]
                .strip()
            )

            return extract_json(raw)

        except (
            httpx.ConnectError,
            httpx.RemoteProtocolError,
            httpx.ReadTimeout,
        ) as e:

            if attempt == 4:
                raise

            wait = 2 ** attempt

            print(
                f"Retry {attempt + 1}/5 "
                f"after {type(e).__name__}; "
                f"sleeping {wait}s"
            )

            await asyncio.sleep(wait)




#    return extract_json(raw)


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------


def print_report(question: str, quality: dict, threshold: float) -> bool:
    scores  = quality.get("scores", {})
    overall = quality.get("overall", 0.0)         
    passed  = overall >= threshold

    q_short = question[:80] + "..." if len(question) > 80 else question
    print(f"\n{'='*60}")
    print(f"  {q_short}")                                
    print(f"{'='*60}")
    print(f"  Specificity:      {scores.get('specificity', 0.0):.2f}")
    print(f"  Atomicity:        {scores.get('atomicity', 0.0):.2f}")
    print(f"  Tier correctness: {scores.get('tier_correctness', 0.0):.2f}")                                                            
    print(f"  Coverage:         {scores.get('coverage', 0.0):.2f}")                                                                   
    print(f"  {'—'*40}")                               
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
        help="Output of stage3 â€” single dict or list of dicts",
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
    n_skipped  = 0

    for record in raw_data:
        prev_quality        = record.get("quality") or {}
        prev_recommendation = record.get("recommendation")

        skip_recheck = prev_recommendation in {"stop", "repair"}

        if skip_recheck:
            quality        = prev_quality
            recommendation = prev_recommendation
            passed         = print_report(record["question"], quality, args.threshold)

            print(
                f"  (carried over â€” already '{prev_recommendation}' "
                "in a previous pass)"
            )
            n_skipped += 1
        else:
            quality        = asyncio.run(check_rubric(record))
            recommendation = asyncio.run(recommend_next_step(quality))
            passed         = print_report(record["question"], quality, args.threshold)
            # print(f"\n  Next step: {recommendation.upper()}")

        if not passed:
            all_passed = False

        results.append({
            **record,
            "quality":        quality,
            "recommendation": recommendation,
        })

    # Summary
    n        = len(results)
#    n_passed = sum(1 for r in results if r["quality"]["overall"] >= args.threshold)

# Change this line:
    # n_passed = sum(1 for r in results if r["quality"]["overall"] >= args.threshold)
    
    # To this safe version:
    n_passed = sum(1 for r in results if r.get("quality", {}).get("overall", 0.0) >= args.threshold)

    recommendations = {"stop": [], "repair": [], "enrich": []}
    for r in results:
        rec = r.get("recommendation", "stop")
        recommendations[rec].append(r["question"][:60])

    print(f"\n{'='*60}")
    print(f"  {n_passed}/{n} rubrics passed (threshold {args.threshold})")
    if n_skipped:
        print(f"  {n_skipped}/{n} carried over from a previous pass (not re-checked)")
    print(f"{'â”€'*60}")
    for action, questions in recommendations.items():
        if questions:
            print(f"  {action.upper()} ({len(questions)}): "
                  + " | ".join(q + "..." for q in questions))
    print(f"{'='*60}\n")

    if args.out:
        output = results if len(results) != 1 else results[0]
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"âœ“ Rubrics + quality written â†’ {args.out}")

    if not all_passed and not args.no_gate:
        print("One or more rubrics failed the quality gate. "
              "Fix issues or rerun with --no-gate to proceed anyway.")
        sys.exit(1)


if __name__ == "__main__":
    main()
"""
Stage 4b: Targeted fact extraction driven by stage 4 quality gaps.

For each rubric that has quality issues, re-reads the original answer files,
runs a targeted extraction prompt per model based on the specific issues and
suggestions from stage 4, clusters and canonicalizes the new facts using the
same agreement logic as stage 2, merges agreed facts into the existing pool,
and re-induces the rubric.

Input:  --checked   rubrics_checked.json  (output of stage 4)
        --out       rubrics_enriched.json
        --min-agreement  int  (default: 2, same as stage 2)

Output: same shape as stage 3 output, with added "enrichment" block
        per question. Ready for stage 4 re-check.

Skips questions whose recommendation is not "enrich".
"""

import asyncio
import json
import argparse
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

from pipeline_shared import (
    WEIGHT_PROFILES,
    MIN_AGREEMENT_DEFAULT,
    extract_json,
    validate_rubric,
    rubric_stats,
    build_gaps_block,
    cluster_facts,
    canonicalize_cluster,
    induce_rubric,
    detect_contradictions,
    ManualReviewFlag,
    _print_flag,
    _flag_to_dict,
)

load_dotenv()

MODEL   = os.environ["ENRICHMENT_MODEL"]
URL     = os.environ["ENRICHMENT_BASE_URL"]
API_KEY = os.environ["ENRICHMENT_API_KEY"]

MAX_TOKENS        = 50000
QUALITY_THRESHOLD = 0.70

ENABLE_CONTRADICTION_CHECK = False

# ---------------------------------------------------------------------------
# Targeted extraction prompt (stage 4b-specific — not in shared)
# ---------------------------------------------------------------------------

TARGETED_EXTRACTION_SYSTEM = """\
You are a precise fact extractor for a financial analysis benchmark.

You will receive:
1. A benchmark question about an S-1 filing
2. A list of known quality gaps in the current grading rubric
3. A single model answer to the question

Your job is to extract atomic facts from the answer that address the
identified gaps. Focus only on facts relevant to the gaps — do not
re-extract facts already covered by the existing rubric.

Rules:
- Each fact must be a single, self-contained, verifiable claim.
- State numerical facts with their exact figures as they appear in the answer.
- If the answer contains a derived calculation (e.g. a ratio, a percentage),
  extract the calculated result as a fact, not just the inputs.
- If the answer does not address a gap, do not invent facts to fill it.
- Preserve hedging language exactly as written.

Return ONLY:
{"extracted_facts": ["fact 1", "fact 2", ...]}

No preamble. No markdown.
"""

TARGETED_EXTRACTION_USER = """\
Question:
{question}

Identified gaps in the current rubric:
{gaps_block}

Model answer:
{answer}

Extract atomic facts from this answer that address the identified gaps.
"""


# ---------------------------------------------------------------------------
# Targeted extraction (stage 4b-specific LLM call)
# ---------------------------------------------------------------------------
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # seconds, exponential backoff
MAX_ANSWER_CHARS = None  # set after checking the diagnostic print below


async def targeted_extract_one(
    client: httpx.AsyncClient,
    question: str,
    gaps_block: str,
    model_name: str,
    answer_text: str,
) -> list[str]:
    if MAX_ANSWER_CHARS is not None and len(answer_text) > MAX_ANSWER_CHARS:
        answer_text = answer_text[:MAX_ANSWER_CHARS]

    prompt = TARGETED_EXTRACTION_USER.format(
        question=question,
        gaps_block=gaps_block,
        answer=answer_text,
    )
    print(
        f"    [{model_name}] answer_chars={len(answer_text):,} "
        f"prompt_chars={len(prompt):,}"
    )

    resp = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.post(
                URL,
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": TARGETED_EXTRACTION_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": MAX_TOKENS,
                    "temperature": 0.0,
                    "thinking": {"type": "disabled"},
                },
                timeout=180.0,
            )
            resp.raise_for_status()
            break
        except Exception as e:
            print(f"    [{model_name}] extraction attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
            if attempt == MAX_RETRIES - 1:
                return []
            await asyncio.sleep(RETRY_BASE_DELAY ** attempt)

    try:
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        facts = extract_json(raw).get("extracted_facts", [])
    except Exception as e:
        print(f"    [{model_name}] failed to parse extraction output: {e}")
        return []

    print(f"    [{model_name}] extracted {len(facts)} targeted facts")
    for f in facts:
        print(f"      · {f}")
    return facts



# ---------------------------------------------------------------------------
# Enrichment logic
# ---------------------------------------------------------------------------

def needs_enrichment(record: dict) -> bool:
    recommendation = record.get("recommendation", "")
    if recommendation:
        return recommendation == "enrich"
    # Fallback if recommendation not stored (e.g. older stage 4 output)
    quality = record.get("quality", {})
    return bool(quality.get("issues")) or quality.get("overall", 0.0) < QUALITY_THRESHOLD


async def enrich_one(
    client: httpx.AsyncClient,
    record: dict,
    min_agreement: int,
) -> dict:
    question     = record["question"]
    category     = record["category"]
    quality      = record.get("quality", {})
    answer_files = record.get("answer_files", [])

    existing_facts = (
        record["rubric"].get("critical",  [])
        + record["rubric"].get("important", [])
        + record["rubric"].get("optional",  [])
    )
    existing_set = set(existing_facts)
    gaps_block   = build_gaps_block(quality)

    # --- Step 1: targeted extraction per model answer ---
    loaded_answers = []
    for path in answer_files:
        p = Path(path)
        if not p.exists():
            print(f"    WARNING: answer file not found: {path}")
            continue

        if p.suffix == ".json":
            source = json.loads(p.read_text(encoding="utf-8"))
            entry = next(
                (item for item in source if item["question"] == question),
                None,
            )
            if entry is None:
                print(f"    WARNING: question not found in {path}")
                continue
            answer_text = entry["answer"]
        else:
            answer_text = p.read_text(encoding="utf-8").strip()

        loaded_answers.append({
            "file":  str(p),
            "model": p.stem,
            "text":  answer_text,
        })

    if not loaded_answers:
        print("    WARNING: no readable answer files — skipping enrichment.")
        return {
            **record,
            "enrichment": {
                "status":              "skipped_no_answers",
                "added_facts":         [],
                "manual_review_flags": [],
            },
        }


    per_model_facts = await asyncio.gather(
      *[
        targeted_extract_one(
            client,
            question,
            gaps_block,
            a["model"],
            a["text"],
        )
        for a in loaded_answers
      ],
      return_exceptions=True,
    )

    cleaned = []
    for result in per_model_facts:
      if isinstance(result, Exception):
        print("    Extraction task failed:", repr(result))
        cleaned.append([])
      else:
        cleaned.append(result)

    per_model_facts = cleaned


    # --- Step 2: filter to new facts only ---

    model_facts = [
        {
            "model": a["model"],
            "facts": [f for f in facts if f not in existing_set],
        }
        for a, facts in zip(loaded_answers, per_model_facts)
        if any(f not in existing_set for f in facts)
    ]

    if not model_facts:
        print("    No new facts extracted from any model answer.")
        return {
            **record,
            "enrichment": {
                "status":              "no_new_facts",
                "added_facts":         [],
                "manual_review_flags": [],
            },
        }

    # --- Step 3: cluster across models ---

    if len(model_facts) == 1:
        if min_agreement > 1:
            flag = ManualReviewFlag(
                reason="gap_unresolved_single_model",
                facts=model_facts[0]["facts"],
                detail=(
                    f"Only '{model_facts[0]['model']}' produced new facts "
                    f"addressing the gaps. min_agreement={min_agreement} "
                    f"cannot be reached with one model."
                ),
            )
            _print_flag(flag)
            return {
                **record,
                "enrichment": {
                    "status":              "manual_review_required",
                    "added_facts":         [],
                    "manual_review_flags": [_flag_to_dict(flag)],
                },
            }
        clusters = [
            [{"model": model_facts[0]["model"], "fact": f}]
            for f in model_facts[0]["facts"]
        ]
    else:
        clusters = await cluster_facts(client, question, model_facts)

    # --- Step 4: agreement filter, contradiction detection, canonicalize ---

    manual_review_flags = []
    added_facts         = []

    cluster_meta = [
        {
            "cluster":   cluster,
            "agreed_by": list({item["model"] for item in cluster}),
            "count":     len({item["model"] for item in cluster}),
        }
        for cluster in clusters
    ]
    flagged_indices = set()

    if ENABLE_CONTRADICTION_CHECK:
        contradictions = await detect_contradictions(client, question, cluster_meta)

        for pair in contradictions:
            idx_a, idx_b = pair["cluster_a"], pair["cluster_b"]
            meta_a, meta_b = cluster_meta[idx_a], cluster_meta[idx_b]

            if meta_a["count"] == meta_b["count"]:
                flag = ManualReviewFlag(
                    reason="contradiction_equal_agreement",
                    facts=[meta_a["cluster"][0]["fact"], meta_b["cluster"][0]["fact"]],
                    detail=(
                        f"Two contradictory facts each agreed by {meta_a['count']} "
                        f"model(s). Cannot resolve automatically.\n"
                        f"  Fact A ({meta_a['count']} models): {meta_a['cluster'][0]['fact']}\n"
                        f"  Fact B ({meta_b['count']} models): {meta_b['cluster'][0]['fact']}"
                    ),
                )
                _print_flag(flag)
                manual_review_flags.append(_flag_to_dict(flag))
                flagged_indices.update([idx_a, idx_b])
            else:
                loser_idx  = idx_a if meta_a["count"] < meta_b["count"] else idx_b
                winner_idx = idx_b if meta_a["count"] < meta_b["count"] else idx_a
                print(
                    f"    Contradiction resolved by agreement: "
                    f"keeping '{cluster_meta[winner_idx]['cluster'][0]['fact']}' "
                    f"({cluster_meta[winner_idx]['count']} models) "
                    f"over '{cluster_meta[loser_idx]['cluster'][0]['fact']}' "
                    f"({cluster_meta[loser_idx]['count']} models)"
                )
                flagged_indices.add(loser_idx)

    canon_tasks  = []
    qualifying   = []
    below_thresh = []

    for i, meta in enumerate(cluster_meta):
        if i in flagged_indices:
            continue
        if meta["count"] < min_agreement:
            below_thresh.append(meta)
            continue
        qualifying.append(meta)
        canon_tasks.append(canonicalize_cluster(client, meta["cluster"]))

    if below_thresh and not qualifying and not added_facts:
        flag = ManualReviewFlag(
            reason="gap_unresolved_below_agreement",
            facts=[m["cluster"][0]["fact"] for m in below_thresh],
            detail=(
                f"The following facts were extracted to fill rubric gaps "
                f"but none reached min_agreement={min_agreement}. "
                f"The gap remains unfilled:\n"
                + "\n".join(
                    f"  [{m['count']} model(s)] {m['cluster'][0]['fact']}"
                    for m in below_thresh
                )
            ),
        )
        _print_flag(flag)
        manual_review_flags.append(_flag_to_dict(flag))
        return {
            **record,
            "enrichment": {
                "status":              "manual_review_required",
                "added_facts":         [],
                "manual_review_flags": manual_review_flags,
            },
        }

    if not canon_tasks:
        print("    No new facts to add after filters.")
        return {
            **record,
            "enrichment": {
                "status":              "below_agreement",
                "added_facts":         [],
                "manual_review_flags": manual_review_flags,
            },
        }

#    canonical_facts = await asyncio.gather(*canon_tasks)

    SEM = asyncio.Semaphore(2)

    async def canonicalize_limited(client, cluster):
      async with SEM:
        return await canonicalize_cluster(client, cluster)

    canonical_facts = await asyncio.gather(
        *(canonicalize_limited(client, m["cluster"])
         for m in qualifying)
    )


    for canon, meta in zip(canonical_facts, qualifying):
        if canon not in existing_set:
            added_facts.append({
                "fact":            canon,
                "agreed_by":       meta["agreed_by"],
                "agreement_count": meta["count"],
            })
            existing_set.add(canon)

    print(f"    {len(added_facts)} new facts added after agreement filter:")
    for item in added_facts:
        print(f"      + [{item['agreement_count']} models] {item['fact']}")

    # --- Step 5: re-induce rubric on enriched pool ---

    enriched_facts = existing_facts + [item["fact"] for item in added_facts]
    rubric         = await induce_rubric(client, question, category, enriched_facts)

    try:
        validate_rubric(rubric, enriched_facts)
    except ValueError as e:
        print(f"    WARNING: rubric validation failed after enrichment: {e}")

    stats = rubric_stats(rubric)

    if stats["total"] > 0 and stats["critical"] > 0.8 * stats["total"]:
        print("    WARNING: >80% of rubric facts marked critical.")
    if stats["total"] > 0 and stats["optional"] == 0:
        print("    WARNING: rubric contains no optional criteria.")

    return {
        "question":     question,
        "category":     category,
        "weights":      WEIGHT_PROFILES[category],
        "rubric":       rubric,
        "stats":        stats,
        "answer_files": answer_files,
        "enrichment": {
            "status":              "enriched",
            "added_facts":         added_facts,
            "manual_review_flags": manual_review_flags,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_all(records: list[dict], threshold: float, min_agreement: int) -> list[dict]:
    results = []
    async with httpx.AsyncClient() as client:
        for record in records:
            q_short = record["question"][:70] + ("..." if len(record["question"]) > 70 else "")
            print(f"\n{'─'*60}")
            print(f"  {q_short}")
            print(f"{'─'*60}")

            if not needs_enrichment(record):
                print("  ✓ Skipping enrichment — recommendation is not ENRICH.")
                results.append({
                    **record,
                    "enrichment": {
                        "status":              "skipped",
                        "added_facts":         [],
                        "manual_review_flags": [],
                    },
                })
                continue

            results.append(await enrich_one(client, record, min_agreement))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checked",       required=True,
                        help="Output of stage 4 (rubrics_checked.json)")
    parser.add_argument("--out",           required=True,
                        help="Enriched rubrics ready for stage 4 re-check")
    parser.add_argument("--threshold",     type=float, default=QUALITY_THRESHOLD,
                        help=f"Quality threshold (default: {QUALITY_THRESHOLD})")
    parser.add_argument("--min-agreement", type=int,   default=MIN_AGREEMENT_DEFAULT,
                        help=f"Minimum model agreement for new facts (default: {MIN_AGREEMENT_DEFAULT})")
    parser.add_argument("--accept-gaps",   action="store_true",
                        help="Proceed even if manual review is required")
    args = parser.parse_args()

    with open(args.checked, encoding="utf-8") as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict):
        raw_data = [raw_data]

    results = asyncio.run(run_all(raw_data, args.threshold, args.min_agreement))

    output = results if len(results) != 1 else results[0]

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    enriched = sum(1 for r in results if r.get("enrichment", {}).get("status") == "enriched")
    print(f"\n{'='*60}")
    print(f"  {enriched}/{len(results)} rubrics enriched → {args.out}")
    print(f"  Run Stage 4c consolidation on this file and then Re-run stage 4 check to verify quality.")
    print(f"{'='*60}")

    # Manual review gate
    needs_review = [
        r for r in results
        if r.get("enrichment", {}).get("status") == "manual_review_required"
    ]
    if needs_review and not args.accept_gaps:
        print(f"\n{'='*60}")
        print(f"  ⚑  MANUAL REVIEW REQUIRED: {len(needs_review)} question(s)")
        for r in needs_review:
            print(f"     · {r['question'][:70]}")
            for flag in r["enrichment"]["manual_review_flags"]:
                print(f"       [{flag['reason']}] {flag['detail'][:100]}")
        print(f"\n  Options:")
        print(f"    1. Rerun with --min-agreement 1 to accept singleton facts")
        print(f"    2. Manually add missing facts to the rubric JSON")
        print(f"    3. Accept the coverage gap and rerun with --accept-gaps")
        print(f"{'='*60}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

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
  - Hallucination is judged on numeric/dollar/percentage payload, not on
    literal sentence wording. The ATOMICITY RULES require splitting compound
    claims into independently-readable criteria, which almost always means
    rewording a clause to stand alone (e.g. prepending "Filing states" to a
    fragment pulled out of a list). A plain substring match on full sentences
    rejects that rewording as a "hallucination" even though no information
    was added — checking only the numeric payload preserves the actual
    safety property without blocking legitimate restructuring.
  - If hallucination detected, original rubric is preserved and
    "recommendation" is left as "repair" so the record surfaces for manual
    review. The discarded attempt and the flagged signals are saved under
    "repair_debug" on the record so the rejection is inspectable directly
    from the output file, without re-running with stdout captured.

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

MAX_TOKENS        = 8000
QUALITY_THRESHOLD = 0.70
MAX_REPAIR_ATTEMPTS = 5

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

REPAIR_SYSTEM = """\
You are a rubric editor for a financial analysis benchmark.

You will receive a grading rubric with identified structural issues and
suggestions for fixing them, along with the dimension scores that failed.
Your job is to repair ONLY the dimensions that scored below threshold.

SCOPE CONSTRAINT (critical — read before touching anything):
You will be told which dimensions need repair. Do not modify dimensions
that are not listed as needing repair.

Specifically:
- If tier_correctness is NOT listed as needing repair, do NOT move any
  criterion between tiers. Preserve every criterion in its current tier.
- If coverage is NOT listed as needing repair, do NOT add or remove any
  substantive facts. Preserve the set of facts as-is.
- If specificity is NOT listed as needing repair, avoid rewording criteria
  except where strictly required to make an atomic split read as a
  standalone sentence.
- If atomicity is NOT listed as needing repair, do NOT split any criteria.

TIER MOVEMENT RULE:
Move a criterion between tiers only if:
1. tier_correctness was listed as needing repair
AND
2. the checker's issues or suggestions explicitly mention that criterion
   or that category of criterion.
Do not move criteria solely because you believe another tier might be
better. When in doubt, preserve the original placement.

TIER SEMANTICS (strict):
- CRITICAL: whatever directly answers the question being asked. This
  includes final calculated numerical answers (growth rates, ratios,
  intensity figures, ARPU values, calculated margins) AND, when the
  question itself asks for a judgment, classification, or determination
  (e.g. "at what stage...", "what can/cannot be determined...", "identify
  which decisions..."), the criterion stating that judgment. Word choice
  never demotes a criterion — a criterion that uses "stage," "indicates,"
  or "suggests" is still CRITICAL if it is the thing the question asked
  for. Check only: is this what the question is asking for?
- IMPORTANT: input figures and supporting facts used to derive or justify
  a critical answer — intermediate values, definitions, exclusions,
  methodology, comparative statements ("X exceeded Y"), average
  calculations used as inputs, and the specific missing/disclosed data
  points that a critical "cannot be determined" conclusion depends on.
- OPTIONAL: interpretive color that goes BEYOND what the question asked —
  implications, comparisons, or framing the question did not request. If
  removing a criterion would remove information the question explicitly
  asked for, it is not optional, regardless of its phrasing.

ATOMICITY RULES (apply only if atomicity needs repair):
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
- You may split existing facts into more atomic claims (only if atomicity
  needs repair).
- You may reassign facts between tiers (only if tier_correctness needs
  repair, and only where issues/suggestions explicitly flag it).
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

Current dimension scores:
- Specificity:      {score_specificity:.2f}
- Atomicity:        {score_atomicity:.2f}
- Tier correctness: {score_tier_correctness:.2f}
- Coverage:         {score_coverage:.2f}
- Overall:          {score_overall:.2f}

Dimensions needing repair (score below {threshold:.2f}):
{repair_focus}

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

Repair ONLY the dimensions listed above. Leave all other dimensions
untouched. Return only the corrected JSON.
"""

REPAIR_USER_RETRY = """\
Question: {question}

Your previous repair attempt scored {prev_score:.2f}, which is lower than
the original score of {original_score:.2f}. It made things worse.

Previous attempt's dimension scores:
- Specificity:      {score_specificity:.2f}
- Atomicity:        {score_atomicity:.2f}
- Tier correctness: {score_tier_correctness:.2f}
- Coverage:         {score_coverage:.2f}
- Overall:          {prev_score:.2f}

Issues the checker found with your previous attempt:
{prev_issues}

This is attempt {attempt} of {max_attempts}. Try a different approach.
Focus only on the issues above. Make minimal changes — do not restructure
the rubric wholesale. If a dimension already scores above {threshold:.2f},
leave it completely untouched.

Dimensions still needing repair (score below {threshold:.2f}):
{repair_focus}

Current rubric (your previous attempt — start from this, not the original):
CRITICAL:
{critical}

IMPORTANT:
{important}

OPTIONAL:
{optional}

Original issues identified by quality checker:
{original_issues}

Original suggestions from quality checker:
{original_suggestions}

Return only the corrected JSON.
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


NUMERIC_RE = re.compile(r"\$?\d[\d,]*\.?\d*%?")


def extract_signals(text: str) -> set[str]:
    """Numeric/dollar/percentage tokens in a fact, punctuation-stripped."""
    return {tok.rstrip(",.") for tok in NUMERIC_RE.findall(text)}


def detect_hallucinations(original: dict, repaired: dict) -> list[str]:
    """
    Return list of facts in the repaired rubric that introduce a number,
    dollar figure, or percentage not present anywhere in the original rubric.
    """
    original_signals = set()
    for fact in all_facts(original):
        original_signals |= extract_signals(fact)

    hallucinated = []
    for fact in all_facts(repaired):
        new_signals = extract_signals(fact) - original_signals
        if new_signals:
            hallucinated.append(fact)
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


def build_repair_focus(scores: dict, threshold: float) -> tuple[list[str], str]:
    """
    Return (list of dimension names needing repair, formatted string for prompt).
    Dimensions with no score entry are treated as passing (don't repair what
    we can't measure).
    """
    dim_map = {
        "specificity":      "specificity",
        "atomicity":        "atomicity",
        "tier_correctness": "tier_correctness",
        "coverage":         "coverage",
    }
    failing = [
        label
        for key, label in dim_map.items()
        if scores.get(key, 1.0) < threshold
    ]
    if failing:
        lines = "\n".join(f"- {d} (score: {scores.get(d, 0.0):.2f})" for d in failing)
    else:
        lines = "- (all dimensions at or above threshold — repair overall structure if needed)"
    return failing, lines


# ---------------------------------------------------------------------------
# Repair (single attempt)
# ---------------------------------------------------------------------------

async def repair_rubric(
    record: dict,
    threshold: float,
    attempt: int = 1,
    prev_quality: dict | None = None,
    original_quality: dict | None = None,
) -> dict:
    """
    Call the model for one repair attempt.

    - attempt=1  → uses REPAIR_USER (first-attempt prompt)
    - attempt>1  → uses REPAIR_USER_RETRY, which shows the model what went
                   wrong in its previous attempt so it can try differently.
                   The rubric passed in is the previous attempt's rubric
                   (not the original), so the model refines rather than
                   starting over from scratch each time.
    """
    quality   = record.get("quality", {})
    scores    = quality.get("scores", {})
    issues    = quality.get("issues", [])
    suggestions = quality.get("suggestions", [])

    _failing_dims, repair_focus_str = build_repair_focus(scores, threshold)

    if attempt == 1 or prev_quality is None:
        user_content = REPAIR_USER.format(
            question=record["question"],
            score_specificity=scores.get("specificity", 0.0),
            score_atomicity=scores.get("atomicity", 0.0),
            score_tier_correctness=scores.get("tier_correctness", 0.0),
            score_coverage=scores.get("coverage", 0.0),
            score_overall=quality.get("overall", 0.0),
            threshold=threshold,
            repair_focus=repair_focus_str,
            critical=fmt_tier(record["rubric"], "critical"),
            important=fmt_tier(record["rubric"], "important"),
            optional=fmt_tier(record["rubric"], "optional"),
            issues="\n".join(f"- {i}" for i in issues),
            suggestions="\n".join(f"- {s}" for s in suggestions),
        )
    else:
        # Retry: pass the *previous attempt's* scores and issues so the model
        # knows specifically what it got wrong, and tell it to start from the
        # previous attempt's rubric (already in record["rubric"]).
        prev_scores  = prev_quality.get("scores", {})
        prev_issues  = prev_quality.get("issues", [])
        orig_issues  = original_quality.get("issues", []) if original_quality else issues
        orig_suggestions = original_quality.get("suggestions", []) if original_quality else suggestions

        _failing_dims, repair_focus_str = build_repair_focus(prev_scores, threshold)

        user_content = REPAIR_USER_RETRY.format(
            question=record["question"],
            prev_score=prev_quality.get("overall", 0.0),
            original_score=original_quality.get("overall", 0.0) if original_quality else 0.0,
            score_specificity=prev_scores.get("specificity", 0.0),
            score_atomicity=prev_scores.get("atomicity", 0.0),
            score_tier_correctness=prev_scores.get("tier_correctness", 0.0),
            score_coverage=prev_scores.get("coverage", 0.0),
            threshold=threshold,
            repair_focus=repair_focus_str,
            attempt=attempt,
            max_attempts=MAX_REPAIR_ATTEMPTS,
            critical=fmt_tier(record["rubric"], "critical"),
            important=fmt_tier(record["rubric"], "important"),
            optional=fmt_tier(record["rubric"], "optional"),
            prev_issues="\n".join(f"- {i}" for i in prev_issues),
            original_issues="\n".join(f"- {i}" for i in orig_issues),
            original_suggestions="\n".join(f"- {s}" for s in orig_suggestions),
        )

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            URL,
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": REPAIR_SYSTEM},
                    {"role": "user",   "content": user_content},
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
# Tier-churn detection
# ---------------------------------------------------------------------------

def detect_tier_churn(original: dict, repaired: dict) -> dict:
    """
    Compute how many criteria moved between tiers. Returns a summary dict.
    Used as a warning signal when tier_correctness was NOT a failing dimension.
    """
    tiers = ["critical", "important", "optional"]
    original_placement = {}
    for tier in tiers:
        for fact in original.get(tier, []):
            original_placement[fact] = tier

    moved = []
    for tier in tiers:
        for fact in repaired.get(tier, []):
            orig_tier = original_placement.get(fact)
            if orig_tier is not None and orig_tier != tier:
                moved.append({"fact": fact, "from": orig_tier, "to": tier})

    total_original = sum(len(original.get(t, [])) for t in tiers)
    churn_rate = len(moved) / total_original if total_original else 0.0

    return {
        "moved_count": len(moved),
        "total_original": total_original,
        "churn_rate": churn_rate,
        "moved": moved,
    }


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
    churn: dict | None = None,
    attempts_used: int = 1,
) -> None:
    q_short = question[:80] + "..." if len(question) > 80 else question
    print(f"\n{'='*60}")
    print(f"  {q_short}")
    print(f"{'='*60}")

    if skipped:
        print(f"  ⊘ Skipped (recommendation: stop, score {original_score:.2f})")
        return

    if reverted:
        print(f"  ✗ Repair reverted — hallucinated signals detected:")
        for h in hallucinated:
            print(f"      · {h}")
        print(f"  → Original rubric preserved (score {original_score:.2f})")
        return

    arrow = "↑" if repaired_score > original_score else ("↓" if repaired_score < original_score else "→")
    print(f"  Score: {original_score:.2f} {arrow} {repaired_score:.2f}  (attempts: {attempts_used})")

    if churn and churn["moved_count"] > 0:
        print(f"  ⚠ Tier churn: {churn['moved_count']}/{churn['total_original']} criteria moved "
              f"({churn['churn_rate']:.0%})")

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
        "--max-attempts",
        type=int,
        default=MAX_REPAIR_ATTEMPTS,
        help=f"Max repair attempts per rubric before giving up (default {MAX_REPAIR_ATTEMPTS})",
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
        original_quality    = quality          # kept for retry prompt context
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

        # ---------------------------------------------------------------
        # Retry loop: attempt repair up to args.max_attempts times.
        # After each attempt we recheck quality and exit early if the
        # rubric reaches threshold. We track the best attempt seen so far
        # and use that as the final output regardless of which attempt
        # was last.
        # ---------------------------------------------------------------

        best_rubric   = None     # rubric dict of the best attempt so far
        best_quality  = None     # quality dict of the best attempt so far
        best_score    = -1.0
        best_churn    = None
        attempts_used = 0

        # The "current" record used as input to each repair call. For
        # attempt 1 this is the original; for retries it's the previous
        # attempt's record (so the model refines rather than restarts).
        current_record   = record
        prev_quality_obj = None

        hallucinated_final = []
        reverted_final     = False

        for attempt in range(1, args.max_attempts + 1):
            attempts_used = attempt

            # --- Run one repair attempt ---
            try:
                repaired_rubric = asyncio.run(repair_rubric(
                    current_record,
                    threshold=args.threshold,
                    attempt=attempt,
                    prev_quality=prev_quality_obj,
                    original_quality=original_quality,
                ))
            except Exception as e:
                print(f"\n  Attempt {attempt}: repair call failed — {type(e).__name__}: {e}")
                break

            # --- Hallucination check ---
            # Always check against the *original* rubric, not the previous
            # attempt, so signals can't accumulate across retries.
            hallucinated = detect_hallucinations(record["rubric"], repaired_rubric)
            if hallucinated:
                print(f"\n  Attempt {attempt}: hallucinated signals — skipping this attempt")
                for h in hallucinated:
                    print(f"      · {h}")
                # Don't update current_record; try again from the same base.
                # If every attempt hallucinates we'll fall through with
                # best_rubric=None and revert to original below.
                hallucinated_final = hallucinated
                reverted_final     = True
                continue

            hallucinated_final = []
            reverted_final     = False

            # --- Recheck quality ---
            candidate_record = {
                **record,
                "rubric": repaired_rubric,
            }

            from stage4_checkrubrics import check_rubric

            print(f"\n  Attempt {attempt}: rechecking...")
            try:
                quality2 = asyncio.run(check_rubric(candidate_record))
            except Exception as e:
                print(f"  Attempt {attempt}: recheck failed — {type(e).__name__}: {e}")
                # Keep trying from the same current_record base.
                continue

            if not isinstance(quality2, dict) or "overall" not in quality2:
                print(f"  Attempt {attempt}: recheck returned malformed response — skipping")
                continue

            candidate_score = quality2.get("overall", 0.0)
            churn           = detect_tier_churn(record["rubric"], repaired_rubric)

            print(f"  Attempt {attempt}: score {candidate_score:.2f} "
                  f"(original {original_score:.2f}, best so far {max(best_score, 0.0):.2f})")

            # Track best regardless of whether it beats original, so we
            # always have something to write even if all attempts regress.
            if candidate_score > best_score:
                best_score   = candidate_score
                best_rubric  = repaired_rubric
                best_quality = quality2
                best_churn   = churn

            # Update for next retry: model refines from this attempt.
            prev_quality_obj = quality2
            current_record = {
                **record,
                "rubric":  repaired_rubric,
                "quality": quality2,
            }

            # Exit early if we've cleared the threshold.
            if candidate_score >= args.threshold:
                print(f"  ✓ Threshold reached on attempt {attempt} — stopping early")
                break

        # ---------------------------------------------------------------
        # Decide what to write for this record.
        # ---------------------------------------------------------------

        if best_rubric is None:
            # Every attempt either hallucinated or errored out — revert.
            print_repair_report(
                record["question"],
                original_score=original_score,
                repaired_score=None,
                skipped=False,
                hallucinated=hallucinated_final,
                reverted=True,
                attempts_used=attempts_used,
            )
            record_out = {
                **record,
                "repair_debug": {
                    "hallucinated_signals": hallucinated_final,
                    "attempts": attempts_used,
                },
            }
            results.append(record_out)
            if original_score < args.threshold:
                all_passed = False
            continue

        # We have at least one valid repaired rubric — use the best one.
        # Only accept it over the original if it's actually better.
        if best_score <= original_score:
            # All attempts made things worse or broke even. Keep original
            # but still log the best attempt for inspection.
            print(f"\n  All {attempts_used} attempt(s) failed to improve score "
                  f"({original_score:.2f}). Keeping original rubric.")
            record_out = {
                **record,
                "repair_debug": {
                    "best_attempted_score": best_score,
                    "best_attempted_rubric": best_rubric,
                    "attempts": attempts_used,
                },
            }
            print_repair_report(
                record["question"],
                original_score=original_score,
                repaired_score=best_score,
                skipped=False,
                hallucinated=[],
                reverted=False,
                churn=best_churn,
                attempts_used=attempts_used,
            )
            results.append(record_out)
            if original_score < args.threshold:
                all_passed = False
            continue

        # Best attempt improved the score — accept it.
        record_repaired = {
            **record,
            "rubric":          best_rubric,
            "rubric_repaired": True,
            "stats":           rubric_stats(best_rubric),
            "quality":         best_quality,
            "recommendation":  None,   # cleared so Stage 4 re-evaluates fresh
            "repair_attempts": attempts_used,
        }
        if best_churn and best_churn["moved_count"] > 0:
            record_repaired["repair_churn"] = best_churn

        print_repair_report(
            record["question"],
            original_score=original_score,
            repaired_score=best_score,
            skipped=False,
            hallucinated=[],
            reverted=False,
            churn=best_churn,
            attempts_used=attempts_used,
        )

        if best_score < args.threshold:
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
"""
Stage 4c: Deduplicate rubric facts via semantic clustering.

Flattens all rubric tiers into a single pool, clusters semantically
equivalent facts using the same cluster_facts + canonicalize_cluster
logic as stage 2, then rebuilds the rubric with each cluster collapsed
to its canonical form. When a cluster spans multiple tiers, the fact
inherits the highest-priority tier (critical > important > optional).

Input:  --rubrics   any rubric JSON file (e.g. rubrics_enriched.json)
Output: --out       deduplicated rubric JSON, same shape as input
"""

import asyncio
import json
import argparse
import os

import httpx
from dotenv import load_dotenv

from pipeline_shared import (
    WEIGHT_PROFILES,
    cluster_facts,
    canonicalize_cluster,
    validate_rubric,
    rubric_stats,
    recommend_next_step,
)

load_dotenv()

MODEL   = os.environ["ENRICHMENT_MODEL"]
URL     = os.environ["ENRICHMENT_BASE_URL"]
API_KEY = os.environ["ENRICHMENT_API_KEY"]

TIER_PRIORITY = {"critical": 0, "important": 1, "optional": 2}


# ---------------------------------------------------------------------------
# Core dedup logic
# ---------------------------------------------------------------------------

async def dedup_rubric(
    client: httpx.AsyncClient,
    record: dict,
) -> dict:
    question = record["question"]
    category = record["category"]
    rubric   = record["rubric"]

    # Flatten tiers, preserving membership
    tiered_facts = []
    for tier in ("critical", "important", "optional"):
        for fact in rubric.get(tier, []):
            tiered_facts.append({"tier": tier, "fact": fact})

    if not tiered_facts:
        print("    No facts found — skipping.")
        return record

    before = len(tiered_facts)

    # Cluster as a single pseudo-model pool
    pool_model_facts = [
        {"model": "rubric", "facts": [f["fact"] for f in tiered_facts]}
    ]
    pool_clusters = await cluster_facts(client, question, pool_model_facts)

    canonical_facts = await asyncio.gather(*[
        canonicalize_cluster(client, cluster)
        for cluster in pool_clusters
    ])

    # Map original fact text -> tier for lookup
    fact_to_tier = {f["fact"]: f["tier"] for f in tiered_facts}

    # Rebuild rubric: each cluster collapses to highest-priority tier
    deduped_rubric = {"critical": [], "important": [], "optional": []}

    for cluster, canon in zip(pool_clusters, canonical_facts):
        best_tier = min(
            (fact_to_tier.get(item["fact"], "optional") for item in cluster),
            key=lambda t: TIER_PRIORITY[t],
        )
        deduped_rubric[best_tier].append(canon)

    after = sum(len(deduped_rubric[t]) for t in ("critical", "important", "optional"))

    print(f"    Before: {before} facts | After: {after} facts | Removed: {before - after}")

    try:
        validate_rubric(deduped_rubric, [f["fact"] for f in tiered_facts])
    except ValueError as e:
        print(f"    WARNING: rubric validation failed after dedup: {e}")

    stats = rubric_stats(deduped_rubric)

    return {
        **record,
        "rubric": deduped_rubric,
        "stats":  stats,
        "weights": WEIGHT_PROFILES[category],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_all(records: list[dict]):
    results = []
    deduped_count = 0
    skipped_count = 0
    async with httpx.AsyncClient() as client:
        for record in records:
            q_short = record["question"][:70] + ("..." if len(record["question"]) > 70 else "")
            print(f"\n{'─'*60}")
            print(f"  {q_short}")
            print(f"{'─'*60}")

            quality        = record.get("quality", {})
            recommendation = await recommend_next_step(quality)

            if recommendation in ("stop", "repair"):
                print(f"  ⊘ Skipped dedup — recommendation is '{recommendation}'")
                results.append(record)
                skipped_count += 1
                continue

            results.append(await dedup_rubric(client, record))
            deduped_count += 1

    return results, deduped_count, skipped_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rubrics", required=True,
                        help="Input rubric JSON file (e.g. rubrics_enriched.json)")
    parser.add_argument("--out",     required=True,
                        help="Output deduplicated rubric JSON")
    args = parser.parse_args()

    with open(args.rubrics, encoding="utf-8") as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict):
        raw_data = [raw_data]

    results, deduped_count, skipped_count = asyncio.run(run_all(raw_data))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total_after = sum(
        r.get("stats", {}).get("total", 0) for r in results
    )

    print(f"\n{'='*60}")
    print(f"  Rubrics processed:       {len(results)}")
    print(f"  Rubrics deduplicated:    {deduped_count}")
    print(f"  Rubrics skipped:         {skipped_count}")
    print(f"  Output:                  {args.out}")
    print(f"  Total facts after dedup: {total_after}")
    print(f"  Run stage 4 on this file to verify quality.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
