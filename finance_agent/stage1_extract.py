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
