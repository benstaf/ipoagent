"""
Stage 3: Induce a grading rubric from consolidated facts + question.

Input:  --consolidated  consolidated_facts.json
        --category      quantitative|disclosure|governance|forensic|modeling|comparative
        --out           rubric.json

Output:
{
    "question": "...",
    "category": "quantitative",
    "weights": {...},
    "rubric": {
        "critical":  ["fact that must be present to pass"],
        "important": ["fact that should be present"],
        "optional":  ["fact that adds depth"]
    }
}
"""

import asyncio
import json
import argparse
import os
import re

import httpx


from dotenv import load_dotenv

load_dotenv()

MODEL = os.environ["ENRICHMENT_MODEL"]
URL = os.environ["ENRICHMENT_BASE_URL"]
API_KEY = os.environ["ENRICHMENT_API_KEY"]




MAX_TOKENS = 50000


# ---------------------------------------------------------------------
# Weight profiles per category
# ---------------------------------------------------------------------

WEIGHT_PROFILES = {
    "quantitative": {
        "critical": 12.0,
        "important": 6.0,
        "optional": 2.0,
        "hallucination": -10.0,
    },
    "disclosure": {
        "critical": 10.0,
        "important": 5.0,
        "optional": 2.0,
        "hallucination": -6.0,
    },
    "governance": {
        "critical": 10.0,
        "important": 5.0,
        "optional": 2.0,
        "hallucination": -6.0,
    },
    "forensic": {
        "critical": 10.0,
        "important": 5.0,
        "optional": 2.0,
        "hallucination": -8.0,
    },
    "modeling": {
        "critical": 10.0,
        "important": 5.0,
        "optional": 2.0,
        "hallucination": -8.0,
    },
    "comparative": {
        "critical": 10.0,
        "important": 5.0,
        "optional": 2.0,
        "hallucination": -6.0,
    },
}


SYSTEM_PROMPT = """\
You are a financial analyst building a grading rubric for a benchmark
question about an S-1 filing.

You will receive:
1. The question
2. A list of consolidated facts (attested by multiple models) that represent
   the ground-truth answer content

Your job is to classify each fact into one of three tiers:

- critical:
  A complete answer MUST include this.
  Missing it means the answer fundamentally fails the question.

- important:
  A good answer should include this.
  Missing it is a notable gap but not a total failure.

- optional:
  Adds depth or precision.
  Bonus credit.

Rules:

- Every consolidated fact must appear in exactly one tier.
- Do not invent new facts not present in the input list.

- For quantitative questions:
  Final calculated answers and primary conclusions should generally be
  treated as critical.
  Supporting inputs and intermediate values are usually important.
  Additional interpretation is optional.

- For disclosure/governance questions:
  What is disclosed vs what is not disclosed may both be critical.

- For forensic questions:
  The accounting mechanism is critical.
  Direction of distortion is important.
  Magnitude estimates are optional.

Return ONLY a JSON object:

{
  "critical": [...],
  "important": [...],
  "optional": [...]
}

Use the fact strings exactly as provided.
No markdown.
No explanation.
"""


USER_TEMPLATE = """\
Question: {question}

Category: {category}

Consolidated facts:
{facts_block}

Classify each fact into critical / important / optional.
"""


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def extract_json(raw: str):
    """
    Robust JSON extraction from model output.
    Handles fenced code blocks and occasional wrapper text.
    """

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
        raise ValueError(
            f"Could not extract JSON from model output:\n"
            f"{raw[:1000]}"
        )

    return json.loads(match.group(0))


def validate_rubric(
    rubric: dict,
    facts: list[str],
):
    """
    Ensure:
      - no missing facts
      - no extra facts
      - no duplicates across tiers
    """

    assigned = (
        rubric.get("critical", [])
        + rubric.get("important", [])
        + rubric.get("optional", [])
    )

    assigned_set = set(assigned)
    fact_set = set(facts)

    missing = fact_set - assigned_set
    extra = assigned_set - fact_set

    if missing:
        raise ValueError(
            f"Rubric omitted facts: {sorted(missing)}"
        )

    if extra:
        raise ValueError(
            f"Rubric introduced unknown facts: {sorted(extra)}"
        )

    if len(assigned) != len(assigned_set):
        raise ValueError(
            "Rubric contains duplicate facts across tiers."
        )


def rubric_stats(rubric: dict):
    total = (
        len(rubric.get("critical", []))
        + len(rubric.get("important", []))
        + len(rubric.get("optional", []))
    )

    return {
        "critical": len(rubric.get("critical", [])),
        "important": len(rubric.get("important", [])),
        "optional": len(rubric.get("optional", [])),
        "total": total,
    }


# ---------------------------------------------------------------------
# Rubric induction
# ---------------------------------------------------------------------

async def induce_rubric(
    question: str,
    category: str,
    facts: list[str],
    api_key: str,
) -> dict:

    facts_block = "\n".join(
        f"- {fact}"
        for fact in facts
    )

    async with httpx.AsyncClient() as client:

        resp = await client.post(
            f"{URL}",
            headers={
                "Authorization": f"Bearer {api_key}"
            },
            json={
                "model": MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": USER_TEMPLATE.format(
                            question=question,
                            category=category,
                            facts_block=facts_block,
                        ),
                    },
                ],
                "max_tokens": MAX_TOKENS,
                "temperature": 0.0,
            },
            timeout=120.0,
        )

        resp.raise_for_status()

        raw = (
            resp.json()["choices"][0]["message"]["content"]
            .strip()
        )

    rubric = extract_json(raw)

    if not isinstance(rubric, dict):
        raise ValueError(
            f"Expected rubric dict, got {type(rubric)}"
        )

    for tier in (
        "critical",
        "important",
        "optional",
    ):
        rubric.setdefault(tier, [])

    return rubric


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--consolidated",
        required=True,
        help="Output of stage2 (consolidated_facts.json)",
    )

    parser.add_argument(
        "--category",
        required=True,
        choices=list(WEIGHT_PROFILES.keys()),
    )

    parser.add_argument(
        "--out",
        required=True,
    )

    parser.add_argument(
        "--min-agreement",
        type=int,
        default=None,
        help=(
            "Override minimum agreement count filter "
            "(default: use value from consolidated file)"
        ),
    )

    args = parser.parse_args()

    api_key = API_KEY

    with open(args.consolidated, encoding="utf-8") as f:
        data = json.load(f)

    question = data["question"]

    min_agreement = (
        args.min_agreement
        if args.min_agreement is not None
        else data.get("min_agreement", 2)
    )

    all_facts = data["core_facts"]

    facts = [
        item["fact"]
        for item in all_facts
        if item["agreement_count"] >= min_agreement
    ]

    if not facts:
        raise ValueError(
            f"No facts survive min_agreement={min_agreement}. "
            f"Available agreement counts: "
            f"{sorted(set(f['agreement_count'] for f in all_facts), reverse=True)}"
        )

    print(
        f"Inducing rubric for category={args.category}, "
        f"{len(facts)} facts "
        f"(min_agreement={min_agreement})..."
    )

    rubric = asyncio.run(
        induce_rubric(
            question=question,
            category=args.category,
            facts=facts,
            api_key=api_key,
        )
    )

    validate_rubric(
        rubric=rubric,
        facts=facts,
    )

    stats = rubric_stats(rubric)

    if (
        stats["total"] > 0
        and stats["critical"] > 0.8 * stats["total"]
    ):
        print(
            "WARNING: more than 80% of rubric facts "
            "were marked critical."
        )

    if (
        stats["total"] > 0
        and stats["optional"] == 0
    ):
        print(
            "WARNING: rubric contains no optional criteria."
        )

    result = {
        "question": question,
        "category": args.category,
        "weights": WEIGHT_PROFILES[args.category],
        "rubric": rubric,
        "stats": stats,
    }

    with open(
        args.out,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    n = sum(
        len(v)
        for v in rubric.values()
        if isinstance(v, list)
    )

    print(
        f"✓ {n} criteria "
        f"({len(rubric['critical'])} critical, "
        f"{len(rubric['important'])} important, "
        f"{len(rubric['optional'])} optional) "
        f"→ {args.out}"
    )


if __name__ == "__main__":
    main()
