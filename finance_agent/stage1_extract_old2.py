"""
Stage 1: Extract atomic facts from a single model answer.

Input:
    --question  "Using FY2025 Connectivity segment figures..."
    --answer    answer.txt  (or inline string)
    --out       facts.json

Output:
{
    "question": "...",
    "model": "...",
    "facts": [
        "fact1",
        "fact2",
        ...
    ]
}

Design goals:
- Facts must be atomic (exactly one claim per fact)
- Facts must be explicit (no inferred information)
- Facts must preserve uncertainty/hedging language
- Duplicate facts from the same answer are removed
"""

import asyncio
import json
import argparse
import os
from pathlib import Path

import httpx


ENRICHMENT_BASE_URL = os.getenv("ENRICHMENT_BASE_URL")
EXTRACTION_MODEL = os.getenv("ENRICHMENT_MODEL")
ENRICHMENT_API_KEY = os.getenv("ENRICHMENT_API_KEY")


MAX_TOKENS = 60000

SYSTEM_PROMPT = """\
You are a financial analyst assistant.

Extract atomic facts from an answer to a financial analysis question about
an S-1 filing.

Definition of an atomic fact:
- A single, self-contained, verifiable claim
- Exactly ONE claim per fact
- Specific: includes numbers, names, dates, metrics, entities, or defined
  terms when present
- Not a restatement of the question
- Not an opinion, speculation, or inference
- Only what the answer explicitly states

IMPORTANT ATOMICITY RULES

If a sentence contains multiple claims, split them.

BAD:
- "Connectivity revenue was $8.241B and grew 14% YoY"

GOOD:
- "Connectivity revenue was $8.241B"
- "Connectivity revenue grew 14% YoY"

BAD:
- "The company disclosed SBC expense and operating margin compression"

GOOD:
- "The company disclosed stock-based compensation expense."
- "Operating margin compressed."

UNCERTAINTY PRESERVATION RULES

Preserve hedging and uncertainty exactly as written.

BAD:
- Answer: "The filing appears to indicate approximately 14% growth."
- Output: "Growth was 14%."

GOOD:
- Output: "The filing appears to indicate approximately 14% growth."

DO NOT EXTRACT:

- Derived calculations
- Ratios computed by the answer
- Conclusions drawn from multiple facts
- Interpretations
- Explanations
- Implications
- Assumptions made by the answer author

Only extract claims explicitly asserted as true in the answer.

DUPLICATE RULES

Do not output duplicate facts.
If two statements assert the same claim, output only one fact.

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

USER_TEMPLATE = """\
Question:
{question}

Answer:
{answer}

Extract all atomic facts as a JSON array of strings.
"""


async def extract(
    question: str,
    answer: str,
    model: str,
    api_key: str,
) -> list[str]:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{ENRICHMENT_BASE_URL}",
            headers={
                "Authorization": f"Bearer {api_key}"
            },
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": USER_TEMPLATE.format(
                            question=question,
                            answer=answer,
                        ),
                    },
                ],
                "max_tokens": MAX_TOKENS,
                "temperature": 0.0,
            },
            timeout=1200,
        )

        resp.raise_for_status()


        data = resp.json()

#        print(json.dumps(data, indent=2)[:3000])

        raw = (
            resp.json()["choices"][0]["message"]["content"]
            .strip()
        )

    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]

        if raw.startswith("json"):
            raw = raw[4:]

        raw = raw.strip()

    facts = json.loads(raw)

    assert isinstance(
        facts, list
    ), f"Expected list, got {type(facts)}"

    # Final cleanup + exact deduplication
    cleaned = []
    seen = set()

    for fact in facts:
        if not isinstance(fact, str):
            continue

        fact = fact.strip()

        if not fact:
            continue

        if fact in seen:
            continue

        seen.add(fact)
        cleaned.append(fact)

    return cleaned


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--question",
        required=True,
    )

    parser.add_argument(
        "--answer",
        required=True,
        help="Path to .txt file or inline string",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Name of model that produced the answer",
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output JSON path",
    )

    args = parser.parse_args()

    api_key = ENRICHMENT_API_KEY

    # Accept file path or inline string
    answer_path = Path(args.answer)

    if answer_path.exists():
        answer = answer_path.read_text()
        answer_file = str(answer_path.resolve())
    else:
        answer = args.answer
        answer_file = None
    facts = asyncio.run(
        extract(
            question=args.question,
            answer=answer,
            model=EXTRACTION_MODEL,
            api_key=api_key,
        )
    )

    result = {
        "question": args.question,
        "model": args.model,
        "facts": facts,
        "answer_file": answer_file,
    }

    with open(args.out, "w") as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"✓ {len(facts)} facts → {args.out}"
    )


if __name__ == "__main__":
    main()
