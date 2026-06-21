"""
Stage 1: Extract atomic facts from a single model answer.

Output (facts.json):
{
    "question": "...",
    "model":    "...",
    "facts": [
        {
            "fact":     "...",
            "evidence": "..."   # verbatim span from the answer
        }
    ]
}
"""

import asyncio
import json
import argparse
import os
from pathlib import Path

import httpx

ENRICHMENT_BASE_URL = os.getenv("ENRICHMENT_BASE_URL")
ENRICHMENT_MODEL = os.getenv("ENRICHMENT_MODEL")


MAX_TOKENS  = 2048   # fact lists rarely need more; keeps latency/cost down
MAX_RETRIES = 5

SYSTEM_PROMPT = """\
You are an expert financial analyst.

Your task is to extract atomic facts from an answer to a financial
analysis question about an S-1 filing.

IMPORTANT RULES:

1. Extract facts ONLY from the answer.
   The question provides context but is NEVER evidence.

2. Every fact must be directly supported by explicit text in the answer.

3. Do NOT:
   - infer missing information
   - strengthen claims
   - resolve ambiguities
   - add numbers not present
   - use outside knowledge

4. Split compound statements into separate atomic facts whenever possible.

5. Preserve wording closely to the original answer.

6. For every fact include a short evidence span copied VERBATIM from the answer.

7. A fact should contain exactly one claim whenever possible.

   Bad:
   "Revenue was $12.4B and gross margin was 41%."

   Good:
   "Revenue was $12.4B."
   "Gross margin was 41%."

Return ONLY valid JSON â€” no preamble, no markdown fences.

Format:
[
  {
    "fact":     "...",
    "evidence": "..."
  }
]

If no verifiable facts exist, return [].
"""

USER_TEMPLATE = """\
Question:
{question}

Answer:
{answer}

Extract all atomic facts.
"""


async def _call_extractor(
    client:   httpx.AsyncClient,
    question: str,
    answer:   str,
    api_key:  str,
) -> str:
    resp = await client.post(
        f"{DEEPINFRA_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": EXTRACTION_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_TEMPLATE.format(
                    question=question,
                    answer=answer,
                )},
            ],
            "temperature": 0.0,
            "max_tokens":  MAX_TOKENS,
        },
        timeout=180.0,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _clean_json(raw: str) -> str:
    """Strip accidental markdown code fences."""
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) >= 2:
            raw = parts[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
    return raw


def _deduplicate(facts: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out:  list[dict] = []
    for item in facts:
        key = item["fact"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


async def extract(
    question: str,
    answer:   str,
    api_key:  str,
) -> list[dict]:
    """
    Returns a deduplicated list of {"fact": ..., "evidence": ...} dicts.
    Retries up to MAX_RETRIES times with exponential back-off.
    """
    async with httpx.AsyncClient() as client:
        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                raw   = await _call_extractor(client, question, answer, api_key)
                raw   = _clean_json(raw)
                try:
                    facts = json.loads(raw)
                except json.JSONDecodeError:
                    repaired = raw.strip()
                    start = repaired.find("[")
                    end   = repaired.rfind("]")
                    if start >= 0 and end > start:
                        facts = json.loads(repaired[start:end + 1])
                    else:
                        raise

                if not isinstance(facts, list):
                    raise ValueError(f"Expected JSON list, got {type(facts)}")

                cleaned: list[dict] = []
                for item in facts:
                    if not isinstance(item, dict):
                        continue
                    fact     = item.get("fact",     "").strip()
                    evidence = item.get("evidence", "").strip()
                    if not fact or not evidence:
                        continue
                    cleaned.append({
                        "fact":     " ".join(fact.split()),
                        "evidence": " ".join(evidence.split()),
                    })

                return _deduplicate(cleaned)

            except (
                httpx.TimeoutException,
                httpx.NetworkError,
                httpx.HTTPStatusError,
                json.JSONDecodeError,
            ) as e:
                if isinstance(e, httpx.HTTPStatusError):
                    if e.response.status_code not in {429, 500, 502, 503, 504}:
                        raise
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)

        if last_error is not None:
            raise last_error
        raise RuntimeError("Extraction failed without an exception")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1 â€” extract atomic facts from a model answer."
    )
    parser.add_argument("--question", required=True,
                        help="The benchmark question text")
    parser.add_argument("--answer",   required=True,
                        help="Path to a .txt file OR an inline answer string")
    parser.add_argument("--model",    required=True,
                        help="Model being evaluated (e.g. deepinfra/Kimi-K2)")
    parser.add_argument("--out",      required=True,
                        help="Output .json path")
    args = parser.parse_args()

    api_key = os.environ["DEEPINFRA_API_KEY"]

    answer_path = Path(args.answer)
    answer = answer_path.read_text(encoding="utf-8") if answer_path.exists() else args.answer

    facts = asyncio.run(extract(args.question, answer, api_key))

    result = {
        "question": args.question,
        "model":    args.model,
        "facts":    facts,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"âœ“ extracted {len(facts)} facts â†’ {args.out}")


if __name__ == "__main__":
    main()
