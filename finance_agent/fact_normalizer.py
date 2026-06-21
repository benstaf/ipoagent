"""
fact_normalizer.py

Deterministic quantitative fact grouping (Option 4) and
incremental qualitative consolidation (Option 2).

Imported by stage2_consolidate.py. No dependency on pipeline_shared
except the constants URL, API_KEY, MODEL, MAX_TOKENS_CLUSTER and
the helpers extract_json / canonicalize_cluster, which are passed in
or imported directly.
"""

import asyncio
import re
from collections import defaultdict

import httpx
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Quantitative normalization
# ---------------------------------------------------------------------------

def parse_value(s: str) -> float | None:
    s = s.replace(",", "")
    match = re.search(r"\$?([\d]+\.?[\d]*)\s*(B|M|K|%)?", s, re.IGNORECASE)
    if not match:
        return None
    val = float(match.group(1))
    suffix = (match.group(2) or "").upper()
    if suffix == "B":
        val *= 1000
    elif suffix == "K":
        val /= 1000
    return round(val, 1)


def extract_period(s: str) -> str:
    match = re.search(
        r"(FY\s*20\d\d|Q[1-4]\s*20\d\d|20\d\d)", s, re.IGNORECASE
    )
    return match.group(0).upper().replace(" ", "") if match else "UNKNOWN"


def is_quantitative(fact: str) -> bool:
    return bool(re.search(r"\d", fact))


def make_quant_key(fact: str) -> tuple | None:
    val = parse_value(fact)
    if val is None:
        return None
    period = extract_period(fact)
    rounded = round(val / 5) * 5
    return (period, rounded)


def split_facts_by_type(
    model_facts: list[dict],
) -> tuple[list[dict], list[dict]]:
    quant, qual = [], []
    for item in model_facts:
        q_facts = [f for f in item["facts"] if is_quantitative(f)]
        t_facts = [f for f in item["facts"] if not is_quantitative(f)]
        if q_facts:
            quant.append({"model": item["model"], "facts": q_facts})
        if t_facts:
            qual.append({"model": item["model"], "facts": t_facts})
    return quant, qual


def group_quant_facts(model_facts: list[dict]) -> list[list[dict]]:
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    ungrouped: list[dict] = []

    for item in model_facts:
        for fact in item["facts"]:
            if not is_quantitative(fact):
                continue
            key = make_quant_key(fact)
            entry = {"model": item["model"], "fact": fact}
            if key:
                buckets[key].append(entry)
            else:
                ungrouped.append(entry)

    clusters = list(buckets.values())
    for entry in ungrouped:
        clusters.append([entry])
    return clusters


# ---------------------------------------------------------------------------
# Incremental qualitative consolidation
# ---------------------------------------------------------------------------

INCREMENTAL_SYSTEM = """\
You are a financial analyst assistant. You have a growing list of canonical facts
extracted from model answers to the same question.

You will receive:
1. The current canonical fact list (may be empty on first call)
2. One new model's facts

Your job:
- For each new fact, decide if it matches an existing canonical fact (same claim).
  If yes: add this model to that fact's supporters. Do NOT duplicate.
- If no match: add it as a new canonical fact.

Return ONLY a JSON array:
[
  {"fact": "canonical phrasing", "models": ["model_a", "model_b"]},
  ...
]

No preamble. No markdown.
"""

INCREMENTAL_USER = """\
Question: {question}

Current canonical facts:
{canonical_block}

New model: {model_name}
New facts:
{new_facts_block}

Merge and return updated canonical list.
"""


async def incremental_consolidate_qual(
    client: httpx.AsyncClient,
    question: str,
    qual_facts: list[dict],
    url: str,
    api_key: str,
    model: str,
    max_tokens: int,
    extract_json_fn,
) -> list[dict]:
    """
    Merge qualitative facts one model at a time.
    Returns list of {fact, models: [...]}.

    Caller passes url/api_key/model/max_tokens/extract_json_fn to avoid
    importing pipeline_shared here (keeps the dependency graph clean).
    """
    canonical: list[dict] = []

    for item in qual_facts:
        if not item["facts"]:
            continue

        canonical_block = (
            "\n".join(
                f"- {c['fact']} (from: {', '.join(c['models'])})"
                for c in canonical
            )
            or "(none yet)"
        )
        new_facts_block = "\n".join(f"- {f}" for f in item["facts"])

        resp = await client.post(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": INCREMENTAL_SYSTEM},
                    {"role": "user", "content": INCREMENTAL_USER.format(
                        question=question,
                        canonical_block=canonical_block,
                        model_name=item["model"],
                        new_facts_block=new_facts_block,
                    )},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "thinking": {"type": "disabled"},
            },
            timeout=1800,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        canonical = extract_json_fn(raw)

    return canonical
