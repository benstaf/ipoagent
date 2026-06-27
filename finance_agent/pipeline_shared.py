"""
pipeline_shared.py

Shared constants, helpers, and async LLM wrappers used by
stages 2, 3, and 4b. Import from here instead of duplicating.

Environment variables read at import time:
    ENRICHMENT_MODEL
    ENRICHMENT_BASE_URL
    ENRICHMENT_API_KEY
"""

import asyncio
import json
import os
import re
from dataclasses import dataclass, field

import httpx
from dotenv import load_dotenv

load_dotenv()

MODEL   = os.environ["ENRICHMENT_MODEL"]
URL     = os.environ["ENRICHMENT_BASE_URL"]
API_KEY = os.environ["ENRICHMENT_API_KEY"]

MAX_TOKENS_CLUSTER = 120000
MAX_TOKENS_CANON   = 3000
MAX_TOKENS_RUBRIC  = 10000

MIN_AGREEMENT_DEFAULT = 2


# ---------------------------------------------------------------------------
# Weight profiles
# ---------------------------------------------------------------------------

WEIGHT_PROFILES = {
    "quantitative": {
        "critical":      12.0,
        "important":      6.0,
        "optional":       2.0,
        "hallucination": -10.0,
    },
    "disclosure": {
        "critical":      10.0,
        "important":      5.0,
        "optional":       2.0,
        "hallucination":  -6.0,
    },
    "governance": {
        "critical":      10.0,
        "important":      5.0,
        "optional":       2.0,
        "hallucination":  -6.0,
    },
    "forensic": {
        "critical":      10.0,
        "important":      5.0,
        "optional":       2.0,
        "hallucination":  -8.0,
    },
    "modeling": {
        "critical":      10.0,
        "important":      5.0,
        "optional":       2.0,
        "hallucination":  -8.0,
    },
    "comparative": {
        "critical":      10.0,
        "important":      5.0,
        "optional":       2.0,
        "hallucination":  -6.0,
    },
}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CLUSTER_SYSTEM = """\
You are a financial analyst assistant. You will receive several lists of \
atomic facts extracted from different model answers to the same question.

Your ONLY job in this step is to identify which facts assert the same claim, \
even if phrased differently. Do NOT merge them yet — just group them.

Rules:
1. Two facts belong in the same cluster if and only if they assert an \
   identical underlying claim. Different numbers or different metrics must \
   never share a cluster.
   EXAMPLE — keep separate:
     "Connectivity revenue was $8,241M"
     "Connectivity revenue grew 14% YoY"
   EXAMPLE — merge:
     "Connectivity revenue was $8,241M"
     "Connectivity revenue reached $8.241B"
2. Each fact appears in exactly one cluster.
3. A singleton cluster (one fact) is fine.

Return ONLY a JSON array of clusters. Each cluster is an array of objects:
[
  [
    {"model": "model_name", "fact": "fact string"},
    {"model": "model_name", "fact": "fact string"}
  ],
  [
    {"model": "model_name", "fact": "fact string"}
  ],
  ...
]

No preamble, no markdown fences.
"""

CLUSTER_USER = """\
Question: {question}

Facts by model:
{facts_block}

Return the clusters as a JSON array.
"""

CANON_SYSTEM = """\
You are a financial analyst assistant. You will receive a small group of \
facts that all assert the same underlying claim, phrased differently by \
different models.

Produce ONE canonical phrasing:
- Prefer the most precise and specific version.
- If exact numbers appear in any variant, preserve them.
- Keep it as a single atomic claim (one number, one metric, one event).

Return ONLY a JSON object:
{"fact": "canonical fact string"}

No preamble, no markdown fences.
"""

CANON_USER = """\
Variants:
{variants_block}

Return the single best canonical phrasing.
"""

RUBRIC_INDUCTION_SYSTEM = """\
You are a financial analyst building a grading rubric for a benchmark
question about an S-1 filing.

You will receive:
1. The question
2. A list of consolidated facts that represent the ground-truth answer content

Your job is to classify each fact into one of three tiers:

- critical:
    A complete answer MUST include this.
    Missing it means the answer fundamentally fails the question.

- important:
    A good answer should include this.
    Missing it is a notable gap but not a total failure.

- optional:
    Adds depth or precision. Bonus credit.

Rules:
- Every fact must appear in exactly one tier.
- Do not invent new facts not present in the input list.
- Every rubric MUST have at least one fact in each tier. If no
  interpretive facts exist in the input, move the least essential
  numeric input to optional.

- For quantitative questions:
    CRITICAL — final numeric answers that directly answer the question:
      growth rates, calculated ratios, derived percentages, absolute
      figures explicitly requested. Derived calculations (e.g. capex
      intensity = capex/revenue) count as final answers and belong here.
    IMPORTANT — raw input values used to derive critical answers,
      intermediate values, methodology explanations, definitions,
      exclusions, and comparative statements ("X exceeded Y").
    OPTIONAL — facts using interpretive or qualitative language to
      characterize what numbers mean: words like "implies", "indicates",
      "reflects", "exhibits", "suggests", "phase", "stage", "maturity",
      "buildout", "leverage". Conclusions about business implications
      belong here.

- For disclosure/governance questions:
    What is disclosed vs what is not disclosed may both be critical.

- For forensic questions:
    The accounting mechanism → critical.
    Direction of distortion → important.
    Magnitude estimates → optional.

- Hard rule (all categories): if a fact characterizes the meaning or
  implication of numbers using qualitative language, it belongs in
  optional, never critical or important. If a fact states a number or
  derived calculation, it belongs in critical or important based on
  whether it directly answers the question.

Return ONLY:
{
  "critical": [...],
  "important": [...],
  "optional": [...]
}

Use fact strings exactly as provided. No markdown. No explanation.
"""

RUBRIC_INDUCTION_USER = """\
Question: {question}

Category: {category}

Consolidated facts:
{facts_block}

Classify each fact into critical / important / optional.
"""

CONTRADICTION_SYSTEM = """\
You are a financial fact checker.

You will receive a list of indexed fact clusters extracted from model answers
to the same question. Each cluster is a group of facts that assert the same
underlying claim.

Your job: identify pairs of clusters that directly contradict each other —
i.e. they refer to the same metric and time period but state different values.

Rules:
- Only flag genuine numerical contradictions (same metric, same period,
  different number).
- Do not flag facts that are about different metrics or different periods.
- Do not flag facts where one is more specific than the other but not
  contradictory (e.g. "$8.2B" vs "$8,241M" — these are the same).

Return ONLY a JSON array of contradicting pairs:
[
  {"cluster_a": 0, "cluster_b": 2, "reason": "one sentence explanation"},
  ...
]

If there are no contradictions, return [].
No preamble. No markdown.
"""

CONTRADICTION_USER = """\
Question: {question}

Fact clusters (indexed):
{clusters_block}

Identify contradicting pairs.
"""


# ---------------------------------------------------------------------------
# Manual review flags
# ---------------------------------------------------------------------------

@dataclass
class ManualReviewFlag:
    reason: str
    facts:  list[str]
    detail: str = ""


def _print_flag(flag: ManualReviewFlag) -> None:
    print(f"\n    ⚑  MANUAL REVIEW REQUIRED: {flag.reason}")
    print(f"       {flag.detail}")
    for f in flag.facts:
        print(f"       · {f}")
    print()


def _flag_to_dict(flag: ManualReviewFlag) -> dict:
    return {
        "reason": flag.reason,
        "facts":  flag.facts,
        "detail": flag.detail,
    }


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def extract_json(raw: str):
    """Extract the first JSON structure from a model response."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw)
        raw = raw.replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Try array first, then object
    match = re.search(r"\[.*\]", raw, flags=re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(
        f"Could not extract JSON from model output:\n{raw[:1000]}"
    )



def repair_rubric(rubric: dict, facts: list[str]) -> dict:
    """
    Two-pass repair:
    1. Replace any rewritten facts with their canonical originals (fuzzy match).
    2. Append any still-missing facts to 'important'.
    """
    # Build a normalizer: lowercase + collapse whitespace
    def normalize(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip().lower())

    canon_by_norm = {normalize(f): f for f in facts}

    # Pass 1: fix rewrites tier by tier
    for tier in ("critical", "important", "optional"):
        repaired = []
        seen_in_tier: set[str] = set()
        for f in rubric.get(tier, []):
            norm = normalize(f)
            if norm in canon_by_norm:
                canonical = canon_by_norm[norm]
                if f != canonical:
                    print(f"  WARNING: rewrite fixed in '{tier}': {f!r} → {canonical!r}")
                if canonical not in seen_in_tier:
                    repaired.append(canonical)
                    seen_in_tier.add(canonical)
            else:
                # Unknown fact — drop it, validate_rubric will still catch
                # anything that survives as a genuine hallucination
                print(f"  WARNING: unknown fact dropped from '{tier}': {f!r}")
        rubric[tier] = repaired

    # Pass 2: append anything still missing
    assigned_set = set(
        rubric["critical"] + rubric["important"] + rubric["optional"]
    )
    missing = [f for f in facts if f not in assigned_set]
    if missing:
        print(
            f"  WARNING: rubric omitted {len(missing)} fact(s), "
            f"appending to 'important': {missing}"
        )
        rubric["important"].extend(missing)

    return rubric

def validate_rubric(rubric: dict, facts: list[str]) -> None:
    assigned = (
        rubric.get("critical",  [])
        + rubric.get("important", [])
        + rubric.get("optional",  [])
    )
    assigned_set = set(assigned)
    fact_set     = set(facts)
    missing = fact_set - assigned_set
    extra   = assigned_set - fact_set
    if missing:
        raise ValueError(f"Rubric omitted facts: {sorted(missing)}")
    if extra:
        raise ValueError(f"Rubric introduced unknown facts: {sorted(extra)}")
    if len(assigned) != len(assigned_set):
        raise ValueError("Rubric contains duplicate facts across tiers.")


def rubric_stats(rubric: dict) -> dict:
    total = (
        len(rubric.get("critical",  []))
        + len(rubric.get("important", []))
        + len(rubric.get("optional",  []))
    )
    return {
        "critical":  len(rubric.get("critical",  [])),
        "important": len(rubric.get("important", [])),
        "optional":  len(rubric.get("optional",  [])),
        "total":     total,
    }


def build_facts_block(model_facts: list[dict]) -> str:
    lines = []
    for item in model_facts:
        lines.append(f"Model: {item['model']}")
        for fact in item["facts"]:
            lines.append(f"  - {fact}")
        lines.append("")
    return "\n".join(lines)


def build_gaps_block(quality: dict) -> str:
    lines = []
    for issue in quality.get("issues", []):
        lines.append(f"Issue: {issue}")
    for suggestion in quality.get("suggestions", []):
        lines.append(f"Suggestion: {suggestion}")
    return "\n".join(lines) if lines else "(none)"


# ---------------------------------------------------------------------------
# Async LLM wrappers
# ---------------------------------------------------------------------------

async def cluster_facts(
    client: httpx.AsyncClient,
    question: str,
    model_facts: list[dict],
) -> list[list[dict]]:
    """Pass 1: group facts from multiple models into same-claim clusters."""
    facts_block = build_facts_block(model_facts)

    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 2.0

    resp = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.post(
                URL,
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": CLUSTER_SYSTEM},
                        {"role": "user",   "content": CLUSTER_USER.format(
                            question=question,
                            facts_block=facts_block,
                        )},
                    ],
                    "max_tokens": MAX_TOKENS_CLUSTER,
                    "temperature": 0.0,
                    "thinking": {"type": "disabled"},
                },
                timeout=2800,
            )
            resp.raise_for_status()
            break
        except Exception as e:
            print(f"    [cluster_facts] attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
            if attempt == MAX_RETRIES - 1:
                raise
            await asyncio.sleep(RETRY_BASE_DELAY ** attempt)

    payload = resp.json()
    if "choices" not in payload:
        raise ValueError(
            f"cluster_facts: API returned no 'choices'. "
            f"Payload: {json.dumps(payload)[:2000]}"
        )
    choice = payload["choices"][0]
    finish_reason = choice.get("finish_reason")

    if finish_reason == "length":
        total_facts = sum(len(m["facts"]) for m in model_facts)
        raise ValueError(
            f"cluster_facts: response truncated (finish_reason=length). "
            f"{total_facts} input facts, MAX_TOKENS_CLUSTER={MAX_TOKENS_CLUSTER}. "
            f"Raise MAX_TOKENS_CLUSTER."
        )

    raw = choice["message"]["content"].strip()
    clusters = extract_json(raw)
    assert isinstance(clusters, list)
    return clusters


async def canonicalize_cluster(
    client: httpx.AsyncClient,
    cluster: list[dict],
) -> str:
    """Pass 2: collapse a cluster of variants into one canonical fact string."""
    if len(cluster) == 1:
        return cluster[0]["fact"]

    variants_block = "\n".join(f"- {item['fact']}" for item in cluster)

    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 2.0

    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.post(
                URL,
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": CANON_SYSTEM},
                        {"role": "user",   "content": CANON_USER.format(
                            variants_block=variants_block,
                        )},
                    ],
                    "max_tokens": MAX_TOKENS_CANON,
                    "temperature": 0.0,
                    "thinking": {"type": "disabled"},
                },
                timeout=2000,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            return extract_json(raw)["fact"]
        except Exception as e:
            print(f"    [canonicalize] attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
            if attempt == MAX_RETRIES - 1:
                # Fallback: return the most verbose variant rather than crash
                return max(cluster, key=lambda x: len(x["fact"]))["fact"]
            await asyncio.sleep(RETRY_BASE_DELAY ** attempt)


def _coerce_list_rubric(items: list) -> dict | None:
    out = {"critical": [], "important": [], "optional": []}

    if all(isinstance(it, dict) and "tier" in it and "facts" in it for it in items):
        for it in items:
            tier = it["tier"].lower()
            if tier in out:
                out[tier].extend(it["facts"])
        return out

    if all(isinstance(it, dict) and "fact" in it and "tier" in it for it in items):
        for it in items:
            tier = it["tier"].lower()
            if tier in out:
                out[tier].append(it["fact"])
        return out

    return None


async def induce_rubric(
    client: httpx.AsyncClient,
    question: str,
    category: str,
    facts: list[str],
) -> dict:
    facts_block = "\n".join(f"- {f}" for f in facts)
    resp = await client.post(
        URL,
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": RUBRIC_INDUCTION_SYSTEM},
                {"role": "user",   "content": RUBRIC_INDUCTION_USER.format(
                    question=question, category=category, facts_block=facts_block,
                )},
            ],
            "max_tokens": MAX_TOKENS_RUBRIC,
            "temperature": 0.0,
            "thinking": {"type": "disabled"},
        },
        timeout=2000,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()
    rubric = extract_json(raw)

    if not isinstance(rubric, dict):
        if isinstance(rubric, list):
            coerced = _coerce_list_rubric(rubric)
            if coerced is not None:
                print(f"  WARNING: coerced list-shaped rubric for {question[:60]!r}")
                rubric = coerced
            else:
                raise ValueError(
                    f"Expected rubric dict, got list for question={question!r}, "
                    f"category={category!r}. Raw response:\n{raw[:2000]}"
                )
        else:
            raise ValueError(
                f"Expected rubric dict, got {type(rubric)} for question={question!r}. "
                f"Raw response:\n{raw[:2000]}"
            )

    for tier in ("critical", "important", "optional"):
        rubric.setdefault(tier, [])
    return rubric

async def detect_contradictions(
    client: httpx.AsyncClient,
    question: str,
    cluster_meta: list[dict],
) -> list[dict]:
    if len(cluster_meta) < 2:
        return []
    clusters_block = "\n".join(
        f"[{i}] {meta['cluster'][0]['fact']} "
        f"(agreed by {meta['count']} model(s))"
        for i, meta in enumerate(cluster_meta)
    )
    resp = await client.post(
        URL,
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": CONTRADICTION_SYSTEM},
                {"role": "user",   "content": CONTRADICTION_USER.format(
                    question=question,
                    clusters_block=clusters_block,
                )},
            ],
            "max_tokens": 2000,
            "temperature": 0.0,
            "thinking": {"type": "disabled"},
        },
        timeout=2000,
    )
    payload = resp.json()
    print("\n=== CONTRADICTION PAYLOAD ===")
    print(json.dumps(payload, indent=2)[:10000])
    choice = payload["choices"][0]
    print("finish_reason =", choice.get("finish_reason"))
    message = choice.get("message", {})
    print("content repr =", repr(message.get("content")))
    print("message keys =", message.keys())
    raw = message.get("content", "").strip()   # ← consolidated here
    resp.raise_for_status()
    # ← removed redundant second resp.json() call

    try:
        result = extract_json(raw)
    except Exception:
        print("\nFAILED CONTRADICTION RESPONSE:")
        print(repr(raw[:5000]))
        raise

    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        return result.get("contradictions", [])
    return []



ROUTING_SYSTEM = """\
You are a rubric pipeline router for a financial analysis benchmark.

You will receive a rubric quality report with scores, issues, and suggestions.

Your job is to decide what the next pipeline step should be:

- "enrich": the rubric is missing facts or has coverage gaps. More facts need
  to be extracted from model answers and added to the rubric.
  Signal: coverage score < 0.90, or issues mention missing criteria, gaps,
  facts never stated, criteria that should exist but don't, implication
  criteria missing, interpretive conclusions absent, question asks for X
  but no criterion addresses X.

- "repair": the rubric has sufficient facts but they are structured incorrectly.
  Existing facts need to be split, reclassified, or deduplicated.
  Signal: coverage >= 0.90, and atomicity or tier_correctness < 0.80, and
  issues mention vague criteria, double-barreled claims, misclassified tiers,
  redundancy, tangential criteria.

- "stop": the rubric is good enough. No further action needed.
  Signal: overall >= 0.95, or only minor cosmetic suggestions with no issues.

Priority rules:
- If coverage < 0.90, always return "enrich" regardless of structural issues.
  Structural problems are easier to fix after facts are complete.
- If coverage >= 0.90 and structural issues exist, return "repair".
- Never return "stop" if overall < 0.85 and there are unresolved issues.
- Never return "repair" if issues describe facts that are absent entirely
  rather than present but misclassified.

Return ONLY a JSON object:
{"recommendation": "enrich" | "repair" | "stop", "reason": "one sentence"}

No preamble. No markdown.
"""

ROUTING_USER = """\
Overall score: {overall}
Scores: {scores}

Issues:
{issues}

Suggestions:
{suggestions}
"""

async def recommend_next_step(quality: dict) -> str:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            URL,
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": ROUTING_SYSTEM},
                    {
                        "role": "user",
                        "content": ROUTING_USER.format(
                            overall=quality.get("overall", 0.0),
                            scores=json.dumps(quality.get("scores", {})),
                            issues="\n".join(
                                f"- {i}" for i in quality.get("issues", [])
                            ) or "(none)",
                            suggestions="\n".join(
                                f"- {s}" for s in quality.get("suggestions", [])
                            ) or "(none)",
                        ),
                    },
                ],
                "max_tokens": 200,
                "temperature": 0.0,
                "thinking": {"type": "disabled"},
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        result = extract_json(raw)
        recommendation = result["recommendation"]
        reason = result.get("reason", "")
        print(f"  Next step: {recommendation.upper()} — {reason}")
        return recommendation




