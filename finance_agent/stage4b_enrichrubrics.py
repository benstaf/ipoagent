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

