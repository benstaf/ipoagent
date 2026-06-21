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
