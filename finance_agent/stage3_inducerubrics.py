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
    failures = []

    async def run_all():
      async with httpx.AsyncClient() as client:
        for i, data in enumerate(raw_data):
            try:
                result = await process_one(
                    data=data,
                    client=client,
                    min_agreement_override=args.min_agreement,
                )
                all_results.append(result)
            except Exception as e:
                question = data.get("question", f"<item {i}>")
                print(f"  ✗ FAILED: {question[:60]!r}: {e}")
                failures.append({"index": i, "question": question, "error": str(e)})

    asyncio.run(run_all())

    output = all_results if len(all_results) != 1 else all_results[0]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n→ {args.out}")
    print(f"\n{len(all_results)} succeeded, {len(failures)} failed.")

    if failures:
        failures_path = args.out.replace(".json", "_failures.json")
        with open(failures_path, "w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2, ensure_ascii=False)
        print(f"→ {failures_path}")



if __name__ == "__main__":
    main()

