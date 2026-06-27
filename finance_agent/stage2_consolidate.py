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

def is_glm_only_agreement(fact):
    models = set(fact["agreed_by"])
    return models == {"glm51", "glm52"}


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


                glm_only = [f for f in all_facts if is_glm_only_agreement(f)]

                core      = [f for f in all_facts if (f["agreement_count"] >= args.min_agreement and not is_glm_only_agreement(f)) ]
                discarded = [f for f in all_facts if (f["agreement_count"] <  args.min_agreement or is_glm_only_agreement(f))]


                results.append({
                     "question":        question,
                     "category":        all_model_data[0][i].get("category", "quantitative"),
                     "min_agreement":   args.min_agreement,
                     "core_facts":      core,
                     "discarded_count": len(discarded),
                     "glm_only_count":  len(glm_only),
                     "answer_files":    answer_files,
                })

                print(f"  -> {len(core)} core facts, {len(discarded)} discarded,{len(glm_only)} glm-only")

            return results

    results = asyncio.run(run_all())

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total_core = sum(len(r["core_facts"]) for r in results)
    print(f"\n✓ {total_core} total core facts across {n_questions} questions -> {args.out}")


if __name__ == "__main__":
    main()
