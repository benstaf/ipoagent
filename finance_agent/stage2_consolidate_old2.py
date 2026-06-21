"""
Stage 2: Consolidate facts from N model fact files.
"""


import asyncio
import json
import argparse
import os

import httpx



from dotenv import load_dotenv

load_dotenv()

CONSOLIDATION_MODEL = os.environ["ENRICHMENT_MODEL"]
CONSOLIDATION_URL = os.environ["ENRICHMENT_BASE_URL"]
CONSOLIDATION_API_KEY = os.environ["ENRICHMENT_API_KEY"]


MAX_TOKENS_CLUSTER  = 50000
MAX_TOKENS_CANON    = 5000
MIN_AGREEMENT_DEFAULT = 2

# ---------------------------------------------------------------------------
# Pass 1 — clustering
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


def build_facts_block(model_facts: list[dict]) -> str:
    lines = []
    for item in model_facts:
        lines.append(f"Model: {item['model']}")
        for fact in item["facts"]:
            lines.append(f"  - {fact}")
        lines.append("")
    return "\n".join(lines)


async def cluster_facts(
    question: str,
    model_facts: list[dict],
    client: httpx.AsyncClient,
    api_key: str,
) -> list[list[dict]]:
    facts_block = build_facts_block(model_facts)

    resp = await client.post(
        f"{CONSOLIDATION_URL}",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": CONSOLIDATION_MODEL,
            "messages": [
                {"role": "system", "content": CLUSTER_SYSTEM},
                {"role": "user",   "content": CLUSTER_USER.format(
                    question=question,
                    facts_block=facts_block,
                )},
            ],
            "max_tokens": MAX_TOKENS_CLUSTER,
            "temperature": 0.0,
        },
        timeout=1800.0,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    clusters = json.loads(raw)
    assert isinstance(clusters, list)
    return clusters


# ---------------------------------------------------------------------------
# Pass 2 — canonicalize
# ---------------------------------------------------------------------------

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


async def canonicalize_cluster(
    cluster: list[dict],
    client: httpx.AsyncClient,
    api_key: str,
) -> str:
    if len(cluster) == 1:
        return cluster[0]["fact"]

    variants_block = "\n".join(f"- {item['fact']}" for item in cluster)

    resp = await client.post(
        f"{CONSOLIDATION_URL}",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": CONSOLIDATION_MODEL,
            "messages": [
                {"role": "system", "content": CANON_SYSTEM},
                {"role": "user",   "content": CANON_USER.format(
                    variants_block=variants_block,
                )},
            ],
            "max_tokens": MAX_TOKENS_CANON,
            "temperature": 0.0,
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    return json.loads(raw)["fact"]


# ---------------------------------------------------------------------------
# Per-question consolidation
# ---------------------------------------------------------------------------

async def consolidate_question(
    question: str,
    model_facts: list[dict],
    client: httpx.AsyncClient,
    api_key: str,
) -> list[dict]:
    clusters = await cluster_facts(question, model_facts, client, api_key)

    canon_tasks = [
        canonicalize_cluster(cluster, client, api_key)
        for cluster in clusters
    ]
    canonical_facts = await asyncio.gather(*canon_tasks)

    all_facts = []
    for cluster, canon in zip(clusters, canonical_facts):
        agreed_by = list({item["model"] for item in cluster})
        all_facts.append({
            "fact":            canon,
            "agreed_by":       agreed_by,
            "agreement_count": len(agreed_by),
        })

    return all_facts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--facts", nargs="+", required=True,
                        help="One facts JSON file per model")
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-agreement", type=int, default=MIN_AGREEMENT_DEFAULT)
    args = parser.parse_args()

    api_key = CONSOLIDATION_API_KEY

    # Load all model facts files — each is a list of {question, model, facts, answer_file}
    all_model_data = []
    for path in args.facts:
        with open(path, encoding="utf-8") as f:
            all_model_data.append(json.load(f))

    # Validate all files have the same number of questions
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
                    {
                        "model": data[i]["model"],
                        "facts": data[i]["facts"],
                    }
                    for data in all_model_data
                ]

                # Collect answer file paths from each model's fact record
                answer_files = [
                    data[i]["answer_file"]
                    for data in all_model_data
                    if data[i].get("answer_file")
                ]

                all_facts = await consolidate_question(
                    question, model_facts, client, api_key
                )

                core      = [f for f in all_facts if f["agreement_count"] >= args.min_agreement]
                discarded = [f for f in all_facts if f["agreement_count"] <  args.min_agreement]

                results.append({
                    "question":        question,
                    "min_agreement":   args.min_agreement,
                    "core_facts":      core,
                    "discarded_count": len(discarded),
                    "answer_files":    answer_files,   # new
                })

                print(f"  -> {len(core)} core facts, {len(discarded)} discarded")

            return results

    results = asyncio.run(run_all())

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total_core = sum(len(r["core_facts"]) for r in results)
    print(f"\n✓ {total_core} total core facts across {n_questions} questions -> {args.out}")


if __name__ == "__main__":
    main()
