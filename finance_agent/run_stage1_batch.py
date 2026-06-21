"""
run_stage1_batch.py

Batch wrapper for stage 1 extraction. Runs classify + extract concurrently
for each question in the input file.

Input:  --input   results/glm-test.json   (list of {question, answer})
        --output  facts/glm-test-facts.json

Output: list of {question, category, model, facts, answer_file}
"""

import json
import asyncio
import argparse
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from stage1_extract import classify_question, extract


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


async def main():
    args = parse_args()

    input_file  = Path(args.input)
    output_file = Path(args.output)
    model_name  = input_file.stem

    with open(input_file, encoding="utf-8") as f:
        answers = json.load(f)

    results = []

    for i, item in enumerate(answers):
        question = item["question"]
        answer   = item["answer"]
        print(f"[{i+1}/{len(answers)}] extracting: {question[:60]}...")

        category, facts = await asyncio.gather(
            classify_question(question),
            extract(question=question, answer=answer),
        )

        print(f"  category: {category}, facts: {len(facts)}")

        results.append({
            "question":    question,
            "category":    category,
            "model":       model_name,
            "facts":       facts,
            "answer_file": str(input_file.resolve()),
            "answer_index": i,
        })

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ {len(results)} questions → {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
