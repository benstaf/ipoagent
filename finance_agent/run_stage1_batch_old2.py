# finance_agent/run_stage1_batch.py
import os
import json
import asyncio
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from stage1_extract import extract

# top of run_stage1_batch.py — replace the hardcoded INPUT/OUTPUT lines
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input",  required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

INPUT_FILE  = Path(args.input)
OUTPUT_FILE = Path(args.output)
MODEL_NAME  = INPUT_FILE.stem




#BASE_DIR   = Path(__file__).resolve().parent.parent
#INPUT_FILE = BASE_DIR / "results" / "glm-5.1.small2.json"
#OUTPUT_FILE = BASE_DIR / "facts" / "glm-5.1.small2-facts.json"

ENRICHMENT_MODEL = os.environ["ENRICHMENT_MODEL"]
MODEL_NAME = INPUT_FILE.stem  # "glm-5.1.small2" — provenance key for stage 2


async def main():
    with open(INPUT_FILE, encoding="utf-8") as f:
        answers = json.load(f)

    api_key = os.environ["ENRICHMENT_API_KEY"]
    results = []

    for i, item in enumerate(answers):
        print(f"[{i+1}/{len(answers)}] extracting...")

        facts = await extract(
            question=item["question"],
            answer=item["answer"],
            model=ENRICHMENT_MODEL,
            api_key=api_key,
        )

        results.append({
            "question": item["question"],
            "model":    MODEL_NAME,   # <-- added
            "facts":    facts,
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"saved {len(results)} questions -> {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
