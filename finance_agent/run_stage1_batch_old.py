# finance_agent/run_stage1_batch.py
import os
import json
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env")


from stage1_extract import extract


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "results" / "glm-5.1.small2.json"

OUTPUT_FILE = BASE_DIR / "facts" / "glm-5.1.small2-facts.json"


ENRICHMENT_MODEL = os.environ["ENRICHMENT_MODEL"]



async def main():
    with open(INPUT_FILE, encoding="utf-8") as f:
        answers = json.load(f)

    api_key = os.environ["ENRICHMENT_API_KEY"]

    results = []

    for i, item in enumerate(answers):

        print(
            f"[{i+1}/{len(answers)}] extracting..."
        )

        facts = await extract(
            question=item["question"],
            answer=item["answer"],
            model=ENRICHMENT_MODEL,
            api_key=api_key,
        )

        results.append(
            {
                "question": item["question"],
                "facts": facts,
            }
        )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"saved {len(results)} questions -> {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    asyncio.run(main())
