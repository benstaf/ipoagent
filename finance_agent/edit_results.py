#!/usr/bin/env python3

import json
from pathlib import Path

from openai import OpenAI

INPUT_FILE = "results.json"
OUTPUT_FILE = "edited_results.json"

MODEL = "gpt-5"

EDITOR_PROMPT = """
You are a third-year investment banking analyst.

Rewrite the answer to improve communication quality.

Requirements:
- Preserve ALL facts.
- Preserve ALL numerical values.
- Preserve ALL calculations.
- Preserve ALL conclusions.
- Preserve ALL source citations and source URLs.
- Preserve any uncertainty expressed in the original answer.

Remove:
- repeated facts
- repeated calculations
- redundant summaries
- unnecessary markdown hierarchy
- boilerplate wording

Prefer:
- answer-first structure
- concise analyst-style writing
- high information density

Do NOT:
- introduce new facts
- recalculate numbers
- change conclusions
- omit material information

Return only the revised answer.
"""


client = OpenAI()


def count_words(text: str) -> int:
    return len(text.split())


def edit_answer(answer: str) -> str:
    response = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "system",
                "content": EDITOR_PROMPT,
            },
            {
                "role": "user",
                "content": answer,
            },
        ],
    )

    return response.output_text.strip()


def main():
    input_path = Path(INPUT_FILE)

    with open(input_path, "r") as f:
        results = json.load(f)

    total_raw_words = 0
    total_edited_words = 0

    edited_count = 0

    for item in results:

        if not item.get("success", False):
            continue

        result = item.get("result", {})

        if "final_answer" not in result:
            continue

        raw_answer = result["final_answer"]

        print(
            f"[{edited_count + 1}] Editing question: "
            f"{item['question'][:80]}..."
        )

        edited_answer = edit_answer(raw_answer)

        raw_words = count_words(raw_answer)
        edited_words = count_words(edited_answer)

        total_raw_words += raw_words
        total_edited_words += edited_words

        result["raw_final_answer"] = raw_answer
        result["edited_final_answer"] = edited_answer

        edited_count += 1

        print(
            f"    {raw_words} -> {edited_words} words "
            f"({100 * edited_words / max(raw_words,1):.1f}% retained)"
        )

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("\nDone.")
    print(f"Edited answers: {edited_count}")
    print(f"Saved to: {OUTPUT_FILE}")

    if total_raw_words:
        compression = total_edited_words / total_raw_words

        print(
            f"Overall compression ratio: "
            f"{compression:.2%}"
        )

        print(
            f"Word count: "
            f"{total_raw_words:,} -> {total_edited_words:,}"
        )


if __name__ == "__main__":
    main()
