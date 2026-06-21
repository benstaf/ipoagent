import json

INPUT  = "../results/glm-5.1-public-spacex-70-20260606-191748.json"
OUTPUT = "../results/glm-test.json"

with open(INPUT, encoding="utf-8") as f:
    data = json.load(f)

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(data[:3], f, indent=2, ensure_ascii=False)

print(f"saved {len(data[:3])} items -> {OUTPUT}")
