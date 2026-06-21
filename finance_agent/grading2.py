#!/usr/bin/env python3

import json
from pathlib import Path

import pandas as pd

# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

TAXONOMY_CSV = ROOT / "data" / "questions_taxonomy.csv"
OUTPUT_DIR = ROOT / "results" / "grading"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GRADE_FILES = {
    "Kimi K2.6": ROOT / "results" / "kimi26_grades.json",
    "Qwen 3.7 Max": ROOT / "results" / "qwen37max_grades.json",
    "Mimo 2.5 Pro": ROOT / "results" / "mimo25pro_grades.json",
    "Nemotron 3 Ultra": ROOT / "results" / "nemotron3ultra_grades.json",
    "GLM 5.1": ROOT / "results" / "glm51_grades.json",
}

# ============================================================
# LOAD TAXONOMY
# ============================================================

questions = pd.read_csv(TAXONOMY_CSV)

required = [
    "question",
    "proposed_domain",
    "proposed_workflow",
]

for col in required:
    if col not in questions.columns:
        raise ValueError(f"Missing column: {col}")

taxonomy_questions = set(
    questions["question"].astype(str).str.strip()
)

print(f"\nLoaded taxonomy: {len(taxonomy_questions)} questions")

# ============================================================
# LOAD SCORES
# ============================================================

all_rows = []

for model_name, json_file in GRADE_FILES.items():

    print(f"\n{'='*60}")
    print(f"LOADING {model_name}")
    print(f"{'='*60}")

    with open(json_file, "r", encoding="utf-8") as f:
        grades = json.load(f)

    score_map = {}

    for result in grades["results"]:
        q = result["question"].strip()
        score_map[q] = result["score"]

    grade_questions = set(score_map.keys())

    matched = taxonomy_questions & grade_questions
    missing_from_grades = taxonomy_questions - grade_questions
    extra_in_grades = grade_questions - taxonomy_questions

    print(f"Taxonomy questions : {len(taxonomy_questions)}")
    print(f"Grade questions    : {len(grade_questions)}")
    print(f"Matched            : {len(matched)}")
    print(f"Missing            : {len(missing_from_grades)}")
    print(f"Extra              : {len(extra_in_grades)}")

    if missing_from_grades:
        print("\nMissing examples:")
        for q in list(sorted(missing_from_grades))[:5]:
            print("  -", q[:120])

    if extra_in_grades:
        print("\nExtra examples:")
        for q in list(sorted(extra_in_grades))[:5]:
            print("  -", q[:120])

    for _, row in questions.iterrows():

        q = str(row["question"]).strip()

        if q not in score_map:
            continue

        all_rows.append(
            {
                "model": model_name,
                "question": q,
                "domain": row["proposed_domain"],
                "workflow": row["proposed_workflow"],
                "score": score_map[q],
            }
        )

df = pd.DataFrame(all_rows)

print(f"\nTotal scored rows: {len(df)}")

# ============================================================
# OVERALL LEADERBOARD
# ============================================================

overall = (
    df.groupby("model")
    .agg(
        mean_score=("score", "mean"),
        median_score=("score", "median"),
        min_score=("score", "min"),
        max_score=("score", "max"),
        n=("score", "count"),
    )
    .sort_values("mean_score", ascending=False)
)

overall["rank"] = range(1, len(overall) + 1)

overall.to_csv(
    OUTPUT_DIR / "overall_leaderboard.csv"
)

# ============================================================
# DOMAIN SCORES
# ============================================================

domain_scores = (
    df.groupby(["domain", "model"])
    .agg(
        mean_score=("score", "mean"),
        median_score=("score", "median"),
        count=("score", "count"),
    )
    .reset_index()
)

domain_scores["rank_within_domain"] = (
    domain_scores
    .groupby("domain")["mean_score"]
    .rank(method="dense", ascending=False)
)

domain_scores.to_csv(
    OUTPUT_DIR / "domain_scores.csv",
    index=False,
)

# ============================================================
# WORKFLOW SCORES
# ============================================================

workflow_scores = (
    df.groupby(["workflow", "model"])
    .agg(
        mean_score=("score", "mean"),
        median_score=("score", "median"),
        count=("score", "count"),
    )
    .reset_index()
)

workflow_scores["rank_within_workflow"] = (
    workflow_scores
    .groupby("workflow")["mean_score"]
    .rank(method="dense", ascending=False)
)

workflow_scores.to_csv(
    OUTPUT_DIR / "workflow_scores.csv",
    index=False,
)

# ============================================================
# DOMAIN WINNERS
# ============================================================

domain_winners = (
    domain_scores
    .sort_values(
        ["domain", "mean_score"],
        ascending=[True, False],
    )
    .groupby("domain")
    .head(1)
)

domain_winners.to_csv(
    OUTPUT_DIR / "domain_winners.csv",
    index=False,
)

# ============================================================
# WORKFLOW WINNERS
# ============================================================

workflow_winners = (
    workflow_scores
    .sort_values(
        ["workflow", "mean_score"],
        ascending=[True, False],
    )
    .groupby("workflow")
    .head(1)
)

workflow_winners.to_csv(
    OUTPUT_DIR / "workflow_winners.csv",
    index=False,
)

# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("OVERALL LEADERBOARD")
print("=" * 60)
print(overall.round(4))

print("\n")
print("=" * 60)
print("DOMAIN WINNERS")
print("=" * 60)
print(
    domain_winners[
        ["domain", "model", "mean_score"]
    ].round(4)
)

print("\n")
print("=" * 60)
print("WORKFLOW WINNERS")
print("=" * 60)
print(
    workflow_winners[
        ["workflow", "model", "mean_score"]
    ].round(4)
)

print("\n")
print("=" * 60)
print("FILES GENERATED")
print("=" * 60)

for f in [
    "overall_leaderboard.csv",
    "domain_scores.csv",
    "workflow_scores.csv",
    "domain_winners.csv",
    "workflow_winners.csv",
]:
    print("✓", OUTPUT_DIR / f)

print("\nDone.")
