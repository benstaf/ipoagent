#!/usr/bin/env python3

import json
import pandas as pd
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

TAXONOMY_CSV = ROOT / "data" / "questions_taxonomy.csv"

GRADE_FILES = {
    "Kimi K2.6": ROOT / "results" / "kimi26_grades.json",
    "Qwen 3.7 Max": ROOT / "results" / "qwen37max_grades.json",
    "Mimo 2.5 Pro": ROOT / "results" / "mimo25pro_grades.json",
    "Nemotron 3 Ultra": ROOT / "results" / "nemotron3ultra_grades.json",
    "GLM 5.1": ROOT / "results" / "glm51_grades.json",
}


# ============================================================
# LOAD QUESTION TAXONOMY
# ============================================================

questions = pd.read_csv(TAXONOMY_CSV)

required_cols = [
    "question",
    "proposed_domain",
    "proposed_workflow"
]

for col in required_cols:
    if col not in questions.columns:
        raise ValueError(f"Missing column: {col}")

# ============================================================
# LOAD ALL MODEL RESULTS
# ============================================================

all_rows = []

for model_name, json_file in GRADE_FILES.items():

    print(f"Loading {model_name}...")

    with open(json_file, "r", encoding="utf-8") as f:
        grades = json.load(f)

    score_map = {}

    for result in grades["results"]:
        score_map[result["question"].strip()] = result["score"]

    matched = 0

    for _, row in questions.iterrows():

        q = row["question"].strip()

        if q not in score_map:
            continue

        matched += 1

        all_rows.append({
            "model": model_name,
            "question": q,
            "domain": row["proposed_domain"],
            "workflow": row["proposed_workflow"],
            "score": score_map[q]
        })

    print(f"  matched {matched} questions")

df = pd.DataFrame(all_rows)

print("\nTotal scored rows:", len(df))

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
          n=("score", "count")
      )
      .sort_values("mean_score", ascending=False)
)

overall["rank"] = range(1, len(overall) + 1)

overall.to_csv("overall_leaderboard.csv")

print("\n==============================")
print("OVERALL LEADERBOARD")
print("==============================")
print(overall.round(4))

# ============================================================
# DOMAIN PERFORMANCE
# ============================================================

domain_perf = (
    df.groupby(["domain", "model"])
      .agg(
          mean_score=("score", "mean"),
          median_score=("score", "median"),
          count=("score", "count")
      )
      .reset_index()
)

domain_perf.to_csv(
    "domain_scores_by_model.csv",
    index=False
)

# ============================================================
# DOMAIN RANKINGS
# ============================================================

domain_rankings = []

for domain in sorted(df["domain"].unique()):

    subset = (
        domain_perf[domain_perf["domain"] == domain]
        .sort_values("mean_score", ascending=False)
        .reset_index(drop=True)
    )

    subset["rank"] = subset.index + 1

    domain_rankings.append(subset)

domain_rankings = pd.concat(domain_rankings)

domain_rankings.to_csv(
    "domain_rankings.csv",
    index=False
)

# ============================================================
# WORKFLOW PERFORMANCE
# ============================================================

workflow_perf = (
    df.groupby(["workflow", "model"])
      .agg(
          mean_score=("score", "mean"),
          median_score=("score", "median"),
          count=("score", "count")
      )
      .reset_index()
)

workflow_perf.to_csv(
    "workflow_scores_by_model.csv",
    index=False
)

# ============================================================
# WORKFLOW RANKINGS
# ============================================================

workflow_rankings = []

for workflow in sorted(df["workflow"].unique()):

    subset = (
        workflow_perf[workflow_perf["workflow"] == workflow]
        .sort_values("mean_score", ascending=False)
        .reset_index(drop=True)
    )

    subset["rank"] = subset.index + 1

    workflow_rankings.append(subset)

workflow_rankings = pd.concat(workflow_rankings)

workflow_rankings.to_csv(
    "workflow_rankings.csv",
    index=False
)

# ============================================================
# DOMAIN ADVANTAGE
# ============================================================

domain_avg = (
    domain_perf
    .groupby("domain")
    .agg(
        domain_average=("mean_score", "mean")
    )
    .reset_index()
)

domain_advantage = domain_perf.merge(
    domain_avg,
    on="domain"
)

domain_advantage["advantage"] = (
    domain_advantage["mean_score"]
    - domain_advantage["domain_average"]
)

domain_advantage = domain_advantage.sort_values(
    ["domain", "advantage"],
    ascending=[True, False]
)

domain_advantage.to_csv(
    "domain_advantage.csv",
    index=False
)

# ============================================================
# BEST DOMAIN SPECIALISTS
# ============================================================

best_domain_specialists = (
    domain_advantage
    .sort_values(
        ["domain", "advantage"],
        ascending=[True, False]
    )
    .groupby("domain")
    .first()
    .reset_index()
)

best_domain_specialists.to_csv(
    "best_domain_specialists.csv",
    index=False
)

# ============================================================
# WORKFLOW ADVANTAGE
# ============================================================

workflow_avg = (
    workflow_perf
    .groupby("workflow")
    .agg(
        workflow_average=("mean_score", "mean")
    )
    .reset_index()
)

workflow_advantage = workflow_perf.merge(
    workflow_avg,
    on="workflow"
)

workflow_advantage["advantage"] = (
    workflow_advantage["mean_score"]
    - workflow_advantage["workflow_average"]
)

workflow_advantage = workflow_advantage.sort_values(
    ["workflow", "advantage"],
    ascending=[True, False]
)

workflow_advantage.to_csv(
    "workflow_advantage.csv",
    index=False
)

# ============================================================
# BEST WORKFLOW SPECIALISTS
# ============================================================

best_workflow_specialists = (
    workflow_advantage
    .sort_values(
        ["workflow", "advantage"],
        ascending=[True, False]
    )
    .groupby("workflow")
    .first()
    .reset_index()
)

best_workflow_specialists.to_csv(
    "best_workflow_specialists.csv",
    index=False
)

# ============================================================
# DOMAIN HEATMAP
# ============================================================

domain_heatmap = pd.pivot_table(
    domain_perf,
    values="mean_score",
    index="domain",
    columns="model"
)

domain_heatmap.to_csv(
    "domain_heatmap.csv"
)

# ============================================================
# WORKFLOW HEATMAP
# ============================================================

workflow_heatmap = pd.pivot_table(
    workflow_perf,
    values="mean_score",
    index="workflow",
    columns="model"
)

workflow_heatmap.to_csv(
    "workflow_heatmap.csv"
)

# ============================================================
# MODEL × DOMAIN RANK MATRIX
# ============================================================

domain_rank_matrix = (
    domain_rankings
    .pivot(
        index="domain",
        columns="model",
        values="rank"
    )
)

domain_rank_matrix.to_csv(
    "domain_rank_matrix.csv"
)

# ============================================================
# MODEL × WORKFLOW RANK MATRIX
# ============================================================

workflow_rank_matrix = (
    workflow_rankings
    .pivot(
        index="workflow",
        columns="model",
        values="rank"
    )
)

workflow_rank_matrix.to_csv(
    "workflow_rank_matrix.csv"
)

# ============================================================
# PRINT PAPER SUMMARY
# ============================================================

print("\n==============================")
print("BEST DOMAIN SPECIALISTS")
print("==============================")
print(
    best_domain_specialists[
        ["domain", "model", "mean_score", "advantage"]
    ].round(4)
)

print("\n==============================")
print("BEST WORKFLOW SPECIALISTS")
print("==============================")
print(
    best_workflow_specialists[
        ["workflow", "model", "mean_score", "advantage"]
    ].round(4)
)

print("\n==============================")
print("FILES GENERATED")
print("==============================")

files = [
    "overall_leaderboard.csv",
    "domain_scores_by_model.csv",
    "workflow_scores_by_model.csv",
    "domain_rankings.csv",
    "workflow_rankings.csv",
    "domain_advantage.csv",
    "workflow_advantage.csv",
    "best_domain_specialists.csv",
    "best_workflow_specialists.csv",
    "domain_heatmap.csv",
    "workflow_heatmap.csv",
    "domain_rank_matrix.csv",
    "workflow_rank_matrix.csv",
]

for f in files:
    print("✓", f)

print("\nDone.")
