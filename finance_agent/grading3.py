#!/usr/bin/env python3

import json
import re
import unicodedata
from difflib import SequenceMatcher
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
    "GLM 5.2": ROOT / "results" / "glm52_grades.json",
}

# Fuzzy matching only kicks in after exact normalized matching fails.
FUZZY_THRESHOLD = 0.90   # minimum text similarity to even consider a match
AMBIGUITY_MARGIN = 0.03  # best candidate must beat the runner-up by this much

# ============================================================
# TEXT MATCHING HELPERS
# ============================================================


def normalize(text: str) -> str:
    """Collapse typographic variants (curly quotes, en/em dashes, EUR/euro
    sign, non-breaking spaces) so questions that differ only in encoding
    compare equal."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u20ac", "EUR").replace("\u00a3", "GBP")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_numbers(text: str):
    """All numeric tokens in a question, in order. Used as a hard guard so
    fuzzy matching never merges two questions that differ only by a date,
    quarter, or dollar figure (e.g. Q1 2026 vs Q2 2026 ARPU questions score
    >0.99 on plain text similarity despite being different questions)."""
    return tuple(re.findall(r"\d+(?:\.\d+)?", text))


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def fuzzy_match(unmatched_taxonomy, unmatched_grades):
    """
    Pair leftover taxonomy questions with leftover grade questions by text
    similarity, restricted to pairs whose embedded numbers are identical.
    A match is only accepted if it's an unambiguous best candidate (no
    close second place), and each side is used at most once.

    Returns (matched_triples, still_unmatched_taxonomy, still_unmatched_grades)
    where matched_triples is a list of (taxonomy_norm, grade_norm, score).
    """
    candidates = []
    for tq in unmatched_taxonomy:
        tq_nums = extract_numbers(tq)
        pool = [
            (gq, similarity(tq, gq))
            for gq in unmatched_grades
            if extract_numbers(gq) == tq_nums
        ]
        pool.sort(key=lambda x: x[1], reverse=True)
        if not pool:
            continue
        best_gq, best_score = pool[0]
        second_score = pool[1][1] if len(pool) > 1 else 0.0
        if best_score >= FUZZY_THRESHOLD and (best_score - second_score) >= AMBIGUITY_MARGIN:
            candidates.append((tq, best_gq, best_score))

    # Resolve globally by confidence so no taxonomy/grade question is used twice.
    candidates.sort(key=lambda x: x[2], reverse=True)

    matched, used_tax, used_grade = [], set(), set()
    for tq, gq, score in candidates:
        if tq in used_tax or gq in used_grade:
            continue
        matched.append((tq, gq, score))
        used_tax.add(tq)
        used_grade.add(gq)

    remaining_tax = unmatched_taxonomy - used_tax
    remaining_grade = unmatched_grades - used_grade
    return matched, remaining_tax, remaining_grade


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

taxonomy_questions_raw = list(questions["question"].astype(str).str.strip())

# normalized text -> canonical (first-seen) raw taxonomy question text
norm_to_taxonomy = {}
for q in taxonomy_questions_raw:
    norm_to_taxonomy.setdefault(normalize(q), q)

taxonomy_norm_set = set(norm_to_taxonomy.keys())

print(f"\nLoaded taxonomy: {len(taxonomy_norm_set)} questions")

if len(taxonomy_norm_set) != len(taxonomy_questions_raw):
    print(
        f"WARNING: {len(taxonomy_questions_raw) - len(taxonomy_norm_set)} "
        "taxonomy questions collapsed onto the same normalized text "
        "(possible duplicates in questions_taxonomy.csv)."
    )

# ============================================================
# LOAD SCORES
# ============================================================

all_rows = []
audit_rows = []  # records every fuzzy (non-exact) match for manual review

for model_name, json_file in GRADE_FILES.items():

    print(f"\n{'='*60}")
    print(f"LOADING {model_name}")
    print(f"{'='*60}")

    with open(json_file, "r", encoding="utf-8") as f:
        grades = json.load(f)

    score_map_raw = {}
    for result in grades["results"]:
        q = result["question"].strip()
        score_map_raw[q] = result["score"]

    norm_to_grade = {}
    for q in score_map_raw.keys():
        norm_to_grade.setdefault(normalize(q), q)

    grade_norm_set = set(norm_to_grade.keys())

    exact_matched = taxonomy_norm_set & grade_norm_set
    unmatched_tax = taxonomy_norm_set - grade_norm_set
    unmatched_grade = grade_norm_set - taxonomy_norm_set

    fuzzy_matched, still_unmatched_tax, still_unmatched_grade = fuzzy_match(
        unmatched_tax, unmatched_grade
    )

    # final lookup: normalized taxonomy question -> score
    norm_score_map = {n: score_map_raw[norm_to_grade[n]] for n in exact_matched}
    for tq_norm, gq_norm, score_conf in fuzzy_matched:
        norm_score_map[tq_norm] = score_map_raw[norm_to_grade[gq_norm]]
        audit_rows.append(
            {
                "model": model_name,
                "taxonomy_question": norm_to_taxonomy[tq_norm],
                "grade_question": norm_to_grade[gq_norm],
                "similarity": round(score_conf, 4),
            }
        )

    print(f"Taxonomy questions : {len(taxonomy_norm_set)}")
    print(f"Grade questions    : {len(grade_norm_set)}")
    print(f"Exact matched      : {len(exact_matched)}")
    print(f"Fuzzy matched      : {len(fuzzy_matched)}")
    print(f"Still missing      : {len(still_unmatched_tax)}")
    print(f"Still extra        : {len(still_unmatched_grade)}")

    if still_unmatched_tax:
        print("\nStill missing (no confident match found, review manually):")
        for n in sorted(still_unmatched_tax):
            print("  -", norm_to_taxonomy[n])

    if still_unmatched_grade:
        print("\nStill extra (no confident match found, review manually):")
        for n in sorted(still_unmatched_grade):
            print("  -", norm_to_grade[n])

    for _, row in questions.iterrows():

        q_norm = normalize(str(row["question"]).strip())

        if q_norm not in norm_score_map:
            continue

        all_rows.append(
            {
                "model": model_name,
                "question": str(row["question"]).strip(),
                "domain": row["proposed_domain"],
                "workflow": row["proposed_workflow"],
                "score": norm_score_map[q_norm],
            }
        )

df = pd.DataFrame(all_rows)

print(f"\nTotal scored rows: {len(df)}")

if audit_rows:
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(OUTPUT_DIR / "fuzzy_match_audit.csv", index=False)
    print(f"\n{len(audit_rows)} fuzzy matches written to fuzzy_match_audit.csv -- spot check these.")

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

generated_files = [
    "overall_leaderboard2.csv",
    "domain_scores2.csv",
    "workflow_scores2.csv",
    "domain_winners2.csv",
    "workflow_winners2.csv",
]
if audit_rows:
    generated_files.append("fuzzy_match_audit2.csv")

for f in generated_files:
    print("\u2713", OUTPUT_DIR / f)

print("\nDone.")
