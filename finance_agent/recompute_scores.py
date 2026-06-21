"""
Recompute dealbreaker-gated (FABv2 "Partial Credit"-equivalent) and All-Pass
scores from existing stage5_grade.py output, without re-calling the judge.

This works because everything needed is already in grades.json:
  - per-criterion "met" verdicts for each tier
  - hallucination_detected flag
  - raw_score / max_score (penalty-adjusted, floored at 0)

Usage:
    python recompute_scores.py grades.json
    python recompute_scores.py grades.json --dealbreaker-tier critical
    python recompute_scores.py grades_dir/ --glob "*.json"   # batch mode
"""

import argparse
import glob
import json
import os
from pathlib import Path


def recompute_one(result: dict, dealbreaker_tier: str) -> dict:
    """
    Mutates a single result record in place, adding:
      - score_breakdown.ungated_normalized   (old behavior, unchanged)
      - score_breakdown.dealbreaker_failed
      - score_breakdown.all_pass
      - score_breakdown.normalized           (now the GATED value)
      - result["score"]                      (now the GATED value, matches normalized)

    Original ungated value is preserved under ungated_normalized so nothing
    is lost.
    """
    verdict = result["verdict"]
    breakdown = result["score_breakdown"]

    tiers = ("critical", "important", "optional")

    # --- All-Pass: every criterion in every tier met, and no hallucination ---
    all_met = all(
        c["met"]
        for tier in tiers
        for c in verdict.get(tier, [])
    )
    hallucination_detected = verdict.get("hallucination_detected", False)
    all_pass = 1.0 if (all_met and not hallucination_detected) else 0.0

    # --- Dealbreaker gate: any unmet criterion in the dealbreaker tier, ---
    # --- or a detected hallucination, zeroes the question.             ---
    dealbreaker_criteria = verdict.get(dealbreaker_tier, [])
    dealbreaker_failed = (
        any(not c["met"] for c in dealbreaker_criteria)
        or hallucination_detected
    )

    # raw_score in the existing breakdown already has the hallucination
    # penalty applied and is floored at 0 (see calculate_score in
    # stage5_grade.py), so it IS the ungated numerator.
    raw_score = breakdown["raw_score"]
    max_score = breakdown["max_score"]

    ungated_normalized = round(raw_score / max_score, 4) if max_score > 0 else 0.0
    gated_score = 0.0 if dealbreaker_failed else raw_score
    gated_normalized = round(gated_score / max_score, 4) if max_score > 0 else 0.0

    breakdown["ungated_normalized"] = ungated_normalized
    breakdown["dealbreaker_failed"] = dealbreaker_failed
    breakdown["all_pass"] = all_pass
    breakdown["normalized"] = gated_normalized  # now the gated value

    result["score"] = gated_normalized
    result["score_breakdown"] = breakdown

    return result


def recompute_file(path: str, dealbreaker_tier: str) -> None:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    for r in results:
        recompute_one(r, dealbreaker_tier)

    gated_scores   = [r["score"] for r in results]
    ungated_scores = [r["score_breakdown"]["ungated_normalized"] for r in results]
    all_pass_scores = [r["score_breakdown"]["all_pass"] for r in results]

    n = len(results)
    data["summary"] = {
        "count": n,
        "avg_gated":   round(sum(gated_scores) / n, 4) if n else 0.0,
        "avg_ungated": round(sum(ungated_scores) / n, 4) if n else 0.0,
        "avg_all_pass": round(sum(all_pass_scores) / n, 4) if n else 0.0,
        "dealbreaker_tier": dealbreaker_tier,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✓ {path}")
    print(f"  model:        {data.get('model', '?')}")
    print(f"  n questions:  {n}")
    print(f"  avg_gated:    {data['summary']['avg_gated']:.4f}  (FABv2-comparable)")
    print(f"  avg_ungated:  {data['summary']['avg_ungated']:.4f}  (original metric)")
    print(f"  avg_all_pass: {data['summary']['avg_all_pass']:.4f}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="grades.json file OR a directory (use --glob to select files)")
    parser.add_argument("--dealbreaker-tier", default="critical",
                         help="Which rubric tier counts as dealbreaker (default: critical)")
    parser.add_argument("--glob", default="*.json",
                         help="Glob pattern when path is a directory (default: *.json)")
    args = parser.parse_args()

    p = Path(args.path)
    if p.is_dir():
        files = sorted(glob.glob(str(p / args.glob)))
        if not files:
            print(f"No files matching {args.glob} in {p}")
            return
        for f in files:
            recompute_file(f, args.dealbreaker_tier)
    else:
        recompute_file(str(p), args.dealbreaker_tier)


if __name__ == "__main__":
    main()
