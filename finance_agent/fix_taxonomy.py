#!/usr/bin/env python3
"""
One-off fix: the taxonomy CSV's wording for the Eutelsat/OneWeb question is
missing a clarifying parenthetical that's present in all 5 grade files
(every model run was generated from the same underlying question text, so
this is a single drift point, not 5 separate ones). This patches the CSV to
match what was actually used at grading time, so the question becomes a
normal exact match instead of falling through the missing/extra path.

Run from inside finance_agent/ (same place grading2.py / grading3.py live)
so the ROOT-relative path resolves the same way.
"""

import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TAXONOMY_CSV = ROOT / "data" / "questions_taxonomy.csv"

OLD_TEXT = (
    "Eutelsat reported LEO (OneWeb) revenues of approximately EUR110.5 million "
    "for H1 FY2025-26, representing approximately 60% year-on-year growth. "
    "Using SpaceX's disclosed Q1 2026 Connectivity revenue and annualizing both "
    "figures to a common calendar-year basis, calculate the implied revenue "
    "multiple by which Starlink's run-rate exceeds OneWeb's. What does this "
    "comparison reveal about the relative commercial maturity of the two LEO "
    "broadband networks, and what disclosure differences between the two "
    "filings limit the precision of a direct comparison?"
)

NEW_TEXT = (
    "Eutelsat reported LEO (OneWeb) revenues of approximately \u20ac110.5 million "
    "for H1 FY2025\u201326 (the six months ending December 31, 2025), representing "
    "approximately 60% year-on-year growth. Using SpaceX's disclosed Q1 2026 "
    "Connectivity revenue and annualizing both figures to a common calendar-year "
    "basis, calculate the implied revenue multiple by which Starlink's run-rate "
    "exceeds OneWeb's. What does this comparison reveal about the relative "
    "commercial maturity of the two LEO broadband networks, and what disclosure "
    "differences between the two filings limit the precision of a direct "
    "comparison?"
)

df = pd.read_csv(TAXONOMY_CSV)

mask = df["question"].astype(str).str.strip() == OLD_TEXT
n_matches = int(mask.sum())

if n_matches != 1:
    raise SystemExit(
        f"Expected exactly 1 row matching OLD_TEXT, found {n_matches}. "
        "Aborting without writing anything -- the wording in the CSV may "
        "have already changed since this script was written. Check by hand."
    )

backup_path = TAXONOMY_CSV.with_suffix(".csv.bak")
shutil.copy(TAXONOMY_CSV, backup_path)
print(f"Backed up original to: {backup_path}")

df.loc[mask, "question"] = NEW_TEXT
df.to_csv(TAXONOMY_CSV, index=False, encoding="utf-8")

print(f"Patched: {TAXONOMY_CSV}")
print("  old:", OLD_TEXT[:90], "...")
print("  new:", NEW_TEXT[:90], "...")
print("\nRe-run grading3.py -- this question should now exact-match in all 5 models.")
