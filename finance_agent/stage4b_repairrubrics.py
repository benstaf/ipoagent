"""
Stage 4b: Auto-repair rubric structural issues flagged by stage 4.

Input:  --rubric  rubrics_checked.json   (output of stage 4 — must have "quality" block)
Output: --out     rubrics_repaired.json

Routing:
  - Records with "recommendation": "repair"  → rubric is repaired and
    "recommendation" is cleared so Stage 4 re-evaluates them fresh.
  - Records with "recommendation": "stop"    → passed through unchanged.
  - All other records                        → passed through unchanged.

Safety:
  - Repair may only restructure or split existing facts.
  - Repair may NOT introduce new facts not present in the original rubric.
  - Hallucination is judged on numeric/dollar/percentage payload, not on
    literal sentence wording. The ATOMICITY RULES require splitting compound
    claims into independently-readable criteria, which almost always means
    rewording a clause to stand alone (e.g. prepending "Filing states" to a
    fragment pulled out of a list). A plain substring match on full sentences
    rejects that rewording as a "hallucination" even though no information
    was added — checking only the numeric payload preserves the actual
    safety property without blocking legitimate restructuring.
  - If hallucination detected, original rubric is preserved and
    "recommendation" is left as "repair" so the record surfaces for manual
    review. The discarded attempt and the flagged signals are saved under
    "repair_debug" on the record so the rejection is inspectable directly
    from the output file, without re-running with stdout captured.

Exit codes:
  0 — all rubrics at or above threshold after repair
  1 — one or more rubrics still below threshold after repair
"""

import asyncio
import json
import argparse
import os
import re
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()

MODEL   = os.environ["ENRICHMENT_MODEL"]
URL     = os.environ["ENRICHMENT_BASE_URL"]
API_KEY = os.environ["ENRICHMENT_API_KEY"]

MAX_TOKENS        = 8000
QUALITY_THRESHOLD = 0.70
MAX_REPAIR_ATTEMPTS = 5

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

REPAIR_SYSTEM = """\
You are a rubric editor for a financial analysis benchmark.

You will receive a grading rubric with identified structural issues and
suggestions for fixing them, along with the dimension scores that failed.
Your job is to repair ONLY the dimensions that scored below threshold.

SCOPE CONSTRAINT (critical — read before touching anything):
You will be told which dimensions need repair. Do not modify dimensions
that are not listed as needing repair.

Specifically:
- If tier_correctness is NOT listed as needing repair, do NOT move any
  criterion between tiers. Preserve every criterion in its current tier.
- If coverage is NOT listed as needing repair, do NOT add or remove any
  substantive facts. Preserve the set of facts as-is.
- If specificity is NOT listed as needing repair, avoid rewording criteria
  except where strictly required to make an atomic split read as a
  standalone sentence.
- If atomicity is NOT listed as needing repair, do NOT split any criteria.

TIER MOVEMENT RULE:
Move a criterion between tiers only if:
1. tier_correctness was listed as needing repair
AND
2. the checker's issues or suggestions explicitly mention that criterion
   or that category of criterion.
Do not move criteria solely because you believe another tier might be
better. When in doubt, preserve the original placement.

TIER SEMANTICS (strict):
- CRITICAL: whatever directly answers the question being asked. This
  includes final calculated numerical answers (growth rates, ratios,
  intensity figures, ARPU values, calculated margins) AND, when the
  question itself asks for a judgment, classification, or determination
  (e.g. "at what stage...", "what can/cannot be determined...", "identify
  which decisions..."), the criterion stating that judgment. Word choice
  never demotes a criterion — a criterion that uses "stage," "indicates,"
  or "suggests" is still CRITICAL if it is the thing the question asked
  for. Check only: is this what the question is asking for?
- IMPORTANT: input figures and supporting facts used to derive or justify
  a critical answer — intermediate values, definitions, exclusions,
  methodology, comparative statements ("X exceeded Y"), average
  calculations used as inputs, and the specific missing/disclosed data
  points that a critical "cannot be determined" conclusion depends on.
- OPTIONAL: interpretive color that goes BEYOND what the question asked —
  implications, comparisons, or framing the question did not request. If
  removing a criterion would remove information the question explicitly
  asked for, it is not optional, regardless of its phrasing.

ATOMICITY RULES (apply only if atomicity needs repair):
- Each criterion must contain exactly ONE verifiable claim.
- Split any criterion that combines a dollar figure with an exclusion
  statement into two separate criteria.
- Split any criterion that combines a calculated value with an
  interpretation into a factual criterion (critical/important) and an
  interpretive criterion (optional).
- Split any criterion that references two time periods into two separate
  criteria unless it is explicitly a change/expansion claim.

REDUNDANCY RULES:
- If the same dollar figure appears in multiple criteria, consolidate to one.
- If individual period values AND an expansion/change criterion all appear
  for the same metric, keep whichever is more informative and remove exact
  overlaps. Do not remove both — keep at least one.
- If two criteria assert the same claim with different phrasing, keep the
  more specific and precise version.

HARD CONSTRAINTS:
- You may only use facts already present in the input rubric.
- You may split existing facts into more atomic claims (only if atomicity
  needs repair).
- You may reassign facts between tiers (only if tier_correctness needs
  repair, and only where issues/suggestions explicitly flag it).
- You may remove genuine duplicates.
- You may NOT introduce any new fact, number, metric, or claim not
  already present in the input rubric.
- You may NOT merge two distinct facts into one if they assert different
  claims.
- Every tier must contain at least one criterion.

Return ONLY a JSON object. No preamble, no markdown fences:
{
  "critical": ["..."],
  "important": ["..."],
  "optional": ["..."]
}
"""

REPAIR_USER = """\
Question: {question}

Current dimension scores:
- Specificity:      {score_specificity:.2f}
- Atomicity:        {score_atomicity:.2f}
- Tier correctness: {score_tier_correctness:.2f}
- Coverage:         {score_coverage:.2f}
- Overall:          {score_overall:.2f}

Dimensions needing repair (score below {threshold:.2f}):
{repair_focus}

Current rubric:
CRITICAL:
{critical}

IMPORTANT:
{important}

OPTIONAL:
{optional}

Issues identified by quality checker:
{issues}

Suggestions from quality checker:
{suggestions}

Repair ONLY the dimensions listed above. Leave all other dimensions
untouched. Return only the corrected JSON.
"""

REPAIR_USER_RETRY = """\
Question: {question}

Your previous repair attempt scored {prev_score:.2f}, which is lower than
the original score of {original_score:.2f}. It made things worse.

Previous attempt's dimension scores:
- Specificity:      {score_specificity:.2f}
- Atomicity:        {score_atomicity:.2f}
- Tier correctness: {score_tier_correctness:.2f}
- Coverage:         {score_coverage:.2f}
- Overall:          {prev_score:.2f}

Issues the checker found with your previous attempt:
{prev_issues}

This is attempt {attempt} of {max_attempts}. Try a different approach.
Focus only on the issues above. Make minimal changes — do not restructure
the rubric wholesale. If a dimension already scores above {threshold:.2f},
leave it completely untouched.

Dimensions still needing repair (score below {threshold:.2f}):
{repair_focus}

Current rubric (your previous attempt — start from this, not the original):
CRITICAL:
{critical}

IMPORTANT:
{important}

OPTIONAL:
{optional}

Original issues identified by quality checker:
{original_issues}

Original suggestions from quality checker:
{original_suggestions}

Return only the corrected JSON.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_json(raw: str):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw)
        raw = raw.replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if not match:
        raise ValueError(f"Could not extract JSON from model output:\n{raw[:1000]}")
    return json.loads(match.group(0))


def fmt_tier(rubric: dict, tier: str) -> str:
    items = rubric.get(tier, [])
    return "\n".join(f"- {f}" for f in items) if items else "(none)"


def all_facts(rubric: dict) -> set:
    return set(
        rubric.get("critical", [])
        + rubric.get("important", [])
        + rubric.get("optional", [])
    )


NUMERIC_RE = re.compile(r"\$?\d[\d,]*\.?\d*%?")


def extract_signals(text: str) -> set[str]:
    """Numeric/dollar/percentage tokens in a fact, punctuation-stripped."""
    return {tok.rstrip(",.") for tok in NUMERIC_RE.findall(text)}


def detect_hallucinations(original: dict, repaired: dict) -> list[str]:
    """
    Return list of facts in the repaired rubric that introduce a number,
    dollar figure, or percentage not present anywhere in the original rubric.
    """
    original_signals = set()
    for fact in all_facts(original):
        original_signals |= extract_signals(fact)

    hallucinated = []
    for fact in all_facts(repaired):
        new_signals = extract_signals(fact) - original_signals
        if new_signals:
            hallucinated.append(fact)
    return hallucinated


def rubric_stats(rubric: dict) -> dict:
    return {
        "critical":  len(rubric.get("critical",  [])),
        "important": len(rubric.get("important", [])),
        "optional":  len(rubric.get("optional",  [])),
        "total":     sum([
            len(rubric.get("critical",  [])),
            len(rubric.get("important", [])),
            len(rubric.get("optional",  [])),
        ]),
    }


def build_repair_focus(scores: dict, threshold: float) -> tuple[list[str], str]:
    """
    Return (list of dimension names needing repair, formatted string for prompt).
    Dimensions with no score entry are treated as passing (don't repair what
    we can't measure).
    """
    dim_map = {
        "specificity":      "specificity",
        "atomicity":        "atomicity",
        "tier_correctness": "tier_correctness",
        "coverage":         "coverage",
    }
    failing = [
        label
        for key, label in dim_map.items()
        if scores.get(key, 1.0) < threshold
    ]
    if failing:
        lines = "\n".join(f"- {d} (score: {scores.get(d, 0.0):.2f})" for d in failing)
    else:
        lines = "- (all dimensions at or above threshold — repair overall structure if needed)"
    return failing, lines


# ---------------------------------------------------------------------------
# Repair (single attempt)
# ---------------------------------------------------------------------------

async def repair_rubric(
    record: dict,
    threshold: float,
    attempt: int = 1,
    prev_quality: dict | None = None,
    original_quality: dict | None = None,
) -> dict:
    """
    Call the model for one repair attempt.

    - attempt=1  → uses REPAIR_USER (first-attempt prompt)
    - attempt>1  → uses REPAIR_USER_RETRY, which shows the model what went
                   wrong in its previous attempt so it can try differently.
                   The rubric passed in is the previous attempt's rubric
                   (not the original), so the model refines rather than
                   starting over from scratch each time.
    """
    quality   = record.get("quality", {})
    scores    = quality.get("scores", {})
    issues    = quality.get("issues", [])
    suggestions = quality.get("suggestions", [])

    _failing_dims, repair_focus_str = build_repair_focus(scores, threshold)

    if attempt == 1 or prev_quality is None:
        user_content = REPAIR_USER.format(
            question=record["question"],
            score_specificity=scores.get("specificity", 0.0),
            score_atomicity=scores.get("atomicity", 0.0),
            score_tier_correctness=scores.get("tier_correctness", 0.0),
            score_coverage=scores.get("coverage", 0.0),
            score_overall=quality.get("overall", 0.0),
            threshold=threshold,
            repair_focus=repair_focus_str,
            critical=fmt_tier(record["rubric"], "critical"),
            important=fmt_tier(record["rubric"], "important"),
            optional=fmt_tier(record["rubric"], "optional"),
            issues="\n".join(f"- {i}" for i in issues),
            suggestions="\n".join(f"- {s}" for s in suggestions),
        )
    else:
        # Retry: pass the *previous attempt's* scores and issues so the model
        # knows specifically what it got wrong, and tell it to start from the
        # previous attempt's rubric (already in record["rubric"]).
        prev_scores  = prev_quality.get("scores", {})
        prev_issues  = prev_quality.get("issues", [])
        orig_issues  = original_quality.get("issues", []) if original_quality else issues
        orig_suggestions = original_quality.get("suggestions", []) if original_quality else suggestions

        _failing_dims, repair_focus_str = build_repair_focus(prev_scores, threshold)

        user_content = REPAIR_USER_RETRY.format(
            question=record["question"],
            prev_score=prev_quality.get("overall", 0.0),
            original_score=original_quality.get("overall", 0.0) if original_quality else 0.0,
            score_specificity=prev_scores.get("specificity", 0.0),
            score_atomicity=prev_scores.get("atomicity", 0.0),
            score_tier_correctness=prev_scores.get("tier_correctness", 0.0),
            score_coverage=prev_scores.get("coverage", 0.0),
            threshold=threshold,
            repair_focus=repair_focus_str,
            attempt=attempt,
            max_attempts=MAX_REPAIR_ATTEMPTS,
            critical=fmt_tier(record["rubric"], "critical"),
            important=fmt_tier(record["rubric"], "important"),
            optional=fmt_tier(record["rubric"], "optional"),
            prev_issues="\n".join(f"- {i}" for i in prev_issues),
            original_issues="\n".join(f"- {i}" for i in orig_issues),
            original_suggestions="\n".join(f"- {s}" for s in orig_suggestions),
        )

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            URL,
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": REPAIR_SYSTEM},
                    {"role": "user",   "content": user_content},
                ],
                "max_tokens": MAX_TOKENS,
                "temperature": 0.0,
                "thinking": {"type": "disabled"},
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        return extract_json(raw)


# ---------------------------------------------------------------------------
# Tier-churn detection
# ---------------------------------------------------------------------------

def detect_tier_churn(original: dict, repaired: dict) -> dict:
    """
    Compute how many criteria moved between tiers. Returns a summary dict.
    Used as a warning signal when tier_correctness was NOT a failing dimension.
    """
    tiers = ["critical", "important", "optional"]
    original_placement = {}
    for tier in tiers:
        for fact in original.get(tier, []):
            original_placement[fact] = tier

    moved = []
    for tier in tiers:
        for fact in repaired.get(tier, []):
            orig_tier = original_placement.get(fact)
            if orig_tier is not None and orig_tier != tier:
                moved.append({"fact": fact, "from": orig_tier, "to": tier})

    total_original = sum(len(original.get(t, [])) for t in tiers)
    churn_rate = len(moved) / total_original if total_original else 0.0

    return {
        "moved_count": len(moved),
        "total_original": total_original,
        "churn_rate": churn_rate,
        "moved": moved,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_repair_report(
    question: str,
    original_score: float,
    repaired_score: float | None,
    skipped: bool,
    hallucinated: list[str],
    reverted: bool,
    churn: dict | None = None,
    attempts_used: int = 1,
) -> None:
    q_short = question[:80] + "..." if len(question) > 80 else question
    print(f"\n{'='*60}")
    print(f"  {q_short}")
    print(f"{'='*60}")

    if skipped:
        print(f"  ⊘ Skipped (recommendation: stop, score {original_score:.2f})")
        return

    if reverted:
        print(f"  ✗ Repair reverted — hallucinated signals detected:")
        for h in hallucinated:
            print(f"      · {h}")
        print(f"  → Original rubric preserved (score {original_score:.2f})")
        return

    arrow = "↑" if repaired_score > original_score else ("↓" if repaired_score < original_score else "→")
    print(f"  Score: {original_score:.2f} {arrow} {repaired_score:.2f}  (attempts: {attempts_used})")

    if churn and churn["moved_count"] > 0:
        print(f"  ⚠ Tier churn: {churn['moved_count']}/{churn['total_original']} criteria moved "
              f"({churn['churn_rate']:.0%})")

    if repaired_score >= QUALITY_THRESHOLD:
        print(f"  ✓ PASS after repair")
    else:
        print(f"  ✗ Still below threshold after repair")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rubric",
        required=True,
        help="Output of stage 4 — must contain 'quality' block per record",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Write repaired rubrics to this path",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=QUALITY_THRESHOLD,
        help="Quality threshold for pass/fail (default 0.70)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=MAX_REPAIR_ATTEMPTS,
        help=f"Max repair attempts per rubric before giving up (default {MAX_REPAIR_ATTEMPTS})",
    )
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="Always exit 0 even if rubrics still fail after repair",
    )
    args = parser.parse_args()

    with open(args.rubric, encoding="utf-8") as f:
        raw_data = json.load(f)
    if isinstance(raw_data, dict):
        raw_data = [raw_data]

    # Validate input has quality blocks
    for record in raw_data:
        if "quality" not in record:
            print(
                f"ERROR: record missing 'quality' block. "
                f"Run stage4_checkrubrics.py first.\n"
                f"Question: {record.get('question', '')[:80]}"
            )
            sys.exit(1)

    all_passed = True
    results    = []

    for record in raw_data:
        quality             = record["quality"]
        original_score      = quality.get("overall", 0.0)
        original_quality    = quality          # kept for retry prompt context
        prev_recommendation = record.get("recommendation")

        # Only repair records Stage 4 explicitly flagged for repair.
        if prev_recommendation != "repair":
            print_repair_report(
                record["question"],
                original_score=original_score,
                repaired_score=None,
                skipped=True,
                hallucinated=[],
                reverted=False,
            )
            results.append(record)
            continue

        # ---------------------------------------------------------------
        # Retry loop: attempt repair up to args.max_attempts times.
        # After each attempt we recheck quality and exit early if the
        # rubric reaches threshold. We track the best attempt seen so far
        # and use that as the final output regardless of which attempt
        # was last.
        # ---------------------------------------------------------------

        best_rubric   = None     # rubric dict of the best attempt so far
        best_quality  = None     # quality dict of the best attempt so far
        best_score    = -1.0
        best_churn    = None
        attempts_used = 0

        # The "current" record used as input to each repair call. For
        # attempt 1 this is the original; for retries it's the previous
        # attempt's record (so the model refines rather than restarts).
        current_record   = record
        prev_quality_obj = None

        hallucinated_final = []
        reverted_final     = False

        for attempt in range(1, args.max_attempts + 1):
            attempts_used = attempt

            # --- Run one repair attempt ---
            try:
                repaired_rubric = asyncio.run(repair_rubric(
                    current_record,
                    threshold=args.threshold,
                    attempt=attempt,
                    prev_quality=prev_quality_obj,
                    original_quality=original_quality,
                ))
            except Exception as e:
                print(f"\n  Attempt {attempt}: repair call failed — {type(e).__name__}: {e}")
                break

            # --- Hallucination check ---
            # Always check against the *original* rubric, not the previous
            # attempt, so signals can't accumulate across retries.
            hallucinated = detect_hallucinations(record["rubric"], repaired_rubric)
            if hallucinated:
                print(f"\n  Attempt {attempt}: hallucinated signals — skipping this attempt")
                for h in hallucinated:
                    print(f"      · {h}")
                # Don't update current_record; try again from the same base.
                # If every attempt hallucinates we'll fall through with
                # best_rubric=None and revert to original below.
                hallucinated_final = hallucinated
                reverted_final     = True
                continue

            hallucinated_final = []
            reverted_final     = False

            # --- Recheck quality ---
            candidate_record = {
                **record,
                "rubric": repaired_rubric,
            }

            from stage4_checkrubrics import check_rubric

            print(f"\n  Attempt {attempt}: rechecking...")
            try:
                quality2 = asyncio.run(check_rubric(candidate_record))
            except Exception as e:
                print(f"  Attempt {attempt}: recheck failed — {type(e).__name__}: {e}")
                # Keep trying from the same current_record base.
                continue

            if not isinstance(quality2, dict) or "overall" not in quality2:
                print(f"  Attempt {attempt}: recheck returned malformed response — skipping")
                continue

            candidate_score = quality2.get("overall", 0.0)
            churn           = detect_tier_churn(record["rubric"], repaired_rubric)

            print(f"  Attempt {attempt}: score {candidate_score:.2f} "
                  f"(original {original_score:.2f}, best so far {max(best_score, 0.0):.2f})")

            # Track best regardless of whether it beats original, so we
            # always have something to write even if all attempts regress.
            if candidate_score > best_score:
                best_score   = candidate_score
                best_rubric  = repaired_rubric
                best_quality = quality2
                best_churn   = churn

            # Update for next retry: model refines from this attempt.
            prev_quality_obj = quality2
            current_record = {
                **record,
                "rubric":  repaired_rubric,
                "quality": quality2,
            }

            # Exit early if we've cleared the threshold.
            if candidate_score >= args.threshold:
                print(f"  ✓ Threshold reached on attempt {attempt} — stopping early")
                break

        # ---------------------------------------------------------------
        # Decide what to write for this record.
        # ---------------------------------------------------------------

        if best_rubric is None:
            # Every attempt either hallucinated or errored out — revert.
            print_repair_report(
                record["question"],
                original_score=original_score,
                repaired_score=None,
                skipped=False,
                hallucinated=hallucinated_final,
                reverted=True,
                attempts_used=attempts_used,
            )
            record_out = {
                **record,
                "repair_debug": {
                    "hallucinated_signals": hallucinated_final,
                    "attempts": attempts_used,
                },
            }
            results.append(record_out)
            if original_score < args.threshold:
                all_passed = False
            continue

        # We have at least one valid repaired rubric — use the best one.
        # Only accept it over the original if it's actually better.
        if best_score <= original_score:
            # All attempts made things worse or broke even. Keep original
            # but still log the best attempt for inspection.
            print(f"\n  All {attempts_used} attempt(s) failed to improve score "
                  f"({original_score:.2f}). Keeping original rubric.")
            record_out = {
                **record,
                "repair_debug": {
                    "best_attempted_score": best_score,
                    "best_attempted_rubric": best_rubric,
                    "attempts": attempts_used,
                },
            }
            print_repair_report(
                record["question"],
                original_score=original_score,
                repaired_score=best_score,
                skipped=False,
                hallucinated=[],
                reverted=False,
                churn=best_churn,
                attempts_used=attempts_used,
            )
            results.append(record_out)
            if original_score < args.threshold:
                all_passed = False
            continue

        # Best attempt improved the score — accept it.
        record_repaired = {
            **record,
            "rubric":          best_rubric,
            "rubric_repaired": True,
            "stats":           rubric_stats(best_rubric),
            "quality":         best_quality,
            "recommendation":  None,   # cleared so Stage 4 re-evaluates fresh
            "repair_attempts": attempts_used,
        }
        if best_churn and best_churn["moved_count"] > 0:
            record_repaired["repair_churn"] = best_churn

        print_repair_report(
            record["question"],
            original_score=original_score,
            repaired_score=best_score,
            skipped=False,
            hallucinated=[],
            reverted=False,
            churn=best_churn,
            attempts_used=attempts_used,
        )

        if best_score < args.threshold:
            all_passed = False

        results.append(record_repaired)

    # Summary
    n        = len(results)
    n_passed = sum(
        1 for r in results
        if r["quality"].get("overall", 0.0) >= args.threshold
    )
    print(f"\n{'='*60}")
    print(f"  {n_passed}/{n} rubrics at or above threshold after repair")
    print(f"{'='*60}\n")

    output = results if len(results) != 1 else results[0]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"✓ Repaired rubrics written → {args.out}")

    if not all_passed and not args.no_gate:
        sys.exit(1)


if __name__ == "__main__":
    main()
