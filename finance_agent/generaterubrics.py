#!/usr/bin/env python3
"""
FABv2 Pipeline Orchestrator
Flow: Stage1 → Stage2 → Stage3 → Stage4 →
      [Enrichment loop: 4b enrich → 4c consolidate → 4 check] (max 5×)
      [Repair loop:     4b repair → 4 check]                   (max 5×)
      → STOP
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
FA       = BASE_DIR

FACTS    = BASE_DIR.parent / "facts"
RUBRIC   = BASE_DIR.parent / "rubric"
LOGS     = BASE_DIR.parent / "logs"



DEFAULT_INPUTS = {
    "glm51":  "results/glm-5.1-public-spacex-70-20260606-191748.json",
    "glm52":  "results/glm-5.2-public-spacex-70-20260624-153058.json",
    "qwen": "results/qwen3.7-max-public-spacex-70-20260609-134629.json",
    "kimi": "results/moonshotai-Kimi-K2.6-public-spacex-70-20260608-060803.json",
    "mimo": "results/mimo-v2.5-pro-public-spacex-70-20260610-142205.json",
}

MAX_ENRICH_LOOPS = 5
MAX_REPAIR_LOOPS = 5

# Key in the stage-4 output JSON that signals routing decision.
# Change to match your actual schema ("action", "recommended_action", etc.)
ROUTING_KEY   = "action"
STOP_VALUE    = "stop"

# ── Helpers ───────────────────────────────────────────────────────────────────

def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)

def run(label: str, cmd: list[str], logfile: Path) -> None:
    """Run a subprocess, stream output to logfile, raise on failure."""
    log(f"START  {label}")
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("w") as lf:
        result = subprocess.run(
            [sys.executable] + [str(c) for c in cmd],
            stdout=lf,
            stderr=subprocess.STDOUT,
        )
    if result.returncode != 0:
        log(f"FAIL   {label} — see {logfile}")
        # Print last 20 lines for quick diagnosis
        lines = logfile.read_text().splitlines()
        for line in lines[-20:]:
            print("  " + line, flush=True)
        sys.exit(result.returncode)
    log(f"OK     {label}")

def run_parallel(jobs: list[tuple[str, list[str], Path]]) -> None:
    """Launch multiple subprocesses in parallel, wait for all."""
    procs = []
    for label, cmd, logfile in jobs:
        log(f"START  {label} (parallel)")
        logfile.parent.mkdir(parents=True, exist_ok=True)
        lf = logfile.open("w")
        p = subprocess.Popen(
            [sys.executable] + [str(c) for c in cmd],
            stdout=lf,
            stderr=subprocess.STDOUT,
        )
        procs.append((label, cmd, logfile, p, lf))

    failed = []
    for label, cmd, logfile, p, lf in procs:
        p.wait()
        lf.close()
        if p.returncode != 0:
            log(f"FAIL   {label} — see {logfile}")
            failed.append(label)
        else:
            log(f"OK     {label}")

    if failed:
        log(f"Parallel stage failed: {failed}")
        sys.exit(1)

def all_done(checked_json: Path) -> bool:
    """
    Return True when every rubric item in the checked JSON has
    ROUTING_KEY == STOP_VALUE (i.e. no further work needed).
    Falls back to True if the file is missing or unparseable (fail-safe).
    """
    if not checked_json.exists():
        log(f"  WARNING: {checked_json} not found — assuming done.")
        return True
    try:
        data = json.loads(checked_json.read_text())
        items = data if isinstance(data, list) else list(data.values())
        pending = [
            item for item in items
            if isinstance(item, dict) and item.get(ROUTING_KEY) != STOP_VALUE
        ]
        if pending:
            log(f"  {len(pending)} rubric(s) still need work ({ROUTING_KEY} ≠ '{STOP_VALUE}').")
            return False
        return True
    except Exception as e:
        log(f"  WARNING: could not parse {checked_json}: {e} — assuming not done.")
        return False

# ── Pipeline ──────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    FACTS.mkdir(parents=True, exist_ok=True)
    RUBRIC.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("FABv2 Pipeline starting")
    log("=" * 60)

    # ── Stage 1: Fact extraction (parallel) ──────────────────
    log("--- Stage 1: Fact extraction ---")
    stage1_jobs = [
        (
            f"stage1_{name}",
            [FA / "run_stage1_batch.py", "--input", inp, "--output", FACTS / f"{name}_facts.json"],
            LOGS / f"stage1_{name}.log",
        )
        for name, inp in args.inputs.items()
    ]
    run_parallel(stage1_jobs)

    fact_files = [str(FACTS / f"{name}_facts.json") for name in args.inputs]

    # ── Stage 2: Fact consolidation ───────────────────────────
    log("--- Stage 2: Fact consolidation ---")
    consolidated_facts = FACTS / "consolidated_facts.json"
    run(
        "stage2_consolidate",
        [FA / "stage2_consolidate.py",
         "--facts", *fact_files,
         "--out", consolidated_facts,
         "--min-agreement", str(args.min_agreement)],
        LOGS / "stage2_consolidate.log",
    )

    # ── Stage 3: Rubric induction ─────────────────────────────
    log("--- Stage 3: Rubric induction ---")
    rubrics_draft = RUBRIC / "rubrics_draft.json"
    run(
        "stage3_inducerubrics",
        [FA / "stage3_inducerubrics.py",
         "--consolidated", consolidated_facts,
         "--out", rubrics_draft],
        LOGS / "stage3_inducerubrics.log",
    )

    # ── Stage 4: Initial check ────────────────────────────────
    log("--- Stage 4: Initial rubric check ---")
    rubrics_checked = RUBRIC / "rubrics_checked_pass0.json"
    run(
        "stage4_check_pass0",
        [FA / "stage4_checkrubrics.py",
         "--rubric", rubrics_draft,
         "--out", rubrics_checked],
        LOGS / "stage4_check_pass0.log",
    )

    # ── Enrichment loop ───────────────────────────────────────
    log(f"--- Enrichment loop (max {args.max_enrich} iterations) ---")
    enrich_in   = rubrics_checked
    enrich_done = False

    for i in range(1, args.max_enrich + 1):
        log(f"  Enrichment iteration {i} / {args.max_enrich}")
        sfx = f"enrich{i}"

        enriched_out    = RUBRIC / f"rubrics_enriched_{sfx}.json"
        consolidated_out = RUBRIC / f"rubrics_consolidated_{sfx}.json"
        checked_out     = RUBRIC / f"rubrics_checked_{sfx}.json"

        run(f"stage4b_enrich_{sfx}",
            [FA / "stage4b_enrichrubrics.py",
             "--checked", enrich_in,
             "--out", enriched_out],
            LOGS / f"stage4b_enrich_{sfx}.log")

        run(f"stage4c_consolidate_{sfx}",
            [FA / "stage4c_consolidaterubrics.py",
             "--rubrics", enriched_out,
             "--out", consolidated_out],
            LOGS / f"stage4c_consolidate_{sfx}.log")

        run(f"stage4_check_{sfx}",
            [FA / "stage4_checkrubrics.py",
             "--rubric", consolidated_out,
             "--out", checked_out],
            LOGS / f"stage4_check_{sfx}.log")

        enrich_in = checked_out

        if all_done(checked_out):
            log(f"  Enrichment complete after {i} iteration(s).")
            enrich_done = True
            break

    if not enrich_done:
        log(f"  Enrichment hit max ({args.max_enrich}). Proceeding with best result.")

    # ── Repair loop ───────────────────────────────────────────
    log(f"--- Repair loop (max {args.max_repair} iterations) ---")
    repair_in   = enrich_in
    repair_done = False

    for i in range(1, args.max_repair + 1):
        log(f"  Repair iteration {i} / {args.max_repair}")
        sfx = f"repair{i}"

        repaired_out = RUBRIC / f"rubrics_repaired_{sfx}.json"
        checked_out  = RUBRIC / f"rubrics_checked_{sfx}.json"

        run(f"stage4b_repair_{sfx}",
            [FA / "stage4b_repairrubrics.py",
             "--checked", repair_in,
             "--out", repaired_out],
            LOGS / f"stage4b_repair_{sfx}.log")

        run(f"stage4_check_{sfx}",
            [FA / "stage4_checkrubrics.py",
             "--rubric", repaired_out,
             "--out", checked_out],
            LOGS / f"stage4_check_{sfx}.log")

        repair_in = checked_out

        if all_done(checked_out):
            log(f"  Repair complete after {i} iteration(s).")
            repair_done = True
            break

    if not repair_done:
        log(f"  Repair hit max ({args.max_repair}). Proceeding with best result.")

    # ── Final output ──────────────────────────────────────────
    final = RUBRIC / "rubrics_final.json"
    final.write_bytes(repair_in.read_bytes())

    log("=" * 60)
    log("Pipeline complete.")
    log(f"Final rubrics → {final}")
    log("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FABv2 pipeline orchestrator")

    parser.add_argument("--glm51",  default=DEFAULT_INPUTS["glm51"],  help="GLM 5.1 result JSON")
    parser.add_argument("--glm52",  default=DEFAULT_INPUTS["glm52"],  help="GLM 5.2 result JSON")
    parser.add_argument("--qwen", default=DEFAULT_INPUTS["qwen"], help="Qwen result JSON")
    parser.add_argument("--kimi", default=DEFAULT_INPUTS["kimi"], help="Kimi result JSON")
    parser.add_argument("--mimo", default=DEFAULT_INPUTS["mimo"], help="MiMo result JSON")

    parser.add_argument("--min-agreement", type=int, default=2,
                        help="Min model agreement for stage 2 (default: 2)")
    parser.add_argument("--max-enrich", type=int, default=MAX_ENRICH_LOOPS,
                        help=f"Max enrichment iterations (default: {MAX_ENRICH_LOOPS})")
    parser.add_argument("--max-repair", type=int, default=MAX_REPAIR_LOOPS,
                        help=f"Max repair iterations (default: {MAX_REPAIR_LOOPS})")

    args = parser.parse_args()
    args.inputs = {
        "glm51":  args.glm51,
        "glm52":  args.glm52,
        "qwen": args.qwen,
        "kimi": args.kimi,
        "mimo": args.mimo,
    }

    main(args)
