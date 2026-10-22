#!/usr/bin/env python
"""
One command to rebuild the whole thesis pipeline and produce one report.

  python run_all.py                       # run everything missing or out of date, then the report
  python run_all.py --dry-run             # print the plan; spends nothing, changes nothing
  python run_all.py --offline             # never call the API: recompute metrics + report from files
  python run_all.py --only analysis report
  python run_all.py --from c4             # start at a stage
  python run_all.py --force --only analysis


Final output: results/thesis_report.html  (one self-contained file; figures embedded).

"""

import argparse
import csv
import hashlib
import itertools
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
R = "results"
CASES = "data/cases/main.csv"
VARIANTS = "data/cases/fairness_variants.csv"
CORPUS = "data/guidelines_clean/corpus.jsonl"
INDEX = "data/indexes/guidelines.index"
FIXED = "data/guidelines_clean/fixed_context.json"
RATED_SHEET = f"{R}/retrieval_rating_sheet_v2_rated.csv"
REPORT = f"{R}/thesis_report.html"

STAGES = ["tests", "data", "corpus", "smoke", "main", "c4", "noise", "analysis", "report"]
CONFIGS = ["baseline", "rag", "rag_masked", "rag_fixed"]
N_RUNS = 3  # runs per config and set, main run included (protocol 5)


@dataclass
class Step:
    stage: str
    name: str
    cmd: list
    inputs: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    api: bool = False
    always: bool = False  # run even if outputs look fresh (tests, report)


def py(script, *args):
    return [PY, f"src/{script}", *args]


def run_files(cfg, tag=""):
    # primary run + rep2..N_RUNS
    files = [f"{R}/{cfg}{tag}_predictions.csv"]
    for i in range(2, N_RUNS + 1):
        files.append(f"{R}/{cfg}{tag}_predictions_rep{i}.csv")
    return files


def build_steps(sleep):
    if sleep:
        s = ["--sleep", str(sleep)]
    else:
        s = []

    steps = [
        # ---- tests: must pass before anything costs money
        Step("tests", "unit tests", [PY, "-m", "pytest", "-q", "tests"], always=True),

        # ---- data (no API, same result every time)
        Step("data", "extract MIMIC cases", py("extract_mimic_cases.py"), [], [CASES]),
        Step("data", "build fairness variants", py("build_fairness_variants.py"), [CASES], [VARIANTS]),

        # ---- corpus + index (no API, same result every time)
        Step("corpus", "build corpus", py("build_corpus.py"), [], [CORPUS]),
        Step("corpus", "build FAISS index", py("build_index.py"), [CORPUS], [INDEX]),

        # ---- smoke gate (~40 calls)
        Step("smoke", "smoke C1", py("run_baseline.py", "--limit", "20", "--output", f"{R}/smoke_baseline.csv", *s),
             [CASES], [f"{R}/smoke_baseline.csv"], api=True),
        Step("smoke", "smoke C2", py("run_rag.py", "--limit", "20", "--output", f"{R}/smoke_rag.csv",
                                     "--log-file", f"{R}/smoke_logs.jsonl", *s),
             [CASES, INDEX], [f"{R}/smoke_rag.csv"], api=True),
    ]
    for name in ("baseline", "rag"):
        steps.append(Step("smoke", f"evaluate smoke {name}",
                          py("evaluate.py", "-p", f"{R}/smoke_{name}.csv", "--output", f"{R}/eval_smoke_{name}.json"),
                          [f"{R}/smoke_{name}.csv"], [f"{R}/eval_smoke_{name}.json"]))

    # ---- main runs: performance set (331) and variants (280)
    steps += [
        Step("main", "C1 performance", py("run_baseline.py", "--output", f"{R}/baseline_predictions.csv", *s),
             [CASES], [f"{R}/baseline_predictions.csv"], api=True),
        Step("main", "C2 performance", py("run_rag.py", *s),
             [CASES, INDEX], [f"{R}/rag_predictions.csv", f"{R}/retrieval_logs.jsonl"], api=True),
        Step("main", "C3 performance", py("run_rag.py", "--mask-demographics", *s),
             [CASES, INDEX], [f"{R}/rag_masked_predictions.csv", f"{R}/retrieval_logs_masked.jsonl"], api=True),
        Step("main", "C1 variants", py("run_baseline.py", "--cases-file", VARIANTS,
                                       "--output", f"{R}/baseline_variants_predictions.csv", *s),
             [VARIANTS], [f"{R}/baseline_variants_predictions.csv"], api=True),
        Step("main", "C2 variants", py("run_rag.py", "--cases-file", VARIANTS,
                                       "--output", f"{R}/rag_variants_predictions.csv",
                                       "--log-file", f"{R}/retrieval_logs_variants.jsonl", *s),
             [VARIANTS, INDEX], [f"{R}/rag_variants_predictions.csv"], api=True),
        Step("main", "C3 variants", py("run_rag.py", "--mask-demographics", "--cases-file", VARIANTS,
                                       "--output", f"{R}/rag_masked_variants_predictions.csv",
                                       "--log-file", f"{R}/retrieval_logs_masked_variants.jsonl", *s),
             [VARIANTS, INDEX], [f"{R}/rag_masked_variants_predictions.csv"], api=True),

        # ---- C4: context comes from the C2/C3 logs, so it has to run after them
        Step("c4", "derive fixed context", py("pick_fixed_context.py"),
             [f"{R}/retrieval_logs.jsonl", f"{R}/retrieval_logs_masked.jsonl",
              f"{R}/retrieval_logs_variants.jsonl", f"{R}/retrieval_logs_masked_variants.jsonl"], [FIXED]),
        Step("c4", "C4 performance", py("run_rag.py", "--fixed-context", "--output", f"{R}/rag_fixed_predictions.csv",
                                        "--log-file", f"{R}/retrieval_logs_fixed.jsonl", *s),
             [CASES, FIXED], [f"{R}/rag_fixed_predictions.csv"], api=True),
    ]

    # ---- noise floor: same runs repeated (protocol 5: >= 3 runs per config and set)
    # main run = run 1, reps 2..N_RUNS write *_repN files. C4 only on the performance set
    run_args = {"baseline": ("run_baseline.py",), "rag": ("run_rag.py",),
                "rag_masked": ("run_rag.py", "--mask-demographics"), "rag_fixed": ("run_rag.py", "--fixed-context")}
    log_stem = {"rag": "retrieval_logs", "rag_masked": "retrieval_logs_masked", "rag_fixed": "retrieval_logs_fixed"}
    label = {"baseline": "C1", "rag": "C2", "rag_masked": "C3", "rag_fixed": "C4"}
    for rep in range(2, N_RUNS + 1):
        for variants in (False, True):
            if variants:
                cfgs = CONFIGS[:3]
                tag = "_variants"
                cases_file = VARIANTS
                set_name = "variants"
            else:
                cfgs = CONFIGS
                tag = ""
                cases_file = CASES
                set_name = "performance"
            for cfg in cfgs:
                out = f"{R}/{cfg}{tag}_predictions_rep{rep}.csv"
                script = run_args[cfg][0]
                flags = list(run_args[cfg][1:])
                args = flags + ["--cases-file", cases_file, "--output", out]
                if cfg in log_stem:
                    args += ["--log-file", f"{R}/{log_stem[cfg]}{tag}_rep{rep}.jsonl"]

                inputs = [cases_file]
                if cfg == "rag_fixed":
                    inputs.append(FIXED)
                elif cfg in log_stem:
                    inputs.append(INDEX)

                name = f"{label[cfg]} {set_name} rep{rep}"
                steps.append(Step("noise", name, py(script, *args, *s), inputs, [out], api=True))

    # ---- analysis (no API, can always run again)
    all_runs = []
    for c in CONFIGS:
        all_runs += run_files(c)
    for c in CONFIGS[:3]:
        all_runs += run_files(c, "_variants")
    steps.append(Step("analysis", "summary table + figures", py("make_summary.py"),
                      all_runs, [f"{R}/results_summary.csv", f"{R}/summary.json"]))

    # main family (protocol 7.1): 6 pairs x 2 metrics on the primary runs, in results/ where
    # build_report.py counts them. C1 vs its own rep2 = the report's robustness row
    pairs = list(itertools.combinations(CONFIGS, 2)) + [("baseline", "baseline_rerun")]
    for a, b in pairs:
        for metric in ("urgency", "specialty"):
            fa = f"{R}/{a}_predictions.csv"
            if b == "baseline_rerun":
                fb = f"{R}/baseline_predictions_rep2.csv"
            else:
                fb = f"{R}/{b}_predictions.csv"
            out = f"{R}/mcnemar_{a}_vs_{b}_{metric}.json"
            steps.append(Step("analysis", f"McNemar {a} vs {b} ({metric})",
                              py("mcnemar.py", "-a", fa, "-b", fb, "--metric", metric, "--output", out),
                              [fa, fb], [out]))

    # RQ1 as a range (7.1): C1 vs C2 and C1 vs C3 for every C1 run, not part of the 12
    for i in range(2, N_RUNS + 1):
        for metric in ("urgency", "specialty"):
            for b in ("rag", "rag_masked"):
                fa = f"{R}/baseline_predictions_rep{i}.csv"
                fb = f"{R}/{b}_predictions.csv"
                out = f"{R}/robustness/mcnemar_baseline_rep{i}_vs_{b}_{metric}.json"
                steps.append(Step("analysis", f"McNemar C1 rep{i} vs {b} ({metric})",
                                  py("mcnemar.py", "-a", fa, "-b", fb, "--metric", metric, "--output", out),
                                  [fa, fb], [out]))

    # sensitivity (6.2.1): the 6 specialty comparisons with sub-specialties mapped to a parent
    for a, b in itertools.combinations(CONFIGS, 2):
        fa = f"{R}/{a}_predictions.csv"
        fb = f"{R}/{b}_predictions.csv"
        out = f"{R}/sensitivity/mcnemar_{a}_vs_{b}_specialty_mapped.json"
        steps.append(Step("analysis", f"McNemar {a} vs {b} (specialty, mapped)",
                          py("mcnemar.py", "-a", fa, "-b", fb, "--metric", "specialty", "--mapped", "--output", out),
                          [fa, fb], [out]))

    # noise floor (6.4) for every config and set, fairness tests (7.2) on the variants
    for cfg in CONFIGS:
        if cfg == "rag_fixed":
            tags = ("",)
        else:
            tags = ("", "_variants")
        for tag in tags:
            out = f"{R}/noise_{cfg}{tag}.json"
            steps.append(Step("analysis", f"noise floor {cfg}{tag}",
                              py("repeat_agreement.py", "--runs", *run_files(cfg, tag), "--output", out),
                              run_files(cfg, tag), [out]))
    for cfg in CONFIGS[:3]:
        out = f"{R}/fairness_{cfg}.json"
        steps.append(Step("analysis", f"fairness tests {cfg}",
                          py("fairness_tests.py", "--runs", *run_files(cfg, "_variants"), "--output", out),
                          run_files(cfg, "_variants"), [out]))
    steps.append(Step("analysis", "retrieval statistics", py("retrieval_stats.py"),
                      [f"{R}/retrieval_logs.jsonl", f"{R}/retrieval_logs_masked.jsonl", FIXED],
                      [f"{R}/retrieval_stats.json"]))
    steps.append(Step("analysis", "error cases", py("error_cases.py", "--config", "rag"),
                      [f"{R}/baseline_predictions.csv", f"{R}/rag_predictions.csv"], [f"{R}/error_cases.csv"]))
    steps.append(Step("analysis", "retrieval relevance scores",
                      py("retrieval_eval.py", "score", "--sheet", RATED_SHEET),
                      [RATED_SHEET], [f"{R}/retrieval_eval.json"]))

    # ---- one-file report
    steps.append(Step("report", "build single-file report", py("build_report.py"), [], [REPORT], always=True))
    return steps


# ---- freshness checks

def path_of(rel):
    return ROOT / rel


def has_error_rows(path):
    if path.suffix != ".csv" or not path.exists():
        return False
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "pred_urgency" not in reader.fieldnames:
            return False
        for r in reader:
            if r["pred_urgency"] == "ERROR":
                return True
    return False


def get_arg(cmd, flag, default=None):
    # value after a flag in a command list
    if flag in cmd:
        return cmd[cmd.index(flag) + 1]
    return default


def missing_rows(step):
    # (prediction file, number of input cases without a finished row) for an API step, else None
    # a prediction file is only complete if every case the step should run (its --cases-file,
    # first --limit rows) has a non-ERROR row. an interrupted run leaves a partial file that is
    # newer than its inputs and has no ERROR rows -> without this check it would be skipped
    if not step.api:
        return None

    pred = None
    for o in step.outputs:
        if o.endswith(".csv"):
            pred = path_of(o)
            break
    if pred is None or not pred.exists():
        return None

    cases_file = get_arg(step.cmd, "--cases-file", CASES)
    limit = get_arg(step.cmd, "--limit")
    csv.field_size_limit(10_000_000)
    with open(path_of(cases_file), encoding="utf-8", newline="") as f:
        cases = list(csv.DictReader(f))
    if limit is not None:
        cases = cases[:int(limit)]

    wanted = set()
    for c in cases:
        wanted.add(c.get("variant_id") or c["case_id"])
    done = set()
    with open(pred, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["pred_urgency"] != "ERROR":
                done.add(r.get("variant_id") or r["case_id"])
    return pred, len(wanted - done)


def is_fresh(step):
    # returns (fresh?, reason)
    if step.always or not step.outputs:
        return False, "always runs"

    outs = []
    for o in step.outputs:
        outs.append(path_of(o))
    for o in outs:
        if not o.exists():
            return False, f"missing {o.relative_to(ROOT)}"

    # any input newer than the oldest output?
    oldest_out = min(o.stat().st_mtime for o in outs)
    for i in step.inputs:
        ip = path_of(i)
        if ip.exists() and ip.stat().st_mtime > oldest_out:
            return False, f"input newer: {i}"

    for o in outs:
        if has_error_rows(o):
            return False, f"ERROR rows in {o.name} (will retry them)"

    incomplete = missing_rows(step)
    if incomplete and incomplete[1]:
        return False, f"{incomplete[1]} cases not yet in {incomplete[0].name} (will resume)"
    return True, "up to date"


def missing_inputs(step):
    missing = []
    for i in step.inputs:
        if not path_of(i).exists():
            missing.append(i)
    return missing


# ---- manifest + logging

RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_DIR = ROOT / R / "runs" / RUN_ID


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def run_logged(cmd, logfile):
    # run cmd, show its output live, keep a full copy (stdout + stderr) in logfile
    # PYTHONUTF8: child processes write UTF-8 to the pipe. on Windows they would use cp1252
    # and progress bars would show up as "█" escapes
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    with open(logfile, "w", encoding="utf-8") as f:
        f.write(f"$ {' '.join(cmd)}\n# started {datetime.now().isoformat(timespec='seconds')}\n\n")
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1, env=env)
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        rc = proc.wait()
        f.write(f"\n# finished {datetime.now().isoformat(timespec='seconds')} exit={rc}\n")
    return rc


def archive_error_rows(step):
    # keep the ERROR rows (with error text) before a retry overwrites them
    for o in step.outputs:
        path = path_of(o)
        if not has_error_rows(path):
            continue
        bad = []
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r["pred_urgency"] == "ERROR":
                    bad.append(r)
            fields = reader.fieldnames
        dest = RUN_DIR / f"errors_before_retry_{path.name}"
        with open(dest, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(bad)


def archive_outputs():
    for name in ("thesis_report.html", "numbers.json"):
        src = path_of(f"{R}/{name}")
        if src.exists():
            shutil.copy2(src, RUN_DIR / name)


def append_index(stages, failed, records):
    index = ROOT / R / "runs" / "INDEX.csv"
    new = not index.exists()
    n_ran = 0
    for r in records:
        if r["status"] == "ok":
            n_ran += 1
    if failed:
        status = "failed"
    else:
        status = "ok"
    with open(index, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["run_id", "stages", "steps_run", "status", "failed_at"])
        w.writerow([RUN_ID, " ".join(stages), n_ran, status, failed or ""])


def sha256_short(rel):
    p = path_of(rel)
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def git_commit():
    # (short commit id or None, uncommitted changes?)
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
        dirty = status.stdout.strip()
        return (out.stdout.strip() or None, bool(dirty))
    except FileNotFoundError:
        return (None, None)


class Log:
    # print a line and add it to results/run_all.log

    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg=""):
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()


def smoke_gate(log):
    # no big API runs unless the smoke test had 0 ERROR and 0 INVALID
    for name in ("baseline", "rag"):
        p = path_of(f"{R}/eval_smoke_{name}.json")
        if not p.exists():
            log(f"  smoke gate: {p.name} missing")
            return False
        ev = json.loads(p.read_text(encoding="utf-8"))
        if ev.get("n_error") or ev.get("n_invalid_urgency"):
            log(f"  smoke gate FAILED for {name}: n_error={ev.get('n_error')} "
                f"n_invalid_urgency={ev.get('n_invalid_urgency')}. Fix the prompt/parser before full runs.")
            return False
    log("  smoke gate passed: 0 ERROR, 0 INVALID")
    return True


def write_manifest(args, stages, started, commit, dirty, records, failed):
    manifest = {
        "started_utc": started.isoformat(timespec="seconds"),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "corpus_sha256": sha256_short(CORPUS),
        "cases_sha256": sha256_short(CASES),
        "variants_sha256": sha256_short(VARIANTS),
        "fixed_context_sha256": sha256_short(FIXED),
        "args": vars(args),
        "stages": stages,
        "steps": records,
        "failed_at": failed,
    }
    manifest["run_id"] = RUN_ID
    text = json.dumps(manifest, indent=2)
    path_of(f"{R}/run_manifest.json").write_text(text, encoding="utf-8")  # latest run
    (RUN_DIR / "manifest.json").write_text(text, encoding="utf-8")        # this run, kept


# ---- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="Print the plan only.")
    ap.add_argument("--offline", action="store_true", help="Skip every API step (use existing predictions).")
    ap.add_argument("--only", nargs="+", choices=STAGES, help="Run only these stages.")
    ap.add_argument("--from", dest="start", choices=STAGES, help="Start at this stage.")
    ap.add_argument("--force", action="store_true", help="Run selected steps even if up to date (non-API only "
                                                         "unless --force-api).")
    ap.add_argument("--force-api", action="store_true", help="Also force API steps (costs money).")
    ap.add_argument("--sleep", type=float, default=0.0, help="Seconds between API calls (rate limits).")
    args = ap.parse_args()
    # if the console can't show a character (e.g. a progress bar block), escape it, don't crash
    sys.stdout.reconfigure(errors="backslashreplace")

    if args.only:
        stages = args.only
    elif args.start:
        stages = STAGES[STAGES.index(args.start):]
    else:
        stages = STAGES

    log = Log(path_of(f"{R}/run_all.log"))
    if not args.dry_run:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
    commit, dirty = git_commit()
    started = datetime.now(timezone.utc)

    # header line
    header = f"\n=== run_all {RUN_ID} ({started.isoformat(timespec='seconds')}) | commit {commit}"
    if dirty:
        header += " (dirty)"
    header += f" | stages: {' '.join(stages)}"
    if args.dry_run:
        header += " | DRY RUN"
    if args.offline:
        header += " | OFFLINE"
    log(header)

    records = []
    failed = None
    gate_ok = False
    planned = set()  # outputs a dry run would make
    for step in build_steps(args.sleep):
        if step.stage not in stages:
            continue
        label = f"[{step.stage:<8}] {step.name}"
        fresh, why = is_fresh(step)
        forced = (args.force and not step.api) or (args.force_api and step.api)

        if step.api and args.offline:
            log(f"{label:<55} SKIP (offline)")
            records.append({"step": step.name, "status": "skipped-offline"})
            continue
        if fresh and not forced:
            log(f"{label:<55} SKIP ({why})")
            records.append({"step": step.name, "status": "up-to-date"})
            continue

        need = []
        for i in missing_inputs(step):
            if i not in planned:
                need.append(i)
        if need:
            msg = f"missing input {need[0]}"
            if step.name.startswith("retrieval relevance"):
                log(f"{label:<55} SKIP ({msg}; rate the sheet by hand first)")
                records.append({"step": step.name, "status": "skipped-manual"})
                continue
            if args.offline:  # input would come from an API step that isn't running
                log(f"{label:<55} SKIP ({msg}; offline)")
                records.append({"step": step.name, "status": "skipped-manual"})
                continue
            log(f"{label:<55} BLOCKED ({msg})")
            failed = step.name
            records.append({"step": step.name, "status": "blocked", "reason": msg})
            break

        if args.dry_run:
            if step.api:
                log(f"{label:<55} WOULD RUN ({why})  [API]")
            else:
                log(f"{label:<55} WOULD RUN ({why})")
            planned.update(step.outputs)
            records.append({"step": step.name, "status": "planned"})
            continue

        # hard gate: no expensive API run before a clean smoke test
        if step.api and step.stage != "smoke" and not gate_ok:
            if not smoke_gate(log):
                failed = "smoke gate"
                records.append({"step": step.name, "status": "blocked", "reason": "smoke gate"})
                break
            gate_ok = True

        if step.stage == "report":  # the report reads the manifest, so write it first
            write_manifest(args, stages, started, commit, dirty, records, failed)
        log(f"{label:<55} RUN ({why})")
        t0 = time.time()
        archive_error_rows(step)
        step_log = RUN_DIR / f"{len(records) + 1:02d}_{slug(step.name)}.log"
        rc = run_logged(step.cmd, step_log)
        dt = round(time.time() - t0, 1)

        if rc == 0:
            status = "ok"
        else:
            status = f"failed rc={rc}"
        records.append({"step": step.name, "status": status, "seconds": dt,
                        "cmd": " ".join(step.cmd[1:]), "log": str(step_log.relative_to(ROOT))})
        if rc != 0:
            log(f"  FAILED after {dt}s (exit {rc}). Full output: {step_log.relative_to(ROOT)}. "
                f"Fix and re-run the same command; finished work is kept.")
            failed = step.name
            break
        log(f"  done in {dt}s")

    if not args.dry_run:
        write_manifest(args, stages, started, commit, dirty, records, failed)
        # even after a failure build the report, so it shows where the run stopped
        if failed and "report" in stages:
            log("Building the report anyway (it will show where the run stopped).")
            run_logged(py("build_report.py"), RUN_DIR / "zz_report_after_failure.log")
        write_manifest(args, stages, started, commit, dirty, records, failed)
        archive_outputs()
        append_index(stages, failed, records)

    # last line
    if failed:
        end = f"=== FAILED at {failed}"
    else:
        end = "=== finished"
    if not failed and not args.dry_run:
        end += " -> " + REPORT
    if not args.dry_run:
        end += " | archived in " + str(RUN_DIR.relative_to(ROOT))
    log(end)

    if failed:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
