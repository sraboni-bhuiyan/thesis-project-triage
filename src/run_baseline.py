# C1: prompt-only baseline (no guideline text). calls the API, resumable
#
# usage:
#   python src/run_baseline.py [--cases-file F] [--output F] [--limit N] [--no-resume] [--sleep S]

import argparse

from common import LLM, build_case_block, load_prompt, read_csv, resolve, run_predictions
from config import CASES_PATH


def main():
    ap = argparse.ArgumentParser(description="C1 prompt-only triage predictions.")
    ap.add_argument("--cases-file", default=str(CASES_PATH))
    ap.add_argument("--output", default="results/baseline_predictions.csv")
    ap.add_argument("--limit", type=int, default=None, help="Only the first N cases (smoke test).")
    ap.add_argument("--no-resume", action="store_true", help="Start the output file from scratch.")
    ap.add_argument("--sleep", type=float, default=0.0, help="Seconds between calls (rate limits).")
    args = ap.parse_args()

    template = load_prompt("baseline.txt")
    cases = read_csv(args.cases_file)[:args.limit]
    llm = LLM()

    def predict(case):
        prompt = template.format(case_block=build_case_block(case))
        return llm.ask(prompt)

    # e.g. "C1 baseline (fairness_variants)"
    desc = f"C1 baseline ({resolve(args.cases_file).stem})"
    run_predictions(cases, args.output, predict, desc=desc, resume=not args.no_resume, sleep=args.sleep)


if __name__ == "__main__":
    main()
