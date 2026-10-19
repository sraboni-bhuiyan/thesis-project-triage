# Retrieval-Augmented Emergency Triage Support

Master's thesis, FH Dortmund (Sraboni Bhuiyan): a fairness-aware evaluation of retrieval-augmented
LLM triage across age and gender groups.

An LLM predicts an urgency colour and a specialty for MIMIC-IV-Ext emergency cases under four
configurations: C1 prompt-only, C2 RAG, C3 RAG with demographic masking, C4 fixed-context control.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # fill in OPENROUTER_BASE_URL and OPENAI_API_KEY
```

The MIMIC-IV-Ext folder (PhysioNet credentialed access) and the guideline PDFs go in place locally;
neither is distributed with this repository.

## Run

```bash
python run_all.py --dry-run       # show the plan, spends nothing
python run_all.py                 # run everything missing or out of date
python run_all.py --offline       # recompute metrics + report without API calls
python -m pytest tests            # unit tests
```

Output: `results/thesis_report.html` and `results/numbers.json`.

## Documents

- `command.md`: full rebuild runbook and project rules
- `PROTOCOL.md`: frozen study design
- `DEVIATIONS.md`: changes against the proposal and protocol
- `docs/`: proposal, results notes, thesis outline
