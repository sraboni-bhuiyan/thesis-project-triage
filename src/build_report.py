# build ONE html report from everything in results/, no API
# only reads files the pipeline already wrote, figures go in as base64
# so it's one file (email, archive, print to pdf)
# outputs:
#   results/thesis_report.html   the report
#   results/numbers.json         every number in the report, flat (for latex/slides)
#
# usage (normally run_all.py calls it):
#   python src/build_report.py

import base64
import csv
import html
import json
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal
from datetime import datetime, timezone

from config import CASES_PATH, CORPUS_FILE, RESULTS_PATH

OUT = RESULTS_PATH / "thesis_report.html"
NUMBERS = RESULTS_PATH / "numbers.json"
CONFIG_ORDER = ["baseline", "rag", "rag_masked", "rag_fixed"]
LABEL = {"baseline": "C1 Baseline", "rag": "C2 RAG", "rag_masked": "C3 RAG + masking",
         "rag_fixed": "C4 Fixed context", "baseline_rerun": "C1 re-run"}
ALPHA = 0.05

numbers = {}  # flat key -> value, saved to numbers.json


def load_json(name):
    p = RESULTS_PATH / name
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def esc(v):
    return html.escape(str(v))


def round_half_up(v, d):
    # round from the decimal string, so 0.5015 -> 50.2% like a reader expects
    return Decimal(str(v)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)


def pct(v):
    try:
        return f"{round_half_up(Decimal(str(float(v))) * 100, 1)}%"
    except (TypeError, ValueError, ArithmeticError):
        return "—"


def num(v, d=3):
    try:
        return f"{round_half_up(float(v), d)}"
    except (TypeError, ValueError, ArithmeticError):
        return "—"


def pval(p):
    if p is None:
        return "—"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def table(headers, rows, cls=""):
    th = ""
    for h in headers:
        th += f"<th>{esc(h)}</th>"
    trs = ""
    for r in rows:
        trs += "<tr>"
        for c in r:
            trs += f"<td>{c}</td>"
        trs += "</tr>"
    return f'<div class="scroll"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>'


def counts_text(d):
    # "a-only / b-only" cell
    return esc(f"{d.get('only_a_correct')} / {d.get('only_b_correct')}")


# ---- sections

def section_provenance():
    m = load_json("run_manifest.json") or {}
    commit = f"{m.get('git_commit', '—')}"
    if m.get("git_dirty"):
        commit += " (uncommitted changes)"
    rows = [
        ("Report built (UTC)", datetime.now(timezone.utc).isoformat(timespec="seconds")),
        ("Pipeline run started", m.get("started_utc", "—")),
        ("Git commit", commit),
        ("Python", m.get("python", "—")),
        ("Corpus sha256", m.get("corpus_sha256", "—")),
        ("Cases sha256", m.get("cases_sha256", "—")),
        ("Variants sha256", m.get("variants_sha256", "—")),
        ("Fixed-context sha256", m.get("fixed_context_sha256", "—")),
    ]
    failed = m.get("failed_at")
    warn = ""
    if failed:
        warn = f'<p class="warn">Last pipeline run stopped at: {esc(failed)}</p>'
    cells = []
    for a, b in rows:
        cells.append((esc(a), f"<code>{esc(b)}</code>"))
    return "<h2>1. Provenance</h2>" + warn + table(["Item", "Value"], cells)


def majority_reference():
    if not CASES_PATH.exists():
        return None, None
    labels = []
    with open(CASES_PATH, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            labels.append(r["ground_truth_urgency"])
    if not labels:
        return None, None
    dist = Counter(labels)
    label, n = dist.most_common(1)[0]
    numbers["ground_truth_distribution"] = dict(dist)
    numbers["majority_class"] = label
    numbers["majority_class_acc"] = n / len(labels)
    return label, n / len(labels)


def section_headline():
    p = RESULTS_PATH / "results_summary.csv"
    if not p.exists():
        return "<h2>2. Headline results</h2><p class='warn'>results_summary.csv not found.</p>"
    with open(p, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    metrics = [
        ("urgency_exact_acc", "Urgency exact", pct), ("urgency_adjacent_acc", "Urgency ±1", pct),
        ("over_triage_rate", "Over-triage", pct), ("under_triage_rate", "Under-triage", pct),
        ("specialty_acc", "Specialty", pct), ("specialty_acc_any_listed", "Specialty (any listed)", pct),
        ("variants_urgency_dpr", "Variant urgency DPR", num), ("urgency_consistency", "Urgency consistency", pct),
        ("n_cases", "n cases", str), ("n_error", "ERROR", str), ("n_invalid_urgency", "INVALID", str),
    ]
    out_rows = []
    for key, name, f in metrics:
        line = [esc(name)]
        for r in rows:
            if r.get(key) not in (None, ""):
                line.append(esc(f(r.get(key))))
            else:
                line.append("—")
        out_rows.append(line)
        for r in rows:
            numbers[f"{r['config']}.{key}"] = r.get(key)

    models = set()
    for r in rows:
        models.add(r.get("model", ""))
    models = sorted(models)

    label, acc = majority_reference()
    ref = ""
    if acc:
        ref = (f'<p class="note">Majority-class reference: always predicting <b>{esc(label)}</b> scores '
               f"<b>{pct(acc)}</b> exact urgency accuracy. Compare every configuration against this.</p>")

    headers = ["Metric"]
    for r in rows:
        headers.append(r["config"])
    return ("<h2>2. Headline results</h2>"
            f"<p>Model(s): <code>{esc(', '.join(models))}</code></p>"
            + table(headers, out_rows) + ref)


def mcnemar_rows(items, flag, thr):
    # table rows, sorted by metric then config order
    order = {}
    all_configs = CONFIG_ORDER + ["baseline_rerun"]
    for i in range(len(all_configs)):
        order[all_configs[i]] = i
    items.sort(key=lambda t: (t[2], order.get(t[0], 9), order.get(t[1], 9)))

    out = []
    for a, b, metric, d in items:
        p = d.get("p_value")
        numbers[f"mcnemar.{a}_vs_{b}.{metric}.p"] = p
        sig = ""
        if flag and p is not None:
            if p < thr:
                sig = '<span class="ok">survives</span>'
            elif p < ALPHA:
                sig = '<span class="mid">p&lt;0.05 only</span>'
            else:
                sig = "n.s."
        line = [esc(metric), esc(f"{LABEL.get(a, a)} vs {LABEL.get(b, b)}"), pct(d.get("acc_a")),
                pct(d.get("acc_b")), counts_text(d), pval(p)]
        if flag:
            line.append(sig)
        out.append(line)
    return out


def section_mcnemar():
    files = sorted(RESULTS_PATH.glob("mcnemar_*.json"))
    if not files:
        return "<h2>3. Significance</h2><p class='warn'>No McNemar results found.</p>"

    main = []
    robustness = []
    for fp in files:
        d = json.loads(fp.read_text(encoding="utf-8"))
        stem = fp.stem[len("mcnemar_"):]
        metric = d.get("metric") or stem.rsplit("_", 1)[-1]
        pair = stem[: -(len(metric) + 1)]
        a, b = pair.split("_vs_")
        if b == "baseline_rerun":
            robustness.append((a, b, metric, d))
        else:
            main.append((a, b, metric, d))

    k = len(main)
    if k:
        thr = ALPHA / k
    else:
        thr = None
    numbers["n_comparisons"] = k
    numbers["bonferroni_alpha"] = thr

    html_ = (f"<h2>3. Significance (paired exact McNemar)</h2>"
             f"<p>{k} pre-specified comparisons. Bonferroni threshold α = {ALPHA}/{k} = <b>{num(thr, 4)}</b>.</p>"
             + table(["Metric", "Comparison", "acc A", "acc B", "A-only / B-only", "p", "Bonferroni"],
                     mcnemar_rows(main, True, thr)))
    if robustness:
        html_ += ("<h3>Robustness: baseline vs its own re-run</h3>"
                  + table(["Metric", "Comparison", "acc A", "acc B", "A-only / B-only", "p"],
                          mcnemar_rows(robustness, False, thr)))
    return html_


def range_text(d, pc=True):
    # "min–max" from a {"min", "max"} dict
    if not d:
        return "—"
    if pc:
        f = pct
    else:
        f = num
    return f"{f(d.get('min'))}–{f(d.get('max'))}"


def section_rq1_range():
    files = sorted((RESULTS_PATH / "robustness").glob("mcnemar_baseline_rep*_vs_*.json"))
    if not files:
        return "<h2>4. RQ1 across C1 runs</h2><p class='warn'>No results/robustness/ files found.</p>"

    rows = []
    pvals = {}
    for b in ("rag", "rag_masked"):
        for metric in ("urgency", "specialty"):
            items = [("run1", load_json(f"mcnemar_baseline_vs_{b}_{metric}.json"))]
            for fp in files:
                rep, rest = fp.stem[len("mcnemar_baseline_"):].split("_vs_", 1)
                if rest == f"{b}_{metric}":
                    items.append((rep, json.loads(fp.read_text(encoding="utf-8"))))
            for run, d in items:
                if not d:
                    continue
                if (b, metric) not in pvals:
                    pvals[(b, metric)] = []
                pvals[(b, metric)].append(d.get("p_value"))
                numbers[f"rq1.baseline_{run}_vs_{b}.{metric}.p"] = d.get("p_value")
                numbers[f"rq1.baseline_{run}_vs_{b}.{metric}.acc_a"] = d.get("acc_a")
                rows.append([esc(metric), esc(f"C1 {run} vs {LABEL[b]}"), pct(d.get("acc_a")), pct(d.get("acc_b")),
                             counts_text(d), pval(d.get("p_value"))])

    summary = ""
    for (b, metric), ps in pvals.items():
        good = []
        for p in ps:
            if p is not None:
                good.append(p)
        numbers[f"rq1.{b}.{metric}.p_min"] = min(good)
        numbers[f"rq1.{b}.{metric}.p_max"] = max(good)
        summary += (f"<li>C1 vs {esc(LABEL[b])} ({esc(metric)}): p = {pval(min(good))} to {pval(max(good))} "
                    f"over {len(good)} C1 runs</li>")

    return ("<h2>4. RQ1 across C1 runs (robustness)</h2>"
            "<p>C1 vs C2 and C1 vs C3, repeated against every complete C1 run. Outside the family of 12; "
            "no Bonferroni label.</p>"
            + table(["Metric", "Comparison", "acc A", "acc B", "A-only / B-only", "p"], rows)
            + "<ul>" + summary + "</ul>")


def section_sensitivity():
    parts = ["<h2>5. Sensitivity: sub-specialties mapped to a parent</h2>",
             "<p>Pre-specified (PROTOCOL §6.2.1). Ground-truth sub-specialties are mapped to a listed parent; "
             "same predictions. Kept apart from the 12 main comparisons; no Bonferroni label.</p>"]

    p = RESULTS_PATH / "results_summary.csv"
    if p.exists():
        with open(p, encoding="utf-8", newline="") as f:
            summ = list(csv.DictReader(f))
        rows = []
        for r in summ:
            for k in ("specialty_acc_mapped", "specialty_acc_any_listed_mapped", "specialty_ceiling"):
                numbers[f"{r['config']}.{k}"] = r.get(k)
            rows.append([esc(LABEL.get(r["config"], r["config"])), pct(r.get("specialty_acc")),
                         pct(r.get("specialty_acc_mapped")), pct(r.get("specialty_acc_any_listed")),
                         pct(r.get("specialty_acc_any_listed_mapped"))])
        parts.append(table(["Config", "Strict", "Strict (mapped)", "Any listed", "Any listed (mapped)"], rows))
        if summ:
            parts.append(f"<p class='note'>Strict-accuracy ceiling (primary analysis): "
                         f"{pct(summ[0].get('specialty_ceiling'))}.</p>")

    # mapped mcnemar files, sorted by config order
    rows = []
    for fp in sorted((RESULTS_PATH / "sensitivity").glob("mcnemar_*_specialty_mapped.json")):
        d = json.loads(fp.read_text(encoding="utf-8"))
        a, b = fp.stem[len("mcnemar_"):-len("_specialty_mapped")].split("_vs_")
        numbers[f"sensitivity.mcnemar.{a}_vs_{b}.specialty_mapped.p"] = d.get("p_value")
        line = [esc(f"{LABEL[a]} vs {LABEL[b]}"), pct(d.get("acc_a")), pct(d.get("acc_b")),
                counts_text(d), pval(d.get("p_value"))]
        rows.append((CONFIG_ORDER.index(a), CONFIG_ORDER.index(b), line))
    if rows:
        rows.sort()
        lines = []
        for r in rows:
            lines.append(r[2])
        parts.append(table(["Comparison (specialty, mapped)", "acc A", "acc B", "A-only / B-only", "p"], lines))
    return "".join(parts)


def section_noise():
    rows = []
    for cfg in CONFIG_ORDER:
        for tag, set_name in (("", "performance"), ("_variants", "variants")):
            d = load_json(f"noise_{cfg}{tag}.json")
            if not d:
                continue
            for k in ("n_runs", "n_compared", "urgency_run_agreement", "specialty_run_agreement",
                      "urgency_all_runs_agree", "urgency_agreement_range", "specialty_agreement_range",
                      "urgency_exact_acc_range", "specialty_acc_range", "consistency_range", "dpr_range",
                      "gender_flip_rate_range", "age_flip_rate_range"):
                if k in d:
                    numbers[f"noise.{cfg}{tag}.{k}"] = d[k]
            rows.append([esc(LABEL[cfg]), esc(set_name), esc(d.get("n_runs")), esc(d.get("n_compared")),
                         f"{pct(d.get('urgency_run_agreement'))} ({range_text(d.get('urgency_agreement_range'))})",
                         f"{pct(d.get('specialty_run_agreement'))} ({range_text(d.get('specialty_agreement_range'))})",
                         pct(d.get("urgency_all_runs_agree")), range_text(d.get("urgency_exact_acc_range")),
                         range_text(d.get("consistency_range")), range_text(d.get("dpr_range"), pc=False)])
    if not rows:
        return "<h2>6. Noise floor</h2><p class='warn'>No noise_*.json found.</p>"
    return ("<h2>6. Noise floor (identical re-runs)</h2>"
            "<p>Same configuration, same inputs, repeated. Agreement is the mean over every pair of runs "
            "(range in brackets); the last three columns are the min–max across runs. Read every fairness "
            "and consistency difference against these numbers.</p>"
            + table(["Config", "Set", "Runs", "n", "Urgency agreement", "Specialty agreement",
                     "All runs agree (urgency)", "Exact acc. range", "Consistency range", "DPR range"], rows))


def section_fairness():
    data = {}
    for cfg in CONFIG_ORDER[:3]:
        d = load_json(f"fairness_{cfg}.json")
        if d:
            data[cfg] = d
    if not data:
        return "<h2>7. Fairness (variant set)</h2><p class='warn'>No fairness_*.json found.</p>"

    rows = []
    effects = []
    groups = []
    pairs = []
    for cfg, d in data.items():
        g = d["gender_flip"]
        a = d["age_flip"]
        nz = d["noise_disagreement"]
        fg = d["fisher_gender_flip_vs_noise"]
        fa = d["fisher_age_flip_vs_noise"]

        numbers[f"fairness.{cfg}.dpr"] = d["dpr"]
        numbers[f"fairness.{cfg}.eod"] = d["eod"]
        numbers[f"fairness.{cfg}.tpr_gap"] = d["tpr_gap"]
        numbers[f"fairness.{cfg}.fpr_gap"] = d["fpr_gap"]
        numbers[f"fairness.{cfg}.consistency"] = d["consistency"]
        numbers[f"fairness.{cfg}.gender_flip_rate"] = g["rate"]
        numbers[f"fairness.{cfg}.age_flip_rate"] = a["rate"]
        numbers[f"fairness.{cfg}.noise_rate"] = nz["rate"]
        numbers[f"fairness.{cfg}.fisher_gender_p"] = fg["p_value"]
        numbers[f"fairness.{cfg}.fisher_age_p"] = fa["p_value"]
        numbers[f"fairness.{cfg}.high_rate_by_group"] = d["high_rate_by_group"]
        numbers[f"fairness.{cfg}.n_by_group"] = d["n_by_group"]

        rows.append([esc(LABEL[cfg]), num(d["dpr"]), f"{num(d['eod'])} ({num(d['tpr_gap'])} / {num(d['fpr_gap'])})",
                     f"{pct(d['consistency'])} ({d['n_consistent']}/{d['n_base_cases']})",
                     f"{pct(g['rate'])} ({g['flips']}/{g['n_pairs']})", f"{pct(a['rate'])} ({a['flips']}/{a['n_pairs']})",
                     f"{pct(nz['rate'])} ({nz['disagree']}/{nz['n_comparisons']})",
                     pval(fg["p_value"]), pval(fa["p_value"])])

        group_line = [esc(LABEL[cfg])]
        for grp, r in d["high_rate_by_group"].items():
            group_line.append(f"{pct(r)} (n={d['n_by_group'][grp]})")
        groups.append(group_line)

        for eff, ref, other in (("gender_main_effect", "male", "female"), ("age_main_effect", "young", "old")):
            m = d[eff]
            for k in ("p_value", "verdict", "rate_difference", "alpha"):
                numbers[f"fairness.{cfg}.{eff}.{k}"] = m[k]
            per_run = []
            for run, r in m["by_run"].items():
                numbers[f"fairness.{cfg}.{eff}.{run}.only_{other}_high"] = r[f"only_{other}_high"]
                numbers[f"fairness.{cfg}.{eff}.{run}.only_{ref}_high"] = r[f"only_{ref}_high"]
                per_run.append(f"{run}: {r[f'only_{other}_high']} vs {r[f'only_{ref}_high']}")
            if m["verdict"] == "claimed":
                cls = "warn"
            elif m["verdict"] == "fragile":
                cls = "mid"
            else:
                cls = ""
            effects.append([esc(LABEL[cfg]), esc(f"{other} vs {ref}"), esc(m["n_pairs"]), esc("; ".join(per_run)),
                            pval(m["p_value"]), f'<span class="{cls}">{esc(m["verdict"])}</span>'])

        for gp in d["group_pairs"]:
            numbers[f"fairness.{cfg}.group_pair.{gp['a']}_vs_{gp['b']}.p"] = gp["p_value"]
            if gp["significant"]:
                result = "significant"
            else:
                result = "n.s."
            pairs.append([esc(LABEL[cfg]), esc(f"{gp['a']} vs {gp['b']}"),
                          esc(f"{gp['only_a_high']} / {gp['only_b_high']}"), pval(gp["p_value"]), result])

    first = list(data.values())[0]
    alpha_main = first["gender_main_effect"]["alpha"]
    alpha_pairs = first["group_pairs_bonferroni_alpha"]
    numbers["fairness.main_effect_alpha"] = alpha_main
    numbers["fairness.group_pairs_alpha"] = alpha_pairs

    group_headers = ["Config"]
    for grp in first["high_rate_by_group"]:
        group_headers.append(esc(grp.replace("_", " ")))

    return ("<h2>7. Fairness (variant set, primary runs)</h2>"
            "<p>Fairness claims rest on this section (variant DPR, consistency, flip rates), read against the "
            "noise floor (section 6). DPR and EOD are max − min across the four age × gender groups; high "
            "acuity = red or orange. Flip rates are compared with the run1-vs-rep2 disagreement rate "
            "(Fisher's exact test).</p>"
            + table(["Config", "DPR", "EOD (TPR gap / FPR gap)", "Consistency", "Gender flips", "Age flips",
                     "Noise (run1 vs rep2)", "p gender vs noise", "p age vs noise"], rows)
            + "<h3>Predicted high-acuity rate by group</h3>"
            + table(group_headers, groups)
            + f"<h3>Main effects (paired exact McNemar, Bonferroni over 6 tests, α = {num(alpha_main, 4)})</h3>"
              "<p>Claimed only if significant in the primary run AND the same direction in every repeat run; "
              "otherwise a significant effect is fragile. Per-run counts: pairs where only the first-named "
              "group is high vs only the second-named group.</p>"
            + table(["Config", "Effect", "Pairs", "Per-run counts", "p (run1)", "Verdict"], effects)
            + f"<h3>Pairwise group comparisons (Bonferroni α = {num(alpha_pairs, 4)})</h3>"
            + table(["Config", "Groups", "A-only / B-only high", "p", "Result"], pairs)
            + "<p class='note'>C3 masks age and gender, so the four C3 prompts of a base case are identical; "
              "any C3 difference between variants is run-to-run noise.</p>")


def section_retrieval():
    parts = ["<h2>8. Retrieval</h2>"]

    # manual relevance scores
    d = load_json("retrieval_eval.json")
    if d:
        for k, v in d.items():
            numbers[f"retrieval.{k}"] = v
        parts.append(table(["Cases rated", "P@k", "MRR", "nDCG@k"],
                           [[esc(d.get("n_cases_rated")), num(d.get("precision_at_k")), num(d.get("mrr")),
                             num(d.get("ndcg_at_k"))]]))
        if d.get("precision_by_rank"):
            precision = ["Precision"]
            for v in d["precision_by_rank"].values():
                precision.append(num(v))
            parts.append(table(["Rank"] + list(d["precision_by_rank"]), [precision]))
    else:
        parts.append("<p class='warn'>retrieval_eval.json not found (rate the sheet, then re-run).</p>")

    # C4 fixed context
    fixed = CORPUS_FILE.parent / "fixed_context.json"
    if fixed.exists():
        fc = json.loads(fixed.read_text(encoding="utf-8"))
        nq = fc.get("n_queries") or 1
        parts.append(f"<h3>C4 fixed context (derived from {esc(fc.get('n_queries'))} logged queries)</h3>")
        chunk_rows = []
        for c in fc.get("chunks", []):
            chunk_rows.append([f"<code>{esc(c['chunk_id'])}</code>", esc(c["retrieval_count"]),
                               pct(c["retrieval_count"] / nq)])
        parts.append(table(["chunk_id", "retrieved", "share"], chunk_rows))
        parts.append("<p class='note'>Log files counted: " + esc(", ".join(fc.get("log_files", {}))) + "</p>")
        numbers["c4.chunk_ids"] = fc.get("chunk_ids")
        numbers["c4.n_queries"] = fc.get("n_queries")

    # descriptive stats
    st = load_json("retrieval_stats.json")
    if st:
        rows = []
        for name, v in st.items():
            if name == "c4_overlap_with_c2":
                continue
            key = "retrieval_stats." + name.replace(" ", "_")
            for k in ("n_queries", "distinct_chunks", "distinct_chunks_rank1", "mean_top1_similarity",
                      "share_by_document"):
                numbers[f"{key}.{k}"] = v[k]
            shares = []
            for doc, x in v["share_by_document"].items():
                shares.append(f"{esc(doc)} {pct(x)}")
            rows.append([esc(name), esc(v["n_queries"]), esc(v["distinct_chunks"]), esc(v["distinct_chunks_rank1"]),
                         num(v["mean_top1_similarity"]), ", ".join(shares)])
        parts.append("<h3>Descriptive retrieval statistics</h3>"
                     + table(["Log", "Queries", "Distinct chunks", "Distinct at rank 1", "Mean top-1 similarity",
                              "Share by document"], rows))

        ov = st.get("c4_overlap_with_c2")
        if ov:
            numbers["c4.overlap_with_c2.cases_by_overlap"] = ov["cases_by_overlap"]
            numbers["c4.overlap_with_c2.mean"] = ov["mean_overlap"]
            k = len(ov["cases_by_overlap"]) - 1
            headers = []
            for i in ov["cases_by_overlap"]:
                headers.append(f"{i} of {k}")
            headers.append("Mean")
            counts = []
            for n in ov["cases_by_overlap"].values():
                counts.append(esc(n))
            counts.append(num(ov["mean_overlap"], 2))
            parts.append(f"<h3>C4 fixed chunks found in C2's top-{k} (performance set, n = {ov['n_cases']})</h3>"
                         + table(headers, [counts]))
    return "".join(parts)


def section_errors():
    p = RESULTS_PATH / "error_cases.csv"
    if not p.exists():
        return ""
    types = Counter()
    with open(p, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            types[r["type"]] += 1
    numbers["error_cases_by_type"] = dict(types)
    rows = []
    for t, n in sorted(types.items()):
        rows.append([esc(t), esc(n)])
    return ("<h2>9. Qualitative candidates</h2><p>Rows in <code>error_cases.csv</code> by type:</p>"
            + table(["Type", "Rows"], rows))


def section_figures():
    figs = sorted((RESULTS_PATH / "figures").glob("*.png"))
    if not figs:
        return ""
    out = "<h2>10. Figures</h2>"
    for fp in figs:
        b64 = base64.b64encode(fp.read_bytes()).decode("ascii")
        out += (f'<figure><img src="data:image/png;base64,{b64}" alt="{esc(fp.stem)}">'
                f"<figcaption>{esc(fp.name)}</figcaption></figure>")
    return out


def section_appendix():
    files = (sorted(RESULTS_PATH.glob("*.json")) + sorted(RESULTS_PATH.glob("robustness/*.json"))
             + sorted(RESULTS_PATH.glob("sensitivity/*.json")))
    out = "<h2>Appendix: raw result files</h2>"
    for fp in files:
        if fp.name == "numbers.json":
            continue
        out += (f"<details><summary><code>{esc(fp.relative_to(RESULTS_PATH).as_posix())}</code></summary>"
                f"<pre>{esc(fp.read_text(encoding='utf-8'))}</pre></details>")
    return out


CSS = """
:root{--bg:#fff;--fg:#1d1d1f;--muted:#666;--line:#e3e3e3;--head:#f5f5f7;--ok:#1a7f37;--mid:#9a6700;--warn:#b42318}
body{background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
max-width:980px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:1.6rem;margin:0 0 4px}h2{margin-top:2.2rem;border-bottom:1px solid var(--line);padding-bottom:4px}
.sub{color:var(--muted);margin:0 0 1rem}.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;margin:.6rem 0;font-variant-numeric:tabular-nums}
th,td{border:1px solid var(--line);padding:6px 9px;text-align:left;white-space:nowrap}th{background:var(--head)}
code{font-size:.88em}.note{color:var(--muted)}.warn{color:var(--warn);font-weight:600}
.ok{color:var(--ok);font-weight:600}.mid{color:var(--mid);font-weight:600}
figure{margin:1rem 0}img{max-width:100%;border:1px solid var(--line)}figcaption{color:var(--muted);font-size:.85em}
pre{background:var(--head);padding:10px;overflow-x:auto;font-size:12px}details{margin:4px 0}
@media print{details{display:none}h2{break-after:avoid}figure,table{break-inside:avoid}}
"""


def main():
    # sections in report order (also the order numbers.json is filled)
    body = ""
    body += section_provenance()
    body += section_headline()
    body += section_mcnemar()
    body += section_rq1_range()
    body += section_sensitivity()
    body += section_noise()
    body += section_fairness()
    body += section_retrieval()
    body += section_errors()
    body += section_figures()
    body += section_appendix()

    doc = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
           f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>Thesis Results Report</title><style>{CSS}</style></head><body>"
           f"<h1>Fairness-Aware RAG Triage: Results Report</h1>"
           f"<p class='sub'>Generated automatically from <code>results/</code>. Do not edit by hand; re-run "
           f"<code>python run_all.py</code>.</p>{body}</body></html>")
    OUT.write_text(doc, encoding="utf-8")
    NUMBERS.write_text(json.dumps(numbers, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size // 1024} KB) and {NUMBERS}")


if __name__ == "__main__":
    main()
