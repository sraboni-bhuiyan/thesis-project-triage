# summary table + figures over the primary runs (protocol 6), no API
# outputs:
#   results/results_summary.csv   one row per config, headline metrics
#   results/summary.json          same + per group detail, run ranges, majority reference
#   results/figures/*.png         accuracy vs majority, predicted colour distribution,
#                                 variant high-acuity rate by group
# run ranges use all runs (primary + _repN files)
# performance-set DPR/EOD only descriptive (case mix, protocol 6.3)
#
# usage:
#   python src/make_summary.py

import csv
from collections import Counter

from common import normalize_urgency, read_csv, write_json
from config import CASES_PATH, ERROR, MODEL_NAME, RESULTS_PATH, URGENCY_ORDER
from fairness_metrics import AGE_PAIRS, GENDER_PAIRS, GROUPS, counterfactual_consistency, dpr, eod, flip_rate
from metrics import evaluate, load_truth

CONFIGS = ["baseline", "rag", "rag_masked", "rag_fixed"]
LABEL = {"baseline": "C1", "rag": "C2", "rag_masked": "C3", "rag_fixed": "C4"}
FIG_DIR = RESULTS_PATH / "figures"
COLOURS = {"red": "#c0392b", "orange": "#e67e22", "yellow": "#f1c40f", "green": "#27ae60", "blue": "#2980b9",
           "INVALID": "#999999"}


def rep_number(path):
    # "x_predictions_rep3" -> 3
    return int(path.stem.rsplit("rep", 1)[1])


def run_files(cfg, tag=""):
    # primary file first, then rep2, rep3, ...
    primary = RESULTS_PATH / f"{cfg}{tag}_predictions.csv"
    if not primary.exists():
        return []
    reps = list(RESULTS_PATH.glob(f"{cfg}{tag}_predictions_rep*.csv"))
    reps.sort(key=rep_number)
    return [primary] + reps


def majority_reference(cases):
    labels = []
    for c in cases:
        labels.append(c["ground_truth_urgency"])
    dist = Counter(labels)
    label, n = dist.most_common(1)[0]
    return {"label": label, "acc": n / len(cases), "distribution": dict(dist)}


def summarise():
    truth_rows = read_csv(CASES_PATH)
    majority = majority_reference(truth_rows)
    rows = []
    detail = {"majority_reference": majority, "configs": {}}

    for cfg in CONFIGS:
        files = run_files(cfg)
        if not files:
            continue

        runs = []
        for f in files:
            runs.append(read_csv(f))
        cases = load_truth(runs[0])
        ev = evaluate(cases, runs[0])

        # accuracy of every run
        accs = []
        spec = []
        for r in runs:
            accs.append(evaluate(cases, r)["urgency_exact_acc"])
        for r in runs:
            spec.append(evaluate(cases, r)["specialty_acc"])

        perf_dpr = dpr(cases, runs[0])
        perf_eod = eod(cases, runs[0])

        models = set()
        for p in runs[0]:
            if p.get("response_model"):
                models.add(p["response_model"])

        row = {"config": cfg, "label": LABEL[cfg], "model": ";".join(sorted(models)), "n_runs": len(runs)}
        # only the plain numbers from evaluate, no dicts
        for k, v in ev.items():
            if not isinstance(v, dict):
                row[k] = v
        row["urgency_exact_acc_min"] = min(accs)
        row["urgency_exact_acc_max"] = max(accs)
        row["specialty_acc_min"] = min(spec)
        row["specialty_acc_max"] = max(spec)
        row["performance_dpr"] = perf_dpr["dpr"]
        row["performance_eod"] = perf_eod["eod"]

        # predicted colours, primary run, no ERROR
        pred_labels = []
        for p in runs[0]:
            if p["pred_urgency"] != ERROR:
                pred_labels.append(normalize_urgency(p["pred_urgency"]))

        file_names = []
        for f in files:
            file_names.append(f.name)

        cfg_detail = {
            "files": file_names,
            "metrics": ev,
            "urgency_exact_acc_by_run": accs,
            "specialty_acc_by_run": spec,
            "performance_dpr": perf_dpr,
            "performance_eod": perf_eod,
            "pred_distribution": dict(Counter(pred_labels)),
        }

        # variant set
        vfiles = run_files(cfg, "_variants")
        if vfiles:
            vruns = []
            for f in vfiles:
                vruns.append(read_csv(f))
            vcases = load_truth(vruns[0])
            v_dpr = dpr(vcases, vruns[0])
            v_eod = eod(vcases, vruns[0])
            cons = counterfactual_consistency(vruns[0])
            g = flip_rate(vruns[0], GENDER_PAIRS)
            a = flip_rate(vruns[0], AGE_PAIRS)

            dprs = []
            for r in vruns:
                dprs.append(dpr(vcases, r)["dpr"])
            consistencies = []
            for r in vruns:
                consistencies.append(counterfactual_consistency(r)["consistency"])

            row["variants_urgency_dpr"] = v_dpr["dpr"]
            row["variants_eod"] = v_eod["eod"]
            row["urgency_consistency"] = cons["consistency"]
            row["gender_flip_rate"] = g["rate"]
            row["age_flip_rate"] = a["rate"]
            row["variants_dpr_min"] = min(dprs)
            row["variants_dpr_max"] = max(dprs)
            row["consistency_min"] = min(consistencies)
            row["consistency_max"] = max(consistencies)

            vfile_names = []
            for f in vfiles:
                vfile_names.append(f.name)
            cfg_detail["variant_files"] = vfile_names
            cfg_detail["variants_dpr"] = v_dpr
            cfg_detail["variants_eod"] = v_eod
            cfg_detail["consistency"] = cons
            cfg_detail["gender_flip"] = g
            cfg_detail["age_flip"] = a

        rows.append(row)
        detail["configs"][cfg] = cfg_detail
    return rows, detail


def write_csv(rows, path):
    # columns = all keys, in order seen
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def figures(rows, detail):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    labels = []
    for r in rows:
        labels.append(r["label"])
    maj = detail["majority_reference"]

    # fig 1: exact urgency acc (primary run) + range over runs, vs majority class
    fig, ax = plt.subplots(figsize=(6, 4))
    acc = []
    err_low = []
    err_high = []
    for r in rows:
        a = r["urgency_exact_acc"]
        acc.append(a)
        err_low.append(a - r["urgency_exact_acc_min"])
        err_high.append(r["urgency_exact_acc_max"] - a)
    ax.bar(labels, acc, yerr=[err_low, err_high], capsize=5, color="#4c72b0")
    ax.axhline(maj["acc"], color="k", ls="--", lw=1, label=f"majority class ({maj['label']})")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Exact urgency accuracy")
    ax.set_title("Urgency accuracy (bar: primary run; whisker: range across runs)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_urgency_accuracy.png", dpi=150)
    plt.close(fig)

    # fig 2: predicted urgency distribution per config vs ground truth
    fig, ax = plt.subplots(figsize=(7, 4))
    cols = ["Ground truth"] + labels
    dists = [maj["distribution"]]
    for r in rows:
        dists.append(detail["configs"][r["config"]]["pred_distribution"])

    # colours that appear somewhere, red first
    levels = []
    for u in reversed(URGENCY_ORDER):
        for d in dists:
            if d.get(u):
                levels.append(u)
                break
    for d in dists:
        if d.get("INVALID"):
            levels.append("INVALID")
            break

    bottom = [0.0] * len(cols)
    for u in levels:
        share = []
        for d in dists:
            share.append(d.get(u, 0) / sum(d.values()))
        ax.bar(cols, share, bottom=bottom, color=COLOURS.get(u, "#777"), label=u, edgecolor="white")
        # new list each time (don't change the one already given to bar)
        new_bottom = []
        for i in range(len(bottom)):
            new_bottom.append(bottom[i] + share[i])
        bottom = new_bottom
    ax.set_ylabel("Share of cases")
    ax.set_title("Predicted urgency distribution (performance set, primary runs)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_urgency_distribution.png", dpi=150)
    plt.close(fig)

    # fig 3: variant set, predicted high-acuity rate by group
    vrows = []
    for r in rows:
        if "variants_dpr" in detail["configs"][r["config"]]:
            vrows.append(r)
    if vrows:
        fig, ax = plt.subplots(figsize=(7, 4))
        width = 0.8 / len(GROUPS)
        for i in range(len(GROUPS)):
            g = GROUPS[i]
            vals = []
            xs = []
            for x in range(len(vrows)):
                vals.append(detail["configs"][vrows[x]["config"]]["variants_dpr"]["high_rate_by_group"][g])
                xs.append(x + i * width)
            ax.bar(xs, vals, width, label=g.replace("_", " "))
        ticks = []
        tick_labels = []
        for x in range(len(vrows)):
            ticks.append(x + width * (len(GROUPS) - 1) / 2)
            tick_labels.append(vrows[x]["label"])
        ax.set_xticks(ticks, tick_labels)
        ax.set_ylim(0, 1)
        ax.set_ylabel("P(predicted red or orange)")
        ax.set_title("Variant set: predicted high-acuity rate by group (primary runs)")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig3_variant_high_acuity_by_group.png", dpi=150)
        plt.close(fig)


def main():
    rows, detail = summarise()
    if not rows:
        raise SystemExit("No prediction files found in results/")
    detail["model_config"] = MODEL_NAME
    write_csv(rows, RESULTS_PATH / "results_summary.csv")
    write_json(RESULTS_PATH / "summary.json", detail)
    figures(rows, detail)

    print(f"Wrote results_summary.csv ({len(rows)} configs), summary.json and figures/")
    for r in rows:
        print(f"  {r['label']}: urgency {r['urgency_exact_acc']:.3f} "
              f"[{r['urgency_exact_acc_min']:.3f}-{r['urgency_exact_acc_max']:.3f}], "
              f"specialty {r['specialty_acc']:.3f}, runs {r['n_runs']}")
    maj = detail["majority_reference"]
    print(f"  majority reference ({maj['label']}): {maj['acc']:.3f}")


if __name__ == "__main__":
    main()
