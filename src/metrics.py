# metrics for urgency and specialty (protocol 6.1, 6.2)
# no API calls here
# ERROR rows are skipped, INVALID rows stay in and count as wrong

from config import ERROR, INVALID, SPECIALTIES, SPECIALTY_PARENT, URGENCY_ORDER, CASES_PATH, FAIRNESS_PATH
from common import normalize_specialty, normalize_urgency, row_key, read_csv

# urgency -> number, blue=0 ... red=4
levels = {}
for i in range(len(URGENCY_ORDER)):
    levels[URGENCY_ORDER[i]] = i

# specialties in our list, lowercase
listed_specialties = set()
for s in SPECIALTIES:
    listed_specialties.add(s.lower())


def urgency_outcome(pred, truth):
    # returns correct / over / under / invalid
    if pred not in levels:
        return "invalid"
    if pred == truth:
        return "correct"
    if levels[pred] > levels[truth]:
        return "over"
    else:
        return "under"


def specialty_labels(case):
    # first specialty + all specialties (saved as A|B|C)
    first = normalize_specialty(case["ground_truth_specialty"])

    all_text = case.get("ground_truth_specialty_all")
    if not all_text:
        all_text = case["ground_truth_specialty"]

    listed = set()
    for s in all_text.split("|"):
        listed.add(normalize_specialty(s))
    return first, listed


def map_to_parent(labels):
    # sub-specialty -> parent, for sensitivity analysis (6.2.1)
    result = set()
    for label in labels:
        parents = SPECIALTY_PARENT.get(label, (label,))
        for p in parents:
            result.add(p)
    return result


def load_truth(preds, cases_file=None):
    # variants file if preds have variant_id, otherwise main.csv
    if cases_file:
        path = cases_file
    else:
        has_variant = False
        for p in preds:
            if p.get("variant_id"):
                has_variant = True
                break
        if has_variant:
            path = FAIRNESS_PATH
        else:
            path = CASES_PATH

    truth = {}
    for c in read_csv(path):
        truth[row_key(c)] = c
    return truth


def paired_rows(cases, preds):
    # (prediction, case) for preds that have a case
    pairs = []
    for p in preds:
        key = row_key(p)
        if key in cases:
            pairs.append((p, cases[key]))
    return pairs


def correct_by_case(cases, preds, metric, mapped=False):
    # key -> True/False, ERROR rows skipped
    # metric = "urgency" or "specialty"
    # mapped only for specialty
    result = {}
    for p, c in paired_rows(cases, preds):
        if p["pred_urgency"] == ERROR:
            continue

        key = row_key(p)
        if metric == "urgency":
            result[key] = normalize_urgency(p["pred_urgency"]) == c["ground_truth_urgency"]
        else:
            first = specialty_labels(c)[0]
            pred = normalize_specialty(p["pred_specialty"])
            if mapped:
                result[key] = pred in map_to_parent([first])
            else:
                result[key] = pred == first
    return result


def evaluate(cases, preds):
    # all metrics for one prediction file
    pairs = paired_rows(cases, preds)

    # remove ERROR rows
    answered = []
    for p, c in pairs:
        if p["pred_urgency"] != ERROR:
            answered.append((p, c))
    n = len(answered)

    def rate(k):
        if n == 0:
            return None
        return k / n

    # urgency counts
    correct = 0
    over = 0
    under = 0
    invalid = 0
    adjacent = 0
    per_class = {}  # true class -> [hits, total]

    for p, c in answered:
        pred = normalize_urgency(p["pred_urgency"])
        truth = c["ground_truth_urgency"]
        outcome = urgency_outcome(pred, truth)

        if outcome == "correct":
            correct += 1
        elif outcome == "over":
            over += 1
        elif outcome == "under":
            under += 1
        else:
            invalid += 1

        # within 1 level
        if outcome != "invalid" and abs(levels[pred] - levels[truth]) <= 1:
            adjacent += 1

        if truth not in per_class:
            per_class[truth] = [0, 0]
        if outcome == "correct":
            per_class[truth][0] += 1
        per_class[truth][1] += 1

    # recall per true class
    recalls = {}
    for u in per_class:
        recalls[u] = per_class[u][0] / per_class[u][1]

    if len(recalls) > 0:
        balanced = sum(recalls.values()) / len(recalls)
    else:
        balanced = None

    n_by_class = {}
    for u in per_class:
        n_by_class[u] = per_class[u][1]

    if "red" in per_class:
        n_red = per_class["red"][1]
    else:
        n_red = 0

    # specialty counts
    strict = 0
    any_listed = 0
    strict_mapped = 0
    any_mapped = 0
    reachable = 0  # first label is in our list
    invalid_spec = 0

    for p, c in answered:
        pred = normalize_specialty(p["pred_specialty"])
        first, listed = specialty_labels(c)
        if pred == first:
            strict += 1
        if pred in listed:
            any_listed += 1
        if pred in map_to_parent([first]):
            strict_mapped += 1
        if pred in map_to_parent(listed):
            any_mapped += 1
        if first in listed_specialties:
            reachable += 1
        if p["pred_specialty"] == INVALID:
            invalid_spec += 1

    return {
        "n_cases": len(pairs),
        "n_error": len(pairs) - n,
        "n_answered": n,
        "n_invalid_urgency": invalid,
        "urgency_exact_acc": rate(correct),
        "urgency_adjacent_acc": rate(adjacent),
        "over_triage_rate": rate(over),
        "under_triage_rate": rate(under),
        "urgency_balanced_acc": balanced,
        "urgency_recall_by_class": recalls,
        "n_by_true_class": n_by_class,
        "red_recall": recalls.get("red"),
        "n_red": n_red,
        "specialty_acc": rate(strict),
        "specialty_acc_any_listed": rate(any_listed),
        "specialty_ceiling": rate(reachable),
        "specialty_acc_mapped": rate(strict_mapped),
        "specialty_acc_any_listed_mapped": rate(any_mapped),
        "n_invalid_specialty": invalid_spec,
    }
