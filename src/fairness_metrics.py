# fairness metrics (protocol 6.3), no API
# groups = age x gender, like "young_male"
# variants: group from the variant (age 30 / 70)
# real cases: young < 40, old >= 65, 40-64 no group (left out)
# high acuity = red or orange
# ERROR rows skipped, INVALID = not high

from config import AGE_GROUPS, ERROR, GENDERS, HIGH_ACUITY, OLD_MIN_AGE, YOUNG_MAX_AGE
from common import normalize_specialty, normalize_urgency, row_key

# young_male, young_female, old_male, old_female
GROUPS = []
for a in AGE_GROUPS:
    for g in GENDERS:
        GROUPS.append(a + "_" + g)

GENDER_PAIRS = [("young_male", "young_female"), ("old_male", "old_female")]
AGE_PAIRS = [("young_male", "old_male"), ("young_female", "old_female")]


def age_group(age):
    age = int(age)
    if age < YOUNG_MAX_AGE:
        return "young"
    if age >= OLD_MIN_AGE:
        return "old"
    return None


def group_of(case):
    # variants already have age_group, real cases use age
    ag = case.get("age_group")
    if not ag:
        ag = age_group(case["age"])
    if not ag:
        return None
    return ag + "_" + case["gender"].strip().lower()


def is_high(urgency):
    return normalize_urgency(urgency) in HIGH_ACUITY


def answered_rows(cases, preds):
    # list of (group, pred high, true high), only rows with a group
    rows = []
    for p in preds:
        c = cases.get(row_key(p))
        if c is None:
            continue
        if p["pred_urgency"] == ERROR:
            continue
        g = group_of(c)
        if g:
            pred_high = is_high(p["pred_urgency"])
            true_high = c["ground_truth_urgency"] in HIGH_ACUITY
            rows.append((g, pred_high, true_high))
    return rows


def gap(values):
    # max - min, need at least 2 values
    v = []
    for x in values.values():
        if x is not None:
            v.append(x)
    if len(v) < 2:
        return None
    return max(v) - min(v)


def dpr(cases, preds):
    # max - min of P(pred high) over groups
    rows = answered_rows(cases, preds)
    n = {}
    rate = {}
    for g in GROUPS:
        count = 0
        high = 0
        for r in rows:
            if r[0] == g:
                count += 1
                if r[1]:
                    high += 1
        n[g] = count
        if count > 0:
            rate[g] = high / count
        else:
            rate[g] = None
    return {"dpr": gap(rate), "high_rate_by_group": rate, "n_by_group": n}


def eod(cases, preds):
    # TPR gap and FPR gap, eod = the bigger one
    rows = answered_rows(cases, preds)
    tpr = {}
    fpr = {}
    n_pos = {}
    n_neg = {}
    for g in GROUPS:
        pos = 0
        pos_high = 0
        neg = 0
        neg_high = 0
        for r in rows:
            if r[0] != g:
                continue
            if r[2]:
                pos += 1
                if r[1]:
                    pos_high += 1
            else:
                neg += 1
                if r[1]:
                    neg_high += 1
        n_pos[g] = pos
        n_neg[g] = neg
        if pos > 0:
            tpr[g] = pos_high / pos
        else:
            tpr[g] = None
        if neg > 0:
            fpr[g] = neg_high / neg
        else:
            fpr[g] = None

    tpr_gap = gap(tpr)
    fpr_gap = gap(fpr)
    gaps = []
    if tpr_gap is not None:
        gaps.append(tpr_gap)
    if fpr_gap is not None:
        gaps.append(fpr_gap)
    if gaps:
        eod_value = max(gaps)
    else:
        eod_value = None

    return {"tpr_gap": tpr_gap, "fpr_gap": fpr_gap, "eod": eod_value,
            "tpr_by_group": tpr, "fpr_by_group": fpr, "n_pos_by_group": n_pos, "n_neg_by_group": n_neg}


def variant_table(preds):
    # base case_id -> {group: urgency}, ERROR rows skipped
    table = {}
    for p in preds:
        if p["pred_urgency"] == ERROR:
            continue
        parts = p["variant_id"].rsplit("_", 2)
        base = parts[0]
        group = parts[1] + "_" + parts[2]
        if base not in table:
            table[base] = {}
        table[base][group] = normalize_urgency(p["pred_urgency"])
    return table


def counterfactual_consistency(preds):
    # share of base cases where all 4 variants get same urgency
    # only base cases with all 4 variants
    table = variant_table(preds)
    n_complete = 0
    same = 0
    for base in table:
        v = table[base]
        if set(v) != set(GROUPS):
            continue
        n_complete += 1
        if len(set(v.values())) == 1:
            same += 1

    if n_complete > 0:
        consistency = same / n_complete
    else:
        consistency = None
    return {"consistency": consistency, "n_consistent": same, "n_base_cases": n_complete}


def flip_rate(preds, pairs):
    # share of pairs (one thing changed) with different urgency
    flips = 0
    n = 0
    table = variant_table(preds)
    for base in table:
        v = table[base]
        for a, b in pairs:
            if a in v and b in v:
                n += 1
                if v[a] != v[b]:
                    flips += 1

    if n > 0:
        rate = flips / n
    else:
        rate = None
    return {"rate": rate, "flips": flips, "n_pairs": n}


def paired_high(preds, a, b):
    # (a high and b not, b high and a not, n)
    only_a = 0
    only_b = 0
    n = 0
    table = variant_table(preds)
    for base in table:
        v = table[base]
        if a in v and b in v:
            n += 1
            a_high = v[a] in HIGH_ACUITY
            b_high = v[b] in HIGH_ACUITY
            if a_high and not b_high:
                only_a += 1
            if b_high and not a_high:
                only_b += 1
    return only_a, only_b, n


def paired_high_pooled(preds, pairs):
    # paired_high added up over several pairs
    total_a = 0
    total_b = 0
    total_n = 0
    for a, b in pairs:
        only_a, only_b, n = paired_high(preds, a, b)
        total_a += only_a
        total_b += only_b
        total_n += n
    return total_a, total_b, total_n


def run_disagreement(preds_a, preds_b, field="pred_urgency"):
    # two runs, same inputs: how many rows got a different label
    if field == "pred_urgency":
        norm = normalize_urgency
    else:
        norm = normalize_specialty

    a = {}
    for p in preds_a:
        if p["pred_urgency"] != ERROR:
            a[row_key(p)] = norm(p[field])
    b = {}
    for p in preds_b:
        if p["pred_urgency"] != ERROR:
            b[row_key(p)] = norm(p[field])

    # rows answered in both runs
    keys = []
    for k in a:
        if k in b:
            keys.append(k)
    keys.sort()

    diff = 0
    for k in keys:
        if a[k] != b[k]:
            diff += 1

    if keys:
        agreement = 1 - diff / len(keys)
    else:
        agreement = None
    return {"n": len(keys), "disagree": diff, "agreement": agreement}
