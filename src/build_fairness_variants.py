# build the counterfactual fairness set data/cases/fairness_variants.csv (protocol 2.3)
# 70 base cases: random sample (seed 42) from main.csv rows with symptoms
# each base case -> 4 variants (young 30 | old 70) x (male | female)
# age + gender fields set, HPI rewritten with demographics.swap_text
# vitals and ground truth copied as they are
#
# also writes data/cases/fairness_variants_audit.json (gitignored, has case ids):
#   - leak check: no female cue left in a male variant and the other way (must be 0)
#   - sex-specific base cases (gender swap not clinically valid), counted for the limitation
#   - base cases with only negated sex-specific words ("no vaginal bleeding"): not counted,
#     but listed as text leak because the swapped text still has the phrase
#
# usage:
#   python src/build_fairness_variants.py

import csv
import hashlib
import json
import random
import re

from common import read_csv
from config import (AGE_GROUPS, CASES_PATH, FAIRNESS_PATH, GENDERS, N_BASE_CASES, OLD_VARIANT_AGE, SEED,
                    YOUNG_VARIANT_AGE)
from demographics import sex_specific_mentions, swap_text

AUDIT_PATH = FAIRNESS_PATH.with_name("fairness_variants_audit.json")
COLUMNS = ["case_id", "variant_id", "age_group", "symptoms", "vitals", "age", "gender",
           "ground_truth_urgency", "ground_truth_specialty", "ground_truth_specialty_all"]
AGES = {"young": YOUNG_VARIANT_AGE, "old": OLD_VARIANT_AGE}

# runbook check for male variants + same check the other way for female variants
# titles case sensitive so "MS" (multiple sclerosis) isn't read as "Ms."
LEAKS = {
    "male": re.compile(r"(?i:\b(she|her|hers|herself|woman|women|female|lady)\b)|\bMs\.|\bMrs\.|\bMiss\b"),
    "female": re.compile(r"(?i:\b(he|him|his|himself|man|men|male|gentleman)\b)|\bMr\."),
}


def sample_base_cases(rows):
    # only rows with symptoms
    eligible = []
    for r in rows:
        if (r.get("symptoms") or "").strip():
            eligible.append(r)
    return random.Random(SEED).sample(eligible, N_BASE_CASES)


def make_variants(base):
    out = []
    for age_group in AGE_GROUPS:
        for gender in GENDERS:
            out.append({
                "case_id": base["case_id"],
                "variant_id": f"{base['case_id']}_{age_group}_{gender}",
                "age_group": age_group,
                "symptoms": swap_text(base["symptoms"], gender, AGES[age_group]),
                "vitals": base["vitals"],
                "age": str(AGES[age_group]),
                "gender": gender.capitalize(),
                "ground_truth_urgency": base["ground_truth_urgency"],
                "ground_truth_specialty": base["ground_truth_specialty"],
                "ground_truth_specialty_all": base["ground_truth_specialty_all"],
            })
    return out


def main():
    base_cases = sample_base_cases(read_csv(CASES_PATH))
    variants = []
    for b in base_cases:
        variants += make_variants(b)

    FAIRNESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FAIRNESS_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(variants)

    # leak check
    leaks = {}
    for g in GENDERS:
        leaks[g] = []
        for v in variants:
            if v["gender"].lower() == g and LEAKS[g].search(v["symptoms"]):
                leaks[g].append(v["variant_id"])

    # sex-specific words in the base cases
    sex_specific = {}
    text_leaks = {}
    for b in base_cases:
        m = sex_specific_mentions(b["symptoms"])
        if m["affirmed"]:
            sex_specific[b["case_id"]] = m["affirmed"]
        elif m["negated"]:
            text_leaks[b["case_id"]] = m["negated"]

    high = 0
    base_ids = []
    for b in base_cases:
        base_ids.append(b["case_id"])
        if b["ground_truth_urgency"] in ("red", "orange"):
            high += 1

    audit = {
        "n_base_cases": len(base_cases),
        "n_variants": len(variants),
        "seed": SEED,
        "base_case_ids": base_ids,
        "n_base_high_acuity": high,
        "leaks_in_male_variants": leaks["male"],
        "leaks_in_female_variants": leaks["female"],
        # counted for the limitation: swapped versions not clinically valid
        "n_sex_specific_base_cases": len(sex_specific),
        "sex_specific_base_cases": sex_specific,
        # not counted: only negated words, same clinical picture, but swapped text keeps a
        # sex-specific phrase (text leak). breast not counted as sex-specific at all
        "n_sex_specific_text_leak_base_cases": len(text_leaks),
        "sex_specific_text_leak_base_cases": text_leaks,
    }
    AUDIT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print(f"{len(base_cases)} base cases (seed {SEED}) -> {len(variants)} variants -> {FAIRNESS_PATH}")
    print(f"High-acuity base cases: {high} of {len(base_cases)}")
    print(f"Leak check: {len(leaks['male'])} male variants with female cues, "
          f"{len(leaks['female'])} female variants with male cues")
    print(f"Sex-specific base cases (counted as a limitation): {len(sex_specific)}; "
          f"negated-only mentions (text leak, not counted): {len(text_leaks)}")
    file_hash = hashlib.sha256(FAIRNESS_PATH.read_bytes()).hexdigest()[:16]
    print(f"sha256 {file_hash}   audit -> {AUDIT_PATH.name}")


if __name__ == "__main__":
    main()
