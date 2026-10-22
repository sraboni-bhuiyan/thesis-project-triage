# build the performance set data/cases/main.csv from MIMIC-IV-Ext CDS (protocol 2.2)
# all clinician-approved specialty referrals, joined on stay_id to triage level (ESI 1-5) -> colour
# race not kept (scope decision, protocol 2.2)
# csv module, sorted by case_id, "\n" line endings -> rebuild gives the exact same file
# to compare with other builds use common.content_sha256 (ignores file format)
#
# usage:
#   python src/extract_mimic_cases.py

import ast
import csv
import hashlib
from collections import Counter

from common import content_sha256
from config import CASES_PATH, PHYSIONET_BASE, URGENCY_MAP

csv.field_size_limit(10_000_000)

APPROVED = PHYSIONET_BASE / "specialty_referral_clinician_approved.csv"
TRIAGE = PHYSIONET_BASE / "triage_level.csv"
SPECIALTY_COL = "specialty clinician approved"
COLUMNS = ["case_id", "symptoms", "vitals", "age", "gender",
           "ground_truth_urgency", "ground_truth_specialty", "ground_truth_specialty_all"]


def parse_patient_info(info):
    # 'Gender: Male, Race: WHITE, Age: 57' -> {'gender': 'Male', 'race': 'WHITE', 'age': '57'}
    out = {}
    for part in (info or "").split(","):
        if ":" in part:
            key, value = part.split(":", 1)
            out[key.strip().lower()] = value.strip()
    return out


def parse_specialties(raw):
    # "['Gastroenterology', 'Gastroenterology', 'Cardiology']" -> ['Gastroenterology', 'Cardiology']
    try:
        items = ast.literal_eval(raw or "[]")
    except (ValueError, SyntaxError):
        items = (raw or "").strip("[]").split(",")
    if not isinstance(items, (list, tuple)):
        items = [items]

    # no duplicates, keep order
    out = []
    for item in items:
        item = str(item).strip().strip("'\"")
        if item and item not in out:
            out.append(item)
    return out


def load_triage():
    # stay_id -> ESI level (int)
    # conflicting duplicates = error, don't just pick one
    levels = {}
    with open(TRIAGE, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            sid = r["stay_id"].strip()
            raw = (r["triage"] or "").strip()
            if not raw:
                continue
            level = int(float(raw))
            if sid in levels and levels[sid] != level:
                raise SystemExit(f"stay_id {sid} has conflicting triage levels {levels[sid]} and {level}")
            levels[sid] = level
    return levels


def case_number(row):
    return int(row["case_id"])


def main():
    print(f"Reading {APPROVED.name} and {TRIAGE.name} from {PHYSIONET_BASE.name}/")
    triage = load_triage()
    with open(APPROVED, encoding="utf-8", newline="") as f:
        approved = list(csv.DictReader(f))

    rows = []
    seen = set()
    dropped = Counter()
    for r in approved:
        sid = r["stay_id"].strip()
        if sid in seen:
            dropped["duplicate stay_id"] += 1
            continue
        seen.add(sid)

        info = parse_patient_info(r["patient_info"])
        specialties = parse_specialties(r[SPECIALTY_COL])
        level = triage.get(sid)

        if not (r["HPI"] or "").strip():
            dropped["no HPI"] += 1
            continue
        if level not in URGENCY_MAP:
            dropped["no triage level"] += 1
            continue
        if not specialties:
            dropped["no specialty"] += 1
            continue

        rows.append({
            "case_id": sid,
            # not stripped: HPI kept exactly like the source (same as September file)
            "symptoms": r["HPI"],
            "vitals": (r["initial_vitals"] or "").strip(),
            "age": info.get("age", ""),
            "gender": info.get("gender", ""),
            "ground_truth_urgency": URGENCY_MAP[level],
            "ground_truth_specialty": specialties[0],
            "ground_truth_specialty_all": "|".join(specialties),
        })

    rows.sort(key=case_number)
    CASES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CASES_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    # print a summary
    dist = Counter()
    genders = Counter()
    for r in rows:
        dist[r["ground_truth_urgency"]] += 1
        genders[r["gender"]] += 1
    urgency = {}
    for c in URGENCY_MAP.values():
        if c not in urgency:
            urgency[c] = dist.get(c, 0)

    print(f"{len(approved)} clinician-approved rows -> {len(rows)} cases written to {CASES_PATH}")
    if dropped:
        print("Dropped:", dict(dropped))
    else:
        print("Dropped:", "none")
    print("Urgency distribution:", urgency)
    print("Gender:", dict(genders))
    file_hash = hashlib.sha256(CASES_PATH.read_bytes()).hexdigest()[:16]
    print(f"file sha256 {file_hash}   content sha256 {content_sha256(CASES_PATH)[:16]}")


if __name__ == "__main__":
    main()
