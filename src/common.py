# shared code for the inference scripts (C1-C4):
# model client, output parsing, label normalisation, patient case block, resumable runner
# inference only: no metrics here, and metric scripts never call the API

import ast
import csv
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from config import (ENV_FILE, ERROR, INVALID, MAX_RETRIES, MAX_TOKENS, MODEL_NAME, PROMPTS_PATH,
                    REQUEST_TIMEOUT_S, ROOT, SPECIALTIES, SPECIALTY_SYNONYMS, SYSTEM_PROMPT,
                    TEMPERATURE, URGENCY_MAP, URGENCY_ORDER, URGENCY_SYNONYMS)
from demographics import mask_text

csv.field_size_limit(10_000_000)  # HPI text can be longer than the csv default


# ---- model client

@dataclass
class ModelReply:
    text: str        # raw message text, saved as it is in every prediction row
    model: str       # model id the provider sends back
    json_mode: bool  # was response_format=json_object sent


class EmptyReply(RuntimeError):
    # API answered but the message was empty -> treat as temporary error
    pass


def is_transient(err):
    # errors worth retrying (rate limit, timeout, connection, server 5xx, empty reply)
    import openai
    if isinstance(err, (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError,
                        openai.InternalServerError, EmptyReply)):
        return True
    return isinstance(err, openai.APIStatusError) and err.status_code >= 500


class LLM:
    # OpenAI-compatible chat client (OpenRouter), retries with backoff, JSON mode fallback
    # asks for JSON mode first. if the provider rejects response_format, JSON mode is
    # switched off for the rest of the run (and it says so). each row saves which mode was used

    def __init__(self):
        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv(ENV_FILE)
        key = os.getenv("OPENAI_API_KEY", "").strip()
        base_url = os.getenv("OPENROUTER_BASE_URL", "").strip()
        if not key or not base_url:
            raise RuntimeError(f"Set OPENAI_API_KEY and OPENROUTER_BASE_URL in {ENV_FILE}")
        # max_retries=0: retries done in ask(), so backoff + logging are in one place
        self.client = OpenAI(api_key=key, base_url=base_url, timeout=REQUEST_TIMEOUT_S, max_retries=0)
        self.json_mode = True

    def ask(self, prompt):
        import openai

        delay = 2.0
        for attempt in range(1, MAX_RETRIES + 1):
            kwargs = dict(model=MODEL_NAME, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                          messages=[{"role": "system", "content": SYSTEM_PROMPT},
                                    {"role": "user", "content": prompt}])
            if self.json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            try:
                resp = self.client.chat.completions.create(**kwargs)
                if resp.choices:
                    text = (resp.choices[0].message.content or "").strip()
                else:
                    text = ""
                if not text:
                    raise EmptyReply("model returned no content")
                model = getattr(resp, "model", "") or ""
                return ModelReply(text=text, model=model, json_mode=self.json_mode)

            except openai.BadRequestError as e:
                # provider doesn't support JSON mode -> try again without it
                if self.json_mode and "response_format" in str(e).lower():
                    tqdm.write("  Provider rejected JSON mode; continuing with prompt-only JSON.")
                    self.json_mode = False
                    continue
                raise

            except Exception as e:
                if attempt == MAX_RETRIES or not is_transient(e):
                    raise
                tqdm.write(f"  Transient error (attempt {attempt}/{MAX_RETRIES}): {str(e)[:120]}; "
                           f"retrying in {delay:.0f}s")
                time.sleep(delay)
                delay = min(delay * 2, 60.0)

        raise RuntimeError(f"Model call failed after {MAX_RETRIES} attempts")


# ---- output parsing

THINK = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL | re.IGNORECASE)
FENCE = re.compile(r"```(?:json)?", re.IGNORECASE)


def extract_json_object(text):
    # first JSON object in the text, after removing <think> blocks and ``` fences, or None
    text = str(text or "")
    text = THINK.sub("", text)
    text = FENCE.sub("", text)
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def parse_model_output(text):
    # returns (urgency, specialty, rationale)
    # urgency normalised, no JSON -> INVALID
    # specialty kept like the model wrote it (so the raw answer can be checked),
    # metric code normalises it later with normalize_specialty()
    obj = extract_json_object(text)
    if obj is None:
        return INVALID, INVALID, "Could not parse a JSON object from the model output"
    urgency = normalize_urgency(obj.get("urgency"))
    specialty = str(obj.get("specialty") or "").strip()
    if not specialty:
        specialty = INVALID
    rationale = str(obj.get("rationale") or "").strip()
    return urgency, specialty, rationale


# ---- label normalisation

COLOUR = re.compile(r"\b(" + "|".join(URGENCY_ORDER) + r")\b")
LEVEL = re.compile(r"\b(?:esi|ctas|level|category|acuity|triage)\s*(?:level\s*)?([1-5])\b")


def normalize_urgency(value):
    # raw urgency answer -> colour, or INVALID. ERROR stays ERROR
    # order: one colour word, a bare number 1-5, one level with prefix ("ESI 2", "level 3"),
    # then URGENCY_SYNONYMS in order. two different levels -> INVALID
    if value is None:
        value = ""
    s = str(value).strip()
    if s == ERROR:
        return ERROR
    s = s.lower()
    if not s:
        return INVALID

    colours = set(COLOUR.findall(s))
    if colours:
        if len(colours) == 1:
            return colours.pop()
        return INVALID

    if re.fullmatch(r"[1-5]", s):
        return URGENCY_MAP[int(s)]

    levels = set(LEVEL.findall(s))
    if levels:
        numbers = set(re.findall(r"\b[1-5]\b", s))  # "ESI 2 or 3" has two levels
        if len(levels) == 1 and len(numbers) == 1:
            return URGENCY_MAP[int(levels.pop())]
        return INVALID

    for pattern, colour in URGENCY_SYNONYMS:
        if re.search(pattern, s):
            return colour
    return INVALID


# lowercase name / synonym -> canonical lowercase specialty
SPECIALTY_LOOKUP = {}
for name in SPECIALTIES:
    SPECIALTY_LOOKUP[name.lower()] = name.lower()
for canon, aliases in SPECIALTY_SYNONYMS.items():
    for alias in aliases:
        SPECIALTY_LOOKUP[alias] = canon


def clean_specialty(s):
    s = re.sub(r"\(.*?\)", " ", s)  # remove (...) parts
    return re.sub(r"\s+", " ", s).strip(" '\".").lower()


def normalize_specialty(value):
    # lowercase canonical specialty. ERROR stays ERROR, empty -> INVALID
    # list answer ("['Cardiology', 'Neurology']") or "Cardiology or Neurology" -> first one
    # unknown names come back cleaned, no guessing
    if value is None:
        value = ""
    s = str(value).strip()
    if s == ERROR:
        return ERROR

    if s.startswith("[") and s.endswith("]"):
        try:
            items = ast.literal_eval(s)
            if isinstance(items, list) and items:
                s = str(items[0])
            else:
                s = s.strip("[]")
        except (ValueError, SyntaxError):
            s = s.strip("[]")

    # whole answer first, e.g. "ear, nose and throat"
    whole = clean_specialty(s)
    if whole in SPECIALTY_LOOKUP:
        return SPECIALTY_LOOKUP[whole]

    # else only the first part
    first_part = re.split(r";|\||,|\s/\s|\bor\b", s, maxsplit=1, flags=re.IGNORECASE)[0]
    first = clean_specialty(first_part)
    if not first:
        return INVALID
    return SPECIALTY_LOOKUP.get(first, first)


# ---- cases and prompts

def build_case_block(case, mask=False):
    # patient case text for the prompt (and the retrieval query for C2/C3)
    # same layout as the September 2026 runs (protocol 3): fields stripped at the ends only,
    # line breaks in the HPI kept, order Symptoms, Vitals, Age, Gender
    # mask=True (C3): no Age/Gender lines, demographic cues in HPI neutralised
    symptoms = str(case.get("symptoms") or "").strip()
    vitals = str(case.get("vitals") or "").strip()
    if mask:
        lines = ["Patient case:", f"- Symptoms: {mask_text(symptoms)}", f"- Vitals: {vitals}"]
        return "\n".join(lines)
    age = str(case.get("age") or "").strip()
    gender = str(case.get("gender") or "").strip()
    lines = ["Patient case:", f"- Symptoms: {symptoms}", f"- Vitals: {vitals}",
             f"- Age: {age}", f"- Gender: {gender}"]
    return "\n".join(lines)


def load_prompt(name):
    return (PROMPTS_PATH / name).read_text(encoding="utf-8").strip()


def resolve(path):
    # command line paths are relative to the project root
    p = Path(path)
    if p.is_absolute():
        return p
    return ROOT / p


def read_csv(path):
    with open(resolve(path), encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def content_sha256(path):
    # sha256 of the csv contents, not the file format (protocol 5.1)
    # header + every row as lists of values, saved as compact UTF-8 JSON, then hashed
    # CRLF inside a value -> LF, so Windows/Linux or pandas/csv module give the same hash
    rows = []
    with open(resolve(path), encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            values = []
            for v in row:
                values.append(v.replace("\r\n", "\n"))
            rows.append(values)
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path, obj):
    p = resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def row_key(row):
    # variant_id on the fairness set, case_id on the performance set
    return str(row.get("variant_id") or row.get("case_id"))


# ---- resumable runner

PRED_FIELDS = ["case_id", "variant_id", "input_age", "input_gender", "pred_urgency", "pred_specialty",
               "pred_rationale", "raw_model_output", "model", "response_model", "json_mode",
               "timestamp_utc"]


def run_predictions(cases, output, predict_fn, desc, resume=True, sleep=0.0):
    # calls predict_fn(case) -> ModelReply for each case, one csv row per case
    # every row flushed right away, a crash loses at most the call that was running
    # resume: finished rows kept, only missing + ERROR rows run again
    # ERROR (no usable reply) and INVALID (reply but no label) are never mixed
    output = resolve(output)

    keys = []
    for c in cases:
        keys.append(row_key(c))
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate case keys in the input; every case_id/variant_id must be unique.")

    # rows from an earlier run
    kept = []
    n_retry = 0
    if resume and output.exists():
        existing = read_csv(output)
        if existing and list(existing[0].keys()) != PRED_FIELDS:
            raise SystemExit(f"{output} has different columns than this code writes; "
                             "move it aside or use --no-resume.")
        for r in existing:
            if r["pred_urgency"] != ERROR:
                kept.append(r)
        n_retry = len(existing) - len(kept)

    done = set()
    for r in kept:
        done.add(row_key(r))
    todo = []
    for c in cases:
        if row_key(c) not in done:
            todo.append(c)
    if done or n_retry:
        print(f"Resuming {output.name}: {len(done)} rows kept, {n_retry} ERROR rows to retry, "
              f"{len(todo)} to run.")

    # write the kept rows again first (drops ERROR rows), replace is atomic
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PRED_FIELDS)
        w.writeheader()
        w.writerows(kept)
    os.replace(tmp, output)

    counts = {"ok": 0, INVALID: 0, ERROR: 0}
    with open(output, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PRED_FIELDS)
        for case in tqdm(todo, desc=desc, unit="case"):
            row = {"case_id": case.get("case_id", ""), "variant_id": case.get("variant_id", ""),
                   "input_age": case.get("age", ""), "input_gender": case.get("gender", ""),
                   "model": MODEL_NAME}
            try:
                reply = predict_fn(case)
                urgency, specialty, rationale = parse_model_output(reply.text)
                row["pred_urgency"] = urgency
                row["pred_specialty"] = specialty
                row["pred_rationale"] = rationale
                row["raw_model_output"] = reply.text
                row["response_model"] = reply.model
                row["json_mode"] = reply.json_mode
                if urgency == INVALID:
                    counts[INVALID] += 1
                else:
                    counts["ok"] += 1
            except Exception as e:  # noqa: BLE001 - any failure here = infrastructure ERROR
                tqdm.write(f"ERROR on {row_key(case)}: {e}")
                row["pred_urgency"] = ERROR
                row["pred_specialty"] = ERROR
                row["pred_rationale"] = f"{type(e).__name__}: {e}"
                row["raw_model_output"] = ""
                row["response_model"] = ""
                row["json_mode"] = ""
                counts[ERROR] += 1
            row["timestamp_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            w.writerow(row)
            f.flush()
            if sleep:
                time.sleep(sleep)

    msg = (f"{output.name}: {counts['ok']} ok, {counts[INVALID]} INVALID, {counts[ERROR]} ERROR this run "
           f"({len(kept)} kept from before).")
    if counts[ERROR]:
        msg += " Re-run the same command to retry ERROR rows."
    print(msg)
    return counts
