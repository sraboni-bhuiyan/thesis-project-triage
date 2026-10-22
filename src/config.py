# all settings in one place: paths, model settings, labels, demographic groups
# other scripts import from here, never define these again
# "labels" part is tested in tests/test_metrics.py -> run the tests after changing it

from pathlib import Path

# ---- paths
ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = ROOT / "src"
PROMPTS_PATH = SRC_PATH / "prompts"
DATA_PATH = ROOT / "data"
GUIDELINES_PATH = DATA_PATH / "guidelines_raw"
CORPUS_FILE = DATA_PATH / "guidelines_clean" / "corpus.jsonl"
FIXED_CONTEXT_FILE = DATA_PATH / "guidelines_clean" / "fixed_context.json"
INDEX_DIR = DATA_PATH / "indexes"
FAISS_INDEX = INDEX_DIR / "guidelines.index"
ID_MAP_FILE = INDEX_DIR / "id_map.json"
CASES_PATH = DATA_PATH / "cases" / "main.csv"
FAIRNESS_PATH = DATA_PATH / "cases" / "fairness_variants.csv"
RESULTS_PATH = ROOT / "results"
ENV_FILE = ROOT / ".env"

# MIMIC-IV-Ext CDS v1.0.2, local only (PhysioNet DUA), never commit
PHYSIONET_BASE = ROOT / "mimic-iv-ext-clinical-decision-support-for-referral-triage-and-diagnosis-1.0.2"

# ---- model and inference (protocol 3)
# fixed on purpose, no env variable override (so the model can't change in the middle of the study)
MODEL_NAME = "nvidia/nemotron-3-ultra-550b-a55b"
TEMPERATURE = 0.0
MAX_TOKENS = 1024
REQUEST_TIMEOUT_S = 180
MAX_RETRIES = 6
SYSTEM_PROMPT = "You are a careful clinical triage assistant. You always answer with a single JSON object."

# ---- retrieval (protocol 2.4)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3
CHUNK_WORDS = 250
OVERLAP_WORDS = 50
MIN_TAIL_WORDS = 100       # last chunk shorter than this is dropped (overlap covers it anyway)
HEADER_FOOTER_RATIO = 0.5  # line on more than this share of pages = running head/footer
# how chunks are joined in the {retrieved_context} slot, C2, C3 and C4 all use it
CHUNK_SEPARATOR = "\n\n---\n\n"

# ---- labels
# row outcomes: ERROR = infrastructure failure (left out), INVALID = answer not readable (wrong)
ERROR = "ERROR"
INVALID = "INVALID"

# ESI level -> urgency colour
URGENCY_MAP = {1: "red", 2: "orange", 3: "yellow", 4: "green", 5: "blue"}
# least -> most severe, index distance used for over/under-triage and adjacent accuracy
URGENCY_ORDER = ["blue", "green", "yellow", "orange", "red"]
# positive class for all binary fairness metrics (ESI 1-2)
HIGH_ACUITY = frozenset({"red", "orange"})

# free-text urgency words -> colour, only used if there's no colour word or level number
# checked in order, first match wins: all phrases with "urgent" as a word must come before
# plain "urgent", otherwise "non-urgent" would be read as yellow
URGENCY_SYNONYMS = [
    (r"\bnon[- ]?urgent\b|\bnot urgent\b", "blue"),
    (r"\bless[- ]urgent\b|\bsemi[- ]urgent\b|\bstandard\b", "green"),
    (r"\bvery urgent\b|\bemergent\b", "orange"),
    (r"\bimmediate\b|\bresuscitation\b", "red"),
    (r"\burgent\b", "yellow"),
]

# specialties the model can choose from (shown in the prompts), canonical names
SPECIALTIES = [
    "General Surgery", "Gastroenterology", "Orthopedics", "Neurosurgery", "Cardiology", "Urology",
    "Neurology", "Vascular Surgery", "Emergency Medicine", "Oncology", "Dermatology",
    "Infectious Disease", "Gynecology", "Pulmonology", "Otolaryngology",
    "Oral and Maxillofacial Surgery", "Nephrology", "Hematology", "Internal Medicine",
    "Psychiatry", "Endocrinology", "Ophthalmology",
]
# other names for the SAME specialty (lowercase). only real synonyms here: mapping a
# sub-specialty to a parent would change what counts as a correct answer
SPECIALTY_SYNONYMS = {
    "orthopedics": ["orthopedic surgery", "orthopaedics", "orthopaedic surgery", "orthopedic", "orthopaedic"],
    "gynecology": ["gynaecology", "obstetrics and gynecology", "obstetrics & gynecology", "ob/gyn", "obgyn"],
    "otolaryngology": ["ent", "ear nose and throat", "ear, nose and throat"],
    "pulmonology": ["pulmonary medicine", "respiratory medicine"],
    "infectious disease": ["infectious diseases"],
    "hematology": ["haematology"],
    "gastroenterology": ["gastro"],
    "emergency medicine": ["emergency department"],
    "oral and maxillofacial surgery": ["maxillofacial surgery", "omfs"],
}

# only for the sensitivity analysis (protocol 6.2.1): ground truth labels not in SPECIALTIES
# -> a listed parent. only used offline on the ground truth, never in prompts or the main metric
# tuple = any of the listed specialties counts as correct
SPECIALTY_PARENT = {
    "rheumatology": ("internal medicine",),
    "allergy and immunology": ("internal medicine",),
    "hematology-oncology": ("hematology", "oncology"),
    "podiatry": ("orthopedics",),
    "addiction medicine": ("psychiatry",),
    "colorectal surgery": ("general surgery",),
    "bariatric surgery": ("general surgery",),
    "hepatology": ("gastroenterology",),
    "transplant nephrology": ("nephrology",),
    "reproductive endocrinology and infertility": ("gynecology",),
    "plastic surgery": ("general surgery",),
    "dentistry": ("oral and maxillofacial surgery",),
    "pediatrics": ("internal medicine",),
}

# ---- demographic groups (protocol 6.3)
GENDERS = ("male", "female")
AGE_GROUPS = ("young", "old")
# performance set (real ages): young < 40, old >= 65, 40-64 no group
YOUNG_MAX_AGE = 40
OLD_MIN_AGE = 65
# variant set (fixed when building the variants)
YOUNG_VARIANT_AGE = 30
OLD_VARIANT_AGE = 70
N_BASE_CASES = 70
SEED = 42

# ---- statistics (protocol 7)
ALPHA = 0.05
# gender + age main effects x C1-C3 = 6 tests, bonferroni (protocol 7.2, fixed 2026-10-14)
N_MAIN_EFFECT_TESTS = 6
MAIN_EFFECT_ALPHA = ALPHA / N_MAIN_EFFECT_TESTS
