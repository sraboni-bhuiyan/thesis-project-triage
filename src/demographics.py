# changing demographics in the HPI text (protocol 2.3, 4)
#   swap_text(text, gender, age)  set every gender cue to `gender`, explicit ages to `age` (variants)
#   mask_text(text)               gender cues -> neutral words, explicit ages -> ___ (C3)
#   sex_specific_mentions(text)   sex-specific clinical words, affirmed vs negated (audit)
#
# MIMIC mostly writes ages as "___", so gender cues are the main signal:
# pronouns, titles, sex words, "yo M" / "y/o F" shorthand
#
# mask_text uses the same wording as the September 2026 C3 prompts (they/them/their, Pt,
# person, "___ y/o") so the model sees the same text. only differs where September was wrong
# (see DEVIATIONS.md)
#
# NOT changed on purpose:
#   - relationship words (husband, wife, daughter, mother ...) -> other people
#   - abbreviations: "MR" (MRI, mitral regurgitation), "MS" (multiple sclerosis), "HE" (hepatic
#     encephalopathy). titles only in Title case. all-caps "HE" is the pronoun only at sentence
#     start before a lowercase word ("HE reports ...")
#   - a bare number before M/F ("101 F" is a temperature): letter = sex only after an age token
#     ("___ F", "45 yo F", "y/o M")
#   - durations ("5 year history"): only phrases with "old" or "yo" are ages

import re

# (male, female) word pairs
PAIRS = [("he", "she"), ("him", "her"), ("his", "her"), ("his", "hers"), ("himself", "herself"),
         ("man", "woman"), ("men", "women"), ("male", "female"), ("males", "females"),
         ("gentleman", "lady"), ("gentlemen", "ladies"), ("boy", "girl")]

# female word -> male word ("her" is decided by context)
TO_MALE = {}
for m, f in PAIRS:
    if f != "her":
        TO_MALE[f] = m

# male word -> female word
TO_FEMALE = {}
for m, f in PAIRS:
    if (m, f) != ("his", "hers"):
        TO_FEMALE[m] = f

# neutral words, like the September 2026 C3 prompts. "her" decided by context (their/them)
NEUTRAL = {
    "he": "the patient", "she": "the patient", "him": "them", "his": "their", "hers": "theirs",
    "himself": "themself", "herself": "themself",
    "man": "person", "woman": "person", "men": "people", "women": "people",
    "male": "patient", "female": "patient", "males": "patients", "females": "patients",
    "gentleman": "person", "lady": "person", "gentlemen": "people", "ladies": "people",
    "boy": "child", "girl": "child",
}

# all gender words, longest first
GENDER_WORD = re.compile(r"\b(" + "|".join(sorted(set(NEUTRAL) | {"her"}, key=len, reverse=True)) + r")\b",
                         re.IGNORECASE)
# titles, Title case only -> "MR" (MRI), "MS" (multiple sclerosis), verb "miss" not touched
TITLE = re.compile(r"\b(Mr|Mrs|Ms|Miss)\b(\.?)")
# age token then sex letter: "___ F", "___ yo M", "45 y/o F", "45yo M"
SEX_LETTER = re.compile(r"((?:___\s*|\b\d{1,3}\s*(?:-\s*)?(?:y/?o|yo|yr?s? old|years?[- ]old)\s*|"
                        r"\by/?o\s*|\byo\s*))\b([MF])\b")
# age + sex in one token: "45M", "70F"
AGE_SEX_COMPACT = re.compile(r"\b(\d{1,3})([MF])\b")
# age phrases: "45 year old", "45-year-old", "45 yo", "45 y/o", "45yo"
# needs "old"/"yo", so durations ("5 year history", "10 yrs") stay
AGE_PHRASE = re.compile(r"\b(\d{1,3})(\s*-?\s*(?:y/?o|yo|yrs? old|years?[- ]old))\b", re.IGNORECASE)
AGE_WORD = re.compile(r"\b(age[d]?\s+)(\d{1,3})\b", re.IGNORECASE)

# if next word is one of these, "her" is the object not possessive ("gave her to")
# case sensitive on purpose: "her SO" (significant other) is possessive, "her so" is not
# "back" and "home" depend on what comes after (see her_is_possessive)
NOT_NOUN = re.compile(r"(?:to|and|or|in|at|on|with|for|from|that|as|is|was|were|by|up|down|out|"
                      r"again|today|yesterday|here|there|after|before|about|if|but|so|"
                      r"when|while|the|a|an)\b")
# "her home" possessive only before a noun ("her home medications"), "walked her home" is not
HOME_NOUN = re.compile(r"home\s+(?:med\w*|o2|oxygen|dose\w*|regimen|health|address|nurs\w*|care|"
                       r"setting|pharmacy|insulin|bipap|cpap)\b", re.IGNORECASE)
PREP_BEFORE = re.compile(r"\b(?:at|to|from|in|into|near|outside|inside|of)\s+$", re.IGNORECASE)
# "her back" = body part, unless it goes on like a direction ("brought her back to the ED")
BACK_DIRECTION = re.compile(r"back\s+(?:to|in|into|home|from|at|out|up|on|over)\b", re.IGNORECASE)


def is_abbreviation(m, text):
    # all-caps "HE" = hepatic encephalopathy, unless sentence start + lowercase word after
    if m.group(1) != "HE":
        return False
    before = text[:m.start()].rstrip()
    starts_sentence = not before or before[-1] in ".:;!?"
    lowercase_next = re.match(r"\s+[a-z]", text[m.end():])
    return not (starts_sentence and lowercase_next)


def match_case(src, repl):
    # same upper/lower case as the original word
    if len(src) > 1 and src.isupper():
        return repl.upper()
    if src[0].isupper():
        return repl[0].upper() + repl[1:]
    return repl


def her_is_possessive(text, end):
    # "her pain" -> possessive, "gave her to", "her.", "her, " -> object
    rest = text[end:end + 25]
    # any whitespace, HPI has line breaks mid sentence
    if not rest[:1].isspace():
        return False
    nxt = rest.lstrip()

    if re.match(r"home\b", nxt, re.IGNORECASE):
        # "at/to/from her home" = the place (possessive), "walked her home" = object
        before = text[max(0, end - 3 - 12):end - 3]
        return bool(HOME_NOUN.match(nxt) or PREP_BEFORE.search(before))

    if re.match(r"back\b", nxt, re.IGNORECASE):
        return not BACK_DIRECTION.match(nxt)

    return bool(nxt[:1].isalpha()) and not NOT_NOUN.match(nxt)


def swap_text(text, gender, age):
    # change every gender cue to `gender` ('male'/'female') and explicit ages to `age`
    if not text:
        return text
    gender = gender.lower()
    if gender not in ("male", "female"):
        raise ValueError(f"gender must be 'male' or 'female', got {gender!r}")

    if gender == "male":
        table = TO_MALE
        title = "Mr"
        letter = "M"
    else:
        table = TO_FEMALE
        title = "Ms"
        letter = "F"

    def word(m):
        w = m.group(1)
        if is_abbreviation(m, text):
            return w
        lw = w.lower()
        if lw == "her" and gender == "male":
            if her_is_possessive(text, m.end()):
                return match_case(w, "his")
            return match_case(w, "him")
        return match_case(w, table.get(lw, lw))

    def title_repl(m):
        return title + m.group(2)

    def letter_repl(m):
        return m.group(1) + letter

    def age_phrase_repl(m):
        return f"{age}{m.group(2)}"

    def age_word_repl(m):
        return f"{m.group(1)}{age}"

    text = GENDER_WORD.sub(word, text)
    text = TITLE.sub(title_repl, text)
    text = SEX_LETTER.sub(letter_repl, text)
    text = AGE_PHRASE.sub(age_phrase_repl, text)
    text = AGE_WORD.sub(age_word_repl, text)
    return text


def mask_text(text):
    # remove gender cues and explicit ages (C3), September wording
    # same order as September: "45M" -> "___", sex letter after age token removed (space kept,
    # like September), age phrases -> "___ y/o", "aged 45" -> "aged ___",
    # gender words -> neutral, titles -> "Pt". whitespace not normalised
    if not text:
        return text

    def letter_repl(m):
        return m.group(1).rstrip() + " "

    def age_word_repl(m):
        return f"{m.group(1)}___"

    text = AGE_SEX_COMPACT.sub("___", text)
    text = SEX_LETTER.sub(letter_repl, text)
    text = AGE_PHRASE.sub("___ y/o", text)
    text = AGE_WORD.sub(age_word_repl, text)

    def word(m):
        w = m.group(1)
        if is_abbreviation(m, text):
            return w
        lw = w.lower()
        if lw == "her":
            if her_is_possessive(text, m.end()):
                return match_case(w, "their")
            return match_case(w, "them")
        return match_case(w, NEUTRAL[lw])

    text = GENDER_WORD.sub(word, text)
    return TITLE.sub("Pt", text)


# sex-specific clinical words (protocol 2.3), only for counting / reporting
# "cervical" left out: in MIMIC almost always cervical spine
# "breast" left out: men can have breast disease, male version still ok
SEX_SPECIFIC = re.compile(
    r"\b(pregnan\w*|gravid\w*|G\d+\s*P\d+|postpartum|post-partum|obstetric\w*|ectopic|miscarriage|"
    r"uter(?:us|ine)|ovar(?:y|ies|ian)|vagin\w*|vulv\w*|endometri\w*|hysterectom\w*|menstru\w*|"
    r"menopaus\w*|gyn\w*|cervix|adnex\w*|fallopian|"
    r"ivf|in[- ]vitro fertili\w*|infertil\w*|gonal[- ]?f|stimulation protocol|oocyte\w*|embryo transfer|"
    r"prostat\w*|testic\w*|testes|scrot\w*|penile|penis|erectile|orchi\w*|epididym\w*)\b", re.IGNORECASE)
# negated = negation word earlier in the same sentence
# ("No vaginal bleeding", "denies fever, dysuria, vaginal discharge")
NEGATION = re.compile(r"\b(?:no|not|denies|denied|denying|without|negative for|absence of)\b", re.IGNORECASE)


def is_negated(text, start):
    # sentence start = last ". ", ".\n" or blank line before the word
    a = text.rfind(". ", 0, start)
    b = text.rfind(".\n", 0, start)
    c = text.rfind("\n\n", 0, start)
    sentence_start = max(a, b, c)
    return bool(NEGATION.search(text[sentence_start + 1:start]))


def sex_specific_mentions(text):
    # {'affirmed': [...], 'negated': [...]}, distinct lowercase words
    # affirmed -> gender swap not clinically valid (counted)
    # negated -> clinical picture same, but sex-specific phrase stays in swapped text (text leak)
    affirmed = set()
    negated = set()
    for m in SEX_SPECIFIC.finditer(text or ""):
        if is_negated(text, m.start()):
            negated.add(m.group(1).lower())
        else:
            affirmed.add(m.group(1).lower())
    return {"affirmed": sorted(affirmed), "negated": sorted(negated - affirmed)}
