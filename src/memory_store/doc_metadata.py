"""Deterministic per-document metadata for ranking: date, category, language, OCR quality.

Everything here is pure Python with no models, network or file-system access. The
extractors are intentionally conservative: a value is only reported when the
evidence is explicit, otherwise ``None`` (date) or ``"other"`` (category,
language) is returned so ranking never acts on a guess.
"""

import re
import unicodedata
from dataclasses import dataclass
from datetime import date

# Bump when extraction rules change; used by the re-index profile.
METADATA_VERSION = 1

CATEGORIES = ("family", "home", "finance", "insurance", "vehicle", "health", "work", "other")
LANGUAGES = ("de", "en", "ru", "other")

DEFAULT_FRONTMATTER_DATE_KEYS = ("document_date", "doc_date", "date", "datum")

MIN_YEAR, MAX_YEAR = 1950, 2100
_DATE_SAMPLE_LINES = 60
_CATEGORY_SAMPLE_CHARS = 3000
_LANGUAGE_SAMPLE_CHARS = 20000
_TITLE_MAX = 80


@dataclass(frozen=True)
class DocumentMetadata:
    title: str
    doc_date: str | None
    category: str
    language: str


def fold(text):
    """Casefold and strip diacritics (ß -> ss, ä -> a)."""
    text = text.casefold().replace("ß", "ss")
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


# --------------------------------------------------------------------------- front matter

_FRONTMATTER = re.compile(r"\A﻿?---[ \t]*\r?\n(.*?)\r?\n(?:---|\.\.\.)[ \t]*(?:\r?\n|\Z)", re.S)


def split_frontmatter(content):
    """Return ``(fields, body)``; ``fields`` holds simple ``key: value`` pairs only."""
    match = _FRONTMATTER.match(content)
    if not match:
        return {}, content
    fields = {}
    for line in match.group(1).splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip() and not key.startswith((" ", "\t", "#", "-")):
            fields[key.strip().casefold()] = value.strip().strip("\"'")
    return fields, content[match.end() :]


# --------------------------------------------------------------------------- dates

_MONTHS = {
    **dict.fromkeys(("januar", "janner", "jan", "january"), 1),
    **dict.fromkeys(("februar", "feb", "february"), 2),
    **dict.fromkeys(("marz", "maerz", "mar", "march"), 3),
    **dict.fromkeys(("april", "apr"), 4),
    **dict.fromkeys(("mai", "may"), 5),
    **dict.fromkeys(("juni", "jun", "june"), 6),
    **dict.fromkeys(("juli", "jul", "july"), 7),
    **dict.fromkeys(("august", "aug"), 8),
    **dict.fromkeys(("september", "sep", "sept"), 9),
    **dict.fromkeys(("oktober", "okt", "october", "oct"), 10),
    **dict.fromkeys(("november", "nov"), 11),
    **dict.fromkeys(("dezember", "dez", "december", "dec"), 12),
}
_MONTH_NAME = "(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + ")"
_ORDINAL = r"(?:st|nd|rd|th)?"

_ISO = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
_DE_NUMERIC = re.compile(r"(?<![\d.])(\d{1,2})\.(\d{1,2})\.(\d{4})(?!\d)")
_DAY_MONTH_YEAR = re.compile(
    rf"(?<!\d)(\d{{1,2}}){_ORDINAL}\.?\s+(?:of\s+)?{_MONTH_NAME}\.?,?\s+(\d{{4}})(?!\d)"
)
_MONTH_DAY_YEAR = re.compile(rf"{_MONTH_NAME}\.?\s+(\d{{1,2}}){_ORDINAL},?\s+(\d{{4}})(?!\d)")
_FILENAME_COMPACT = re.compile(r"(?<!\d)(\d{4})[-_.]?(\d{2})[-_.]?(\d{2})(?!\d)")
_FILENAME_DE = re.compile(r"(?<!\d)(\d{2})[-_.](\d{2})[-_.](\d{4})(?!\d)")


def _valid_date(year, month, day):
    try:
        value = date(int(year), int(month), int(day))
    except ValueError:
        return None
    return value.isoformat() if MIN_YEAR <= value.year <= MAX_YEAR else None


def parse_date_prefix(text):
    """Parse a date that starts *text* (ISO, DD.MM.YYYY, '12. März 2024', 'March 3rd, 2024')."""
    text = fold(text.lstrip(" \t|*_`\"'()[]"))
    match = _ISO.match(text)
    if match:
        return _valid_date(*match.groups())
    match = _DE_NUMERIC.match(text)
    if match:
        day, month, year = match.groups()
        return _valid_date(year, month, day)
    match = _DAY_MONTH_YEAR.match(text)
    if match:
        day, month, year = match.groups()
        return _valid_date(year, _MONTHS[month], day)
    match = _MONTH_DAY_YEAR.match(text)
    if match:
        month, day, year = match.groups()
        return _valid_date(year, _MONTHS[month], day)
    return None


_DATE_LABEL = re.compile(
    r"^[\s>#*_|`-]*(?:datum|date|rechnungsdatum|belegdatum|ausstellungsdatum|ausgestellt am|"
    r"invoice date|issue date|date of issue|dated)[\s:|*_`-]+(.+)$",
    re.I,
)
_LETTER_DATE_LINE = re.compile(
    r"^([^\W\d_][^\W\d_ .'-]*(?: [^\W\d_][^\W\d_.'-]*){0,3}),\s*(den\s+)?(.+?)\s*$"
)
_NON_DOCUMENT_DATE_LABEL = re.compile(
    r"geburt|geboren|\bgeb\b|birth|born|\bdob\b|fallig|\bdue\b|zahlbar|zahlungsziel|frist|"
    r"deadline|ablauf|expir|gultig|valid|konver|convert|erstellt|created|modif|geandert|"
    r"updated|aktualis|stichtag|beginn|\bende\b|\bend\b|\bstart|laufzeit|verfall|"
    r"lieferdatum|leistungsdatum|termin|appointment|\bvon\b|\bbis\b|\bfrom\b|\buntil\b"
)


def _date_from_content(body):
    lines = [line for line in body.splitlines() if line.strip()][:_DATE_SAMPLE_LINES]
    for line in lines:
        labelled = _DATE_LABEL.match(fold(line))
        if labelled:
            parsed = parse_date_prefix(labelled.group(1))
            if parsed:
                return parsed
        letter = _LETTER_DATE_LINE.match(fold(line.strip(" *_`|#>")))
        if letter:
            # "Name, 15.03.1985" is usually a birth date and "Fällig, ..." a due date: skip
            # known non-document labels and require "den" or a month name.
            label, den, candidate = letter.group(1), letter.group(2), letter.group(3).rstrip(" .")
            if _NON_DOCUMENT_DATE_LABEL.search(fold(label)):
                continue
            parsed = parse_date_prefix(candidate)
            if parsed and len(candidate) <= 30 and (den or re.search(r"[a-z]", candidate)):
                return parsed
    return None


def _date_from_filename(rel_path):
    name = rel_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    match = _FILENAME_COMPACT.search(name)
    if match:
        parsed = _valid_date(*match.groups())
        if parsed:
            return parsed
    match = _FILENAME_DE.search(name)
    if match:
        day, month, year = match.groups()
        return _valid_date(year, month, day)
    return None


def extract_doc_date(content, rel_path, frontmatter_keys=DEFAULT_FRONTMATTER_DATE_KEYS):
    """Return an ISO ``YYYY-MM-DD`` document date or ``None``.

    Sources, in order: front matter, an explicitly labelled date near the top of the
    text (``Datum:``, ``Invoice date:``, ``Berlin, den ...``), a date in the file name.
    File mtime, conversion time and bare unlabelled dates in the body are never used.
    """
    fields, body = split_frontmatter(content)
    for key in frontmatter_keys:
        if key in fields:
            parsed = parse_date_prefix(fields[key])
            if parsed:
                return parsed
    return _date_from_content(body) or _date_from_filename(rel_path)


# --------------------------------------------------------------------------- title


def extract_title(content, rel_path):
    fields, body = split_frontmatter(content)
    title = fields.get("title", "")
    if not title:
        heading = re.search(r"^#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$", body, re.M)
        title = heading.group(1) if heading else ""
    if not title:
        stem = rel_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        title = re.sub(r"[_\-\s]+", " ", stem)
    title = re.sub(r"[\s|]+", " ", title).strip()
    return title[:_TITLE_MAX].rstrip()


# --------------------------------------------------------------------------- category

# Precedence (also the tie-break order) when several categories score equally.
_CATEGORY_ORDER = ("insurance", "vehicle", "health", "finance", "work", "home", "family")
_CATEGORY_KEYWORDS = {
    "insurance": (
        "versicherung",
        "versicherungen",
        "insurance",
        "police",
        "policy",
        "haftpflicht",
        "hausrat",
        "rechtsschutz",
        "schadenmeldung",
        "versicherungsschein",
        "beitragsrechnung",
    ),
    "vehicle": (
        "auto",
        "kfz",
        "fahrzeug",
        "vehicle",
        "car",
        "tuv",
        "zulassung",
        "fuhrerschein",
        "fahrzeugschein",
        "werkstatt",
        "reifen",
        "leasing",
        "motorrad",
        "fahrrad",
    ),
    "health": (
        "arzt",
        "arzte",
        "hausarzt",
        "zahnarzt",
        "health",
        "gesundheit",
        "krankenhaus",
        "klinik",
        "rezept",
        "befund",
        "impfung",
        "impfpass",
        "diagnose",
        "medical",
        "doctor",
        "praxis",
        "therapie",
        "vorsorge",
        "labor",
    ),
    "finance": (
        "bank",
        "konto",
        "steuer",
        "steuerbescheid",
        "tax",
        "finanzamt",
        "finance",
        "depot",
        "kredit",
        "darlehen",
        "loan",
        "invoice",
        "rechnung",
        "kontoauszug",
        "gehaltsabrechnung",
        "payslip",
        "rente",
        "pension",
        "sparplan",
        "mahnung",
        "zahlung",
        "payment",
    ),
    "work": (
        "arbeit",
        "arbeitgeber",
        "arbeitsvertrag",
        "work",
        "job",
        "bewerbung",
        "gehalt",
        "salary",
        "employer",
        "employment",
        "kundigung",
        "zeugnis",
        "lebenslauf",
        "resume",
        "cv",
        "contract",
        "projekt",
        "dienstreise",
        "reisekosten",
    ),
    "home": (
        "haus",
        "home",
        "house",
        "wohnung",
        "miete",
        "mietvertrag",
        "nebenkosten",
        "strom",
        "gas",
        "wasser",
        "energie",
        "heizung",
        "internet",
        "grundsteuer",
        "eigentum",
        "hausverwaltung",
        "umzug",
        "handwerker",
        "apartment",
        "rent",
        "lease",
        "utility",
        "immobilie",
    ),
    "family": (
        "familie",
        "family",
        "kinder",
        "kind",
        "kita",
        "schule",
        "school",
        "kindergarten",
        "geburtsurkunde",
        "heiratsurkunde",
        "eltern",
        "oma",
        "opa",
        "schulzeugnis",
        "elterngeld",
        "kindergeld",
        "children",
    ),
}
_COMPOUND_MIN = 6  # keywords this long also match inside German compounds
_CONTENT_MIN_HITS = 2  # body-only evidence needs two distinct keywords


def _keyword_hits(tokens, category):
    hits = 0
    for keyword in _CATEGORY_KEYWORDS[category]:
        compound = len(keyword) >= _COMPOUND_MIN
        if any(t == keyword or (compound and keyword in t) for t in tokens):
            hits += 1
    return hits


def classify_category(rel_path, content):
    """Return one of ``CATEGORIES``; any path keyword outranks body-only evidence."""
    path_tokens = set(re.findall(r"[^\W_]+", fold(rel_path.rsplit(".", 1)[0])))
    _, body = split_frontmatter(content)
    body_tokens = set(re.findall(r"[^\W_]+", fold(body[:_CATEGORY_SAMPLE_CHARS])))
    best, best_key = "other", (0, 0)
    for category in _CATEGORY_ORDER:
        content_hits = _keyword_hits(body_tokens, category)
        key = (
            _keyword_hits(path_tokens, category),
            content_hits if content_hits >= _CONTENT_MIN_HITS else 0,
        )
        if key > best_key:
            best, best_key = category, key
    return best


# --------------------------------------------------------------------------- language

_STOPWORDS_DE = frozenset(
    "der die das den dem des und oder ist sind nicht ich du wir ihr mit von fur auf im zu zum "
    "zur bei nach aus uber auch wie als wenn dass eine einen einem einer ein sich es sie er "
    "wird werden wurde haben habe kann konnen noch nur aber dann diese dieser dieses vom um "
    "ihre ihren unser unsere sehr bitte vielen dank geehrte herr frau".split()
)
_STOPWORDS_EN = frozenset(
    "the and of to is are were for with that this on be by as at from or not have has had it "
    "its you your we our they their would can could please dear regards thank thanks yours "
    "sincerely which should been".split()
)
_STOPWORDS_RU = frozenset("и в не на я что с по это как для от из за но он она мы вы они".split())
_STOPWORDS_ALL = _STOPWORDS_DE | _STOPWORDS_EN | _STOPWORDS_RU
_MIN_STOPWORD_HITS = 3
_MIN_STOPWORD_SHARE = 0.06
_WORD = re.compile(r"[^\W\d_]+")
_CYRILLIC = re.compile(r"[Ѐ-ӿ]")


def detect_language(text):
    """Heuristic ``de``/``en``/``ru``/``other`` from script and function-word frequency."""
    sample = text[:_LANGUAGE_SAMPLE_CHARS]
    letters = _WORD.findall(sample)
    if not letters:
        return "other"
    cyrillic = sum(len(_CYRILLIC.findall(word)) for word in letters)
    total_letters = sum(len(word) for word in letters)
    if cyrillic / total_letters > 0.3:
        return "ru"
    folded = [fold(word) for word in letters]
    de_hits = sum(1 for word in folded if word in _STOPWORDS_DE)
    en_hits = sum(1 for word in folded if word in _STOPWORDS_EN)
    de_hits += sum(sample.count(char) for char in "äöüßÄÖÜ") // 3
    best, hits = ("de", de_hits) if de_hits > en_hits else ("en", en_hits)
    if hits < _MIN_STOPWORD_HITS or hits / len(folded) < _MIN_STOPWORD_SHARE or de_hits == en_hits:
        return "other"
    return best


# --------------------------------------------------------------------------- OCR quality

_VOWELS = frozenset(
    "aeiouyäöüéèêàáâíìîóòôúùûæœøåÿ" + "\u0430\u0435\u0451\u0438\u043e\u0443\u044b\u044d\u044e\u044f"
)
_SINGLE_OK = frozenset(
    "aAiI"
    + "\u0430\u0410\u0438\u0418\u043e\u041e\u0432\u0412\u0441\u0421\u043a\u041a\u0443\u0423\u044f\u042f"
)
_GARBAGE_SYMBOLS = frozenset("¦¬¤¶†‡■□▪◊¿¡~^´`¨�")
_SYMBOL_OK = frozenset("©®°™§")
_GARBAGE_RUN = re.compile(r"([|!:;,'\"^~`¦])\1{3,}")
_TOKEN = re.compile(r"[^\W_]+")
_MIXED_OK = re.compile(r"^(?:[^\W\d_]{1,5}\d{1,8}|\d{1,8}[^\W\d_]{1,3})$")
_ONLY_LOOKALIKES = re.compile(r"^[Iil1]{3,}$")
_REPEATED_CHAR = re.compile(r"(.)\1{3,}", re.I)
_MAX_TOKEN = 40
_MAX_CONSONANT_RUN = 6
_ACRONYM_MAX = 5
_PENALTY_PER_GARBAGE_RATIO = 4.0
_SW_FACTOR_MIN_TOKENS = 30
_SW_FULL_SHARE = 0.15


def _is_garbage_char(char):
    if char in _GARBAGE_SYMBOLS:
        return True
    category = unicodedata.category(char)
    if category == "So":
        return char not in _SYMBOL_OK
    return category[0] == "C"


def _plausible_alpha(word):
    if len(word) > _MAX_TOKEN:
        return False
    if len(word) == 1:
        return word in _SINGLE_OK
    if _ONLY_LOOKALIKES.match(word) or _REPEATED_CHAR.search(word):
        return False
    if sum(1 for a, b in zip(word, word[1:], strict=False) if a.islower() and b.isupper()) >= 2:
        return False
    run = longest = 0
    for char in word:
        run = 0 if char.casefold() in _VOWELS else run + 1
        longest = max(longest, run)
    if longest == len(word):  # no vowel: only short abbreviations (Nr, Str, bzw) or acronyms (GmbH)
        acronym = len(word) <= _ACRONYM_MAX and sum(c.isupper() for c in word) >= 2
        return acronym or (len(word) <= 3 and len(set(word.casefold())) > 1)
    return longest <= _MAX_CONSONANT_RUN


def _plausible_token(token):
    if token.isdigit():
        return True
    if token.isalpha():
        return _plausible_alpha(token)
    return bool(_MIXED_OK.match(token))


def ocr_quality(text):
    """Cheap OCR-quality score in ``[0, 1]`` (``None`` when the text has no content).

    Combines the share of plausibly shaped word tokens, a garbage-character penalty and,
    for longer passages, the density of German/English/Russian function words. Higher is
    cleaner. Works without a dictionary so German and English need no resources.
    """
    visible = [c for c in text if not c.isspace()]
    if not visible:
        return None
    garbage = sum(1 for c in visible if _is_garbage_char(c))
    garbage += sum(len(m.group(0)) for m in _GARBAGE_RUN.finditer(text))
    tokens = _TOKEN.findall(text)
    if not tokens:
        return 0.0 if garbage else None
    shape = sum(1 for token in tokens if _plausible_token(token)) / len(tokens)
    penalty = max(0.0, 1.0 - _PENALTY_PER_GARBAGE_RATIO * min(1.0, garbage / len(visible)))
    factor = 1.0
    alpha = [fold(t) for t in tokens if t.isalpha()]
    if len(alpha) >= _SW_FACTOR_MIN_TOKENS:
        stop = sum(1 for t in alpha if t in _STOPWORDS_ALL)
        factor = 0.8 + 0.2 * min(1.0, stop / len(alpha) / _SW_FULL_SHARE)
    return round(max(0.0, min(1.0, shape * penalty * factor)), 3)


# --------------------------------------------------------------------------- orchestration


def extract_document_metadata(content, rel_path, frontmatter_keys=DEFAULT_FRONTMATTER_DATE_KEYS):
    return DocumentMetadata(
        title=extract_title(content, rel_path),
        doc_date=extract_doc_date(content, rel_path, frontmatter_keys),
        category=classify_category(rel_path, content),
        language=detect_language(split_frontmatter(content)[1]),
    )


def context_header(meta):
    """Short ``title | date | category`` line prepended to chunk text when enabled."""
    parts = [meta.title, meta.doc_date, meta.category if meta.category != "other" else None]
    return " | ".join(part for part in parts if part)
