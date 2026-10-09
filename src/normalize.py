"""Text normalisation for business names and addresses.

Design rules
- Country-agnostic: the same code runs for US, India, France and any unseen country.
  Country-specific knowledge lives only in the abbreviation dictionaries below, and every
  dictionary is applied to every record.
- Both sides of a pair go through identical normalisation, so an imperfect mapping
  (e.g. French "st" = saint, mapped to "street") still matches consistently.
- No external data, no geocoding, no lookups: only the provided records.
"""

import re
import unicodedata

import pandas as pd

from config import ADDR, COUNTRY, NAME

try:  # transliterate Devanagari / Gujarati / other scripts to Latin letters (ISC licence)
    from anyascii import anyascii
except ImportError:  # pipeline still runs without it, Indian-script names just match worse
    anyascii = None

# ------------------------------------------------------------------ dictionaries
NAME_ABBR = {
    "pvt": "private", "pte": "private", "prvt": "private",
    "ltd": "limited", "ltda": "limited", "lmtd": "limited", "ltd.": "limited",
    "corp": "corporation", "corpn": "corporation", "inc": "incorporated",
    "co": "company", "cos": "companies", "cie": "company", "cia": "company",
    "intl": "international", "int'l": "international", "natl": "national",
    "mfg": "manufacturing", "mfrs": "manufacturers", "mfr": "manufacturer",
    "svc": "service", "svcs": "services", "serv": "services",
    "bros": "brothers", "assoc": "associates", "assocs": "associates",
    "mgmt": "management", "mgt": "management", "dept": "department",
    "ent": "enterprises", "entp": "enterprises", "ents": "enterprises",
    "inds": "industries", "ind": "industries", "indus": "industries",
    "tech": "technologies", "techs": "technologies", "technology": "technologies",
    "sys": "systems", "sol": "solutions", "solns": "solutions",
    "grp": "group", "hldgs": "holdings", "hldg": "holding",
    "ctr": "center", "centre": "center", "cntr": "center",
    "univ": "university", "inst": "institute", "assn": "association",
    "&": "and", "et": "and",
    "st": "saint", "ste": "sainte",
}
# Legal-form tokens, removed from the "core" name and kept separately as a feature.
LEGAL = {
    "private", "limited", "incorporated", "corporation", "company", "companies",
    "llc", "llp", "lp", "plc", "pllc", "pc", "opc", "gmbh", "ag", "bv", "nv", "pty",
    "sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "scop", "selarl", "spa", "srl",
    "holding", "holdings", "group",
}
NAME_STOP = {"the", "and", "of", "a", "an", "le", "la", "les", "l", "de", "du", "des", "d", "ms"}

ADDR_ABBR = {
    # US / generic English
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue",
    "avn": "avenue", "blvd": "boulevard", "bd": "boulevard", "bvd": "boulevard",
    "dr": "drive", "ln": "lane", "hwy": "highway", "pkwy": "parkway", "fwy": "freeway",
    "ste": "suite", "apt": "apartment", "fl": "floor", "flr": "floor", "bldg": "building",
    "sq": "square", "ct": "court", "pl": "place", "cir": "circle", "ter": "terrace",
    "trl": "trail", "expy": "expressway", "mt": "mount", "ft": "fort",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    # India
    "nagr": "nagar", "ngr": "nagar", "sec": "sector", "sect": "sector", "sctr": "sector",
    "ph": "phase", "extn": "extension", "ext": "extension", "colny": "colony",
    "clny": "colony", "col": "colony", "mkt": "market", "bazar": "bazaar", "bazzar": "bazaar",
    "chk": "chowk", "cross": "cross", "crs": "cross", "mn": "main", "opp": "opposite",
    "nr": "near", "dist": "district", "distt": "district", "teh": "tehsil", "vill": "village",
    "vpo": "village", "po": "postoffice", "ps": "policestation", "bldng": "building",
    "indl": "industrial", "ind": "industrial", "estt": "estate", "est": "estate",
    "marg": "marg", "salai": "road", "rasta": "road", "path": "road",
    "bengaluru": "bangalore", "mumbai": "bombay", "chennai": "madras", "kolkata": "calcutta",
    "gurugram": "gurgaon", "puducherry": "pondicherry",
    # France
    "r": "rue", "ch": "chemin", "che": "chemin", "imp": "impasse", "all": "allee",
    "fbg": "faubourg", "rte": "route", "qu": "quai", "bis": "bis",
    "cedex": "", "zi": "zoneindustrielle", "za": "zoneactivite", "zac": "zoneactivite",
}
ADDR_STOP = {
    "no", "number", "num", "the", "of", "and", "de", "du", "des", "la", "le", "les", "d", "l",
    "india", "usa", "us", "u", "united", "states", "america", "france", "republic",
}
LANDMARK_RE = re.compile(
    r"\b(near|nr|opp|opposite|behind|beside|besides|next to|adjacent to|adj to|adj|"
    r"in front of|infront of|close to|above|below|opposite to|pres de|en face de|a cote de)\b[^,;]*"
)
DBA_RE = re.compile(
    r"\b(dba|d/b/a|doing business as|trading as|t/a|aka|a/k/a|formerly|operating as)\b"
)
MS_PREFIX_RE = re.compile(r"^\s*m\s*/\s*s\b\.?")       # Indian "M/s." prefix
POSTAL_SPACED_RE = re.compile(r"\b(\d{3})\s(\d{3})\b")  # "110 001" -> "110001"
POSTAL_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b|\b\d{6}\b")
NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]+")
SPACES_RE = re.compile(r"\s+")
ORDINAL_RE = re.compile(r"^(\d+)(st|nd|rd|th)$")
HOUSE_RE = re.compile(r"^(\d+)[a-z]?$")
WEB_RE = re.compile(r"(?:www\.)?([a-z0-9][a-z0-9-]*)\.(?:com|net|org|in|co|biz|info|us|io)\b")
ORDINAL_WORDS = {
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6",
    "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10", "eleventh": "11",
    "twelfth": "12", "thirteenth": "13", "fourteenth": "14", "fifteenth": "15",
    "sixteenth": "16", "seventeenth": "17", "eighteenth": "18", "nineteenth": "19",
    "twentieth": "20", "thirtieth": "30", "fortieth": "40", "fiftieth": "50",
}

COUNTRY_MAP = {
    "us": "us", "usa": "us", "u s": "us", "u s a": "us", "united states": "us",
    "united states of america": "us", "america": "us",
    "india": "india", "in": "india", "ind": "india", "bharat": "india",
    "france": "france", "fr": "france", "fra": "france", "republique francaise": "france",
}


# ------------------------------------------------------------------ helpers
def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def base_clean(s: str) -> str:
    s = str(s)
    if anyascii is not None and not s.isascii():
        s = anyascii(s)
    s = strip_accents(s.lower())
    s = s.replace("&", " and ").replace("@", " at ")
    s = re.sub(r"[’'`]", "", s)          # o'brien -> obrien
    s = s.replace(".", "")               # pvt. ltd. -> pvt ltd ; l.l.c. -> llc
    return s


def tokens_of(s: str):
    s = NON_ALNUM_RE.sub(" ", s)
    return [t for t in SPACES_RE.sub(" ", s).strip().split(" ") if t]


def norm_country(c: str) -> str:
    c = SPACES_RE.sub(" ", NON_ALNUM_RE.sub(" ", strip_accents(str(c).lower()))).strip()
    return COUNTRY_MAP.get(c, c)  # unknown countries pass through unchanged (open set)


# ------------------------------------------------------------------ names
def normalize_name(raw: str) -> dict:
    s = MS_PREFIX_RE.sub(" ", base_clean(raw))
    parts = DBA_RE.split(s)
    main = parts[0]
    alt = parts[-1] if len(parts) >= 3 else ""   # text after "dba"
    out = {}
    for key, text in (("main", main), ("alt", alt)):
        toks = [NAME_ABBR.get(t, t) for t in tokens_of(text)]
        toks = [t for t in toks if t not in NAME_STOP]
        legal = [t for t in toks if t in LEGAL]
        core = [t for t in toks if t not in LEGAL]
        if not core:                 # e.g. the whole name is "Company Limited"
            core = toks
        out[key] = (toks, core, legal)
    toks, core, legal = out["main"]
    alt_core = out["alt"][1]
    initials = "".join(t[0] for t in core if t and not t.isdigit())
    web = WEB_RE.search(str(raw).lower())
    name_web = web.group(1).replace("-", "") if web else ""
    return {
        "name_web": name_web,
        "name_norm": " ".join(toks),
        "name_core": " ".join(core),
        "name_alt": " ".join(alt_core),
        "name_compact": "".join(core),
        "name_initials": initials,
        "name_legal": " ".join(sorted(set(legal))),
        "name_first": core[0] if core else "",
        "name_digits": " ".join(t for t in core if any(ch.isdigit() for ch in t)),
    }


# ------------------------------------------------------------------ addresses
def normalize_address(raw: str) -> dict:
    s = base_clean(raw)
    landmark = " ".join(m.group(0) for m in LANDMARK_RE.finditer(s))
    main = LANDMARK_RE.sub(" ", s)
    main = POSTAL_SPACED_RE.sub(r"\1\2", main)
    postals = POSTAL_RE.findall(main)
    postal = postals[-1].split("-")[0] if postals else ""
    if postal:
        main = main.replace(postals[-1], " ")
    toks = []
    for t in tokens_of(main):
        t = ORDINAL_RE.sub(r"\1", t)
        t = ORDINAL_WORDS.get(t, t)
        t = ADDR_ABBR.get(t, t)
        if t and t not in ADDR_STOP:
            toks.append(t)
    # primary house number (first number-like token) and the token right after it (street)
    house, street = "", ""
    for k, t in enumerate(toks):
        m = HOUSE_RE.match(t)
        if m:
            house = m.group(1).lstrip("0") or "0"
            street = toks[k + 1] if k + 1 < len(toks) else ""
            break
    numbers = sorted({t for t in toks if any(ch.isdigit() for ch in t)})
    words = [t for t in toks if not any(ch.isdigit() for ch in t)]
    lm_toks = [ADDR_ABBR.get(t, t) for t in tokens_of(landmark)]
    lm_toks = [t for t in lm_toks if t not in ADDR_STOP and t not in {"near", "opposite"}]
    return {
        "addr_norm": " ".join(toks),
        "addr_words": " ".join(words),
        "addr_numbers": " ".join(numbers),
        "addr_postal": postal,
        "addr_landmark": " ".join(lm_toks),
        "addr_tail": " ".join(words[-3:]),   # usually city / state end of the address
        "addr_house": house,
        "addr_street": street,
    }


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add all normalised columns to a copy of a source dataframe."""
    out = df.copy()
    names = pd.DataFrame([normalize_name(x) for x in out[NAME]], index=out.index)
    addrs = pd.DataFrame([normalize_address(x) for x in out[ADDR]], index=out.index)
    out = pd.concat([out, names, addrs], axis=1)
    out["country_norm"] = [norm_country(c) for c in out[COUNTRY]]
    # text used by the name blocker; fall back to the full name if the core is empty
    out["block_name"] = [c if c else n for c, n in zip(out["name_core"], out["name_norm"])]
    out["block_full"] = (out["block_name"] + " " + out["addr_norm"]).str.strip()
    return out
