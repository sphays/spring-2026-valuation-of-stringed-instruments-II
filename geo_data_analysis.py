import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

# ----------------------------- Normalization -----------------------------

CHAR_MAP = str.maketrans({
    "ø": "o", "Ø": "O",
    "ł": "l", "Ł": "L",
    "đ": "d", "Đ": "D",
    "ð": "d", "Ð": "D",
    "þ": "th", "Þ": "Th",
    "æ": "ae", "Æ": "Ae",
    "œ": "oe", "Œ": "Oe",
})

def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def norm_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.translate(CHAR_MAP)
    s = strip_accents(s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Cf")
    s = s.casefold().strip()
    s = re.sub(r"[’`]", "'", s)
    s = re.sub(r"[\s\-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


# ----------------------------- Parsing helpers -----------------------------

US_STATE_ABBR = set("""
AL AK AZ AR CA CO CT DE FL GA HI IA ID IL IN KS KY LA MA MD ME MI MN
MO MS MT NC ND NE NH NJ NM NV NY OH OK OR PA RI SC SD TN TX UT VA VT
WA WI WV WY DC
""".split())

US_STATE_NAME_TO_ABBR = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC",
}

def parse_place(raw: str) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """
    Returns: (place_core, subregion_hint, state_hint, country_hint)

    - "City, ST" -> state_hint=ST, country_hint='US'
    - "City, StateName" -> state_hint=abbr, country_hint='US'
    - "Name (Qualifier)" -> place_core=Name, subregion_hint=Qualifier
    - Non-US comma suffix like "Worthing, Sussex" -> subregion_hint="Sussex"
    """
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return ("", None, None, None)

    s = str(raw).strip()
    if not s:
        return ("", None, None, None)

    m_paren = re.match(r"^(.*?)\s*\((.*?)\)\s*$", s)
    if m_paren:
        core = m_paren.group(1).strip()
        qual = m_paren.group(2).strip()
        return (core, qual or None, None, None)

    if "," in s:
        left, right = s.rsplit(",", 1)
        left = left.strip()
        right_stripped = right.strip()

        m_abbr = re.match(r"^([A-Za-z]{2})$", right_stripped)
        if m_abbr:
            st = m_abbr.group(1).upper()
            if st in US_STATE_ABBR:
                return (left, None, st, "US")

        right_norm = norm_text(right_stripped)
        if right_norm in US_STATE_NAME_TO_ABBR:
            st = US_STATE_NAME_TO_ABBR[right_norm]
            return (left, None, st, "US")

        # treat as subregion hint (e.g. Sussex, Haute-Saone, etc.)
        return (left, right_stripped or None, None, None)

    return (s, None, None, None)


# ----------------------------- GeoNames loading -----------------------------

@dataclass(frozen=True)
class Candidate:
    geonameid: str
    name: str
    asciiname: str
    country: str
    admin1: str
    admin2: str
    feature_class: str
    feature_code: str
    population: int

def load_admin1(admin1_path: str) -> Dict[str, str]:
    m = {}
    with open(admin1_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0].strip():
                m[parts[0].strip()] = parts[1].strip()
    return m

def load_admin2(admin2_path: str) -> Dict[str, str]:
    m = {}
    with open(admin2_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0].strip():
                m[parts[0].strip()] = parts[1].strip()
    return m

def build_candidates_index(
    geonames_path: str,
    target_keys: set,
    include_admin_features: bool = True,
) -> Dict[str, List[Candidate]]:
    allowed_A_codes = {"ADM1", "ADM2", "ADM3", "ADM4", "ADMD", "ADM5"}
    cand_by_key: Dict[str, List[Candidate]] = {k: [] for k in target_keys}

    def maybe_add(key: str, parts: List[str]) -> None:
        fclass = parts[6]
        fcode = parts[7]
        if fclass == "P":
            pass
        elif include_admin_features and fclass == "A" and fcode in allowed_A_codes:
            pass
        else:
            return

        try:
            pop = int(parts[14]) if parts[14] else 0
        except ValueError:
            pop = 0

        cand_by_key[key].append(
            Candidate(
                geonameid=parts[0],
                name=parts[1],
                asciiname=parts[2],
                country=parts[8],
                admin1=parts[10],
                admin2=parts[11],
                feature_class=fclass,
                feature_code=fcode,
                population=pop,
            )
        )

    # pass 1: name/asciiname
    with open(geonames_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 15:
                continue

            k_ascii = norm_text(parts[2])
            if k_ascii in target_keys:
                maybe_add(k_ascii, parts)

            k_name = norm_text(parts[1])
            if k_name in target_keys and k_name != k_ascii:
                maybe_add(k_name, parts)

    unmatched = {k for k, v in cand_by_key.items() if len(v) == 0}
    if not unmatched:
        return cand_by_key

    # pass 2: alternatenames for remaining unmatched keys
    with open(geonames_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 15:
                continue
            alts = parts[3]
            if not alts or not unmatched:
                continue

            for alt in alts.split(","):
                k = norm_text(alt)
                if k in unmatched:
                    maybe_add(k, parts)
                    unmatched.discard(k)
                    if not unmatched:
                        break

    return cand_by_key


# ----------------------------- Overrides (unchanged) -----------------------------

OVERRIDES: Dict[str, Dict[str, Union[str, List[str], bool, List[str]]]] = {
    "gerona":      {"country": "IT", "lookup": ["fosso gerona"]},
    "mantua":      {"country": "IT", "lookup": ["mantova", "mantua"], "prefer_class": "A", "prefer_codes": ["ADM2", "ADM1"]},
    "cheshire":    {"country": "GB", "prefer_class": "A"},
    "padua":       {"country": "IT", "lookup": ["padova", "padua"]},
    "beaconsfield":{"country": "GB"},
    "lugo":        {"country": "IT"},
    "san marino":  {"country": "SM"},
    "devonport":   {"country": "GB"},
    "tournay":     {"country": "BE", "lookup": ["tournai", "tournay"]},
    "berne":       {"country": "CH", "lookup": ["bern", "berne"]},
    "antwerp":     {"country": "BE"},
    "valencia":    {"country": "ES"},
    "kew":         {"country": "GB"},
    "gand":        {"country": "BE", "lookup": ["gent", "ghent", "gand"]},
    "salo":        {"country": "IT"},
    "surrey":      {"country": "GB", "prefer_class": "A"},
    "hanwell":     {"country": "GB"},
    "kensington":  {"country": "GB"},
    "hull":        {"country": "GB"},
    "bernal":      {"country": "AR"},
    "moravia":     {"country": "CZ", "prefer_class": "A"},
    "orleans":     {"country": "FR"},
    "aquila":      {"country": "IT", "lookup": ["l'aquila", "laquila", "l aquila", "aquila"]},
    "leith":       {"country": "GB"},
    "pesth":       {"country": "HU", "lookup": ["pest", "pesth"]},
    "leonardo":    {"country": "IT"},
    "geneva":      {"country": "CH"},
    "san remo":    {"country": "IT", "lookup": ["sanremo", "san remo"]},
    "cumberland":  {"country": "GB", "prefer_class": "A"},
    "lancaster":   {"country": "US", "state_hint": "PA"},
    "alva":        {"country": "GB"},
    "hanley":      {"country": "GB"},
    "franklin":    {"blank": True},
}


# ----------------------------- Main enrichment -----------------------------

def enrich_city_maker(
    df: pd.DataFrame,
    city_col: str = "city_maker",
    geonames_path: str = "Data/Geo_Data/allCountries.txt",
    admin1_path: str = "Data/Geo_Data/admin1CodesASCII.txt",
    admin2_path: str = "Data/Geo_Data/admin2Codes.txt",
    print_ambiguous: bool = True,
    top_k_print: int = 5,
    use_subregion_hint: bool = True,
    hint_bonus: float = 4.0,
) -> pd.DataFrame:
    """
    Adds:
      - country_iso1
      - admin1_name
      - admin2_name
      - candidate_count
      - is_ambiguous

    Uses subregion hint from parentheses or non-US comma suffix to filter/boost candidates.
    """
    out = df.copy()
    if city_col not in out.columns:
        raise ValueError(f"Column '{city_col}' not found.")

    # Prepare unique places
    uniques = pd.Series(out[city_col].dropna().unique())
    parsed = []
    for raw in uniques.tolist():
        core, sub_hint, st_hint, c_hint = parse_place(raw)
        core_key = norm_text(core)
        if core_key:
            parsed.append((raw, core, core_key, sub_hint, st_hint, c_hint))

    if not parsed:
        out["country_iso1"] = pd.NA
        out["admin1_name"] = pd.NA
        out["admin2_name"] = pd.NA
        out["candidate_count"] = pd.NA
        out["is_ambiguous"] = False
        return out

    places = pd.DataFrame(parsed, columns=["place_raw", "place_core", "place_key", "subregion_hint", "state_hint", "country_hint"])

    # Build target keys (include override lookup keys)
    target_keys = set(places["place_key"].unique())
    for _, spec in OVERRIDES.items():
        lookup = spec.get("lookup", None)
        if isinstance(lookup, str):
            target_keys.add(norm_text(lookup))
        elif isinstance(lookup, list):
            for name in lookup:
                target_keys.add(norm_text(name))

    admin1_map = load_admin1(admin1_path)
    admin2_map = load_admin2(admin2_path)
    cand_by_key = build_candidates_index(geonames_path, target_keys, include_admin_features=True)

    def admin_names_for(c: Candidate) -> Tuple[str, str]:
        a1 = admin1_map.get(f"{c.country}.{c.admin1}", "") if c.country and c.admin1 else ""
        a2 = admin2_map.get(f"{c.country}.{c.admin1}.{c.admin2}", "") if c.country and c.admin1 and c.admin2 else ""
        return a1, a2

    def hint_matches(c: Candidate, hint_norm: str) -> bool:
        if not hint_norm:
            return False
        a1, a2 = admin_names_for(c)
        return (hint_norm in norm_text(a1)) or (hint_norm in norm_text(a2))

    def score_candidate(c: Candidate, prefer_class: Optional[str], prefer_codes: Optional[List[str]], hint_norm: str) -> float:
        s = math.log1p(max(c.population, 0))
        if c.feature_class == "P":
            s += 2.0
        if prefer_class and c.feature_class == prefer_class:
            s += 3.0
        if prefer_codes and c.feature_code in set(prefer_codes):
            s += 2.0
        if use_subregion_hint and hint_norm and hint_matches(c, hint_norm):
            s += hint_bonus
        return s

    resolved_rows = []
    ambiguous_reports = []

    for _, r in places.iterrows():
        raw = r["place_raw"]
        core_key = r["place_key"]
        hint_norm = norm_text(r["subregion_hint"]) if (use_subregion_hint and r["subregion_hint"]) else ""

        spec = OVERRIDES.get(core_key, {})
        if spec.get("blank", False):
            resolved_rows.append({
                "place_raw": raw,
                "country_iso1": pd.NA,
                "admin1_name": pd.NA,
                "admin2_name": pd.NA,
                "candidate_count": 0,
                "is_ambiguous": False,
            })
            continue

        forced_country = spec.get("country", None) or r["country_hint"]
        forced_state   = spec.get("state_hint", None) or r["state_hint"]
        prefer_class   = spec.get("prefer_class", None)
        prefer_codes   = spec.get("prefer_codes", None)

        lookup_names = spec.get("lookup", None)
        if isinstance(lookup_names, str):
            lookup_keys = [norm_text(lookup_names)]
        elif isinstance(lookup_names, list):
            lookup_keys = [norm_text(x) for x in lookup_names]
        else:
            lookup_keys = [core_key]

        # gather candidates (dedupe)
        candidates = []
        seen = set()
        for lk in lookup_keys:
            for c in cand_by_key.get(lk, []):
                if c.geonameid not in seen:
                    seen.add(c.geonameid)
                    candidates.append(c)

        # forced country filter (hard)
        if forced_country:
            candidates = [c for c in candidates if c.country == forced_country]

        # forced US state filter when possible (hard)
        if forced_state:
            candidates2 = [c for c in candidates if c.country == "US" and c.admin1 == forced_state]
            if candidates2:
                candidates = candidates2

        # if forced country yields nothing: blank (your rule)
        if forced_country and not candidates:
            resolved_rows.append({
                "place_raw": raw,
                "country_iso1": pd.NA,
                "admin1_name": pd.NA,
                "admin2_name": pd.NA,
                "candidate_count": 0,
                "is_ambiguous": False,
            })
            continue

        # NEW: subregion hint narrowing
        if use_subregion_hint and hint_norm and len(candidates) > 1:
            matching = [c for c in candidates if hint_matches(c, hint_norm)]
            if matching:
                candidates = matching  # only narrow if it actually helps

        cand_count = len(candidates)
        is_amb = cand_count > 1

        if candidates:
            scored = [(score_candidate(c, prefer_class, prefer_codes, hint_norm), c) for c in candidates]
            scored.sort(key=lambda t: t[0], reverse=True)
            best = scored[0][1]

            country_iso1 = best.country or pd.NA
            a1_key = f"{best.country}.{best.admin1}" if best.country and best.admin1 else None
            admin1_name = admin1_map.get(a1_key, pd.NA) if a1_key else pd.NA
            a2_key = f"{best.country}.{best.admin1}.{best.admin2}" if best.country and best.admin1 and best.admin2 else None
            admin2_name = admin2_map.get(a2_key, pd.NA) if a2_key else pd.NA

            # Print remaining ambiguous cases (excluding overrides)
            if print_ambiguous and is_amb and core_key not in OVERRIDES:
                top = scored[:top_k_print]
                opts = []
                for s_val, c in top:
                    a1, a2 = admin_names_for(c)
                    opts.append((s_val, c.name, c.asciiname, c.country, a1, a2, c.feature_class, c.feature_code, c.population))
                ambiguous_reports.append((raw, cand_count, r["subregion_hint"], opts))
        else:
            country_iso1 = pd.NA
            admin1_name = pd.NA
            admin2_name = pd.NA

        resolved_rows.append({
            "place_raw": raw,
            "country_iso1": country_iso1,
            "admin1_name": admin1_name,
            "admin2_name": admin2_name,
            "candidate_count": cand_count,
            "is_ambiguous": bool(is_amb),
        })

    mapping = pd.DataFrame(resolved_rows).drop_duplicates("place_raw")
    out = out.merge(mapping, left_on=city_col, right_on="place_raw", how="left").drop(columns=["place_raw"])
    out["is_ambiguous"] = out["is_ambiguous"].fillna(False).astype(bool)

    if print_ambiguous and ambiguous_reports:
        print("\n--- Ambiguous place names (not covered by overrides) ---")
        for raw, n, hint, opts in ambiguous_reports:
            hint_txt = f" | hint='{hint}'" if hint else ""
            print(f"\nAMBIGUOUS: '{raw}' (candidates={n}{hint_txt})")
            for s_val, name, asciiname, cc, a1, a2, fcl, fcd, pop in opts:
                a1_txt = f" | admin1={a1}" if a1 else ""
                a2_txt = f" | admin2={a2}" if a2 else ""
                print(f"  score={s_val:0.3f} | {name} ({asciiname}) | {cc}{a1_txt}{a2_txt} | {fcl}/{fcd} | pop={pop}")

    return out

def fill_location_from_maker(
    df, method,  # "shared_level" or "mode"
    maker_col="maker_name",
    city_col="city_maker",
    country_col="country_iso1",
    admin1_col="admin1_name",
    admin2_col="admin2_name",
    filled_col="location_filled",
    only_when_city_and_location_missing=True,
    verbose=True,
):

    out = df.copy()

    # Ensure columns exist
    for c in [maker_col, city_col, country_col, admin1_col, admin2_col]:
        if c not in out.columns:
            out[c] = pd.NA

    # Treat empty strings as missing
    for c in [city_col, country_col, admin1_col, admin2_col]:
        out[c] = out[c].replace(r"^\s*$", pd.NA, regex=True)

    before = out[[city_col, country_col, admin1_col, admin2_col]].copy()

    def _mode_nonnull(s):
        s2 = s.dropna()
        return pd.NA if s2.empty else s2.value_counts().idxmax()

    def _shared_or_na(s):
        s2 = s.dropna()
        u = pd.unique(s2)
        return u[0] if len(u) == 1 else pd.NA

    # Eligibility mask
    loc_all_missing = out[[country_col, admin1_col, admin2_col]].isna().all(axis=1)
    city_missing = out[city_col].isna()

    if only_when_city_and_location_missing:
        eligible = city_missing & loc_all_missing & out[maker_col].notna()
    else:
        eligible = (city_missing | loc_all_missing) & out[maker_col].notna()

    method = str(method).strip().casefold()
    if method not in {"shared_level", "mode"}:
        raise ValueError("method must be 'shared_level' or 'mode'")

    if method == "mode":
        # Per maker, compute independent modes for each column
        maker_fill = (
            out[out[maker_col].notna()]
            .groupby(maker_col, sort=False)
            .agg(
                _fill_city=(city_col, _mode_nonnull),
                _fill_country=(country_col, _mode_nonnull),
                _fill_admin1=(admin1_col, _mode_nonnull),
                _fill_admin2=(admin2_col, _mode_nonnull),
            )
            .reset_index()
        )

        out = out.merge(maker_fill, on=maker_col, how="left")

        # Fill NaNs only
        out.loc[eligible, country_col] = out.loc[eligible, country_col].fillna(out.loc[eligible, "_fill_country"])
        out.loc[eligible, admin1_col]  = out.loc[eligible, admin1_col].fillna(out.loc[eligible, "_fill_admin1"])
        out.loc[eligible, admin2_col]  = out.loc[eligible, admin2_col].fillna(out.loc[eligible, "_fill_admin2"])
        out.loc[eligible, city_col]    = out.loc[eligible, city_col].fillna(out.loc[eligible, "_fill_city"])

        out = out.drop(columns=["_fill_city", "_fill_country", "_fill_admin1", "_fill_admin2"], errors="ignore")

    else:
        # shared_level method
        out["_city_key"] = out[city_col].map(lambda x: norm_text(x) if pd.notna(x) else pd.NA)

        def _maker_fill_rule(g):
            city_keys = pd.unique(g["_city_key"].dropna())

            # Single-city maker: fill from that city's rows (mode within that city)
            if len(city_keys) == 1:
                key = city_keys[0]
                gg = g[g["_city_key"] == key]
                return pd.Series(
                    {
                        "_fill_city": _mode_nonnull(gg[city_col]),
                        "_fill_country": _mode_nonnull(gg[country_col]),
                        "_fill_admin1": _mode_nonnull(gg[admin1_col]),
                        "_fill_admin2": _mode_nonnull(gg[admin2_col]),
                        "_single_city": True,
                    }
                )

            # Multi-city maker: fill only highest shared admin level across all rows
            fill_country = _shared_or_na(g[country_col])
            fill_admin1 = _shared_or_na(g[admin1_col]) if pd.notna(fill_country) else pd.NA
            fill_admin2 = _shared_or_na(g[admin2_col]) if pd.notna(fill_admin1) else pd.NA

            return pd.Series(
                {
                    "_fill_city": pd.NA,
                    "_fill_country": fill_country,
                    "_fill_admin1": fill_admin1,
                    "_fill_admin2": fill_admin2,
                    "_single_city": False,
                }
            )

        maker_fill = (
            out[out[maker_col].notna()]
            .groupby(maker_col, sort=False)
            .apply(_maker_fill_rule)
            .reset_index()
        )

        out = out.merge(maker_fill, on=maker_col, how="left")

        out.loc[eligible, country_col] = out.loc[eligible, country_col].fillna(out.loc[eligible, "_fill_country"])
        out.loc[eligible, admin1_col]  = out.loc[eligible, admin1_col].fillna(out.loc[eligible, "_fill_admin1"])
        out.loc[eligible, admin2_col]  = out.loc[eligible, admin2_col].fillna(out.loc[eligible, "_fill_admin2"])

        # Only fill city for single-city makers
        city_fill_mask = eligible & out[city_col].isna() & out["_single_city"].fillna(False)
        out.loc[city_fill_mask, city_col] = out.loc[city_fill_mask, "_fill_city"]

        out = out.drop(columns=["_city_key", "_fill_city", "_fill_country", "_fill_admin1", "_fill_admin2", "_single_city"], errors="ignore")

    # Filled flag
    after = out[[city_col, country_col, admin1_col, admin2_col]]
    sentinel = "__NA__"
    out[filled_col] = (before.fillna(sentinel) != after.fillna(sentinel)).any(axis=1)

    if verbose:
        n = int(out[filled_col].sum())
        m = out.loc[out[filled_col], maker_col].nunique(dropna=True)
        print(f"Filled location for {n} rows across {m} makers (method='{method}').")

    return out

def fill_location_fallback(
    df,
    city_col="city_maker",
    admin2_col="admin2_name",
    admin1_col="admin1_name",
    country_col="country_iso1",
    level_col="highest_level_known",
    best_value_col="location_best",
):
    out = df.copy()

    # Ensure columns exist
    for c in [city_col, admin2_col, admin1_col, country_col]:
        if c not in out.columns:
            out[c] = pd.NA

    # Treat empty strings as missing
    for c in [city_col, admin2_col, admin1_col, country_col]:
        out[c] = out[c].replace(r"^\s*$", pd.NA, regex=True)

    has_city = out[city_col].notna()
    has_a2   = out[admin2_col].notna()
    has_a1   = out[admin1_col].notna()
    has_ctry = out[country_col].notna()

    # Most specific level known
    out[level_col] = "none"
    out.loc[has_ctry, level_col] = "country"
    out.loc[has_a1,   level_col] = "admin1"
    out.loc[has_a2,   level_col] = "admin2"
    out.loc[has_city, level_col] = "city"

    # Best available value (single column)
    out[best_value_col] = out[city_col]
    out[best_value_col] = out[best_value_col].fillna(out[admin2_col])
    out[best_value_col] = out[best_value_col].fillna(out[admin1_col])
    out[best_value_col] = out[best_value_col].fillna(out[country_col])

    # Fill missing fields with next coarser level (no overwrites)
    out[admin1_col] = out[admin1_col].fillna(out[country_col])
    out[admin2_col] = out[admin2_col].fillna(out[admin1_col])
    out[city_col]   = out[city_col].fillna(out[admin2_col])

    return out

def fit_maker_location_rules(
    train_df,
    maker_col="maker_name",
    city_col="city_maker",
    country_col="country_iso1",
    admin1_col="admin1_name",
    admin2_col="admin2_name",
    method="shared_level",   # "shared_level" or "mode"
):
    """
    Fit maker-level location fill rules using TRAIN ONLY.
    Returns a dataframe keyed by maker_col with columns:
      _fill_city, _fill_country, _fill_admin1, _fill_admin2, _single_city
    Assumes norm_text exists.
    """
    t = train_df.copy()

    for c in [maker_col, city_col, country_col, admin1_col, admin2_col]:
        if c not in t.columns:
            t[c] = pd.NA

    for c in [city_col, country_col, admin1_col, admin2_col]:
        t[c] = t[c].replace(r"^\s*$", pd.NA, regex=True)

    def _mode_nonnull(s):
        s2 = s.dropna()
        return pd.NA if s2.empty else s2.value_counts().idxmax()

    def _shared_or_na(s):
        s2 = s.dropna()
        u = pd.unique(s2)
        return u[0] if len(u) == 1 else pd.NA

    method = str(method).strip().casefold()
    if method not in {"shared_level", "mode"}:
        raise ValueError("method must be 'shared_level' or 'mode'")

    if method == "mode":
        rules = (
            t[t[maker_col].notna()]
            .groupby(maker_col, sort=False)
            .agg(
                _fill_city=(city_col, _mode_nonnull),
                _fill_country=(country_col, _mode_nonnull),
                _fill_admin1=(admin1_col, _mode_nonnull),
                _fill_admin2=(admin2_col, _mode_nonnull),
            )
            .reset_index()
        )
        rules["_single_city"] = True  # in mode method we allow filling city directly
        return rules

    # shared_level
    t["_city_key"] = t[city_col].map(lambda x: norm_text(x) if pd.notna(x) else pd.NA)

    def _maker_rule(g):
        city_keys = pd.unique(g["_city_key"].dropna())

        if len(city_keys) == 1:
            key = city_keys[0]
            gg = g[g["_city_key"] == key]
            return pd.Series(
                {
                    "_fill_city": _mode_nonnull(gg[city_col]),
                    "_fill_country": _mode_nonnull(gg[country_col]),
                    "_fill_admin1": _mode_nonnull(gg[admin1_col]),
                    "_fill_admin2": _mode_nonnull(gg[admin2_col]),
                    "_single_city": True,
                }
            )

        fill_country = _shared_or_na(g[country_col])
        fill_admin1 = _shared_or_na(g[admin1_col]) if pd.notna(fill_country) else pd.NA
        fill_admin2 = _shared_or_na(g[admin2_col]) if pd.notna(fill_admin1) else pd.NA

        return pd.Series(
            {
                "_fill_city": pd.NA,
                "_fill_country": fill_country,
                "_fill_admin1": fill_admin1,
                "_fill_admin2": fill_admin2,
                "_single_city": False,
            }
        )

    rules = (
        t[t[maker_col].notna()]
        .groupby(maker_col, sort=False)
        .apply(_maker_rule)
        .reset_index()
    )
    return rules.drop(columns=["_city_key"], errors="ignore")


def apply_maker_location_rules(
    df,
    rules_df,
    maker_col="maker_name",
    city_col="city_maker",
    country_col="country_iso1",
    admin1_col="admin1_name",
    admin2_col="admin2_name",
    filled_col="location_filled",
    only_when_city_and_location_missing=True,
):
    """
    Apply pre-fit maker rules to any dataframe (train/val/test) without refitting.
    Only fills NaNs; does not overwrite existing values.
    Adds filled_col indicating if any location field was imputed.
    """
    out = df.copy()

    for c in [maker_col, city_col, country_col, admin1_col, admin2_col]:
        if c not in out.columns:
            out[c] = pd.NA
    for c in [city_col, country_col, admin1_col, admin2_col]:
        out[c] = out[c].replace(r"^\s*$", pd.NA, regex=True)

    before = out[[city_col, country_col, admin1_col, admin2_col]].copy()

    out = out.merge(rules_df, on=maker_col, how="left")

    loc_all_missing = out[[country_col, admin1_col, admin2_col]].isna().all(axis=1)
    city_missing = out[city_col].isna()

    if only_when_city_and_location_missing:
        eligible = city_missing & loc_all_missing & out[maker_col].notna()
    else:
        eligible = (city_missing | loc_all_missing) & out[maker_col].notna()

    out.loc[eligible, country_col] = out.loc[eligible, country_col].fillna(out.loc[eligible, "_fill_country"])
    out.loc[eligible, admin1_col]  = out.loc[eligible, admin1_col].fillna(out.loc[eligible, "_fill_admin1"])
    out.loc[eligible, admin2_col]  = out.loc[eligible, admin2_col].fillna(out.loc[eligible, "_fill_admin2"])

    city_fill_mask = eligible & out[city_col].isna() & out["_single_city"].fillna(False)
    out.loc[city_fill_mask, city_col] = out.loc[city_fill_mask, "_fill_city"]

    cols = [city_col, country_col, admin1_col, admin2_col]
    sentinel = "__NA__"

    before_arr = before.reindex(columns=cols).fillna(sentinel).to_numpy()
    after_arr  = out[cols].fillna(sentinel).to_numpy()

    out[filled_col] = (before_arr != after_arr).any(axis=1)

    out = out.drop(columns=["_fill_city", "_fill_country", "_fill_admin1", "_fill_admin2", "_single_city"], errors="ignore")
    return out