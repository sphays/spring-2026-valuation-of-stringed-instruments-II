#!/usr/bin/env python3
import argparse
import re
import unicodedata
from typing import List, Tuple, Optional
import pandas as pd

# ---------- Utils sûrs ----------
def safe_str(x) -> str:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)) or pd.isna(x):
            return ""
    except Exception:
        if x is None:
            return ""
    return str(x)

def _norm(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower().strip()

def split_list(cell: str) -> List[str]:
    s = safe_str(cell).strip()
    if not s:
        return []
    return [x.strip() for x in s.split(";") if x.strip()]

def normalize_type(s: str) -> str:
    return safe_str(s).strip().lower()

def extract_year4(s: str) -> Optional[int]:
    if not s:
        return None
    m = re.search(r"\b(1[6-9]\d{2}|20\d{2})\b", str(s))
    return int(m.group(1)) if m else None

# ---------- Maisons ----------
CANON_HOUSES = [
    "Tarisio","Bongartz's","Sotheby's","Bonhams","Phillip's","Christie's","Gardiner-Houlgate",
    "Vichy-Enchères","Ingles & Hayday","Guy Laurent","da Salo Auction","Skinner","Claude Aguttes",
    "Ader Tajan","T2 Auctions","Brompton's","Dorotheum","Puttick & Simpson","Sotheby Parke Bernet",
    "Freeman's Auctions","Hôtel Drouot","Palais Galliera","Millon & Associés (Gilles Chancereul)",
    "Christie & Manson","Kestenbaum & Co.","Tepper Galleries","J & A Beare Auctions","Tajan",
    "Butterfield & Butterfield","Rubinacci Casa d'Aste","Drouot-Richelieu","Casa d'Aste Babuino",
    "Delorme & Collin du Bocage","Adam Partridge Auctioneers","Neuilly St. James","Leseuil & Rambert",
    "Anderson Galleries Inc.","Foster Auction House","Glendining & Co.","Rambert Michel"
]

VARIANT_MAP = {
    "t2": "T2 Auctions", "t2 auctions": "T2 Auctions",
    "sotheby": "Sotheby's", "sothebys": "Sotheby's", "sotheby’s": "Sotheby's",
    "sotheby parke bernet": "Sotheby Parke Bernet",
    "christie": "Christie's", "christies": "Christie's",
    "christie & manson": "Christie & Manson",
    "bonhams": "Bonhams",
    "brompton": "Brompton's", "brompton's": "Brompton's",
    "gardiner houlgate": "Gardiner-Houlgate", "gardiner-houlgate": "Gardiner-Houlgate",
    "vichy encheres": "Vichy-Enchères", "vichy enchères": "Vichy-Enchères", "vichy-encheres": "Vichy-Enchères",
    "ingles & hayday": "Ingles & Hayday", "ingles and hayday": "Ingles & Hayday",
    "guy laurent": "Guy Laurent",
    "da salo auction": "da Salo Auction",
    "skinner": "Skinner",
    "claude aguttes": "Claude Aguttes", "aguttes": "Claude Aguttes",
    "ader tajan": "Ader Tajan", "tajan": "Tajan",
    "bongartz": "Bongartz's", "bongartz's": "Bongartz's",
    "puttick & simpson": "Puttick & Simpson", "puttick and simpson": "Puttick & Simpson",
    "freeman's auctions": "Freeman's Auctions", "freemans auctions": "Freeman's Auctions",
    "freeman's": "Freeman's Auctions", "freemans": "Freeman's Auctions",
    "hotel drouot": "Hôtel Drouot", "hôtel drouot": "Hôtel Drouot",
    "palais galliera": "Palais Galliera",
    "millon & associes (gilles chancereul)": "Millon & Associés (Gilles Chancereul)",
    "millon & associés (gilles chancereul)": "Millon & Associés (Gilles Chancereul)",
    "kestenbaum & co.": "Kestenbaum & Co.", "kestenbaum and co.": "Kestenbaum & Co.",
    "tepper galleries": "Tepper Galleries",
    "j & a beare auctions": "J & A Beare Auctions", "j and a beare auctions": "J & A Beare Auctions",
    "butterfield & butterfield": "Butterfield & Butterfield", "butterfield and butterfield": "Butterfield & Butterfield",
    "rubinacci casa d'aste": "Rubinacci Casa d'Aste",
    "drouot-richelieu": "Drouot-Richelieu",
    "casa d'aste babuino": "Casa d'Aste Babuino",
    "delorme & collin du bocage": "Delorme & Collin du Bocage",
    "adam partridge auctioneers": "Adam Partridge Auctioneers",
    "neuilly st. james": "Neuilly St. James",
    "leseuil & rambert": "Leseuil & Rambert",
    "anderson galleries inc.": "Anderson Galleries Inc.",
    "foster auction house": "Foster Auction House",
    "glendining & co.": "Glendining & Co.", "glendining and co.": "Glendining & Co.",
    "rambert michel": "Rambert Michel",
    "tarisio": "Tarisio",
    "phillip's": "Phillip's", "phillips": "Phillip's",
    # Pont demandé
    "tarisio private sales": "Tarisio", "tarisio private sale": "Tarisio"
}

def canonicalize_house(name: str) -> str:
    pn = _norm(name or "")
    if pn in VARIANT_MAP:
        return VARIANT_MAP[pn]
    for h in CANON_HOUSES:
        if _norm(h) == pn:
            return h
    return (name or "").strip()

def houses_match(a: str, b: str) -> bool:
    return _norm(canonicalize_house(a)) == _norm(canonicalize_house(b))

# ---------- Dates ----------
MONTHS_FULL = {
    "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
    "july":7,"august":8,"september":9,"october":10,"november":11,"december":12
}
MONTHS_ABBR = {
    "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
    "jul":7,"aug":8,"sep":9,"sept":9,"oct":10,"nov":11,"dec":12
}

def parse_sale_date(s: str) -> Optional[Tuple[int,int,int]]:
    s = safe_str(s).strip()
    if not s:
        return None
    m = re.match(r"^\s*([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})\s*$", s)
    if not m:
        return None
    mon_s, day_s, year_s = m.group(1), m.group(2), m.group(3)
    mon = MONTHS_ABBR.get(mon_s.lower()) or MONTHS_FULL.get(mon_s.lower())
    if not mon:
        return None
    return (int(year_s), int(mon), int(day_s))

MONTHS_RE = r"(January|February|March|April|May|June|July|August|September|October|November|December)"
PAT_SINGLE = re.compile(rf"\b{MONTHS_RE}\s+\d{{1,2}},\s*\d{{4}}\b", re.I)
PAT_RANGE = re.compile(rf"\b{MONTHS_RE}\s+\d{{1,2}}\s*(?:-|–|—|to)\s*\d{{1,2}},\s*\d{{4}}\b", re.I)
PAT_MULTI = re.compile(rf"\b{MONTHS_RE}\s+\d{{1,2}}(?:\s*,\s*\d{{1,2}})*(?:\s*&\s*\d{{1,2}})?\s*,\s*\d{{4}}\b", re.I)
PAT_MONTH_YEAR = re.compile(rf"\b{MONTHS_RE}\s+\d{{4}}\b", re.I)

def month_name_to_int(mon: str) -> Optional[int]:
    return MONTHS_FULL.get(mon.lower())

def parse_catalog_date_entry(s: str) -> Optional[Tuple[int,int,Optional[List[int]]]]:
    s = safe_str(s).strip()
    if not s:
        return None
    m = PAT_RANGE.search(s)
    if m:
        mon = month_name_to_int(m.group(1))
        nums = [int(x) for x in re.findall(r"\d+", s)]
        if len(nums) >= 3:
            d1, d2, y = nums[-3], nums[-2], nums[-1]
            return (y, mon, list(range(min(d1,d2), max(d1,d2)+1)))
    m = PAT_MULTI.search(s)
    if m:
        mon = month_name_to_int(m.group(1))
        nums = [int(x) for x in re.findall(r"\d+", s)]
        if len(nums) >= 2:
            y = nums[-1]
            days = nums[:-1]
            return (y, mon, days)
    m = PAT_SINGLE.search(s)
    if m:
        mon = month_name_to_int(m.group(1))
        nums = [int(x) for x in re.findall(r"\d+", s)]
        if len(nums) == 2:
            day, y = nums[0], nums[1]
            return (y, mon, [day])
    m = PAT_MONTH_YEAR.search(s)
    if m:
        mon = month_name_to_int(m.group(1))
        y = int(re.findall(r"\d{4}", s)[-1])
        return (y, mon, None)
    return None

def sale_matches_catalog_date(sale: Tuple[int,int,int], cat: Tuple[int,int,Optional[List[int]]]) -> bool:
    sy, sm, sd = sale
    cy, cm, cdays = cat
    if sy != cy or sm != cm:
        return False
    if cdays is None:
        return True
    return sd in set(cdays)

# ---------- Ponts de types (seulement) ----------
RARE_TYPES = {
    normalize_type("Small Violin"),
    normalize_type("Bass Bow"),
    normalize_type("Bass"),
    normalize_type("Tenor Viol"),
    normalize_type("Bass Viol"),
    normalize_type("Treble Viol"),
    normalize_type("Miscellaneous"),
    normalize_type("Viola d'Amore")
}

def types_compatible(sales_type: str, instr_type: str) -> bool:
    st = normalize_type(sales_type)
    it = normalize_type(instr_type)
    if not st or not it:
        return False
    if it == st:
        return True
    if it == "other" and st in RARE_TYPES:
        return True
    if it == "bow" and "bow" in st:
        return True
    return False

# ---------- Index instruments (par maker uniquement) ----------
def build_instrument_index(df_ins: pd.DataFrame):
    groups = {}
    for _, row in df_ins.iterrows():
        maker_id = safe_str(row.get("maker_id")).strip()
        itype = normalize_type(safe_str(row.get("type")))
        inst_id = safe_str(row.get("instrument_id")).strip()
        if not maker_id or not itype or not inst_id:
            continue

        cat_dates_raw = split_list(safe_str(row.get("auction_catalog_date")))
        cat_houses_raw = split_list(safe_str(row.get("auction_catalog_house")))
        cat_dates_parsed = [parse_catalog_date_entry(x) for x in cat_dates_raw]

        year_sale = split_list(safe_str(row.get("year_sale")))
        house_sale = split_list(safe_str(row.get("house_sale")))

        entry = {
            "instrument_id": inst_id,
            "maker_id": maker_id,
            "type_norm": itype,
            "cat_dates_raw": cat_dates_raw,
            "cat_houses_raw": cat_houses_raw,
            "cat_dates_parsed": cat_dates_parsed,
            "year_sale": year_sale,
            "house_sale": house_sale,
        }
        groups.setdefault(maker_id, []).append(entry)
    return groups

# ---------- Matching strict (avec ponts de types) ----------
def match_sale_to_instruments_strict(row, candidates):
    sale_house = canonicalize_house(safe_str(row.get("auction_house")).strip())
    sale_date_str = safe_str(row.get("sale_date")).strip()
    sale_dt = parse_sale_date(sale_date_str)
    sale_year = sale_dt[0] if sale_dt else None
    sale_type = safe_str(row.get("type"))

    matches = []

    for inst in candidates:
        # Filtre type (avec ponts)
        if not types_compatible(sale_type, inst["type_norm"]):
            continue

        # 1) Catalogue strict (même index date_i + house_i)
        cat_dates = inst["cat_dates_parsed"]
        cat_houses = inst["cat_houses_raw"]
        if sale_dt and cat_dates and cat_houses:
            n = min(len(cat_dates), len(cat_houses))
            for i in range(n):
                cdate = cat_dates[i]
                chouse = cat_houses[i]
                if not chouse or not cdate:
                    continue
                if not houses_match(chouse, sale_house):
                    continue
                if sale_matches_catalog_date(sale_dt, cdate):
                    matches.append(inst["instrument_id"])
                    break  # cet instrument est déjà un match

        # 2) Provenance strict (même index année_i + house_i)
        ys = inst["year_sale"]
        hs = inst["house_sale"]
        if sale_year is not None and ys and hs:
            n2 = min(len(ys), len(hs))
            for j in range(n2):
                y4 = extract_year4(ys[j])
                h = hs[j]
                if y4 is None or not h:
                    continue
                if y4 == sale_year and houses_match(h, sale_house):
                    matches.append(inst["instrument_id"])
                    break

    # Déduplication
    matches = list(dict.fromkeys(matches))
    return matches

# ---------- Pipeline ----------
def main():
    ap = argparse.ArgumentParser(description="Relie sales.instrument_id en restant strict, avec ponts de type: instruments 'other' → types rares; instruments 'bow' → tout type 'Bow'.")
    ap.add_argument("--sales-csv", required=True)
    ap.add_argument("--instruments-csv", required=True)
    ap.add_argument("--out-sales-csv", required=True)
    ap.add_argument("--ambiguous-report-csv", required=True)
    args = ap.parse_args()

    df_sales = pd.read_csv(args.sales_csv, dtype=str, keep_default_na=False, na_filter=False)
    df_ins = pd.read_csv(args.instruments_csv, dtype=str, keep_default_na=False, na_filter=False)

    if "instrument_id" not in df_sales.columns:
        df_sales["instrument_id"] = ""

    idx = build_instrument_index(df_ins)

    amb_rows = []
    filled = 0
    total = len(df_sales)

    for ix, row in df_sales.iterrows():
        if safe_str(row.get("instrument_id")).strip():
            continue

        maker_id = safe_str(row.get("maker_id")).strip()
        if not maker_id:
            continue

        # Préconditions mini: type, maison, date
        if not safe_str(row.get("type")).strip():
            continue
        if not safe_str(row.get("auction_house")).strip():
            continue
        if not safe_str(row.get("sale_date")).strip():
            continue

        candidates = idx.get(maker_id, [])
        if not candidates:
            continue

        matches = match_sale_to_instruments_strict(row, candidates)
        if len(matches) == 1:
            df_sales.at[ix, "instrument_id"] = matches[0]
            filled += 1
        elif len(matches) > 1:
            amb_rows.append({
                "row_index": ix,
                "maker_id": maker_id,
                "type": safe_str(row.get("type")),
                "auction_house": safe_str(row.get("auction_house")),
                "sale_date": safe_str(row.get("sale_date")),
                "candidate_instrument_ids": "; ".join(matches)
            })

        if (ix+1) % 500 == 0 or (ix+1) == total:
            print(f"Traitement ventes: {ix+1}/{total} (remplies: {filled})")

    df_sales.to_csv(args.out_sales_csv, index=False, encoding="utf-8-sig")
    print(f"OK: écrit {args.out_sales_csv} (ventes liées: {filled})")

    if amb_rows:
        pd.DataFrame(amb_rows).to_csv(args.ambiguous_report_csv, index=False, encoding="utf-8-sig")
        print(f"Ambiguïtés: {len(amb_rows)} → {args.ambiguous_report_csv}")
    else:
        pd.DataFrame(columns=["row_index","maker_id","type","auction_house","sale_date","candidate_instrument_ids"]) \
          .to_csv(args.ambiguous_report_csv, index=False, encoding="utf-8-sig")
        print("Aucune vente ambiguë.")

if __name__ == "__main__":
    main()