#!/usr/bin/env python3
import argparse, csv, sys, time, requests
from typing import Iterable, List, Dict, Optional, Tuple
try:
    from Bio import Entrez
except Exception:
    Entrez = None

IDCONV_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
ELINK_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
UA = "pmid2pmcid-cli/1.2 (+https://example.org)"

# -------------------- IO --------------------
def read_study_pmids_from_csv(path: str) -> List[Dict[str, str]]:
    """
    Expect a CSV with at least: studyId, pmid
    Falls back to any column named pmid/PMID/id if studyId is missing.
    Returns list of dicts: {'studyId': <str or ''>, 'pmid': <str>}
    """
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("No header row found in CSV.")

        # Discover columns
        study_col = None
        if "studyId" in reader.fieldnames:
            study_col = "studyId"

        pmid_col = None
        for c in ("pmid", "PMID", "id", "Id", "ID"):
            if c in reader.fieldnames:
                pmid_col = c
                break
        if not pmid_col:
            raise ValueError(f"No pmid-like column in {reader.fieldnames}")

        for row in reader:
            pmid = (row.get(pmid_col) or "").strip()
            if not pmid:
                continue
            study = (row.get(study_col) or "").strip() if study_col else ""
            rows.append({"studyId": study, "pmid": pmid})
    return rows

def normalize_pmid(p: str) -> str:
    p = str(p).strip()
    if not p:
        return ""
    if p.lower().startswith("pmid"):
        p = "".join(ch for ch in p if ch.isdigit())
    return p

def unique_pmids(rows: List[Dict[str, str]]) -> List[str]:
    seen = set()
    out = []
    for r in rows:
        p = normalize_pmid(r["pmid"])
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out

# -------------------- NIH resolvers --------------------
def idconv_batch(pmids: List[str], email: Optional[str], verbose: bool) -> Dict[str, Dict]:
    """Return mapping {pmid(str): {pmid, pmcid, doi, status, errmsg}} for a batch."""
    params = {
        "ids": ",".join(pmids),
        "format": "json",
        "tool": "pmid2pmcid-cli",
    }
    if email:
        params["email"] = email

    r = requests.get(IDCONV_URL, params=params, timeout=60,
                     headers={"User-Agent": UA})
    r.raise_for_status()
    j = r.json()
    if verbose:
        print("[idconv] records:", len(j.get("records", [])))

    out: Dict[str, Dict] = {}
    for rec in j.get("records", []):
        # Force string key so lookup matches normalized inputs
        pmid = str(rec.get("pmid") or rec.get("requested-id") or "").strip()
        out[pmid] = {
            "pmid": pmid,
            "pmcid": rec.get("pmcid") or "",
            "doi": rec.get("doi") or "",
            "status": rec.get("status") or "ok",
            "errmsg": rec.get("errmsg") or "",
        }

    # Ensure every input pmid has an entry (keys are strings)
    for p in pmids:
        ps = str(p).strip()
        out.setdefault(ps, {"pmid": ps, "pmcid": "", "doi": "", "status": "", "errmsg": ""})
    return out

def resolve_idconv_all(pmids: List[str], email: Optional[str], sleep=0.34, verbose=False) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    B = 200  # NIH allows up to 200 per request
    for i in range(0, len(pmids), B):
        batch = pmids[i:i+B]
        m = idconv_batch(batch, email=email, verbose=verbose)
        out.update(m)
        time.sleep(sleep)
    return out

def elink_pubmed_to_pmc(pmid: str, email: Optional[str], api_key: Optional[str]) -> str:
    if Entrez:
        if not email:
            raise ValueError("E-utilities elink requires --email when Biopython is installed.")
        Entrez.email = email
        if api_key:
            Entrez.api_key = api_key
        h = Entrez.elink(dbfrom="pubmed", db="pmc", id=pmid, retmode="xml")
        recs = Entrez.read(h); h.close()
        try:
            links = recs[0]["LinkSetDb"][0]["Link"]
            if links:
                return "PMC" + links[0]["Id"]
        except Exception:
            return ""
        return ""
    else:
        params = {"dbfrom": "pubmed", "db": "pmc", "id": pmid, "retmode": "json", "tool": "pmid2pmcid-cli"}
        if email:
            params["email"] = email
        r = requests.get(ELINK_URL, params=params, timeout=30, headers={"User-Agent": UA})
        r.raise_for_status()
        j = r.json()
        try:
            links = j["linksets"][0]["linksetdbs"][0]["links"]
            if links:
                return "PMC" + str(links[0])
        except Exception:
            return ""
        return ""

def resolve_pmids(pmids: List[str], email: Optional[str], force_elink: bool,
                  fallback: bool, api_key: Optional[str], verbose: bool) -> Dict[str, Dict]:
    """Return mapping pmid(str) -> resolved fields."""
    pmids = [normalize_pmid(p) for p in pmids if normalize_pmid(p)]
    mapping: Dict[str, Dict] = {p: {"pmid": p, "pmcid": "", "doi": "", "status": "", "errmsg": ""} for p in pmids}

    if not force_elink:
        idc = resolve_idconv_all(pmids, email=email, verbose=verbose)
        mapping.update(idc)

    if force_elink or fallback:
        for p in pmids:
            need = force_elink or not mapping[p]["pmcid"]
            if need:
                try:
                    pmcid = elink_pubmed_to_pmc(p, email=email, api_key=api_key)
                except Exception as e:
                    pmcid = ""
                    if verbose:
                        print(f"[elink] {p} error: {e}")
                if pmcid:
                    mapping[p]["pmcid"] = pmcid
                    mapping[p]["status"] = (mapping[p]["status"] + ";elink").strip(";")

    return mapping

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser(description="Convert PMID → PMCID/DOI and keep studyId in the output when provided.")
    g_in = ap.add_mutually_exclusive_group(required=True)
    g_in.add_argument("--pmids", help="Comma-separated PMIDs, e.g. 29625048,37261122 (no studyId)")
    g_in.add_argument("--in-csv", help="CSV with columns 'studyId' and 'pmid' (at minimum)")
    ap.add_argument("--out", default="pmid_to_pmcid.csv", help="Output CSV path")
    ap.add_argument("--email", help="Your email (recommended; passed to NIH)")
    ap.add_argument("--api-key", help="NCBI API key (optional)")
    ap.add_argument("--fallback-elink", action="store_true", help="If ID Converter has no PMCID, try E-utilities")
    ap.add_argument("--force-elink", action="store_true", help="Use E-utilities for ALL IDs (skip ID Converter)")
    ap.add_argument("--verbose", action="store_true", help="Print raw API info summary")
    args = ap.parse_args()

    # Build an input list with optional studyId
    if args.pmids:
        input_rows = [{"studyId": "", "pmid": p.strip()} for p in args.pmids.split(",") if p.strip()]
    else:
        input_rows = read_study_pmids_from_csv(args.in_csv)

    if not input_rows:
        print("No input rows found.", file=sys.stderr)
        sys.exit(1)

    # Resolve unique PMIDs once
    pmids = unique_pmids(input_rows)
    pmid_map = resolve_pmids(pmids, email=args.email, force_elink=args.force_elink,
                             fallback=args.fallback_elink, api_key=args.api_key, verbose=args.verbose)

    # Re-expand to one row per studyId from input
    out_rows = []
    for r in input_rows:
        p = normalize_pmid(r["pmid"])
        res = pmid_map.get(p, {"pmcid": "", "doi": "", "status": "", "errmsg": ""})
        out_rows.append({
            "studyId": r.get("studyId", ""),
            "pmid": p,
            "pmcid": res.get("pmcid", ""),
            "doi": res.get("doi", ""),
            "status": res.get("status", ""),
            "errmsg": res.get("errmsg", ""),
        })

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["studyId", "pmid", "pmcid", "doi", "status", "errmsg"])
        w.writeheader()
        w.writerows(out_rows)

    print(f"[✓] Wrote {len(out_rows)} rows to {args.out}")

if __name__ == "__main__":
    main()
