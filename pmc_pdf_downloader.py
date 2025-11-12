#!/usr/bin/env python3
"""
Download NIH/PMC PDFs in batches from a CSV produced by pmid2pmcid.py.

Input CSV schema (minimum):
  pmid,pmcid,doi,status,errmsg
Only rows with a non-empty PMCID are attempted (NIH/PMC full text).

Features
- Batch processing with per-batch delay
- Concurrency (threaded) with polite throttling
- Robust URL strategy (handles /pdf/ and /pdf/<PMCID>.pdf)
- Retries with backoff for 429/5xx
- Resume: skips already-downloaded files unless --overwrite
- Manifest CSV of successes/failures

Examples
  python pmc_pdf_downloader.py --in pmid_to_pmcid.csv --out-dir ./pmc_pdfs \
      --batch-size 40 --concurrency 4 --delay 1.0 --email you@org.edu

  # Overwrite existing PDFs and be extra slow/polite
  python pmc_pdf_downloader.py --in map.csv --out-dir ./pmc_pdfs --overwrite \
      --batch-size 20 --concurrency 2 --delay 2.0
"""
from __future__ import annotations

import argparse
import csv
import os
import time
import sys
import math
import re
import pathlib
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

UA = "pmc-pdf-downloader/1.0 (+https://example.org)"
PMC_HOSTS = [
    "https://www.ncbi.nlm.nih.gov",
    "https://pmc.ncbi.nlm.nih.gov",
]
# Strategies we try in order for each host
PMC_PATH_PATTERNS = [
    "/pmc/articles/{pmcid}/pdf/",             # canonical "directory" that serves main PDF
    "/pmc/articles/{pmcid}/pdf/{pmcid}.pdf",  # explicit filename
]

def read_rows(csv_path: str) -> List[Dict[str, str]]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        if not rd.fieldnames:
            raise ValueError("CSV appears to have no header row.")
        for r in rd:
            rows.append({k: (v or "").strip() for k, v in r.items()})
    return rows

def valid_pmcid(pmcid: str) -> bool:
    # Accept forms like "PMC12345" (case-insensitive)
    return bool(re.fullmatch(r"(?i)PMC\d+", pmcid or ""))

def ensure_dir(path: str):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._\-]+", "_", name)

def pick_filename(pmcid: str, pmid: str = "", doi: str = "") -> str:
    base = pmcid.upper() if pmcid else (pmid or "UNKNOWN")
    return sanitize_filename(f"{base}.pdf")

def stream_download(url: str, dest_path: str, timeout: int, session: requests.Session) -> Tuple[bool, str]:
    """Stream to disk; returns (ok, message)."""
    with session.get(url, stream=True, timeout=timeout) as r:
        if r.status_code != 200 or "application/pdf" not in r.headers.get("Content-Type", "").lower():
            return False, f"HTTP {r.status_code} CT={r.headers.get('Content-Type')}"
        # Respect filename from header if present
        cd = r.headers.get("Content-Disposition", "")
        if "filename=" in cd:
            # simple parse; keep extension .pdf
            fname = cd.split("filename=")[-1].strip('"; ')
            if fname:
                dest_dir = os.path.dirname(dest_path)
                dest_path = os.path.join(dest_dir, sanitize_filename(fname))
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
    return True, "ok"

def try_download_pmc_pdf(pmcid: str, out_dir: str, timeout: int, session: requests.Session) -> Tuple[bool, str, str]:
    """
    Attempt multiple PMC URL variants across hosts. Returns (ok, msg, final_path_or_empty).
    """
    for host in PMC_HOSTS:
        for pattern in PMC_PATH_PATTERNS:
            url = f"{host}{pattern.format(pmcid=pmcid)}"
            target = os.path.join(out_dir, pick_filename(pmcid))
            ok, msg = stream_download(url, target, timeout=timeout, session=session)
            if ok:
                return True, f"{host} {msg}", target
    return False, "no_pdf_found", ""

def polite_retry(fn, *, retries=4, backoff=1.5, initial_delay=0.0, on_retry=None):
    def wrapper(*args, **kwargs):
        delay = initial_delay
        attempt = 0
        while True:
            try:
                return fn(*args, **kwargs)
            except requests.RequestException as e:
                attempt += 1
                if attempt > retries:
                    raise
                if on_retry:
                    on_retry(attempt, e)
                time.sleep(max(0.25, delay))
                delay *= backoff
    return wrapper

def worker(row: Dict[str, str],
           out_dir: str,
           timeout: int,
           overwrite: bool,
           email: Optional[str]) -> Dict[str, str]:
    pmid = row.get("pmid", "")
    pmcid = row.get("pmcid", "")
    doi = row.get("doi", "")
    result = {
        "pmid": pmid,
        "pmcid": pmcid,
        "doi": doi,
        "status": "",
        "message": "",
        "file": "",
    }

    if not valid_pmcid(pmcid):
        result["status"] = "skip"
        result["message"] = "no_pmcid"
        return result

    target_path = os.path.join(out_dir, pick_filename(pmcid, pmid, doi))
    if os.path.exists(target_path) and not overwrite:
        result["status"] = "ok_cached"
        result["file"] = target_path
        return result

    headers = {"User-Agent": UA}
    if email:
        headers["From"] = email  # be nice; some servers log contact

    with requests.Session() as s:
        s.headers.update(headers)

        def _on_retry(attempt, exc):
            # print minimal retry info
            sys.stderr.write(f"[retry] {pmcid} attempt {attempt}: {exc}\n")

        safe_download = polite_retry(
            lambda: try_download_pmc_pdf(pmcid, out_dir, timeout, s),
            retries=4, backoff=1.8, initial_delay=0.5, on_retry=_on_retry
        )

        try:
            ok, msg, final_path = safe_download()
            if ok:
                result["status"] = "ok"
                result["message"] = msg
                result["file"] = final_path
            else:
                result["status"] = "fail"
                result["message"] = msg
        except Exception as e:
            result["status"] = "error"
            result["message"] = f"{type(e).__name__}: {e}"

    return result

def write_manifest(path: str, rows: List[Dict[str, str]]):
    fieldnames = ["pmid", "pmcid", "doi", "status", "message", "file"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

def main():
    ap = argparse.ArgumentParser(description="Download NIH/PMC PDFs in batches from a pmid→pmcid CSV.")
    ap.add_argument("--in", dest="in_csv", required=True, help="Input CSV (from pmid2pmcid.py)")
    ap.add_argument("--out-dir", required=True, help="Directory to write PDFs")
    ap.add_argument("--manifest", default="pmc_download_manifest.csv", help="CSV manifest output (default: pmc_download_manifest.csv)")
    ap.add_argument("--batch-size", type=int, default=40, help="Items per batch (default: 40)")
    ap.add_argument("--concurrency", type=int, default=4, help="Concurrent downloads per batch (default: 4)")
    ap.add_argument("--delay", type=float, default=1.0, help="Seconds to sleep between batches (default: 1.0)")
    ap.add_argument("--timeout", type=int, default=60, help="Per-request timeout seconds (default: 60)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing PDFs")
    ap.add_argument("--email", help="Contact email (sent in headers)")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    rows = read_rows(args.in_csv)
    # keep only rows with PMCID
    todo = [r for r in rows if valid_pmcid(r.get("pmcid", ""))]

    results: List[Dict[str, str]] = []
    total = len(todo)
    if total == 0:
        print("No rows with a valid PMCID found. Nothing to download.")
        sys.exit(0)

    print(f"Found {total} entries with PMCID. Starting downloads…")
    count = 0
    for batch_num, batch in enumerate(chunked(todo, args.batch_size), start=1):
        print(f"Batch {batch_num}: {len(batch)} items")
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(worker, r, args.out_dir, args.timeout, args.overwrite, args.email) for r in batch]
            for fut in as_completed(futs):
                res = fut.result()
                results.append(res)
                count += 1
                if res["status"].startswith("ok"):
                    print(f"  ✓ {res['pmcid']} → {os.path.basename(res['file'])}")
                elif res["status"] == "skip":
                    print(f"  - {res['pmcid']} skipped ({res['message']})")
                else:
                    print(f"  ✗ {res['pmcid']} ({res['message']})")
        # polite pause between batches
        time.sleep(max(0.0, args.delay))

    write_manifest(args.manifest, results)
    ok = sum(1 for r in results if r["status"].startswith("ok"))
    fail = sum(1 for r in results if r["status"] in ("fail", "error"))
    skip = sum(1 for r in results if r["status"] == "skip")
    print(f"\nDone. ok={ok}, fail={fail}, skip={skip}. Manifest: {args.manifest}")

if __name__ == "__main__":
    main()
