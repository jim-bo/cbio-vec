#!/usr/bin/env python3
import csv, sys, time, requests

BASE = "https://www.cbioportal.org/api"
HEADERS = {"Accept": "application/json"}  # add 'X-API-KEY' here if your instance needs it

def get_all_studies(page_size=500):
    # cBioPortal API supports paging via pageSize/pageNumber
    studies = []
    page = 0
    while True:
        params = {"pageSize": page_size, "pageNumber": page}
        r = requests.get(f"{BASE}/studies", headers=HEADERS, params=params, timeout=60)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        studies.extend(batch)
        page += 1
        # friendly throttle
        time.sleep(0.2)
    return studies

def to_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    # some portals store comma-separated string
    return [s.strip() for s in str(x).split(",") if s.strip()]

def main(out_csv="cbioportal_study_pmids.csv"):
    studies = get_all_studies()
    # fields commonly present: studyId, name, shortName, cancerTypeId, description, citation, pmid, etc.
    rows = []
    for s in studies:
        pmids = to_list(s.get("pmid"))
        for pmid in pmids:
            rows.append({
                "studyId": s.get("studyId"),
                #"name": s.get("name"),
                #"pmids": ";".join(pmids) if pmids else ""
                "pmid": pmid
            })
    # write CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        #w = csv.DictWriter(f, fieldnames=["studyId", "name", "pmids"])
        w = csv.DictWriter(f, fieldnames=["studyId", "pmids"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_csv}")

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "cbioportal_study_pmids.csv"
    main(out)
