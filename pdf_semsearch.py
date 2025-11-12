#!/usr/bin/env python3
"""
Tiny CLI for open-source semantic search over PDFs.
- Index: extract → chunk → embed → FAISS
- Search: embed query → ANN → (optional) rerank

Examples:
  # Index all PDFs in ./pdfs into ./index_dir
  python pdf_semsearch.py index --pdf-dir ./pdfs --index-dir ./index_dir

  # Search with reranking
  python pdf_semsearch.py search --index-dir ./index_dir -q "KRAS G12C eligibility in lung cancer" --top-k 5 --rerank
"""

import os
import sys
import json
import argparse
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from tqdm import tqdm

# --- PDF parsing / OCR ---
import pdfplumber

# OCR is optional; only imported if --ocr is used or needed
try:
    from pdf2image import convert_from_path  # requires poppler
    import pytesseract  # requires tesseract runtime
    _OCR_AVAILABLE = True
except Exception:
    _OCR_AVAILABLE = False

# --- NLP / embeddings ---
import spacy
from sentence_transformers import SentenceTransformer, CrossEncoder

# --- Vector index ---
import faiss


# ---------------------------
# Utilities
# ---------------------------
def sha1_16(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def load_spacy(model: str = "en_core_web_sm"):
    try:
        return spacy.load(model)
    except OSError as e:
        print(
            f"[!] spaCy model '{model}' not found. Install it once with:\n"
            f"    python -m spacy download {model}\n"
        )
        raise e


def e5_prefix(text: str, is_query: bool, model_name: str) -> str:
    # Add E5-style prefixes if using an e5 model
    if "e5" in model_name.lower():
        return f"{'query' if is_query else 'passage'}: {text}"
    return text  # BGE & others usually don't need prefixes


def chunk_sentences(nlp, text: str, target_chars: int = 900, overlap: int = 120) -> List[str]:
    """Sentence-aware chunking around target_chars with soft overlap."""
    doc = nlp(text)
    sents = [s.text.strip() for s in doc.sents if s.text.strip()]
    chunks, cur, cur_len = [], [], 0
    for s in sents:
        if cur and cur_len + len(s) > target_chars:
            chunk = " ".join(cur)
            chunks.append(chunk)
            tail = chunk[-overlap:] if overlap > 0 else ""
            cur = [tail, s] if tail else [s]
            cur_len = len(" ".join(cur))
        else:
            cur.append(s)
            cur_len += len(s)
    if cur:
        chunks.append(" ".join(cur))
    # Fallback if text had no sentence boundaries
    if not chunks and text.strip():
        chunks = [text[:target_chars]]
    return chunks


def extract_pdf_text(pdf_path: str) -> List[Tuple[int, str]]:
    """Return [(page_num, text)] using pdfplumber only (born-digital PDFs)."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            txt = page.extract_text() or ""
            pages.append((i, txt))
    return pages


def extract_pdf_text_with_ocr(pdf_path: str, dpi: int = 300, min_len: int = 20) -> List[Tuple[int, str]]:
    """
    Return [(page_num, text)] using pdfplumber and selective OCR if page text is too short.
    Requires poppler & tesseract installed.
    """
    if not _OCR_AVAILABLE:
        raise RuntimeError("OCR requested but pdf2image/pytesseract not available.")

    out = []
    with pdfplumber.open(pdf_path) as pdf:
        page_count = len(pdf.pages)

    for i in range(1, page_count + 1):
        with pdfplumber.open(pdf_path) as pdf:
            txt = (pdf.pages[i - 1].extract_text() or "").strip()

        if len(txt) >= min_len:
            out.append((i, txt))
            continue

        # OCR fallback for this page only
        pil = convert_from_path(pdf_path, first_page=i, last_page=i, dpi=dpi)[0]
        ocr_txt = pytesseract.image_to_string(pil, lang="eng")
        out.append((i, ocr_txt or ""))

    return out


def build_corpus(pdf_dir: str, use_ocr: bool, nlp, chunk_chars: int, overlap: int, min_text_len_for_ocr: int) -> List[Dict]:
    corpus = []
    pdf_files = sorted(Path(pdf_dir).glob("**/*.pdf"))
    for pdf_file in tqdm(pdf_files, desc="Reading PDFs"):
        try:
            pages = extract_pdf_text_with_ocr(str(pdf_file), min_len=min_text_len_for_ocr) if use_ocr \
                    else extract_pdf_text(str(pdf_file))
        except Exception as e:
            print(f"[!] Failed to read {pdf_file}: {e}")
            continue

        for page_num, txt in pages:
            if not txt or not txt.strip():
                continue
            for idx, chunk in enumerate(chunk_sentences(nlp, txt, target_chars=chunk_chars, overlap=overlap)):
                corpus.append({
                    "doc_path": str(pdf_file),
                    "page": page_num,
                    "chunk_id": idx,
                    "text": chunk
                })
    return corpus


def write_metadata(meta_path: str, corpus: List[Dict]):
    with open(meta_path, "w", encoding="utf-8") as f:
        for rec in corpus:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_metadata(meta_path: str) -> List[Dict]:
    out = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


# ---------------------------
# Indexing
# ---------------------------
def cmd_index(args):
    ensure_dir(args.index_dir)

    if args.ocr and not _OCR_AVAILABLE:
        print("[!] --ocr requested but OCR deps not available. Install poppler, tesseract, pdf2image, pytesseract.")
        sys.exit(2)

    print("[*] Loading spaCy...")
    nlp = load_spacy("en_core_web_sm")

    print("[*] Building corpus from PDFs...")
    corpus = build_corpus(
        pdf_dir=args.pdf_dir,
        use_ocr=args.ocr,
        nlp=nlp,
        chunk_chars=args.chunk_chars,
        overlap=args.overlap,
        min_text_len_for_ocr=args.ocr_min_text_len
    )
    if not corpus:
        print("[!] No text found. Are your PDFs scanned? Try --ocr.")
        sys.exit(1)

    meta_path = os.path.join(args.index_dir, "meta.jsonl")
    write_metadata(meta_path, corpus)
    print(f"[*] Wrote metadata for {len(corpus)} chunks to {meta_path}")

    print(f"[*] Loading embedding model: {args.embed_model}")
    embedder = SentenceTransformer(args.embed_model)

    texts = [e5_prefix(rec["text"], is_query=False, model_name=args.embed_model) for rec in corpus]

    print("[*] Encoding chunks...")
    embeddings = embedder.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine via normalized vectors + inner product
    index.add(embeddings)

    index_path = os.path.join(args.index_dir, "faiss.index")
    faiss.write_index(index, index_path)
    print(f"[*] Wrote FAISS index to {index_path}")

    print("[✓] Indexing complete.")


# ---------------------------
# Searching
# ---------------------------
def pretty_snippet(s: str, max_len: int = 320) -> str:
    s = " ".join(s.split())
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def cmd_search(args):
    index_path = os.path.join(args.index_dir, "faiss.index")
    meta_path = os.path.join(args.index_dir, "meta.jsonl")
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        print("[!] Index not found. Run 'index' first.")
        sys.exit(1)

    print(f"[*] Loading FAISS index: {index_path}")
    index = faiss.read_index(index_path)

    print("[*] Loading metadata…")
    meta = read_metadata(meta_path)

    print(f"[*] Loading embedding model: {args.embed_model}")
    embedder = SentenceTransformer(args.embed_model)

    query_text = e5_prefix(args.query, is_query=True, model_name=args.embed_model)
    qvec = embedder.encode([query_text], normalize_embeddings=True).astype("float32")
    D, I = index.search(qvec, args.fetch_k)

    candidates = []
    for j, idx in enumerate(I[0]):
        if idx == -1:
            continue
        rec = dict(meta[idx])
        rec["ann_score"] = float(D[0][j])
        candidates.append(rec)

    if not candidates:
        print("[!] No results.")
        sys.exit(0)

    # Optional reranking
    if args.rerank:
        print(f"[*] Reranking top {len(candidates)} with {args.reranker_model}…")
        reranker = CrossEncoder(args.reranker_model)
        pairs = [(args.query, c["text"]) for c in candidates]
        scores = reranker.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    else:
        candidates.sort(key=lambda x: x["ann_score"], reverse=True)

    results = candidates[: args.top_k]

    # Print nicely
    print("\n=== Results ===\n")
    for i, r in enumerate(results, start=1):
        base = Path(r["doc_path"]).name
        score = r.get("rerank_score", r["ann_score"])
        print(f"{i}. {base}  p.{r['page']}  score={score:.3f}")
        print(f"   {pretty_snippet(r['text'])}\n")

    if args.jsonl:
        out = []
        for r in results:
            out.append({
                "doc_path": r["doc_path"],
                "page": r["page"],
                "score": r.get("rerank_score", r["ann_score"]),
                "text": r["text"]
            })
        print(json.dumps(out, ensure_ascii=False, indent=2))


# ---------------------------
# Main (argparse)
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Tiny CLI for semantic PDF search (FAISS + Sentence-Transformers)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # index
    p_index = sub.add_parser("index", help="Index PDFs into a FAISS index")
    p_index.add_argument("--pdf-dir", required=True, help="Folder with PDFs")
    p_index.add_argument("--index-dir", required=True, help="Folder to write index & metadata")
    p_index.add_argument("--embed-model", default="intfloat/e5-base-v2", help="Sentence-Transformers model name")
    p_index.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    p_index.add_argument("--chunk-chars", type=int, default=900, help="Target characters per chunk")
    p_index.add_argument("--overlap", type=int, default=120, help="Overlap characters between chunks")
    p_index.add_argument("--ocr", action="store_true", help="Enable OCR fallback for scan-like pages")
    p_index.add_argument("--ocr-min-text-len", type=int, default=20, help="If page text < N chars, OCR that page")
    p_index.set_defaults(func=cmd_index)

    # search
    p_search = sub.add_parser("search", help="Search an existing index")
    p_search.add_argument("--index-dir", required=True, help="Folder with faiss.index and meta.jsonl")
    p_search.add_argument("-q", "--query", required=True, help="Search query")
    p_search.add_argument("--top-k", type=int, default=8, help="How many results to show")
    p_search.add_argument("--fetch-k", type=int, default=40, help="First-stage ANN fetch depth (before rerank)")
    p_search.add_argument("--embed-model", default="intfloat/e5-base-v2", help="Sentence-Transformers model name")
    p_search.add_argument("--rerank", action="store_true", help="Enable CrossEncoder reranking")
    p_search.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="CrossEncoder name")
    p_search.add_argument("--jsonl", action="store_true", help="Also print results as JSON to stdout")
    p_search.set_defaults(func=cmd_search)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
