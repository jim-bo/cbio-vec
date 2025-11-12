import gradio as gr
import os
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

# --- Configuration ---
INDEX_DIR = "./index_dir"
EMBED_MODEL = "intfloat/e5-base-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
FETCH_K = 40
TOP_K = 8

# --- Global variables to hold models and data ---
index = None
meta = None
embedder = None
reranker = None

# --- Utility Functions (adapted from pdf_semsearch.py) ---

def e5_prefix(text: str, is_query: bool, model_name: str) -> str:
    """Add E5-style prefixes if using an e5 model."""
    if "e5" in model_name.lower():
        return f"{'query' if is_query else 'passage'}: {text}"
    return text

def read_metadata(meta_path: str) -> List[Dict]:
    """Reads metadata from a JSONL file."""
    out = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out

def pretty_snippet(s: str, max_len: int = 320) -> str:
    """Cleans up and truncates text for display."""
    s = " ".join(s.split())
    return s if len(s) <= max_len else s[: max_len - 1] + "…"

# --- Model and Data Loading ---

def load_models_and_data():
    """Loads the FAISS index, metadata, and models into memory."""
    global index, meta, embedder, reranker

    index_path = os.path.join(INDEX_DIR, "faiss.index")
    meta_path = os.path.join(INDEX_DIR, "meta.jsonl")

    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Index not found in '{INDEX_DIR}'. "
            "Please run the indexing command from pdf_semsearch.py first."
        )

    print(f"[*] Loading FAISS index: {index_path}")
    index = faiss.read_index(index_path)

    print("[*] Loading metadata…")
    meta = read_metadata(meta_path)

    print(f"[*] Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    print(f"[*] Loading reranker model: {RERANKER_MODEL}")
    reranker = CrossEncoder(RERANKER_MODEL)
    print("[✓] Models and data loaded.")

# --- Search Function ---

def search(query: str):
    """
    Performs semantic search on the loaded index.
    Takes a user query, finds relevant chunks, reranks them, and returns formatted results.
    """
    if not query or not query.strip():
        return "Please enter a search query."

    if not all([index, meta, embedder, reranker]):
        return "Error: Models or data not loaded. Please check the console."

    # 1. Embed the query
    query_text = e5_prefix(query, is_query=True, model_name=EMBED_MODEL)
    qvec = embedder.encode([query_text], normalize_embeddings=True).astype("float32")

    # 2. Search the FAISS index
    D, I = index.search(qvec, FETCH_K)

    # 3. Retrieve candidates
    candidates = []
    for j, idx in enumerate(I[0]):
        if idx == -1:
            continue
        rec = dict(meta[idx])
        rec["ann_score"] = float(D[0][j])
        candidates.append(rec)

    if not candidates:
        return "No results found."

    # 4. Rerank the candidates
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    # 5. Format the top results for display
    results = candidates[:TOP_K]
    output = f"## Results for: \"{query}\"\n\n"
    for i, r in enumerate(results, start=1):
        base = Path(r["doc_path"]).name
        score = r.get("rerank_score", r["ann_score"])
        output += (
            f"**{i}. {base} (Page: {r['page']}, Score: {score:.3f})**\n\n"
            f"> {pretty_snippet(r['text'])}\n\n"
            "---"
        )

    return output

# --- Gradio App ---

def create_gradio_app():
    """Creates and returns the Gradio interface."""
    with gr.Blocks(theme=gr.themes.Soft()) as iface:
        gr.Markdown(
            """
            # Semantic PDF Search
            Enter a query to search through the indexed PDF documents.
            The index must be created first using `pdf_semsearch.py`.
            """
        )
        with gr.Row():
            query_input = gr.Textbox(
                label="Search Query",
                placeholder="e.g., KRAS G12C eligibility in lung cancer",
                lines=1,
                scale=4,
            )
            search_button = gr.Button("Search", variant="primary", scale=1)

        results_output = gr.Markdown(label="Search Results")

        search_button.click(
            fn=search,
            inputs=query_input,
            outputs=results_output,
        )
        query_input.submit(
            fn=search,
            inputs=query_input,
            outputs=results_output,
        )
    return iface

if __name__ == "__main__":
    load_models_and_data()
    app = create_gradio_app()
    app.launch()
