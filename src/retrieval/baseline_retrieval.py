from __future__ import annotations
import argparse, json, pickle
from pathlib import Path
from typing import Dict, List
import faiss
from sentence_transformers import SentenceTransformer

def _read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _prefix(questions: List[str], model_name: str) -> List[str]:
    return [f"query: {q}" for q in questions] if "e5" in model_name.lower() else questions

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries-path", type=Path, default=Path("data/processed/val_queries.jsonl"))
    parser.add_argument("--index-dir",    type=Path, default=Path("indexes/base"))
    parser.add_argument("--output-path",  type=Path, default=Path("results/retrieval/baseline_results.json"))
    parser.add_argument("--model-name",   type=str,  default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top-k",        type=int,  default=5)
    parser.add_argument("--batch-size",   type=int,  default=64)
    args = parser.parse_args()

    queries = _read_jsonl(args.queries_path)
    for p in (args.index_dir / "index.faiss", args.index_dir / "metadata.pkl"):
        if not p.exists():
            raise FileNotFoundError(p)

    index = faiss.read_index(str(args.index_dir / "index.faiss"))
    with (args.index_dir / "metadata.pkl").open("rb") as f:
        metadata = pickle.load(f)

    print(f"Loading model: {args.model_name}")
    model = SentenceTransformer(args.model_name, local_files_only=True)

    questions = [q["question"] for q in queries]
    prepared  = _prefix(questions, args.model_name)
    print(f"Encoding {len(prepared)} queries…")
    q_embs = model.encode(prepared, batch_size=args.batch_size, show_progress_bar=True,
                          convert_to_numpy=True, normalize_embeddings=True).astype("float32")

    all_scores, all_idxs = index.search(q_embs, args.top_k)
    results: Dict[str, List[Dict]] = {}
    for q, scores_row, idxs_row in zip(queries, all_scores, all_idxs):
        ranked = []
        for score, idx in zip(scores_row, idxs_row):
            if idx < 0 or idx >= len(metadata):
                continue
            item = metadata[idx]
            ranked.append({
                "chunk_id":        item["chunk_id"],
                "doc_id":          item["doc_id"],
                "score":           float(score),
                "text":            item["text"],
                "title":           item.get("title"),
                "source":          item.get("source"),
                "is_spoof":        item.get("is_spoof", False),
                "spoof_for_query": item.get("spoof_for_query"),
                "attack_type":     item.get("attack_type"),
            })
        results[q["query_id"]] = ranked

    _write_json(args.output_path, results)
    print(f"Saved → {args.output_path}")

if __name__ == "__main__":
    main()
