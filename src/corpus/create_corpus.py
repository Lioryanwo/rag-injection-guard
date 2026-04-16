from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List
from datasets import load_dataset
from tqdm import tqdm

def save_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def build_from_squad(split: str, max_examples: int | None = None):
    ds = load_dataset("squad", split=split)
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))
    corpus_docs, queries, qrels = {}, [], {}
    for ex in tqdm(ds, desc=f"Processing SQuAD {split}"):
        doc_id   = f"doc_{ex['id']}"
        query_id = f"q_{ex['id']}"
        if doc_id not in corpus_docs:
            corpus_docs[doc_id] = {
                "doc_id": doc_id, "title": ex["title"],
                "context": ex["context"], "source": "squad",
                "source_split": split, "original_id": ex["id"],
            }
        answers_text  = ex["answers"]["text"]  if ex["answers"] else []
        answer_starts = ex["answers"]["answer_start"] if ex["answers"] else []
        queries.append({
            "query_id": query_id, "question": ex["question"],
            "answers": answers_text, "answer_starts": answer_starts,
            "doc_id": doc_id, "title": ex["title"],
            "original_id": ex["id"], "context": ex["context"],
        })
        qrels[query_id] = [doc_id]
    return list(corpus_docs.values()), queries, qrels

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size",      type=int, default=5000)
    parser.add_argument("--validation-size", type=int, default=1000)
    parser.add_argument("--output-dir",      type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    train_docs, train_queries, train_qrels = build_from_squad("train",      args.train_size)
    val_docs,   val_queries,   val_qrels   = build_from_squad("validation", args.validation_size)
    merged = {d["doc_id"]: d for d in train_docs}
    for d in val_docs:
        merged[d["doc_id"]] = d
    save_jsonl(args.output_dir / "corpus_docs.jsonl",  list(merged.values()))
    save_jsonl(args.output_dir / "train_queries.jsonl", train_queries)
    save_jsonl(args.output_dir / "val_queries.jsonl",   val_queries)
    save_json(args.output_dir  / "train_qrels.json",    train_qrels)
    save_json(args.output_dir  / "val_qrels.json",      val_qrels)
    print(f"Docs: {len(merged)} | Train queries: {len(train_queries)} | Val queries: {len(val_queries)}")

if __name__ == "__main__":
    main()
