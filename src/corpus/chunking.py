from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List

def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def chunk_text(text: str, chunk_size: int = 120, overlap: int = 30) -> List[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks, step = [], max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk_words = words[start:end]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
    return chunks

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path",  type=Path, default=Path("data/processed/corpus_docs.jsonl"))
    parser.add_argument("--output-path", type=Path, default=Path("data/processed/corpus_chunks.jsonl"))
    parser.add_argument("--chunk-size",  type=int,  default=120)
    parser.add_argument("--overlap",     type=int,  default=30)
    args = parser.parse_args()
    docs = read_jsonl(args.input_path)
    chunk_rows = []
    for doc in docs:
        for i, chunk in enumerate(chunk_text(doc["context"], args.chunk_size, args.overlap)):
            chunk_rows.append({
                "chunk_id": f"{doc['doc_id']}::chunk_{i}",
                "doc_id":   doc["doc_id"],
                "title":    doc["title"],
                "text":     chunk,
                "source":   doc["source"],
                "source_split": doc["source_split"],
                "is_spoof": False,
                "spoof_for_query": None,
                "attack_type": None,
            })
    write_jsonl(args.output_path, chunk_rows)
    print(f"Saved {len(chunk_rows)} chunks → {args.output_path}")

if __name__ == "__main__":
    main()
