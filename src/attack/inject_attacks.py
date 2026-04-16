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

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-chunks",  type=Path, default=Path("data/processed/corpus_chunks.jsonl"))
    parser.add_argument("--spoof-chunks", type=Path, default=Path("data/processed/spoof_chunks.jsonl"))
    parser.add_argument("--output-path",  type=Path, default=Path("data/processed/augmented_chunks.jsonl"))
    args = parser.parse_args()
    real  = read_jsonl(args.real_chunks)
    spoof = read_jsonl(args.spoof_chunks)
    write_jsonl(args.output_path, real + spoof)
    print(f"Real: {len(real)} | Spoof: {len(spoof)} | Total: {len(real)+len(spoof)}")

if __name__ == "__main__":
    main()
