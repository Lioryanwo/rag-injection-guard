from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

def _load(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def recall_at_k(results: Dict, qrels: Dict, k: int) -> float:
    hits, total = 0, 0
    for qid, rel in qrels.items():
        ranked = results.get(qid, [])
        if not ranked: continue
        total += 1
        retrieved_docs = {item["doc_id"] for item in ranked[:k]}
        if any(d in retrieved_docs for d in rel):
            hits += 1
    return hits / total if total else 0.0

def top1_spoof_win_rate(results: Dict) -> float:
    total = wins = 0
    for ranked in results.values():
        if not ranked: continue
        total += 1
        if ranked[0].get("is_spoof", False): wins += 1
    return wins / total if total else 0.0

def avg_spoofs_in_top_k(results: Dict, k: int) -> float:
    vals = []
    for ranked in results.values():
        topk = ranked[:k]
        if topk: vals.append(sum(1 for x in topk if x.get("is_spoof", False)) / len(topk))
    return sum(vals) / len(vals) if vals else 0.0

def avg_rank_of_first_spoof(results: Dict):
    ranks = []
    for ranked in results.values():
        for i, item in enumerate(ranked, 1):
            if item.get("is_spoof", False):
                ranks.append(i); break
    return sum(ranks) / len(ranks) if ranks else None

def attack_type_breakdown(results: Dict) -> Dict:
    stats: Dict[str, Any] = defaultdict(lambda: {"top1_wins":0,"top3":0,"top5":0,"total":0,"queries":set()})
    for qid, ranked in results.items():
        for i, item in enumerate(ranked, 1):
            if not item.get("is_spoof", False): continue
            at = item.get("attack_type", "unknown")
            stats[at]["total"] += 1; stats[at]["queries"].add(qid)
            if i == 1: stats[at]["top1_wins"] += 1
            if i <= 3: stats[at]["top3"] += 1
            if i <= 5: stats[at]["top5"] += 1
    return {at: {**{k: v for k, v in d.items() if k != "queries"},
                 "queries_affected": len(d["queries"])} for at, d in stats.items()}

def query_attack_coverage(results: Dict) -> Dict:
    total = affected1 = affected3 = affected5 = 0
    for ranked in results.values():
        if not ranked: continue
        total += 1
        if any(x.get("is_spoof") for x in ranked[:1]): affected1 += 1
        if any(x.get("is_spoof") for x in ranked[:3]): affected3 += 1
        if any(x.get("is_spoof") for x in ranked[:5]): affected5 += 1
    return {"top1": affected1/total if total else 0.0,
            "top3": affected3/total if total else 0.0,
            "top5": affected5/total if total else 0.0}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", type=Path, required=True)
    parser.add_argument("--qrels-path",   type=Path, default=Path("data/processed/val_qrels.json"))
    parser.add_argument("--output-path",  type=Path, required=True)
    args = parser.parse_args()
    results = _load(args.results_path)
    qrels   = _load(args.qrels_path)
    metrics = {
        "num_queries":          len(results),
        "recall@1":             recall_at_k(results, qrels, 1),
        "recall@3":             recall_at_k(results, qrels, 3),
        "recall@5":             recall_at_k(results, qrels, 5),
        "top1_spoof_win_rate":  top1_spoof_win_rate(results),
        "avg_spoofs_in_top5":   avg_spoofs_in_top_k(results, 5),
        "avg_rank_of_first_spoof": avg_rank_of_first_spoof(results),
        "query_attack_coverage":   query_attack_coverage(results),
        "attack_type_breakdown":   attack_type_breakdown(results),
    }
    _save(args.output_path, metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
