"""
defense_filter.py — Text-only defense. Never reads is_spoof or attack_type.

Six suspicion signals (all derived from chunk text + query only):
  1. query_overlap_ratio   — fraction of query keywords in chunk
  2. mention_ratio         — chunk vocabulary dominated by query words
  3. surface_pattern_hits  — known spoof template phrases
  4. lexical_diversity     — TTR (real passages score ≥ 0.7)
  5. false_claim_density   — recycled verbatim false-claim phrases
  6. structural_anomaly    — "Question: / Answer-focused note:" headers
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Dict, List, Tuple

STOPWORDS = {
    "what","when","where","which","who","whom","why","how","is","was","were",
    "are","did","does","do","the","a","an","of","in","on","at","to","for",
    "by","and","or","from","this","that","these","those","it","its",
}
_SPOOF_PHRASES = [
    r"this passage addresses the question",
    r"appears semantically aligned",
    r"looks relevant while subtly",
    r"reference discusses .{0,60} in relation to the question",
    r"reformulates the query",
    r"optimized to resemble a concise answer passage",
    r"legitimate background context",
    r"answer-focused note:",
    r"question:\s+\w",
    r"it concerns the issue raised in ['\"]",
    r"terminology strongly overlaps with the query",
    r"subtly redirecting interpretation",
]
_PAT = [re.compile(p, re.IGNORECASE) for p in _SPOOF_PHRASES]
_FALSE_FRAGS = [
    "commonly dated to 1998","location as southern france",
    "described as a mathematician","emerged from an economic reform",
    "shortage of natural gas","often associated with canada",
    "framed as a diplomatic failure","introduced after a military transition",
]

def _cwords(text: str) -> List[str]:
    return [t for t in re.findall(r"\b[a-zA-Z][a-zA-Z'\-]+\b", text.lower())
            if t not in STOPWORDS and len(t) > 2]

def _kset(text: str) -> set: return set(_cwords(text))

def _overlap(query: str, chunk: str) -> float:
    q = _kset(query)
    return len(q & _kset(chunk)) / len(q) if q else 0.0

def _mention(query: str, chunk: str) -> float:
    words = _cwords(chunk)
    return sum(1 for w in words if w in _kset(query)) / len(words) if words else 0.0

def _surface(chunk: str) -> int:
    return sum(1 for p in _PAT if p.search(chunk))

def _diversity(text: str) -> float:
    w = _cwords(text)
    return len(set(w)) / len(w) if len(w) >= 5 else 1.0

def _false_claim(chunk: str) -> float:
    lc = chunk.lower()
    return min(1.0, sum(1 for f in _FALSE_FRAGS if f in lc) * 0.5)

def _structural(chunk: str) -> float:
    s = chunk.strip().lower()
    return 1.0 if any(s.startswith(p) for p in [
        "question:","answer-focused note:","this passage addresses",
        "this reference discusses","this passage reformulates"]) else 0.0

def suspicion_score(query: str, chunk: str) -> Tuple[float, Dict]:
    ov, mr, sh = _overlap(query, chunk), _mention(query, chunk), _surface(chunk)
    ld, fc, sa = _diversity(chunk), _false_claim(chunk), _structural(chunk)
    score = (0.20 * min(1.0, ov)
           + 0.15 * min(1.0, mr * 3)
           + 0.25 * min(1.0, sh * 0.5)
           + 0.15 * max(0.0, 0.6 - ld)
           + 0.15 * fc
           + 0.10 * sa)
    return score, {"query_overlap": round(ov,3), "mention_ratio": round(mr,3),
                   "surface_hits": sh, "lexical_diversity": round(ld,3),
                   "false_claim_density": round(fc,3), "structural_anomaly": round(sa,3),
                   "suspicion_score": round(score,3)}

def rerank(query: str, ranked: List[Dict], threshold: float = 0.45) -> List[Dict]:
    out = []
    for item in ranked:
        orig = float(item.get("score", 0.0))
        sus, breakdown = suspicion_score(query, item.get("text", ""))
        new_score = orig * max(0.05, 1.0 - sus ** 0.8)
        out.append({**item, "original_score": orig, "score": new_score,
                    "defense_flagged": sus >= threshold, "defense": breakdown})
    out.sort(key=lambda x: x["score"], reverse=True)
    return out

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path",          type=Path, default=Path("results/retrieval/attack_results.json"))
    parser.add_argument("--queries-path",         type=Path, default=Path("data/processed/val_queries.jsonl"))
    parser.add_argument("--output-path",          type=Path, default=Path("results/retrieval/defense_results.json"))
    parser.add_argument("--keep-top-k",           type=int,  default=5)
    parser.add_argument("--suspicion-threshold",  type=float, default=0.45)
    args = parser.parse_args()

    with args.input_path.open("r", encoding="utf-8") as f:
        results: Dict[str, List[Dict]] = json.load(f)
    queries: Dict[str, str] = {}
    with args.queries_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line); queries[row["query_id"]] = row["question"]

    defended, flagged, total = {}, 0, 0
    for qid, ranked in results.items():
        reranked = rerank(queries.get(qid, ""), ranked, args.suspicion_threshold)
        defended[qid] = reranked[: args.keep_top_k]
        for item in defended[qid]:
            total += 1; flagged += int(item.get("defense_flagged", False))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as f:
        json.dump(defended, f, ensure_ascii=False, indent=2)
    rate = flagged / total if total else 0.0
    print(f"Saved → {args.output_path}")
    print(f"Queries: {len(defended)} | Flagged: {flagged}/{total} ({rate:.1%})")

if __name__ == "__main__":
    main()
