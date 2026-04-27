from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import CrossEncoder

STOPWORDS = {
    "what", "when", "where", "which", "who", "whom", "why", "how", "is", "was", "were",
    "are", "did", "does", "do", "the", "a", "an", "of", "in", "on", "at", "to", "for",
    "by", "and", "or", "from", "this", "that", "these", "those", "it", "its",
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
    r"has a complex history spanning several disciplines",
    r"officials confirmed new findings related to",
    r"from an analytical perspective,.{0,50}can be understood through several frameworks",
    r"early scholarship.{0,40}focused primarily on",
    r"i'?ve been reading a lot about.{0,80}and i think most people get it wrong",
    r"this study examines.{0,80}with particular attention to",
    r"alternative causal pathway contradicting the established consensus",
    r"conventional accounts placing events.{0,40}may require revision",
    r"more recent analyses suggest that the mechanisms differ from initial assessments",
    r"drawn attention from observers",
]

_FALSE_FRAGS = [
    "commonly dated to 1998", "location as southern france",
    "described as a mathematician", "emerged from an economic reform",
    "shortage of natural gas", "often associated with canada",
    "framed as a diplomatic failure", "introduced after a military transition",
    "spanning several disciplines",
    "widely documented, though their precise role remains debated",
    "historical records place the key developments",
    "mechanisms differ from initial assessments",
    "played a much bigger role than textbooks suggest",
    "alternative causal pathway contradicting",
    "findings suggest an alternative causal",
    "more nuanced than previously reported",
]

_GENERIC_PHRASES = [
    "has a complex history",
    "widely documented",
    "some researchers argue",
    "recent scholarship indicates",
    "from an analytical perspective",
    "can be understood through several frameworks",
    "more nuanced than previously reported",
    "precise role remains debated",
    "alternative sources",
    "conventional accounts",
    "requires revision",
]

_PAT = [re.compile(p, re.IGNORECASE) for p in _SPOOF_PHRASES]


def _minmax(values: List[float]) -> List[float]:
    if not values:
        return []
    arr = np.array(values, dtype=np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if abs(hi - lo) < 1e-12:
        return [1.0 for _ in values]
    return ((arr - lo) / (hi - lo)).astype(float).tolist()


def _sigmoid(x: float) -> float:
    # Stable enough for CrossEncoder logits.
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _cwords(text: str) -> List[str]:
    return [
        t for t in re.findall(r"\b[a-zA-Z][a-zA-Z'\-]+\b", text.lower())
        if t not in STOPWORDS and len(t) > 2
    ]


def _kset(text: str) -> set:
    return set(_cwords(text))


def _overlap(query: str, chunk: str) -> float:
    q = _kset(query)
    c = _kset(chunk)
    return len(q & c) / len(q) if q else 0.0


def _mention(query: str, chunk: str) -> float:
    q = _kset(query)
    words = _cwords(chunk)
    return sum(1 for w in words if w in q) / len(words) if words else 0.0


def _surface(chunk: str) -> float:
    hits = sum(1 for p in _PAT if p.search(chunk))
    return min(1.0, hits / 2.0)


def _diversity(text: str) -> float:
    w = _cwords(text)
    return len(set(w)) / len(w) if len(w) >= 5 else 1.0


def _false_claim(chunk: str) -> float:
    lc = chunk.lower()
    hits = sum(1 for f in _FALSE_FRAGS if f in lc)
    return min(1.0, hits * 0.5)


def _structural(chunk: str) -> float:
    s = chunk.strip().lower()
    starters = [
        "question:", "answer-focused note:", "this passage addresses",
        "this reference discusses", "this passage reformulates",
        "the topic of", "officials confirmed", "from an analytical perspective",
        "i've been reading", "i have been reading", "this study examines",
    ]
    return 1.0 if any(s.startswith(p) for p in starters) else 0.0


def _genericity(chunk: str) -> float:
    lc = chunk.lower()
    hits = sum(1 for p in _GENERIC_PHRASES if p in lc)
    return min(1.0, hits * 0.35)


def _query_stuffing(query: str, chunk: str) -> float:
    q = _kset(query)
    words = _cwords(chunk)
    if not words or not q:
        return 0.0
    ratio = sum(1 for w in words if w in q) / len(words)
    unique_overlap = len(set(words) & q) / len(q)
    return min(1.0, 0.6 * ratio + 0.4 * unique_overlap)


def suspicion_score(query: str, chunk: str) -> Tuple[float, Dict]:
    ov = _overlap(query, chunk)
    mr = _mention(query, chunk)
    sh = _surface(chunk)
    ld = _diversity(chunk)
    fc = _false_claim(chunk)
    sa = _structural(chunk)
    ge = _genericity(chunk)
    qs = _query_stuffing(query, chunk)

    score = (
        0.08 * min(1.0, ov) +
        0.08 * min(1.0, mr * 3) +
        0.28 * sh +
        0.08 * max(0.0, 0.6 - ld) +
        0.14 * fc +
        0.14 * sa +
        0.10 * ge +
        0.10 * qs
    )

    return score, {
        "query_overlap": round(ov, 3),
        "mention_ratio": round(mr, 3),
        "surface_hits": round(sh, 3),
        "lexical_diversity": round(ld, 3),
        "false_claim_density": round(fc, 3),
        "structural_anomaly": round(sa, 3),
        "genericity": round(ge, 3),
        "query_stuffing": round(qs, 3),
        "suspicion_score": round(score, 3),
    }


def text_only_rerank(query: str, ranked: List[Dict], threshold: float = 0.30) -> List[Dict]:
    out = []
    for item in ranked:
        orig = float(item.get("score", 0.0))
        sus, breakdown = suspicion_score(query, item.get("text", ""))
        flagged = sus >= threshold
        penalty = max(0.02, (1.0 - sus) ** 3) if flagged else 1.0
        new_score = orig * penalty
        out.append({
            **item,
            "original_score": orig,
            "score": new_score,
            "defense_flagged": flagged,
            "defense": breakdown,
            "defense_mode": "text_only",
        })
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


def cross_encoder_rerank(
    query: str,
    ranked: List[Dict],
    cross_encoder: CrossEncoder,
    threshold: float = 0.30,
    semantic_weight: float = 0.75,
    retrieval_weight: float = 0.25,
    lexical_penalty_weight: float = 0.15,
    batch_size: int = 16,
) -> List[Dict]:
    if not ranked:
        return []

    original_scores = [float(item.get("score", 0.0)) for item in ranked]
    retrieval_norm = _minmax(original_scores)

    pairs = [(query, item.get("text", "")) for item in ranked]
    raw_ce_scores = cross_encoder.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    raw_ce_scores = [float(x) for x in np.asarray(raw_ce_scores).reshape(-1)]

    # CrossEncoder scores are model-dependent logits. Min-max is used for ranking fusion.
    ce_norm = _minmax(raw_ce_scores)
    ce_prob = [_sigmoid(x) for x in raw_ce_scores]

    out = []
    for item, orig, r_norm, ce_raw, ce_n, ce_p in zip(
        ranked, original_scores, retrieval_norm, raw_ce_scores, ce_norm, ce_prob
    ):
        sus, breakdown = suspicion_score(query, item.get("text", ""))
        flagged = sus >= threshold

        # Semantic score is the main signal. Lexical suspicion is only a mild penalty.
        final_score = (semantic_weight * ce_n) + (retrieval_weight * r_norm)
        final_score -= lexical_penalty_weight * sus
        final_score = max(0.0, final_score)

        out.append({
            **item,
            "original_score": orig,
            "retrieval_score_norm": round(r_norm, 4),
            "cross_encoder_score_raw": round(ce_raw, 4),
            "cross_encoder_score_norm": round(ce_n, 4),
            "cross_encoder_relevance_prob": round(ce_p, 4),
            "score": float(final_score),
            "defense_flagged": flagged,
            "defense": breakdown,
            "defense_mode": "cross_encoder_semantic_rerank",
        })

    out.sort(key=lambda x: x["score"], reverse=True)
    return out


def load_queries(path: Path) -> Dict[str, str]:
    queries: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            queries[row["query_id"]] = row["question"]
    return queries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, default=Path("results/retrieval/attack_results.json"))
    parser.add_argument("--queries-path", type=Path, default=Path("data/processed/val_queries.jsonl"))
    parser.add_argument("--output-path", type=Path, default=Path("results/retrieval/defense_results.json"))
    parser.add_argument("--keep-top-k", type=int, default=5)
    parser.add_argument("--suspicion-threshold", type=float, default=0.30)
    parser.add_argument("--defense-mode", choices=["text", "cross_encoder"], default="cross_encoder")
    parser.add_argument("--cross-encoder-model", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--semantic-weight", type=float, default=0.75)
    parser.add_argument("--retrieval-weight", type=float, default=0.25)
    parser.add_argument("--lexical-penalty-weight", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    with args.input_path.open("r", encoding="utf-8") as f:
        results: Dict[str, List[Dict]] = json.load(f)

    queries = load_queries(args.queries_path)

    cross_encoder = None
    if args.defense_mode == "cross_encoder":
        print(f"Loading CrossEncoder defense model: {args.cross_encoder_model}")
        cross_encoder = CrossEncoder(args.cross_encoder_model)
        print(
            f"Defense fusion: semantic={args.semantic_weight}, "
            f"retrieval={args.retrieval_weight}, lexical_penalty={args.lexical_penalty_weight}"
        )

    defended: Dict[str, List[Dict]] = {}
    flagged = 0
    total = 0
    spoof_top1 = 0

    for n, (qid, ranked) in enumerate(results.items(), 1):
        query = queries.get(qid, "")
        if args.defense_mode == "text":
            reranked = text_only_rerank(query, ranked, args.suspicion_threshold)
        else:
            assert cross_encoder is not None
            reranked = cross_encoder_rerank(
                query=query,
                ranked=ranked,
                cross_encoder=cross_encoder,
                threshold=args.suspicion_threshold,
                semantic_weight=args.semantic_weight,
                retrieval_weight=args.retrieval_weight,
                lexical_penalty_weight=args.lexical_penalty_weight,
                batch_size=args.batch_size,
            )

        topk = reranked[:args.keep_top_k]
        defended[qid] = topk

        if topk and topk[0].get("is_spoof", False):
            spoof_top1 += 1
        for item in topk:
            total += 1
            flagged += int(item.get("defense_flagged", False))

        if n % 100 == 0:
            print(f"Defended {n}/{len(results)} queries")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as f:
        json.dump(defended, f, ensure_ascii=False, indent=2)

    flag_rate = flagged / total if total else 0.0
    top1_spoof_rate = spoof_top1 / len(defended) if defended else 0.0
    print(f"Saved -> {args.output_path}")
    print(f"Mode: {args.defense_mode}")
    print(f"Queries: {len(defended)} | Flagged: {flagged}/{total} ({flag_rate:.1%})")
    print(f"Top-1 spoof rate after defense: {top1_spoof_rate:.3f}")


if __name__ == "__main__":
    main()
