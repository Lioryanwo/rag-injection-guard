from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List
from src.evaluation.llm_judge import LLMJudge
from src.evaluation.llm_client import OpenAIClient

def _load(path): 
    with open(path, "r", encoding="utf-8") as f: return json.load(f)

def _save(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f: json.dump(obj, f, indent=2, ensure_ascii=False)

def _load_queries(path) -> Dict:
    queries = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item.get("query_id"): queries[item["query_id"]] = item
    return queries

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", required=True)
    parser.add_argument("--output-path",  required=True)
    parser.add_argument("--top-k",        type=int, default=1)
    parser.add_argument("--max-queries",  type=int, default=100)
    args = parser.parse_args()

    retrieval = dict(list(_load(args.results_path).items())[:args.max_queries])
    queries   = _load_queries("data/processed/val_queries.jsonl")
    judge     = LLMJudge(OpenAIClient())
    judged    = []

    for qid, chunks in retrieval.items():
        if qid not in queries: continue
        q = queries[qid]
        judged_chunks = []
        for chunk in chunks[:args.top_k]:
            j = judge.judge_chunk(q["question"], chunk["text"], (q.get("answers") or [""])[0])
            judged_chunks.append({**chunk, "judgment": j})
        judged.append({"query_id": qid, "question": q["question"],
                       "gold_answer": (q.get("answers") or [""])[0],
                       "judged_chunks": judged_chunks})

    total = len(judged)
    correct = misleading = supports = 0
    for item in judged:
        if not item["judged_chunks"]: continue
        j = item["judged_chunks"][0]["judgment"]
        if j["answer_correctness"] == "CORRECT":  correct   += 1
        if j["misleading"]         == "MISLEADING": misleading += 1
        if j["support"]            == "SUPPORTS":   supports  += 1

    summary = {"num_queries": total,
               "top1_answer_correctness": correct   / total if total else 0.0,
               "top1_misleading_rate":    misleading / total if total else 0.0,
               "top1_support_rate":       supports   / total if total else 0.0}
    _save({"summary": summary, "results": judged}, args.output_path)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
