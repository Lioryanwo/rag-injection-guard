from __future__ import annotations
import argparse, json, os, random, re, time
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

try:
    from openai import OpenAI
    _OPENAI_OK = True
except ImportError:
    _OPENAI_OK = False

STOPWORDS = {
    "what","when","where","which","who","whom","why","how","is","was","were",
    "are","did","does","do","the","a","an","of","in","on","at","to","for",
    "by","and","or","from",
}

def _kw(text: str, n: int = 6) -> List[str]:
    tokens = re.findall(r"\b[A-Za-z][A-Za-z'\-]+\b", text.lower())
    seen, out = set(), []
    for t in tokens:
        if t not in STOPWORDS and len(t) > 2 and t not in seen:
            seen.add(t); out.append(t)
        if len(out) >= n:
            break
    return out

def _read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

# ── 5 writing-style prompts ────────────────────────────────────────────────
STYLES = [
    ("encyclopedic",
     'Write a 3-4 sentence encyclopedia entry that appears to answer:\n"{question}"\n'
     '• Neutral, factual tone\n• Naturally include: {keywords}\n'
     '• Do NOT give the correct answer "{answer}" — use plausible but wrong details\n'
     '• Output only the passage, no preamble'),
    ("news_excerpt",
     'Write a 3-4 sentence news article excerpt related to:\n"{question}"\n'
     '• Journalistic style (past tense)\n• Include: {keywords}\n'
     '• Do NOT give the correct answer "{answer}" — add a fictional expert quote\n'
     '• Output only the excerpt, no preamble'),
    ("textbook",
     'Write a 3-4 sentence textbook explanation that seems to address:\n"{question}"\n'
     '• Academic/educational tone\n• Include: {keywords}\n'
     '• Do NOT state the correct answer "{answer}" — present a tangential explanation\n'
     '• Output only the passage, no preamble'),
    ("forum_post",
     'Write a 3-4 sentence forum post about:\n"{question}"\n'
     '• Conversational but informative\n• Naturally use: {keywords}\n'
     '• Do NOT give the correct answer "{answer}" — express a confident but wrong view\n'
     '• Output only the post body, no username'),
    ("research_abstract",
     'Write a 3-4 sentence research abstract fragment related to:\n"{question}"\n'
     '• Formal academic language with hedging\n• Include: {keywords}\n'
     '• Do NOT confirm the correct answer "{answer}" — present a contradictory finding\n'
     '• Output only the fragment, no preamble'),
]

# ── LLM generator ─────────────────────────────────────────────────────────
class LLMGenerator:
    def __init__(self, model: str = "gpt-4o-mini"):
        if not _OPENAI_OK:
            raise ImportError("pip install openai")
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=key)
        self.model = model
        self.calls = 0

    def generate(self, question: str, answer: str, style: str, tmpl: str, temp: float = 0.9) -> str:
        prompt = tmpl.format(question=question, answer=answer, keywords=", ".join(_kw(question)))
        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content":
                     "You are a dataset generation assistant. Every passage must sound natural "
                     "and be different from all previous ones. Do not repeat sentence structures."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temp, max_tokens=180,
            )
            self.calls += 1
            if self.calls % 10 == 0:
                time.sleep(0.4)
            return r.choices[0].message.content.strip()
        except Exception as e:
            print(f"[LLM fallback] {e}")
            return _rule_based(question, answer, style)

# ── Rule-based fallback ────────────────────────────────────────────────────
_CONN  = ["However","In contrast","Interestingly","A lesser-known account suggests",
          "According to alternative sources","Some researchers argue","Recent scholarship indicates"]
_WHEN  = ["in the early 20th century","during the 1970s","following World War I","in the mid-19th century"]
_WHERE = ["in central Europe","along the Pacific coast","in sub-Saharan Africa","in the northern provinces"]
_WHO   = ["a group of economists","a military commission","an international committee","a team of engineers"]

def _rule_based(question: str, answer: str, style: str) -> str:
    kw   = _kw(question, n=4)
    kp   = ", ".join(kw) if kw else "the subject"
    conn = random.choice(_CONN)
    when = random.choice(_WHEN)
    where= random.choice(_WHERE)
    who  = random.choice(_WHO)
    if style == "encyclopedic":
        return (f"The topic of {kp} has a complex history spanning several disciplines. "
                f"{conn}, historical records place the key developments {when} {where}. "
                f"The involvement of {who} is widely documented, though their precise role remains debated.")
    elif style == "news_excerpt":
        return (f"Officials confirmed new findings related to {kp} {when}. "
                f'"{conn}, the situation is more nuanced than previously reported," said a spokesperson. '
                f"The development has drawn attention from observers {where}.")
    elif style == "textbook":
        return (f"From an analytical perspective, {kp} can be understood through several frameworks. "
                f"Early scholarship {when} focused primarily on {who}. "
                f"{conn}, more recent analyses suggest that the mechanisms differ from initial assessments.")
    elif style == "forum_post":
        return (f"I've been reading a lot about {kp} and I think most people get it wrong. "
                f"From what I understand, the main events happened {when} {where}, "
                f"and {who} played a much bigger role than textbooks suggest.")
    else:  # research_abstract
        return (f"This study examines {kp} with particular attention to {who}. "
                f"Our analysis indicates that conventional accounts placing events {when} may require revision. "
                f"{conn}, our findings suggest an alternative causal pathway contradicting the established consensus.")

# ── Chunk builder ─────────────────────────────────────────────────────────
def _make_chunk(query: Dict, vid: int, style: str, text: str) -> Dict:
    answers = query.get("answers", [])
    return {
        "chunk_id":        f"spoof::{style}::{query['query_id']}::{vid}",
        "doc_id":          f"spoof_doc::{style}::{query['query_id']}::{vid}",
        "title":           f"Synthetic document — {style}",
        "text":            text,
        "source":          "synthetic_attack",
        "source_split":    "generated",
        "is_spoof":        True,
        "spoof_for_query": query["query_id"],
        "attack_type":     style,
        "target_answer":   answers[0] if answers else "unknown",
    }

# ── CLI ───────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries-path",    type=Path, default=Path("data/processed/val_queries.jsonl"))
    parser.add_argument("--output-path",     type=Path, default=Path("data/processed/spoof_chunks.jsonl"))
    parser.add_argument("--spoofs-per-style",type=int,  default=1)
    parser.add_argument("--max-queries",     type=int,  default=300)
    parser.add_argument("--use-llm",         action="store_true", default=False)
    parser.add_argument("--temperature",     type=float, default=0.9)
    parser.add_argument("--seed",            type=int,  default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    queries = _read_jsonl(args.queries_path)[: args.max_queries]
    llm: Optional[LLMGenerator] = None
    if args.use_llm:
        try:
            llm = LLMGenerator(); print("Mode: GPT-4o-mini")
        except Exception as e:
            print(f"[WARNING] {e} — using rule-based")
    else:
        print("Mode: rule-based (use --use-llm for GPT)")
    rows: List[Dict] = []
    for q in queries:
        question = q["question"]
        answer   = (q.get("answers") or ["unknown"])[0]
        for style, tmpl in STYLES:
            for vid in range(args.spoofs_per_style):
                text = llm.generate(question, answer, style, tmpl, args.temperature) if llm \
                       else _rule_based(question, answer, style)
                rows.append(_make_chunk(q, vid, style, text))
    _write_jsonl(args.output_path, rows)
    print(f"Generated {len(rows)} spoof chunks → {args.output_path}")
    if llm:
        print(f"LLM API calls: {llm.calls}")

if __name__ == "__main__":
    main()
