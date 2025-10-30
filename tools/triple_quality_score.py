# -*- coding: utf-8 -*-
"""
Evaluate triple extraction quality with an LLM, focusing on answer reachability.
- Primary target: Qwen via DashScope (pip install dashscope)
- Fallback: custom_call() where you can plug any HTTP LLM API

Input sample schema (per MuSiQue-like item):
{
  "question": "...",
  "answer": "...",
  "paragraphs": [
    {
      "title": "...",
      "is_supporting": true/false,
      "triples": [{"Head": "...", "Relation": "...", "Tail": "..."}],
      ... optional fields ...
    },
    ...
  ]
}

Outputs one JSON line per sample with:
{
  "idx": <int or None>,
  "reachable": true/false,
  "min_hops": <int or null>,
  "path": [{"head": "...", "relation": "...", "tail": "..."}, ...] or [],
  "confidence": <float 0..1>,
  "notes": "<model brief note>"
}
"""
from __future__ import annotations
import os, sys, json, time, re, math
from typing import Dict, List, Any, Optional, Tuple

# ---------- (A) Prompt builder ----------

SYSTEM_INSTRUCTION = (
    "You are a precise knowledge-graph reasoning judge. "
    "Only use the provided triples as your world. "
    "Determine whether the gold answer is reachable/entailed from the triples within at most 4 hops. "
    "If reachable, return one shortest plausible path (sequence of triples). "
    "Be strict about entity names but allow minor surface variants (e.g., punctuation/case). "
    "Do NOT invent new facts beyond the triples."
)

USER_PROMPT_TEMPLATE = """\
Question: {question}
Gold Answer: {answer}

Triples (each line is Head \u2192[Relation]\u2192 Tail):
{triple_lines}

Your task:
1) Decide if the Gold Answer is reachable from the triples via a chain of 1-4 triples.
   - A path is a sequence where Tail of step k equals Head of step k+1, or the Gold Answer is directly a Tail (or Head) if that resolves the question unambiguously.
   - Prefer the **shortest** valid path.
2) Output STRICT JSON with this schema (no extra keys, no commentary):
{{
  "reachable": true|false,
  "min_hops": <integer or null>,
  "path": [{{"head":"...","relation":"...","tail":"..."}}, ...],
  "confidence": <0.0..1.0>,
  "notes": "<brief reason under 30 words>"
}}
"""

def _norm_txt(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def build_prompt(sample: Dict[str, Any]) -> str:
    q = _norm_txt(sample["question"])
    a = _norm_txt(sample["answer"])
    triples: List[Tuple[str,str,str]] = []
    for p in sample.get("paragraphs", []):
        if p.get("is_supporting", False):
            for t in p.get("triples", []):
                h = _norm_txt(t.get("Head",""))
                r = _norm_txt(t.get("Relation",""))
                v = _norm_txt(t.get("Tail",""))
                if h and r and v:
                    triples.append((h,r,v))
    triple_lines = "\n".join([f"- {h} →[{r}]→ {v}" for h,r,v in triples]) or "(no triples)"
    return USER_PROMPT_TEMPLATE.format(question=q, answer=a, triple_lines=triple_lines)

# ---------- (B) LLM callers ----------

def call_qwen_dashscope(prompt: str,
                        api_key: Optional[str] = None,
                        model: str = "qwen2.5-72b-instruct",
                        temperature: float = 0.2,
                        max_tokens: int = 800) -> str:
    """
    Requires: pip install dashscope
    Env: export DASHSCOPE_API_KEY=sk-xxx
    """
    try:
        from http import HTTPStatus
        from dashscope import Generation

        api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is missing.")
        os.environ["DASHSCOPE_API_KEY"] = api_key

        # 关键修复：移除或改为 'message'
        resp = Generation.call(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            # result_format="json",  # ❌ 旧写法会触发 400
            result_format="message",  # ✅ 新写法（也可以完全移除此参数）
            max_output_tokens=max_tokens,
        )

        if resp.status_code != HTTPStatus.OK:
            # 更可读的错误信息
            msg = getattr(resp, "message", "") or getattr(resp, "code", "")
            raise RuntimeError(f"DashScope error {resp.status_code}: {msg}")

        # 统一抽取文本：兼容不同形态
        out = resp.output or {}
        # 一些模型会直接提供 out["text"]
        if isinstance(out.get("text"), str) and out["text"].strip():
            return out["text"]

        # 标准 message 结构：choices -> message -> content
        choices = out.get("choices") or []
        if choices:
            msg = choices[0].get("message", {})
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # content 可能是 [{"role":"assistant","content":[{"text":"..."}]}] 这样
                pieces = []
                for seg in content:
                    if isinstance(seg, dict) and "text" in seg and isinstance(seg["text"], str):
                        pieces.append(seg["text"])
                    elif isinstance(seg, str):
                        pieces.append(seg)
                if pieces:
                    return "\n".join(pieces)

        # 兜底：把整个结构转成字符串（不推荐，但不让你卡死）
        return str(out)

    except ImportError:
        raise RuntimeError("dashscope not installed. Run: pip install dashscope")
def call_qwen_dashscope(prompt: str,
                        api_key: Optional[str] = None,
                        model: str = "qwen2.5-72b-instruct",
                        temperature: float = 0.2,
                        max_tokens: int = 800) -> str:
    """
    Requires: pip install dashscope
    Env: export DASHSCOPE_API_KEY=sk-xxx
    """
    try:
        from http import HTTPStatus
        from dashscope import Generation

        api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is missing.")
        os.environ["DASHSCOPE_API_KEY"] = api_key

        # 关键修复：移除或改为 'message'
        resp = Generation.call(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            # result_format="json",  # ❌ 旧写法会触发 400
            result_format="message",  # ✅ 新写法（也可以完全移除此参数）
            max_output_tokens=max_tokens,
        )

        if resp.status_code != HTTPStatus.OK:
            # 更可读的错误信息
            msg = getattr(resp, "message", "") or getattr(resp, "code", "")
            raise RuntimeError(f"DashScope error {resp.status_code}: {msg}")

        # 统一抽取文本：兼容不同形态
        out = resp.output or {}
        # 一些模型会直接提供 out["text"]
        if isinstance(out.get("text"), str) and out["text"].strip():
            return out["text"]

        # 标准 message 结构：choices -> message -> content
        choices = out.get("choices") or []
        if choices:
            msg = choices[0].get("message", {})
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # content 可能是 [{"role":"assistant","content":[{"text":"..."}]}] 这样
                pieces = []
                for seg in content:
                    if isinstance(seg, dict) and "text" in seg and isinstance(seg["text"], str):
                        pieces.append(seg["text"])
                    elif isinstance(seg, str):
                        pieces.append(seg)
                if pieces:
                    return "\n".join(pieces)

        # 兜底：把整个结构转成字符串（不推荐，但不让你卡死）
        return str(out)

    except ImportError:
        raise RuntimeError("dashscope not installed. Run: pip install dashscope")


def call_custom(prompt: str) -> str:
    """
    Plug ANY LLM API here (e.g. OpenAI, Azure, Volc, Moonshot).
    Must return raw string of the model output.
    """
    raise NotImplementedError("Implement your HTTP call here and return the model text.")

# ---------- (C) Output parsing & validation ----------

def parse_json_strict(text: str) -> Dict[str, Any]:
    # Try to find the first {...} JSON block
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("No JSON object found in model output.")
    obj = json.loads(m.group(0))
    # Minimal schema check
    for k in ["reachable","min_hops","path","confidence","notes"]:
        if k not in obj:
            raise ValueError(f"Missing key: {k}")
    if not isinstance(obj["reachable"], bool):
        raise ValueError("reachable must be bool")
    if obj["min_hops"] is not None and not isinstance(obj["min_hops"], int):
        raise ValueError("min_hops must be int or null")
    if not isinstance(obj["path"], list):
        raise ValueError("path must be a list")
    if not (isinstance(obj["confidence"], float) or isinstance(obj["confidence"], int)):
        raise ValueError("confidence must be number")
    return obj

# ---------- (D) Single-sample evaluation ----------

def eval_sample_with_llm(sample: Dict[str, Any],
                         use_qwen: bool = True,
                         qwen_api_key: Optional[str] = None,
                         qwen_model: str = "qwen2.5-72b-instruct",
                         max_retries: int = 3,
                         backoff: float = 2.0) -> Dict[str, Any]:
    prompt = build_prompt(sample)
    last_err = None
    for t in range(max_retries):
        try:
            if use_qwen:
                raw = call_qwen_dashscope(prompt, api_key=qwen_api_key, model=qwen_model)
                # print(f"[DEBUG] Qwen prompt:\n{prompt} \n raw={raw}")
            else:
                raw = call_custom(prompt)
            obj = parse_json_strict(raw)
            return {
                "idx": sample.get("idx"),
                "reachable": bool(obj["reachable"]),
                "min_hops": obj["min_hops"],
                "path": obj["path"],
                "confidence": float(obj["confidence"]),
                "notes": obj.get("notes","")[:200]
            }
        except Exception as e:
            last_err = e
            time.sleep(backoff ** t)
    # Fallback if model keeps failing: mark unreachable with reason
    return {
        "idx": sample.get("idx"),
        "reachable": False,
        "min_hops": None,
        "path": [],
        "confidence": 0.0,
        "notes": f"LLM_error: {repr(last_err)}"
    }

# ---------- (E) Batch runner (JSON/JSONL in, JSONL out) ----------

def load_dataset(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str,Any]] = []
    if path.endswith(".jsonl"):
        with open(path,"r",encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
    else:
        with open(path,"r",encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                items = data
            else:
                items = data.get("data", [])
    # add running index if none
    for i, it in enumerate(items):
        it.setdefault("idx", i)
    return items

def save_jsonl(path: str, rows: List[Dict[str,Any]]) -> None:
    with open(path,"w",encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def run_eval(input_path: str,
             output_path: str,
             use_qwen: bool = True,
             qwen_api_key: Optional[str] = None,
             qwen_model: str = "qwen2.5-72b-instruct") -> None:
    data = load_dataset(input_path)
    results: List[Dict[str,Any]] = []
    print(f"[INFO] Total samples: {len(data)}")
    for i, samp in enumerate(data):
        # if samp['id'] != "2hop__821197_368148":
        #     continue
        res = eval_sample_with_llm(
            samp,
            use_qwen=use_qwen,
            qwen_api_key=qwen_api_key,
            qwen_model=qwen_model
        )
        results.append(res)
        # simple running log
        if i % 10 == 0:
            print(f"[{i}/{len(data)}] idx={res.get('idx')} reachable={res['reachable']} hops={res['min_hops']}")
    save_jsonl(output_path, results)
    # summary
    reach_cnt = sum(1 for r in results if r["reachable"])
    avg_hops = round(sum(r["min_hops"] or 0 for r in results) / max(1, sum(1 for r in results if r["min_hops"])), 3)
    print(f"Done. Reachable: {reach_cnt}/{len(results)} ({reach_cnt/len(results):.1%}), avg hops={avg_hops}")
    print(f"Saved to: {output_path}")

# ---------- (F) CLI ----------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to dataset (.json or .jsonl)")
    ap.add_argument("--output", required=True, help="Output JSONL path for LLM judgments")
    ap.add_argument("--use-qwen", action="store_true", help="Use Qwen DashScope API")
    ap.add_argument("--qwen-model", default="qwen2.5-72b-instruct")
    ap.add_argument("--qwen-key", default=None, help="Override DASHSCOPE_API_KEY")
    args = ap.parse_args()

    run_eval(
        input_path=args.input,
        output_path=args.output,
        use_qwen=args.use_qwen,
        qwen_api_key=args.qwen_key,
        qwen_model=args.qwen_model
    )
