"""
Compare the model with and without RAG on the Kubermatic/Benchmark-Questions set
(41 multiple-choice questions about Cilium, Istio and Envoy).

Run it once per model to fill the 2x2 table (base Gemma vs DeepCNCF, with/without RAG):
    python src/rag/evaluate.py                              # model from config / .env
    python src/rag/evaluate.py --model gemma2b --limit 5    # quick check with another model

Metrics:
    exact    - predicted letter set equals the gold letter set
    jaccard  - partial credit for "select all that apply" questions
Raw model outputs are written to src/rag/output/ so the letter parsing can be checked.
"""

import argparse
import csv
import re
from datetime import datetime

from datasets import load_dataset
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm

import config

MC_INSTRUCTION = (
    "The following is a multiple-choice question; one or more options may be correct.\n"
    "Reply with only the letter(s) of the correct option(s), comma-separated, e.g. \"Answer: a, c\"."
)

PLAIN_MC_PROMPT = ChatPromptTemplate.from_messages([(
    "human", MC_INSTRUCTION + "\n\nQuestion: {question}",
)])

RAG_MC_PROMPT = ChatPromptTemplate.from_messages([(
    "human",
    "You are DeepCNCF, an assistant for projects of the Cloud Native Computing Foundation (CNCF).\n"
    "Use the context below to answer.\n\nContext:\n{context}\n\n"
    + MC_INSTRUCTION + "\n\nQuestion: {question}",
)])

LETTER = re.compile(r"(?<![a-z])([a-f])(?![a-z])")


def gold_letters(answer: str) -> set[str]:
    """'Answer: a, b, c' / 'Answer: b.' -> {'a', 'b', 'c'} / {'b'}"""
    return set(LETTER.findall(answer.lower().replace("answer", "")))


def predicted_letters(output: str) -> set[str]:
    """Take the letters right after 'Answer:' (or from the first line if it starts with an option)."""
    text = re.sub(r"[*_`]", "", output.lower()).strip()  # drop markdown emphasis
    match = re.search(r"answer\s*[:\-]?\s*(.*)", text)
    if match:
        span = match.group(1)
    elif re.match(r"\(?[a-f](?:[.),]|\s|$)", text):  # e.g. "b." / "a, c" - not prose like "Cilium is a ..."
        span = text
    else:
        return set()
    span = span.splitlines()[0] if span else ""
    # Letters only count if they look like option labels: "a", "a.", "a)", "(a)", "a, b".
    return set(re.findall(r"(?:^|[\s,(])([a-f])(?=$|[\s,.)])", span[:40]))


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DeepCNCF with and without RAG.")
    parser.add_argument("--modes", nargs="+", choices=["plain", "rag"], default=["plain", "rag"])
    parser.add_argument("--model", help="override RAG_LLM_MODEL for this run")
    parser.add_argument("--limit", type=int, help="only use the first N questions")
    parser.add_argument("-k", type=int, default=config.TOP_K)
    args = parser.parse_args()

    if args.model:
        config.LLM_MODEL = args.model
    from rag_chain import build_plain_chain, build_rag_chain  # after config override

    dataset = load_dataset(config.BENCHMARK_DATASET_ID, split="train")
    answer_col = next(c for c in dataset.column_names if c.strip().lower() == "answer")
    rows = list(dataset)[: args.limit] if args.limit else list(dataset)

    chains = {}
    if "plain" in args.modes:
        chains["plain"] = build_plain_chain(prompt=PLAIN_MC_PROMPT)
    if "rag" in args.modes:
        chains["rag"] = build_rag_chain(k=args.k, prompt=RAG_MC_PROMPT)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = config.OUTPUT_DIR / f"eval_{config.LLM_MODEL.replace(':', '_').replace('/', '_')}_{stamp}.csv"

    scores = {mode: {"exact": 0, "jaccard": 0.0} for mode in chains}
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "mode", "gold", "pred", "exact", "jaccard", "sources", "raw_output"])
        for idx, row in enumerate(tqdm(rows, desc="Questions")):
            gold = gold_letters(row[answer_col])
            for mode, chain in chains.items():
                result = chain.invoke(row["Question"])
                if mode == "rag":
                    raw = result["answer"]
                    sources = "; ".join(f"{d.metadata.get('project_name')}/{d.metadata.get('file_name')}"
                                        for d in result["docs"])
                else:
                    raw, sources = result, ""
                pred = predicted_letters(raw)
                exact, jac = pred == gold, jaccard(pred, gold)
                scores[mode]["exact"] += exact
                scores[mode]["jaccard"] += jac
                writer.writerow([idx, mode, ",".join(sorted(gold)), ",".join(sorted(pred)),
                                 int(exact), f"{jac:.2f}", sources, raw])

    n = len(rows)
    print(f"\nModel: {config.LLM_MODEL}   questions: {n}   k: {args.k}")
    for mode, score in scores.items():
        print(f"  {mode:5s}  exact: {score['exact'] / n:6.1%}   jaccard: {score['jaccard'] / n:6.1%}")
    print(f"Details: {out_path}")


if __name__ == "__main__":
    main()
