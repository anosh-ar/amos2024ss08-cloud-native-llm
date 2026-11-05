"""
Ask DeepCNCF questions from the terminal.

Examples (run from the repository root):
    python src/rag/chat_cli.py                                   # interactive, with RAG
    python src/rag/chat_cli.py -q "How does Cilium use eBPF?"    # single question
    python src/rag/chat_cli.py --project istio                   # only search Istio docs
    python src/rag/chat_cli.py --retrieve-only                   # no LLM needed: show retrieved chunks
    python src/rag/chat_cli.py --no-rag                          # plain model, for comparison
"""

import argparse

import config
from rag_chain import build_plain_chain, build_rag_chain, format_sources, get_retriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with DeepCNCF (+ RAG).")
    parser.add_argument("-q", "--question", help="ask one question and exit")
    parser.add_argument("-k", type=int, default=config.TOP_K, help="number of chunks to retrieve")
    parser.add_argument("--project", help="restrict retrieval to one project_name")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--retrieve-only", action="store_true",
                      help="only show the retrieved chunks (no LLM call)")
    mode.add_argument("--no-rag", action="store_true", help="ask the model without retrieval")
    parser.add_argument("--show-context", action="store_true",
                        help="print the retrieved chunks along with the answer")
    return parser.parse_args()


def print_docs(docs) -> None:
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        print(f"\n--- [{i}] {meta.get('project_name')} / {meta.get('file_name')} ---")
        print(doc.page_content[:800] + ("..." if len(doc.page_content) > 800 else ""))


def main() -> None:
    args = parse_args()

    if args.retrieve_only:
        retriever = get_retriever(k=args.k, project=args.project)
        ask = lambda q: print_docs(retriever.invoke(q))
    elif args.no_rag:
        chain = build_plain_chain()
        ask = lambda q: print(f"\n{chain.invoke(q)}")
    else:
        chain = build_rag_chain(k=args.k, project=args.project)

        def ask(question: str) -> None:
            result = chain.invoke(question)
            if args.show_context:
                print_docs(result["docs"])
            print(f"\n{result['answer']}\n\nSources:")
            print("\n".join(format_sources(result["docs"])))

    if args.question:
        ask(args.question)
        return

    print(f"DeepCNCF chat  (LLM: {config.LLM_MODEL} @ {config.LLM_BASE_URL})  - empty line to quit")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        ask(question)


if __name__ == "__main__":
    main()
