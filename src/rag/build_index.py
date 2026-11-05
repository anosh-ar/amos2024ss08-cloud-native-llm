"""
Phase A of the RAG pipeline: build the vector index.

Loads the scraped CNCF corpus from Hugging Face, turns every chunk into a LangChain
`Document` (keeping project/file metadata for citations and filtering), re-splits
chunks to fit the embedding model, embeds them locally and stores them in Chroma.

Re-running is safe: every piece of text gets a deterministic id, so existing
entries are overwritten instead of duplicated.

Examples (run from the repository root):
    python src/rag/build_index.py --list-projects
    python src/rag/build_index.py --projects cilium istio envoy      # pilot index
    python src/rag/build_index.py --all                              # full index (slow on CPU)
"""

import argparse
import hashlib
import shutil
import sys
from collections import Counter

from datasets import load_dataset
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from transformers import AutoTokenizer

import config
from vectorstore import get_vectorstore

MIN_CHARS = 50  # skip near-empty fragments (headers, stray lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the DeepCNCF RAG vector index.")
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--projects", nargs="+", metavar="NAME",
                       help="only index these CNCF projects (case-insensitive project_name)")
    scope.add_argument("--all", action="store_true", help="index the whole corpus")
    scope.add_argument("--list-projects", action="store_true",
                       help="print project names with their file counts and exit")
    parser.add_argument("--limit-files", type=int, default=None,
                        help="stop after this many files (quick smoke tests)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="documents embedded and written per batch")
    parser.add_argument("--reset", action="store_true",
                        help="delete the existing index before building")
    return parser.parse_args()


def clean_metadata(tag: dict) -> dict:
    """Chroma only accepts str/int/float/bool metadata values (no None)."""
    return {key: ("" if value is None else str(value)) for key, value in tag.items()}


def doc_id(meta: dict, chunk_idx: int, piece_idx: int, text: str) -> str:
    # File names are not unique within a project, so the text is part of the id.
    raw = f"{meta.get('project_name')}|{meta.get('file_name')}|{chunk_idx}|{piece_idx}|{text}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def iter_documents(dataset, splitter):
    """Yield (id, Document) for every re-split piece of every chunk, skipping duplicates."""
    seen = set()
    for row in dataset:
        meta = clean_metadata(row["tag"])
        for chunk_idx, chunk in enumerate(row["content"]):
            text = (chunk.get("data") or "").strip()
            if len(text) < MIN_CHARS:
                continue
            for piece_idx, piece in enumerate(splitter.split_text(text)):
                if len(piece) < MIN_CHARS:
                    continue
                _id = doc_id(meta, chunk_idx, piece_idx, piece)
                if _id in seen:
                    continue
                seen.add(_id)
                yield _id, Document(
                    page_content=piece,
                    metadata={**meta, "chunk_index": chunk_idx, "piece_index": piece_idx},
                )


def main() -> None:
    args = parse_args()

    print(f"Loading dataset {config.DATASET_ID} ...")
    dataset = load_dataset(config.DATASET_ID, split="train")

    if args.list_projects:
        counts = Counter(tag["project_name"] for tag in dataset["tag"])
        for name, count in sorted(counts.items(), key=lambda item: (-item[1], str(item[0]))):
            print(f"{count:6d}  {name}")
        print(f"\n{len(counts)} projects, {len(dataset)} files")
        return

    if args.projects:
        wanted = {name.lower() for name in args.projects}
        dataset = dataset.filter(lambda row: str(row["tag"]["project_name"]).lower() in wanted)
        found = {str(name).lower() for name in set(row["project_name"] for row in dataset["tag"])}
        missing = wanted - found
        if missing:
            print(f"Warning: no files found for {sorted(missing)} "
                  f"(see --list-projects for valid names)", file=sys.stderr)
        if len(dataset) == 0:
            sys.exit("Nothing to index.")

    if args.limit_files:
        dataset = dataset.select(range(min(args.limit_files, len(dataset))))
    print(f"Indexing {len(dataset)} files")

    if args.reset and config.INDEX_DIR.exists():
        print(f"Removing existing index at {config.INDEX_DIR}")
        shutil.rmtree(config.INDEX_DIR)

    # Split by *embedding-model tokens* so no piece is silently truncated at embed time.
    tokenizer = AutoTokenizer.from_pretrained(config.EMBED_MODEL)
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, chunk_size=config.CHUNK_TOKENS, chunk_overlap=config.CHUNK_OVERLAP
    )

    store = get_vectorstore()
    ids, docs, total = [], [], 0
    progress = tqdm(iter_documents(dataset, splitter), unit=" chunks", desc="Embedding")
    for _id, doc in progress:
        ids.append(_id)
        docs.append(doc)
        if len(docs) >= args.batch_size:
            store.add_documents(docs, ids=ids)
            total += len(docs)
            ids, docs = [], []
    if docs:
        store.add_documents(docs, ids=ids)
        total += len(docs)

    print(f"Done: {total} chunks written. Collection now holds "
          f"{store._collection.count()} chunks in {config.INDEX_DIR}")


if __name__ == "__main__":
    main()
