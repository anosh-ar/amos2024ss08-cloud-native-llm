# DeepCNCF + RAG (LangChain)

Retrieval-Augmented Generation on top of DeepCNCF. Before the model answers, relevant passages
from the scraped CNCF corpus ([cncf-raw-data-for-llm-training](https://huggingface.co/datasets/Kubermatic/cncf-raw-data-for-llm-training))
are retrieved and put into the prompt, so answers are grounded in the documentation and cite their sources.
Everything runs locally on CPU. No GPU or HPC is needed.

```
PHASE A - indexing (build_index.py, once / when data changes)
  HF dataset ─► Documents + metadata ─► ~400-token chunks ─► bge-small embeddings ─► Chroma (src/rag/index)

PHASE B - querying (rag_chain.py, every question)
  question ─► embed ─► top-k similar chunks ─► prompt with numbered context ─► LLM ─► answer + sources
```

| File | Purpose |
|---|---|
| `config.py` | All settings (overridable via env vars / `src/rag/.env`, see `.env.example`) |
| `vectorstore.py` | Embedding model + Chroma store |
| `build_index.py` | Phase A: build the vector index |
| `rag_chain.py` | Phase B: LangChain retrieval + generation chains (RAG and no-RAG baseline) |
| `chat_cli.py` | Ask questions from the terminal |
| `evaluate.py` | Accuracy on [Benchmark-Questions](https://huggingface.co/datasets/Kubermatic/Benchmark-Questions) with vs. without RAG |
| `ollama/` | Modelfile + script to serve DeepCNCFQuantized with Ollama |

## Setup

From the repository root (Windows paths shown, use `.venv-rag/bin/` on Linux/macOS):

```powershell
python -m venv .venv-rag
.venv-rag\Scripts\python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
.venv-rag\Scripts\python -m pip install -r src/rag/requirements-rag.txt
```

## 1. Build the index

```powershell
.venv-rag\Scripts\python src/rag/build_index.py --list-projects                          # 640 projects
.venv-rag\Scripts\python src/rag/build_index.py --projects cilium envoy istio hubble      # pilot (benchmark projects)
.venv-rag\Scripts\python src/rag/build_index.py --all                                    # full corpus, hours on CPU
```

Re-running is idempotent (deterministic chunk ids), so you can add projects to an existing index.
Use `--reset` to start from scratch.

## 2. Serve the LLM

Any OpenAI-compatible server works. Set `RAG_LLM_BASE_URL` and `RAG_LLM_MODEL` accordingly.

**Ollama (no Docker needed):** install from https://ollama.com/download, then

```powershell
powershell -ExecutionPolicy Bypass -File src\rag\ollama\setup_ollama.ps1   # downloads 5.3 GB GGUF, creates "deepcncf"
```

> Note: `DeepCNCFQuantized` is based on **Gemma 1.1 7B IT** (see its `config.json`), not 2B.
> On a laptop CPU, expect a few tokens per second.

**LocalAI (Docker):** run `src/scripts/GUI/preparation_scripts.sh`, then set
`RAG_LLM_BASE_URL=http://localhost:8080/v1` and `RAG_LLM_MODEL=DeepCNCFQuantized`.
`context_size` in `model_configuration.yaml` was raised from 512 to 4096, because the retrieved context does not fit in 512 tokens.

## 3. Ask questions

```powershell
.venv-rag\Scripts\python src/rag/chat_cli.py --retrieve-only -q "What is Hubble?"   # retrieval only, no LLM
.venv-rag\Scripts\python src/rag/chat_cli.py                                        # interactive chat with RAG
.venv-rag\Scripts\python src/rag/chat_cli.py --project Istio --show-context         # filter by project (exact name)
.venv-rag\Scripts\python src/rag/chat_cli.py --no-rag                               # model alone
```

## 4. Evaluate

```powershell
.venv-rag\Scripts\python src/rag/evaluate.py --limit 5        # quick check
.venv-rag\Scripts\python src/rag/evaluate.py                  # all 41 questions, plain + rag
.venv-rag\Scripts\python src/rag/evaluate.py --model <base-gemma-model-in-ollama>
```

Running it for DeepCNCF and for the base Gemma model fills this table:

| | No retrieval | With RAG |
|---|---|---|
| Base Gemma 1.1 7B IT | | |
| DeepCNCF (fine-tuned) | | |

Per-question outputs (gold, prediction, sources, raw answer) are saved to `src/rag/output/`.
