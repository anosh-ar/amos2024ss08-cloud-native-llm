"""
Phase B of the RAG pipeline: retrieve relevant chunks and let the LLM answer from them.

    question ─► retriever (Chroma, top-k) ─► prompt with numbered context ─► LLM ─► answer + sources

The LLM is any OpenAI-compatible server (Ollama, LocalAI, llama.cpp), configured in config.py.
"""

from typing import Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

import config
from vectorstore import get_vectorstore

# Gemma has no "system" role, so all instructions go into the single user turn.
RAG_PROMPT = ChatPromptTemplate.from_messages([(
    "human",
    "You are DeepCNCF, an assistant for projects of the Cloud Native Computing Foundation (CNCF).\n"
    "Answer the question using only the context below. If the context does not contain the answer, "
    "say that you don't know instead of guessing. Refer to the context passages by their number, "
    "like [1].\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}",
)])

PLAIN_PROMPT = ChatPromptTemplate.from_messages([("human", "{question}")])


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
    )


def get_retriever(k: int = config.TOP_K, project: Optional[str] = None):
    """Similarity retriever; `project` restricts results to one CNCF project."""
    search_kwargs = {"k": k}
    if project:
        search_kwargs["filter"] = {"project_name": project}
    return get_vectorstore().as_retriever(search_kwargs=search_kwargs)


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{i}] ({doc.metadata.get('project_name')} / {doc.metadata.get('file_name')})\n"
        f"{doc.page_content}"
        for i, doc in enumerate(docs, start=1)
    )


def format_sources(docs: list[Document]) -> list[str]:
    return [f"[{i}] {d.metadata.get('project_name')} / {d.metadata.get('file_name')}"
            for i, d in enumerate(docs, start=1)]


def build_rag_chain(k: int = config.TOP_K, project: Optional[str] = None,
                    prompt: ChatPromptTemplate = RAG_PROMPT) -> Runnable:
    """
    Chain: question (str) -> {"question", "docs", "answer"}.

    The retrieved docs are kept in the output so callers can show citations.
    """
    retriever = get_retriever(k=k, project=project)
    generate = (
        RunnableLambda(lambda x: {"context": format_docs(x["docs"]), "question": x["question"]})
        | prompt
        | get_llm()
        | StrOutputParser()
    )
    return (
        {"docs": retriever, "question": RunnablePassthrough()}
        | RunnablePassthrough.assign(answer=generate)
    )


def build_plain_chain(prompt: ChatPromptTemplate = PLAIN_PROMPT) -> Runnable:
    """Baseline without retrieval: question (str) -> answer (str)."""
    return {"question": RunnablePassthrough()} | prompt | get_llm() | StrOutputParser()
