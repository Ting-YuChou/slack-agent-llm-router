#!/usr/bin/env python3
"""Retrieval evaluation for the RAG pipeline on a small bilingual corpus.

Ingests ``tests/fixtures/rag_retrieval/corpus.json`` through the real
``RagService`` (chunker, vector store, context builder) and asks every query
in ``queries.json``. A query is answered when the chunk holding its
``answer_span`` comes back, so the scores do not depend on chunk ids.

Per query it records three things, each a different stage of the pipeline:

* ``indexed``    - the answer survived chunking and is in the index at all
* ``rank``       - where the answer chunk came back (recall@k and MRR)
* ``in_context`` - the answer is in the text ``build_context`` hands the model

and ``sources_in_context``: how many of the retrieved sources made it into
that text (``coverage`` in the summary is their share).

The default run is offline and deterministic: in-memory store, no embedding
provider, so it measures chunking, the keyword branch and the context budget.
``--backend redis`` runs the same corpus on a Redis Stack under its own key
prefix and removes it afterwards; ``--embedding configured`` uses the embedding
server from the config file to measure hybrid retrieval.
"""

import argparse
import asyncio
import hashlib
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.memory import NoOpEmbeddingProvider  # noqa: E402
from src.rag.parser import (  # noqa: E402
    DocumentBlock,
    DocumentParser,
    ParsedDocument,
)
from src.rag.service import RagService  # noqa: E402

FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "rag_retrieval"
DEFAULT_CONFIG = REPO_ROOT / "config" / "config.yaml"
KNOWLEDGE_BASE_ID = "retrieval-eval"


class KeywordOnlyEmbedding(NoOpEmbeddingProvider):
    """No vectors. Unlike the no-op provider it lets ingestion finish.

    The no-op provider's ``embed_many`` raises, which ingestion treats as a
    failed document; without a batch method every chunk is simply stored
    without an embedding.
    """

    embed_many = None  # type: ignore[assignment]


class FixtureParser(DocumentParser):
    """Turns one corpus entry (already split into blocks) into a document."""

    def parse_bytes(
        self,
        *,
        content: bytes,
        filename: str,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ParsedDocument:
        payload = json.loads(content.decode("utf-8"))
        resolved = document_id or str(payload["document_id"])
        blocks = [
            DocumentBlock(
                doc_id=resolved,
                page=int(block.get("page", 1)),
                bbox=None,
                block_type=str(block["type"]),
                text=str(block["text"]),
                reading_order=order,
            )
            for order, block in enumerate(payload["blocks"], start=1)
        ]
        return ParsedDocument(
            document_id=resolved,
            filename=filename,
            content_hash=hashlib.sha256(content).hexdigest(),
            blocks=blocks,
            metadata=dict(metadata or {}),
        )


def load_fixture(
    directory: Path = FIXTURE_DIR,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    corpus = json.loads((directory / "corpus.json").read_text(encoding="utf-8"))
    queries = json.loads((directory / "queries.json").read_text(encoding="utf-8"))
    return corpus, queries


def pipeline_settings(config_path: Path = DEFAULT_CONFIG) -> Dict[str, Any]:
    """The checked-in chunking, retrieval and embedding settings."""
    rag = dict(
        (yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}).get("rag", {})
    )
    return {
        "chunking": dict(rag.get("chunking", {}) or {}),
        "retrieval": dict(rag.get("retrieval", {}) or {}),
        "embedding": dict(rag.get("embedding", {}) or {}),
    }


def _normalize(text: str) -> str:
    return " ".join(str(text).split())


def _service_config(
    settings: Dict[str, Any],
    *,
    backend: str,
    redis_url: Optional[str],
    key_prefix: str,
    embedding: str,
) -> Dict[str, Any]:
    retrieval = dict(settings.get("retrieval", {}) or {})
    if embedding == "none":
        # Without vectors, recency alone would rank chunks nothing matched.
        retrieval["recency_weight"] = 0.0
    embedding_config = (
        dict(settings.get("embedding", {}) or {})
        if embedding == "configured"
        else {"provider": "none"}
    )
    config: Dict[str, Any] = {
        "enabled": True,
        "backend": "redis_stack" if backend == "redis" else "memory",
        "layout_normalization": {"enabled": False},
        "chunking": dict(settings.get("chunking", {}) or {}),
        "retrieval": retrieval,
        "embedding": embedding_config,
        "rerank": {"enabled": False},
    }
    if backend == "redis":
        config["redis"] = {"url": redis_url, "key_prefix": key_prefix}
    return config


async def _drop_redis_namespace(store: Any, key_prefix: str) -> None:
    client = getattr(store, "client", None)
    if client is None:
        return
    try:
        await client.execute_command("FT.DROPINDEX", store.index_name)
    except Exception:
        pass
    cursor = 0
    while True:
        cursor, keys = await client.scan(cursor, match=f"{key_prefix}:*", count=500)
        if keys:
            await client.delete(*keys)
        if not cursor:
            break


async def run_eval(
    corpus: Sequence[Dict[str, Any]],
    queries: Sequence[Dict[str, Any]],
    *,
    settings: Optional[Dict[str, Any]] = None,
    backend: str = "memory",
    redis_url: Optional[str] = None,
    embedding: str = "none",
) -> Dict[str, Any]:
    """Ingest the corpus, ask every query, and return rows plus a summary."""
    settings = settings if settings is not None else pipeline_settings()
    key_prefix = f"rag_eval_{uuid.uuid4().hex[:12]}"
    service = RagService(
        _service_config(
            settings,
            backend=backend,
            redis_url=redis_url,
            key_prefix=key_prefix,
            embedding=embedding,
        ),
        parser=FixtureParser(),
        embedding_provider=KeywordOnlyEmbedding() if embedding == "none" else None,
    )
    indexed_text: Dict[str, List[str]] = {}
    upsert = service.vector_store.upsert_chunks

    async def recording_upsert(chunks: Any, *args: Any, **kwargs: Any) -> Any:
        for chunk in chunks:
            indexed_text.setdefault(chunk.document_id, []).append(
                _normalize(chunk.text)
            )
        return await upsert(chunks, *args, **kwargs)

    service.vector_store.upsert_chunks = recording_upsert
    await service.initialize()
    try:
        chunk_count = 0
        for document in corpus:
            job = await service.ingest_document(
                content=json.dumps(document, ensure_ascii=False).encode("utf-8"),
                filename=str(document.get("filename") or document["document_id"]),
                knowledge_base_id=KNOWLEDGE_BASE_ID,
                metadata={"title": document.get("title")},
                document_id=str(document["document_id"]),
            )
            if job.status not in {"completed", "completed_with_warnings"}:
                raise RuntimeError(
                    f"ingesting {document['document_id']} failed: {job.error}"
                )
            chunk_count += int(job.chunks_indexed or 0)

        rows = []
        for query in queries:
            span = _normalize(query["answer_span"])
            document_id = str(query["document_id"])
            outcome = await service.retrieve_with_outcome(
                str(query["query"]), knowledge_base_ids=[KNOWLEDGE_BASE_ID]
            )
            results = list(outcome.results)
            rank = next(
                (
                    position
                    for position, result in enumerate(results, start=1)
                    if result.chunk.document_id == document_id
                    and span in _normalize(result.chunk.text)
                ),
                None,
            )
            context = service.build_context(results)
            rows.append(
                {
                    "id": query["id"],
                    "language": query["language"],
                    "kind": query["kind"],
                    "status": outcome.status,
                    "results": len(results),
                    "indexed": any(
                        span in text for text in indexed_text.get(document_id, [])
                    ),
                    "rank": rank,
                    "in_context": span in _normalize(context),
                    "sources_in_context": sum(
                        f"[S{position}] " in context
                        for position in range(1, len(results) + 1)
                    ),
                }
            )
    finally:
        if backend == "redis":
            await _drop_redis_namespace(service.vector_store, key_prefix)
        await service.shutdown()
    return {
        "backend": backend,
        "embedding": embedding,
        "chunking": dict(settings.get("chunking", {}) or {}),
        "max_context_chars": (settings.get("retrieval", {}) or {}).get(
            "max_context_chars"
        ),
        "chunks": chunk_count,
        "rows": rows,
        "summary": summarize(rows),
    }


def _metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    count = len(rows)
    if not count:
        return {
            "n": 0,
            "indexed": 0.0,
            "recall": 0.0,
            "mrr": 0.0,
            "in_context": 0.0,
            "coverage": 0.0,
        }
    retrieved = sum(int(row["results"]) for row in rows)
    return {
        "n": count,
        "indexed": sum(bool(row["indexed"]) for row in rows) / count,
        "recall": sum(row["rank"] is not None for row in rows) / count,
        "mrr": sum(1.0 / row["rank"] for row in rows if row["rank"]) / count,
        "in_context": sum(bool(row["in_context"]) for row in rows) / count,
        "coverage": (
            sum(int(row["sources_in_context"]) for row in rows) / retrieved
            if retrieved
            else 0.0
        ),
    }


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Metrics for every query, per language, and per language and kind."""
    groups: Dict[str, List[Dict[str, Any]]] = {"all": list(rows)}
    for row in rows:
        groups.setdefault(str(row["language"]), []).append(row)
        groups.setdefault(f"{row['language']}/{row['kind']}", []).append(row)
    return {name: _metrics(members) for name, members in sorted(groups.items())}


def format_report(report: Dict[str, Any]) -> str:
    lines = [
        f"backend={report['backend']} embedding={report['embedding']} "
        f"chunks={report['chunks']} chunking={report['chunking']} "
        f"max_context_chars={report['max_context_chars']}",
        f"{'group':<14}{'n':>4}{'indexed':>10}{'recall':>9}{'mrr':>8}{'in_ctx':>9}"
        f"{'coverage':>10}",
    ]
    for name, metrics in report["summary"].items():
        lines.append(
            f"{name:<14}{metrics['n']:>4}{metrics['indexed']:>10.2f}"
            f"{metrics['recall']:>9.2f}{metrics['mrr']:>8.2f}"
            f"{metrics['in_context']:>9.2f}{metrics['coverage']:>10.2f}"
        )
    missed = [
        f"{row['id']}("
        + ",".join(
            label
            for label, bad in (
                ("not indexed", not row["indexed"]),
                ("not retrieved", row["rank"] is None),
                ("not in context", not row["in_context"]),
            )
            if bad
        )
        + ")"
        for row in report["rows"]
        if not (row["indexed"] and row["rank"] and row["in_context"])
    ]
    lines.append("missed: " + (" ".join(missed) if missed else "none"))
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixture", type=Path, default=FIXTURE_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--backend", choices=("memory", "redis"), default="memory")
    parser.add_argument("--redis-url", default="redis://127.0.0.1:6380/0")
    parser.add_argument("--embedding", choices=("none", "configured"), default="none")
    parser.add_argument("--json", action="store_true", help="print the full report")
    args = parser.parse_args(argv)

    corpus, queries = load_fixture(args.fixture)
    report = asyncio.run(
        run_eval(
            corpus,
            queries,
            settings=pipeline_settings(args.config),
            backend=args.backend,
            redis_url=args.redis_url,
            embedding=args.embedding,
        )
    )
    print(
        json.dumps(report, ensure_ascii=False, indent=1)
        if args.json
        else format_report(report)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
