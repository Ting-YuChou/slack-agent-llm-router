# Text-first multimodal RAG

PDF ingestion keeps Docling as the parser and normalizes its typed block stream
before chunking:

```text
Docling blocks
  -> header/footer suppression
  -> footnote linking
  -> page-seam stitching
  -> cross-page table reconciliation
  -> figure/text relationships
  -> structure-aware chunks
```

Figures are cropped at 180 DPI and stored as generation-scoped private assets.
OCR, caption, chart summary, and diagram summary are indexed through the same
BGE-M3 text embedding and BM25 paths as document text. There is no visual
embedding or visual vector search.

At query time, direct figure hits precede figures related to retrieved text or
tables. At most three unique assets are loaded with workload IAM, verified,
resized to a 1568-pixel long edge and 5 MB limit, and sent as Base64 image
blocks to a model that declares the `vision` capability. Public responses expose
only `rag-asset://` references. If no vision model or image is available, the
answer uses indexed OCR/caption text and returns a `rag_vision_fallback`
warning.

## Legacy visual-index removal

Deploy this reader and reindex all documents before deleting legacy visual
vectors. Preview the deletion first:

```bash
python scripts/cleanup_rag_visual_index.py --redis-url "$RAG_REDIS_URL"
```

After reviewing the key count, apply it explicitly:

```bash
python scripts/cleanup_rag_visual_index.py --redis-url "$RAG_REDIS_URL" --apply
```

The cleanup only targets visual chunk keys, visual document sets, and the
legacy visual RediSearch index.
