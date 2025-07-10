# 📥 Tyra MCP – Enhanced Document Ingestion Refactor Specification

**Date:** 2025-07-10  
**Author:** Creator  
**Purpose:** Upgrade Tyra MCP's ingestion capabilities to support *all common document formats* via API or n8n, with chunking, embedding, and memory logging via PostgreSQL/pgvector and Memgraph. This file gives **Claude** all instructions to avoid ambiguity or implementation gaps. **No guesswork allowed**.

---

## ✅ Goals

1. Ingest documents from n8n or HTTP API (local file, URL, or base64).
2. Support formats: `.pdf`, `.docx`, `.pptx`, `.txt`, `.md`, `.html`, `.json`, `.csv`, `.epub`.
3. Chunk content based on file type (dynamic strategy per type).
4. Embed chunks locally (HuggingFace model) with LLM-enhanced context.
5. Store vectors in `pgvector`, metadata in PostgreSQL, graph relationships in Memgraph.
6. Return success/failure with optional hallucination risk flag.
7. Use Pydantic for all request and response validation.
8. Enable modularity: new formats/tools easily added later.

---

## 📦 Input API Requirements

### Endpoint

```
POST /v1/ingest/document
```

### Request Schema (Pydantic)

```python
class IngestRequest(BaseModel):
    source_type: Literal["file", "url", "base64"]
    file_name: str
    file_type: Literal["pdf", "docx", "pptx", "txt", "md", "html", "json", "csv", "epub"]
    content: Optional[str] = None  # base64 or plain text
    file_url: Optional[HttpUrl] = None
    source_agent: Optional[str] = "tyra"
    description: Optional[str] = ""
    chunking_strategy: Optional[str] = "auto"
    chunk_size: Optional[int] = 512
    chunk_overlap: Optional[int] = 50
```

---

## 🧠 Embedding Pipeline

### Logic Flow

1. Validate with Pydantic.
2. Route to file-type-specific loader:
   - PDF → `pdfminer.six` or `PyMuPDF`
   - DOCX → `python-docx`
   - PPTX → `python-pptx`
   - TXT/MD/HTML/CSV/JSON → native + `html2text` or `json.loads`
3. Auto-detect and apply best chunking strategy:
   - Paragraph → `.docx`, `.md`, `.txt`
   - Slide → `.pptx`
   - Semantic → `.pdf`, `.epub`
   - Line/token → `.csv`, `.json`
4. Use local embedding model:
   - Default: `sentence-transformers/all-MiniLM-L6-v2`
   - If fails: fallback to `intfloat/e5-large` or similar
   - Optional reranking: `BM25 + semantic reranker`
5. Inject LLM-enhanced context (using vLLM), example format:
   ```
   {chunk_text} — Contextualized by Tyra’s understanding of prior related memory and topic.
   ```
6. Store each chunk with:
   - Vector → pgvector
   - Metadata → PostgreSQL
   - Graph → Memgraph (document relations, themes, origin)

---

## 🧾 Metadata Schema Example

```python
class EmbeddedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_text: str
    tokens: int
    embedding: List[float]
    similarity_score: Optional[float]
    hallucination_score: Optional[float]
    created_at: datetime
    file_type: str
    chunk_index: int
    source: str
    ingestion_method: str
```

---

## 🛠 Storage + Graph Updates

- Use `pgvector` for embeddings (`vector_chunks` table)
- Use `PostgreSQL` for metadata (`chunk_metadata`, `documents`, `sources`)
- Use `Memgraph` with `graphiti-memgraph` for document + concept linking:
  - Example nodes:
    - `Document`, `Agent`, `Concept`, `Chunk`
  - Edges:
    - `EMBEDS`, `AUTHORED_BY`, `REFERS_TO`, `DERIVED_FROM`

---

## 🔒 Validation + Feedback

- All endpoints must return:
  - `status`: `success | fail`
  - `summary`: brief log string
  - `chunks_ingested`: int
  - `warnings`: optional (e.g., chunk failure, fallback triggered)
- Confidence scoring returned (hallucination flags)

---

## 🔁 Future Add-ons (Design For)

- OCR (`tesserocr`, `pytesseract`) for images/PDFs with no text layer
- Audio transcription (`whisper`)
- Web crawling ingestion via `crawl4ai`
- User-defined chunking or pre-ingested HTML content (e.g., from n8n)

---

## ✅ Claude Must Ensure

- Code modularity — easy to add more formats and embeddings
- Use try/except for ALL ingest/parse operations
- Ensure logs are structured via `structlog`
- Enable concurrent ingest jobs (FastAPI + async I/O)
- Use streaming embedding pipeline if file > 10MB
- Only local models (no OpenAI or remote calls)
- Embed + rerank enabled by default (BM25 fallback)
- LLM context enrichment MUST wrap chunks before embedding
- All failures must log chunk ID, type, and failure point

---

## 🧪 Testing

Create `test_ingest.py` covering:
- All file types
- Missing data / invalid types
- Chunk edge cases (short files, empty)
- Invalid encoding
- Fallback triggering
- Timeout handling

---

## 🚀 Submission

Once complete:
- Update `requirements.txt` if needed
- Document new routes in `README.md`
- Add changelog line `✔ Document ingestion fully refactored`

---

End of specification.
