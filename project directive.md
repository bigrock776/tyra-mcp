# Claude Integration Prompt: Merge Advanced RAG into Cole's mem0 MCP Template for Tyra

## 🧠 Objective

You will **fully merge** the current refactored **Advanced RAG system** into Cole Medin’s [`mem0` MCP memory server template](https://github.com/coleam00/mem0), replacing all memory-related logic and tools with Tyra's architecture. The result will be a **modular, agent-ready MCP server** powering Tyra’s genius-level long-term memory.

---

## 🛠️ Core Goals

* Replace all `mem0` internal memory logic with Tyra’s current **Advanced RAG** architecture.
* Preserve full agent-based functionality of `mem0`.
* Implement local **HuggingFace embeddings with fallback**.
* Use **PostgreSQL + pgvector** for memory vector store.
* Use **Memgraph** for the knowledge graph backend.
* Retain and improve on **hallucination detection, reranking, and scoring**.
* Expose endpoints through FastAPI with clean modular structure.
* Enable multi-agent access (Tyra, Claude, Archon, etc.).
* Allow easy ingestion from n8n and document APIs.

---

## 🔁 Replacement Strategy

### 1. Replace `mem0.memory` Logic

Remove or disable the existing `Supabase`, `Chroma`, or abstracted memory modules and replace with:

* `postgres_memory.py` (uses pgvector, Psycopg2 or SQLAlchemy)
* `retriever.py` (with FAISS or raw cosine sim from pgvector)
* `embedder.py` (uses `sentence-transformers` + fallback logic)

> 🔍 Example:
>
> ```python
> def get_embedding(text): 
>     try:
>         return embedder.encode(text)
>     except Exception:
>         return fallback_model.encode(text)
> ```

---

### 2. Replace Graph Backend with `Memgraph`

Update all `graph.py` and Cypher-related logic to use `Memgraph`. Do not use Neo4j or any hosted solutions.

> 🔍 Example:
>
> ```python
> from gqlalchemy import Memgraph
> memgraph = Memgraph()
> memgraph.execute("MATCH (n)-[r]->(m) RETURN n,r,m")
> ```

---

### 3. Move RAG Flow into `mem0.agentic_rag`

Port over the RAG logic:

* Reranker with hallucination filter
* Chunk merging
* Source scoring

Preserve:

* Structured JSON answers
* Scoring keys: `confidence_score`, `hallucination_flag`, `safe_to_act_on`

---

### 4. Expose via FastAPI Routes

Expose FastAPI routes using the structure below:

```
/memory/embed
/memory/retrieve
/memory/rerank
/memory/hallucination
/memory/graph-query
```

Each should:

* Accept and return JSON
* Log timestamps and agent ID (optional)
* Return a default structure with:

  * `answer`
  * `confidence_score`
  * `source_chunks`
  * `hallucination_flag`

---

### 5. Remove All `mem0` Memory Tooling

Remove or disable:

* Supabase clients
* Abstract embedding tools that aren't HuggingFace
* Any mention of Chroma, Pinecone, or Langchain

---

### 6. Enhance for Tyra Use

* Agent-aware scoring (tag data by agent only if required)
* Shared memory between Tyra, Claude, Archon (default: YES)
* Add `hallucination_confidence: 1-100` and `hallucination_flag: True/False` to responses
* Create shared `config.yaml` to toggle components

> 🔍 Example config:
>
> ```yaml
> memory_backend: postgres
> vector_backend: pgvector
> graph_backend: memgraph
> embedder: BAAI/bge-small-en-v1.5
> hallucination_scoring: true
> fallback_enabled: true
> rerank_enabled: true
> ```

---

## 🧠 Tyra Integration

Tyra will access this MCP server using an internal `memory_client.py` module.

Endpoints must:

* Be local-only (no external exposure)
* Accept structured queries and return scores
* Be resilient to failure and fallbacks

Tyra will:

* Ask `/retrieve` to get top memory chunks
* Inject retrieved memory into LLM prompt
* Ask `/hallucination` to score her answer
* Use `/graph-query` to understand relationships
* Only act on `safe_to_act_on: true`

---

## 📈 Observability: OpenTelemetry Integration

The MCP must be fully instrumented using **OpenTelemetry** for full tracing, debugging, performance analysis, and hallucination inspection.

### 📌 Requirements:

* Instrument all FastAPI endpoints and internal logic
* Add trace spans for: embedding, fallback, vector search, reranking, hallucination scoring, graph querying, memory logging
* Add `agent_id`, `tool_used`, `model_name`, `fallback_triggered`, `hallucination_score` as trace attributes
* Use `.env` or `settings.yaml` for export target (console, Jaeger, Prometheus)

> 🔍 Example:
>
> ```python
> with tracer.start_as_current_span("reranker_process"):
>     result = rerank_top_k(...)
> ```

* Optionally expose `/telemetry/status` for inspection
* Must be fail-safe and switchable off in config

---

## 🧱 Scalability & Extensibility Requirements

To ensure this MCP memory server is maintainable and extensible for future tools, agents, and components, **follow these architecture principles**:

### ✅ 1. Modular Folder Structure

Organize all logic into versioned, atomic modules:

```
/mcp/
├── api/                 # FastAPI routers
│   ├── retrieve.py
│   ├── embed.py
│   ├── hallucinate.py
├── core/                # Core memory and logic engines
│   ├── vectorstore/
│   ├── graph/
│   ├── reranker/
│   ├── hallucination/
│   ├── embedder/
│   └── utils/
├── config/
│   ├── config.yaml
│   └── agents.yaml
├── clients/             # Memory clients for agents like Tyra
│   └── memory_client.py
```

### ✅ 2. Config-Driven Everything

Move all model names, endpoint settings, and agent behavior rules into `config.yaml`. Tyra or Claude should never need to modify code just to change an embedder or scoring flag.

> 🔍 Example:
>
> ```yaml
> embedder: BAAI/bge-small-en-v1.5
> fallback_enabled: true
> scoring:
>   rerank_enabled: true
>   hallucination_threshold: 75
> ```

### ✅ 3. Pluggable Tools

Every functional unit (embedder, hallucination engine, reranker, etc.) must be swappable via interfaces or classes. For example:

```python
class Embedder:
    def embed(self, text): ...
```

### ✅ 4. Versioning Support

Add:

```yaml
api_version: v1
```

And expose routes like:

```
/v1/memory/retrieve
```

Prepare for future `/v2/` versions or agent-specific overrides.

### ✅ 5. Logging + Audit Trail

Store every request/response pair in PostgreSQL or local JSON logs for future fine-tuning, bug tracing, or backtesting.

### ✅ 6. Compatible with n8n and other AI tools

Ensure the API endpoints:

* Accept standard JSON
* Can be called by n8n, Curl, Postman
* Return machine-readable results with `score`, `answer`, `source`, and `flags`

### ✅ 7. Hot-Swappable Components

Design the architecture so future agents, tools, or LLMs can be integrated without major refactoring:

* All major tools (embedder, hallucination checker, reranker) must be injectable from config or constructor
* Graph backend and vector search backend must be modular and swappable without logic rewrites

---

## 📌 Final Note

The system must be **modular, scalable, versioned, config-driven, and future-proof**. Prioritize extensibility in all designs and make it easy for future agents or tools (like Archon, n8n, or new retrievers) to plug into the memory MCP with minimal refactoring.

> Always use the **most current, modular, and high-performance architecture** available at the time of implementation. Prioritize clean, local-first, open-source tools and frameworks optimized for accuracy, transparency, and speed.

> Tyra must always be capable of adapting, self-learning, and expanding her cognitive footprint without architectural rewrites.

