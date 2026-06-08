# Milvus → 26ai Migration TODO

## Context

Milvus is used as the vector store backend for CA-RAG (Context-Aware RAG). All Milvus-specific
code lives in the `context-aware-rag` package. VSS itself only passes connection config to
`ContextManager` and manages the milvus-server process lifecycle.

---

## context-aware-rag changes (primary effort)

### 1. Create `26ai_db.py` (new file)
**Path:** `src/vss_ctx_rag/tools/storage/26ai_db.py`

Extend `StorageTool` and implement the full interface that the rest of the codebase depends on.

Methods required:
- `add_summary(summary: str, metadata: dict)` — called in batch.py:158, 277
- `aadd_summary(summary: str, metadata: dict)` — async variant
- `add_summaries(batch_summary: list, batch_metadata: list)` — batch ingest
- `aget_text_data(fields=["*"], filter="pk > 0") -> list` — called in batch.py:204
- `search(search_query, top_k=1) -> list[dict]` — called in vector_retrieval_func.py:122
- `drop_data(expr="pk > 0")` — called in batch.py:321, vector_retrieval_func.py:103
- `drop_data_filtered(filter) -> int` — called during stream reset
- `drop_collection()` — called in context_manager_handler.py

Attributes required:
- `self.reranker` — NVIDIARerank instance (unchanged, not Milvus-specific)
- `self.vector_db` — must expose `.as_retriever(search_kwargs={"filter": ..., "k": ...})`
  used in vector_retrieval_func.py:67

Milvus-specific patterns being replaced:
```python
# These internal col accesses must be reimplemented via 26ai API
self.vector_db.col.query(expr=filter, output_fields=fields)  # aget_text_data
self.vector_db.col.delete(expr=expr)                          # drop_data / drop_data_filtered
self.vector_db.col.flush()                                    # after delete
self.vector_db.as_retriever(search_kwargs={"filter": {"doc_type": "caption"}})
```

Filter expressions used across the codebase (must be supported or translated):
- `"pk > 0"` — delete all
- `"doc_type == 'caption_summary' and batch_i in [1,2,3]"` — query by type + index
- `"doc_type == 'caption'"` — retriever filter
- `{"uuid": stream_id}` — per-stream isolation (passed as dict in some resets)

### 2. Update `tools/storage/__init__.py`
**Path:** `src/vss_ctx_rag/tools/storage/__init__.py`

- Export new `26AIDBTool` class
- Remove or keep `MilvusDBTool` export (remove once migration is confirmed working)

### 3. Update `context_manager_handler.py`
**Path:** `src/vss_ctx_rag/context_manager/context_manager_handler.py`

- Line 38: swap `MilvusDBTool` import for `26AIDBTool`
- Line 95: update type annotation `self.milvus_db: MilvusDBTool` → `26AIDBTool`
- Lines 159-167: replace `MilvusDBTool(...)` instantiation with `26AIDBTool(...)`
- Config keys: swap `config["milvus_db_host"]` / `config["milvus_db_port"]` for 26ai equivalents

### 4. Update `vector_retrieval_func.py`
**Path:** `src/vss_ctx_rag/functions/rag/vector_rag/vector_retrieval_func.py`

- Line 31: update import from `milvus_db` to new module
- Line 42: update type annotation `vector_db: MilvusDBTool`
- Lines 63/67: verify `self.vector_db.reranker` and `self.vector_db.vector_db.as_retriever(...)` still work

### 5. Update `pyproject.toml`
**Path:** `pyproject.toml`

- Remove: `langchain_milvus==0.1.5`, `pymilvus==2.4.4`
- Add: 26ai Python client package + version

---

## vss-oci changes (minimal, plumbing only)

### 6. Update `start_via.sh`
**Path:** `src/vss-engine/start_via.sh`

- Remove `start_milvus()` function (lines 180-206)
- Remove `check_milvus()` function (lines 161-178)
- Remove `MILVUS_DB_HOST` / `MILVUS_DB_PORT` env vars (lines 25-31)
- Remove `start_milvus` call from `start_processes()` (line 370)
- Remove milvus arg from `EXTRA_ARGS` (line 294)
- Remove "Disabling milvus" log line (lines 355-356)
- Add 26ai connection env vars and any required startup steps

### 7. Update `via_stream_handler.py`
**Path:** `src/vss-engine/src/via_stream_handler.py`

- Lines 562-563: replace `config["milvus_db_host"]` / `config["milvus_db_port"]` with 26ai config keys
- Lines 2774-2784: rename `--milvus-db-port` / `--milvus-db-host` CLI args
- Line 65: rename / verify `MAX_MILVUS_STRING_LEN` — confirm 26ai field length limit

### 8. Update all `compose.yaml` files
**Paths:** `deploy/docker/*/compose.yaml` (5 files)

- Remove `MILVUS_DATA_DIR` volume mount
- Remove `MILVUS_DB_HOST` / `MILVUS_DB_PORT` env vars
- Add 26ai connection env vars

### 9. Update all `.env` files
**Paths:** `deploy/docker/*/.env` (5 files)

- Replace Milvus vars with 26ai equivalents

### 10. Update `deploy/helm/scripts/override_remote_endpoints.sh`
**Path:** `deploy/helm/scripts/override_remote_endpoints.sh`

- Lines 85-93: remove `check-milvus-up` init-container
- Add 26ai readiness check if needed

### 11. Update `Dockerfile`
**Path:** `src/vss-engine/docker/Dockerfile`

- Line 19: update `context-aware-rag` clone to point to forked repo / new version with 26ai support

---

## Blockers / open questions

- [ ] What is 26ai's Python client library name and connection interface? -- oracledb
- [ ] Does 26ai have a LangChain vector store adapter? (determines implementation complexity) -- yes
- [ ] How does 26ai handle filtered queries / deletes? (need to translate Milvus expression syntax) -- common sql, it's Oracle AI Database
- [ ] Does 26ai have a concept of named collections for per-stream isolation? -- it should
- [ ] What are 26ai's field length limits? (replaces `MAX_MILVUS_STRING_LEN = 65535`) --they should be similar
- [ ] Is the Helm chart binary (`nvidia-blueprint-vss-2.3.1.tgz`) hardcoded with a Milvus sidecar? -- unsure
