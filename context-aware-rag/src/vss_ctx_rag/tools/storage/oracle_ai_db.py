# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import os
import re
from typing import Optional

import oracledb
from langchain_community.vectorstores import oraclevs
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.docstore.document import Document
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, NVIDIARerank
from langchain.text_splitter import RecursiveCharacterTextSplitter

from vss_ctx_rag.tools.storage.storage_tool import StorageTool
from vss_ctx_rag.utils.ctx_rag_logger import TimeMeasure, logger


def _milvus_expr_to_oracle_sql(expr) -> str:
    """Translate a Milvus-style filter expression to an Oracle SQL WHERE fragment.

    Supported patterns:
      - ``"pk > 0"``                                  -> ``"1=1"`` (match all)
      - dict ``{"doc_type": "caption"}``              -> JSON_VALUE conditions
      - ``"field == 'val' and other in [1,2,3]"``     -> JSON_VALUE conditions
    """
    if isinstance(expr, dict):
        parts = []
        for k, v in expr.items():
            if isinstance(v, str):
                parts.append(f"JSON_VALUE(METADATA, '$.{k}') = '{v}'")
            else:
                parts.append(f"JSON_VALUE(METADATA, '$.{k}') = {v}")
        return " AND ".join(parts) if parts else "1=1"

    if not isinstance(expr, str) or expr.strip() == "pk > 0":
        return "1=1"

    result = expr

    # field == 'value'  or  field == "value"
    def _eq_sub(m):
        val = m.group(2).replace('"', "'")
        return f"JSON_VALUE(METADATA, '$.{m.group(1)}') = {val}"

    result = re.sub(
        r'(\w+)\s*==\s*(\'(?:[^\'\\]|\\.)*\'|"(?:[^"\\]|\\.)*")',
        _eq_sub,
        result,
    )

    # field in [1, 2, 3]
    result = re.sub(
        r"(\w+)\s+in\s+\[([^\]]+)\]",
        lambda m: f"JSON_VALUE(METADATA, '$.{m.group(1)}') IN ({m.group(2)})",
        result,
        flags=re.IGNORECASE,
    )

    # lowercase 'and' -> uppercase AND
    result = re.sub(r"\band\b", "AND", result, flags=re.IGNORECASE)

    return result


class OracleAIDBTool(StorageTool):
    """Handler for Oracle AI Vector Search that stores video embeddings
    mapped to summary text embeddings for context-aware RAG retrieval.

    Implements the StorageTool interface as a drop-in replacement for
    MilvusDBTool.
    """

    def __init__(
        self,
        collection_name: str,
        host: Optional[str] = None,
        port: str = "1521",
        embedding_model_name: str = "nvidia/llama-3.2-nv-embedqa-1b-v2",
        embedding_base_url: str = "https://integrate.api.nvidia.com/v1",
        reranker_model_name: str = "nvidia/llama-3.2-nv-rerankqa-1b-v2",
        reranker_base_url: str = (
            "https://ai.api.nvidia.com/v1/retrieval/nvidia/"
            "llama-3_2-nv-rerankqa-1b-v2/reranking"
        ),
        name: str = "oracle_ai_db",
    ) -> None:
        super().__init__(name)

        api_key = os.getenv("NVIDIA_API_KEY") or "NOAPIKEYSET"

        db_user = os.getenv("ORACLE_AI_DB_USER", "admin")
        db_password = os.getenv("ORACLE_AI_DB_PASSWORD", "")
        db_host = host or os.getenv("ORACLE_AI_DB_HOST", "127.0.0.1")
        db_port = port or os.getenv("ORACLE_AI_DB_PORT", "1521")
        db_service = os.getenv("ORACLE_AI_DB_SERVICE_NAME", "orcl")

        # Oracle identifiers: uppercase, alphanumeric + underscore, max 128 chars
        self.collection_name = re.sub(r"[^A-Z0-9_]", "_", collection_name.upper())[:128]

        dsn = f"{db_host}:{db_port}/{db_service}"
        self._pool = oracledb.create_pool(
            user=db_user,
            password=db_password,
            dsn=dsn,
            min=2,
            max=10,
            increment=1,
        )

        self.embedding = NVIDIAEmbeddings(
            model=embedding_model_name,
            truncate="END",
            api_key=api_key,
            base_url=embedding_base_url,
        )

        drop_old = (
            os.getenv("VIA_CTX_RAG_ENABLE_RET", "True").lower() not in ["true", "1"]
        )
        if drop_old:
            try:
                with self._pool.acquire() as conn:
                    oraclevs.drop_table_purge(conn, self.collection_name)
            except Exception:
                pass

        # Acquire a dedicated connection for OracleVS to hold for its lifetime.
        # Direct SQL operations each acquire their own connection from the pool.
        self._vector_db_conn = self._pool.acquire()
        self.vector_db = OracleVS(
            client=self._vector_db_conn,
            embedding_function=self.embedding,
            table_name=self.collection_name,
            distance_strategy=DistanceStrategy.DOT_PRODUCT,
        )

        self.reranker = NVIDIARerank(
            model=reranker_model_name,
            api_key=api_key,
            base_url=reranker_base_url,
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", ";", ",", " ", ""],
        )

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_summary(self, summary: str, metadata: dict):
        with TimeMeasure("oracleaidb/add_summary", "blue"):
            doc = Document(page_content=summary, metadata=metadata)
        return self.vector_db.add_documents([doc])

    async def aadd_summary(self, summary: str, metadata: dict):
        with TimeMeasure("oracleaidb/aadd_summary", "blue"):
            doc = Document(page_content=summary, metadata=metadata)
            return await self.vector_db.aadd_documents([doc])

    def add_summaries(self, batch_summary: list, batch_metadata: list):
        with TimeMeasure("OracleAIDB/add_summaries", "yellow"):
            if len(batch_summary) != len(batch_metadata):
                raise ValueError(
                    "Incorrect param. The length of batch_summary and "
                    "metadata batch should match."
                )
            docs = [
                Document(page_content=batch_summary[i], metadata=batch_metadata[i])
                for i in range(len(batch_summary))
            ]
            document_chunks = self.text_splitter.split_documents(docs)
            self.vector_db.add_documents(document_chunks)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def aget_text_data(self, fields: Optional[list] = None, filter: str = "pk > 0") -> list:
        if fields is None:
            fields = ["*"]
        await asyncio.sleep(0.001)
        sql_filter = _milvus_expr_to_oracle_sql(filter)
        try:
            with self._pool.acquire() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        f"SELECT METADATA, TEXT FROM {self.collection_name} "  # noqa: S608
                        f"WHERE {sql_filter}"
                    )
                    rows = cursor.fetchall()
        except Exception as e:
            logger.error(f"aget_text_data query failed: {e}")
            return []

        results = []
        for metadata_raw, text in rows:
            try:
                meta = (
                    json.loads(metadata_raw)
                    if isinstance(metadata_raw, str)
                    else (metadata_raw or {})
                )
            except (json.JSONDecodeError, TypeError):
                meta = {}
            meta["text"] = text
            if fields == ["*"]:
                results.append(dict(meta.items()))
            else:
                results.append({k: meta[k] for k in fields if k in meta})
        return results

    def search(self, search_query, top_k=1) -> list:
        search_results = self.vector_db.similarity_search(search_query, k=top_k)
        return [result.metadata for result in search_results]

    # ------------------------------------------------------------------
    # Delete / reset operations
    # ------------------------------------------------------------------

    def drop_data(self, expr="pk > 0"):
        sql_filter = _milvus_expr_to_oracle_sql(expr)
        try:
            with self._pool.acquire() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        f"DELETE FROM {self.collection_name} WHERE {sql_filter}"  # noqa: S608
                    )
                conn.commit()
        except Exception as e:
            logger.error(f"drop_data failed: {e}")

    def drop_data_filtered(self, filter) -> int:
        sql_filter = _milvus_expr_to_oracle_sql(filter)
        try:
            with self._pool.acquire() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        f"DELETE FROM {self.collection_name} WHERE {sql_filter}"  # noqa: S608
                    )
                    count = cursor.rowcount
                conn.commit()
            return count
        except Exception as e:
            logger.error(f"drop_data_filtered failed: {e}")
            return 0

    def drop_collection(self):
        try:
            with self._pool.acquire() as conn:
                oraclevs.drop_table_purge(conn, self.collection_name)
        except Exception as e:
            logger.warning(f"drop_collection: could not purge table: {e}")
        self.vector_db = OracleVS(
            client=self._vector_db_conn,
            embedding_function=self.embedding,
            table_name=self.collection_name,
            distance_strategy=DistanceStrategy.DOT_PRODUCT,
        )
