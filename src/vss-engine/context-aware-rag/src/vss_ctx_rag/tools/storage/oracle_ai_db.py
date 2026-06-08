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
import os
import re
from typing import Any, Dict, List, Optional

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.oraclevs import OracleVS, drop_table_purge
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, NVIDIARerank

from vss_ctx_rag.tools.storage import StorageTool
from vss_ctx_rag.utils.ctx_rag_logger import TimeMeasure, logger


class OracleAIDBTool(StorageTool):
    """Handler for Oracle AI Vector Search which stores video embeddings mapped
    using summary text embeddings for retrieval.

    Drop-in replacement for MilvusDBTool using Oracle Database 23ai.
    Metadata is stored flat in the Oracle JSON column, mirroring Milvus field layout.
    """

    def __init__(
        self,
        collection_name: str,
        host: str = "127.0.0.1",
        port: str = "1521",
        service_name: str = "orcl",
        username: str = "admin",
        password: str = "",
        embedding_model_name: str = "nvidia/llama-3.2-nv-embedqa-1b-v2",
        embedding_base_url: str = "https://integrate.api.nvidia.com/v1",
        reranker_model_name: str = "nvidia/llama-3.2-nv-rerankqa-1b-v2",
        reranker_base_url: str = "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking",
        name: str = "oracle_ai_db",
    ) -> None:
        super().__init__(name)

        api_key = os.getenv("NVIDIA_API_KEY") if os.getenv("NVIDIA_API_KEY") else "NOAPIKEYSET"

        self.collection_name = collection_name
        self._host = host
        self._port = port
        self._service_name = service_name

        self.embedding = NVIDIAEmbeddings(
            model=embedding_model_name,
            truncate="END",
            api_key=api_key,
            base_url=embedding_base_url,
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

        self._connection = self._create_connection(host, port, service_name, username, password)
        self.vector_db = OracleVS(
            client=self._connection,
            embedding_function=self.embedding,
            table_name=collection_name,
        )

    @staticmethod
    def _create_connection(host: str, port: str, service_name: str, username: str, password: str):
        import oracledb
        dsn = f"{host}:{port}/{service_name}"
        return oracledb.connect(user=username, password=password, dsn=dsn)

    @staticmethod
    def _escape(val: str) -> str:
        """Escape single quotes for Oracle SQL string literals."""
        return val.replace("'", "''")

    @staticmethod
    def _translate_filter(milvus_expr: str) -> Optional[str]:
        """Translate a Milvus filter expression to an Oracle SQL WHERE clause.

        Handles the patterns used in vss_ctx_rag:
          - "pk > 0"                                    → None (match all)
          - "field == 'value'"                          → JSON_VALUE = 'value'
          - "field in [1,2,3]"                          → JSON_VALUE IN (1,2,3)
          - combined with 'and' / 'or'
        """
        if not milvus_expr or milvus_expr.strip() == "pk > 0":
            return None

        expr = milvus_expr.strip()

        # Normalize boolean operators
        expr = re.sub(r"\band\b", "AND", expr)
        expr = re.sub(r"\bor\b", "OR", expr)

        # field == 'string_value'
        expr = re.sub(
            r"(\w+)\s*==\s*'([^']*)'",
            lambda m: f"JSON_VALUE(metadata, '$.{m.group(1)}') = '{m.group(2)}'",
            expr,
        )

        # field in [n1, n2, ...]  (numeric list)
        expr = re.sub(
            r"(\w+)\s+in\s+\[([^\]]+)\]",
            lambda m: (
                f"JSON_VALUE(metadata, '$.{m.group(1)}' RETURNING NUMBER) "
                f"IN ({m.group(2)})"
            ),
            expr,
            flags=re.IGNORECASE,
        )

        return expr

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_summary(self, summary: str, metadata: dict):
        with TimeMeasure("oracleai/add caption", "blue"):
            doc = Document(page_content=summary, metadata=metadata)
        try:
            return self.vector_db.add_documents([doc])
        except Exception as e:
            logger.error(f"Error adding document to Oracle AI DB: {metadata}")
            raise e

    async def aadd_summary(self, summary: str, metadata: dict):
        with TimeMeasure("oracleai/add caption", "blue"):
            doc = Document(page_content=summary, metadata=metadata)
            return await self.vector_db.aadd_documents([doc])

    def add_summaries(self, batch_summary: List[str], batch_metadata: List[dict]):
        with TimeMeasure("oracleai/AddSummaries", "yellow"):
            if len(batch_summary) != len(batch_metadata):
                raise ValueError(
                    "Length of batch_summary and batch_metadata must match."
                )
            docs = [
                Document(page_content=text, metadata=meta)
                for text, meta in zip(batch_summary, batch_metadata)
            ]
            document_chunks = self.text_splitter.split_documents(docs)
            self.vector_db.add_documents(document_chunks)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def aget_text_data(
        self, fields: List[str] = ["*"], filter: str = "pk > 0"
    ) -> List[Dict[str, Any]]:
        """Retrieve documents matching a Milvus-style filter expression."""
        await asyncio.sleep(0.001)
        try:
            where_clause = self._translate_filter(filter)
            select_clause = "text, metadata"
            sql = f"SELECT {select_clause} FROM {self.collection_name}"
            if where_clause:
                sql += f" WHERE {where_clause}"

            with self._connection.cursor() as cursor:
                cursor.execute(sql)
                rows = cursor.fetchall()

            results = []
            for text_val, meta_val in rows:
                text = text_val.read() if hasattr(text_val, "read") else (text_val or "")
                meta = dict(meta_val) if isinstance(meta_val, dict) else {}
                row = {"text": text, **meta}
                # Return only requested fields if not wildcard
                if fields != ["*"]:
                    row = {k: row[k] for k in fields if k in row}
                results.append(row)
            return results
        except Exception as e:
            logger.warning(f"Error getting text data: {e}")
            return []

    def search(self, search_query: str, top_k: int = 1) -> List[Dict[str, Any]]:
        search_results = self.vector_db.similarity_search(search_query, k=top_k)
        return [result.metadata for result in search_results]

    def query(self, search_query: str, top_k: int = 1) -> List[Dict[str, Any]]:
        search_results = self.vector_db.similarity_search(search_query, k=top_k)
        return [result.metadata for result in search_results]

    # ------------------------------------------------------------------
    # Delete / lifecycle
    # ------------------------------------------------------------------

    def drop_data(self, expr: str = "pk > 0"):
        try:
            where_clause = self._translate_filter(expr)
            sql = f"DELETE FROM {self.collection_name}"
            if where_clause:
                sql += f" WHERE {where_clause}"
            with self._connection.cursor() as cursor:
                cursor.execute(sql)
            self._connection.commit()
        except Exception as e:
            logger.warning(f"Error dropping data: {e}")

    def drop_data_filtered(self, filter: str) -> int:
        try:
            where_clause = self._translate_filter(filter)
            sql = f"DELETE FROM {self.collection_name}"
            if where_clause:
                sql += f" WHERE {where_clause}"
            with self._connection.cursor() as cursor:
                cursor.execute(sql)
                count = cursor.rowcount
            self._connection.commit()
            return count
        except Exception as e:
            logger.warning(f"Error dropping filtered data: {e}")
            return 0

    def drop_collection(self):
        try:
            drop_table_purge(self._connection, self.collection_name)
        except Exception as e:
            logger.warning(f"Error dropping collection: {e}")
        # Recreate empty table
        self.vector_db = OracleVS(
            client=self._connection,
            embedding_function=self.embedding,
            table_name=self.collection_name,
        )
