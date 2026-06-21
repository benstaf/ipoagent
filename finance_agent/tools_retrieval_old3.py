import logging
import os
import re
from typing import Any

import aiohttp
import numpy as np
from rank_bm25 import BM25Okapi

from model_library.agent import Tool, ToolOutput
from model_library.base import LLM
from .embeddings import get_embeddings_api

RERANKER_MODEL = os.getenv("MODEL_RERANKER", "Qwen/Qwen3-Reranker-0.6B")

async def rerank_api(query: str, documents: list[str]) -> list[float]:
    api_key = os.getenv("DEEPINFRA_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "queries": [query],
        "documents": documents,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"https://api.deepinfra.com/v1/inference/{RERANKER_MODEL}",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data["scores"]

class RetrieveInformation(Tool):
    name = "retrieve_information"
    description = (
        "Retrieves critical data segments from a previously ingested document via hybrid "
        "lexical and semantic calculations. Crucial Requirement: The search prompt argument "
        "MUST contain the token placeholder containing your storage key inside double curly brackets, "
        "exactly format formatted as: {{key_name}} alongside your query description text."
    )
    parameters: dict[str, Any] = {
        "prompt": {
            "type": "string",
            "description": "Your target question containing the explicit target reference mapping (e.g., 'What were the total research costs in {{my_document_key}}?').",
        },
        "preserve_document_order": {
            "type": "boolean",
            "default": False,
            "description": "Set to True if processing complex structural data dependencies across sequential disclosures.",
        },
    }
    required = ["prompt"]

    def __init__(self, llm: LLM):
        self._llm = llm

    def _validate_inputs(self, prompt: str, state: dict[str, Any]) -> list[str]:
        if not re.search(r"{{[^{}]+}}", prompt):
            raise ValueError("ERROR: Prompt must include a {{key_name}} reference.")
        keys = re.findall(r"{{([^{}]+)}}", prompt)
        for key in keys:
            if f"{key}_index" not in state:
                raise KeyError(f"ERROR: Key '{key}' not found in runtime state.")
        return keys

    async def execute(
        self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger
    ) -> ToolOutput:
        try:
            prompt                  = args["prompt"]
            preserve_document_order = args.get("preserve_document_order", False)

            referenced_keys = self._validate_inputs(prompt, state)
            primary_key     = referenced_keys[0]
            search_query    = re.sub(r"\{\{[^{}]+\}\}", "", prompt).strip()

            if f"{primary_key}_index" not in state:
                logger.warning("Cache index missing — falling back to flat text.")
               # doc_content  = state[primary_key]
                doc_content = state[f"{primary_key}_index"]["raw_text"]
                final_prompt = prompt.replace(f"{{{{{primary_key}}}}}", doc_content[:50000])
                response = await self._llm.query(final_prompt)
                return ToolOutput(output=response.output_text_str)

            index_assets      = state[f"{primary_key}_index"]
            chunks            = index_assets["chunks"]
            cached_embeddings = index_assets["embeddings"]
            tokenized_corpus  = index_assets["tokenized_corpus"]

            # Lexical Search Matrix
            bm25 = BM25Okapi(tokenized_corpus)
            bm25_scores  = bm25.get_scores(search_query.lower().split())
            top_bm25_idx = list(np.argsort(bm25_scores)[-40:][::-1])

            # Dense Vector Semantic Search
            query_vector  = await get_embeddings_api([search_query])
            query_vector  = query_vector[0]
            vector_scores = np.dot(cached_embeddings, query_vector)
            top_vec_idx   = list(np.argsort(vector_scores)[-40:][::-1])

            # Reciprocal Rank Fusion (RRF) Execution 
            rrf_scores: dict[int, float] = {}
            for rank, idx in enumerate(top_bm25_idx):
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (60 + rank)
            for rank, idx in enumerate(top_vec_idx):
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (60 + rank)

            rrf_candidates = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)[:60]

            candidate_chunks = [chunks[idx] for idx in rrf_candidates]
            rerank_scores    = await rerank_api(search_query, candidate_chunks)

            sorted_pairs = sorted(zip(rrf_candidates, rerank_scores), key=lambda x: x[1], reverse=True)
            top_20_pairs = sorted_pairs[:20]

            # Adaptive Sequential Chunk Consolidation
            retrieved_chunks_map = {idx: chunks[idx] for idx, _ in top_20_pairs}
            target_indices = list(retrieved_chunks_map.keys())

            if preserve_document_order:
                target_indices.sort()

            consolidated_passages = []
            visited = set()

            for idx in target_indices:
                if idx in visited:
                    continue

                current_passage = retrieved_chunks_map[idx]
                visited.add(idx)

                next_idx = idx + 1
                while next_idx in retrieved_chunks_map:
                    current_passage += f"\n[Contiguous Block Extension]\n{retrieved_chunks_map[next_idx]}"
                    visited.add(next_idx)
                    next_idx += 1

                consolidated_passages.append(current_passage)

            retrieved_context = "\n\n[... Structural Fragment Break ...]\n\n".join(consolidated_passages)

            final_prompt = prompt.replace(f"{{{{{primary_key}}}}}", retrieved_context)
            response     = await self._llm.query(final_prompt)
            return ToolOutput(output=response.output_text_str)

        except Exception as e:
            logger.error(f"Retrieval tool error: {e}")
            return ToolOutput(output=str(e), error=str(e))
