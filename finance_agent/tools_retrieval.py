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

# Sentence-starting words that happen to be capitalized — not document identifiers
COMMON_UPPERCASE = {
    "The", "What", "When", "Where", "How", "Did", "Does",
    "Was", "Were", "Has", "Have", "Which", "Who", "Why",
    "And", "But", "For", "With", "That", "This", "From",
}


def extract_exact_terms(query: str) -> list[str]:
    """
    Extract tokens from a query that are likely to appear verbatim in a
    financial document — identifiers, versioned names, hyphenated compounds,
    and quoted phrases — regardless of filing domain.

    Captures:
      - Quoted phrases:        "Adjusted EBITDA", "non-GAAP"
      - Digit-containing:      V3, 10-Q, S-1, BLA-123, Q3 2024, Series B-1
      - Hyphenated compounds:  non-GAAP, sale-leaseback, mark-to-market
      - Proper identifiers:    Starship, Falcon, EBITDA, NASDAQ
        (len > 3, not all-lowercase, not a common sentence-starter)
    """
    # Quoted phrases — always treat as exact targets
    quoted = re.findall(r'"([^"]+)"', query)

    # All word/hyphen tokens in the query
    tokens = re.findall(r"\b[\w\-]+\b", query)

    identifiers = [
        t for t in tokens
        if (
            any(c.isdigit() for c in t)           # V3, 10-Q, BLA-123, Q3
            or "-" in t                             # non-GAAP, sale-leaseback
            or (
                len(t) > 3                          # skip short noise
                and not t.islower()                 # skip plain lowercase words
                and t not in COMMON_UPPERCASE       # skip sentence-starters
            )
        )
    ]

    return [t.lower() for t in set(quoted + identifiers) if len(t) > 2]


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
            "description": (
                "Your target question containing the explicit target reference mapping "
                "(e.g., 'What were the total research costs in {{my_document_key}}?')."
            ),
        },
        "preserve_document_order": {
            "type": "boolean",
            "default": False,
            "description": (
                "Set to True if processing complex structural data dependencies "
                "across sequential disclosures."
            ),
        },
    }
    required = ["prompt"]

    def __init__(self, llm: LLM):
        self._llm = llm

    def _validate_inputs(self, prompt: str, state: dict[str, Any]) -> tuple[list[str], str]:
      if not re.search(r"{{[^{}]+}}", prompt):
        # Attempt auto-repair: if exactly one index key exists in state, inject it
        available = [k.removesuffix("_index") for k in state if k.endswith("_index")]
        if len(available) == 1:
            # Append the key reference to the prompt rather than failing
            prompt = prompt + f" {{{{{{available[0]}}}}}}".replace("{available[0]}", available[0])
            # simpler:
            prompt = f"{prompt} {{{{{available[0]}}}}}"
        else:
            example_key = available[0] if available else "my_doc"
            raise ValueError(
                f"ERROR: Prompt must include a {{{{key_name}}}} reference. "
                f"Available keys: {available}. "
                f"Example: 'What is the revenue in {{{{{example_key}}}}}?'"
            )
      keys = re.findall(r"{{([^{}]+)}}", prompt)
      resolved_keys = []
      for key in keys:
        if f"{key}_index" in state:
            resolved_keys.append(key)
            continue
        normalized = re.sub(r"_+", "_", key.replace(" ", "_")).strip("_")
        candidate = next(
            (k.removesuffix("_index") for k in state
             if k.endswith("_index") and
             re.sub(r"_+", "_", k.removesuffix("_index")) == normalized),
            None,
        )
        if candidate:
            prompt = prompt.replace(f"{{{{{key}}}}}", f"{{{{{candidate}}}}}")
            resolved_keys.append(candidate)
        else:
            available = [k.removesuffix("_index") for k in state if k.endswith("_index")]
            example_key = available[0] if available else "my_doc"
            raise KeyError(
                f"ERROR: Key '{key}' not found in runtime state. "
                f"Available keys: {available}. "
                f"Example usage: '{{{{{example_key}}}}}'"
            )
      return resolved_keys, prompt

    async def execute(
        self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger
    ) -> ToolOutput:
        try:
            prompt                  = args["prompt"]
            preserve_document_order = args.get("preserve_document_order", False)

            referenced_keys, prompt = self._validate_inputs(prompt, state)
            primary_key = referenced_keys[0]
            search_query = re.sub(r"\{\{[^{}]+\}\}", "", prompt).strip()


            if f"{primary_key}_index" not in state:
                logger.warning("Cache index missing — falling back to flat text.")
                doc_content  = state[f"{primary_key}_index"]["raw_text"]
                final_prompt = prompt.replace(f"{{{{{primary_key}}}}}", doc_content[:50000])
                response     = await self._llm.query(final_prompt)
                return ToolOutput(output=response.output_text_str)

            index_assets      = state[f"{primary_key}_index"]
            chunks            = index_assets["chunks"]
            cached_embeddings = index_assets["embeddings"]
            tokenized_corpus  = index_assets["tokenized_corpus"]

            # ----------------------------------------------------------------
            # Channel 1: Lexical Search (BM25)
            # ----------------------------------------------------------------
            bm25         = BM25Okapi(tokenized_corpus)
            bm25_scores  = bm25.get_scores(search_query.lower().split())
            top_bm25_idx = list(np.argsort(bm25_scores)[-40:][::-1])

            # ----------------------------------------------------------------
            # Channel 2: Dense Vector Semantic Search
            # ----------------------------------------------------------------
            query_vector  = await get_embeddings_api([search_query])
            query_vector  = query_vector[0]
            vector_scores = np.dot(cached_embeddings, query_vector)
            top_vec_idx   = list(np.argsort(vector_scores)[-40:][::-1])

            # ----------------------------------------------------------------
            # Channel 3: Exact Identifier / Term Search
            # Boosts chunks that contain verbatim query identifiers
            # (versioned names, hyphenated compounds, quoted phrases, etc.)
            # ----------------------------------------------------------------
            exact_terms  = extract_exact_terms(search_query)
            exact_scores: dict[int, int] = {}
            if exact_terms:
                for i, chunk in enumerate(chunks):
                    chunk_lower = chunk.lower()
                    hits = sum(1 for term in exact_terms if term in chunk_lower)
                    if hits > 0:
                        exact_scores[i] = hits

            logger.info(f"Exact terms extracted: {exact_terms}")
            logger.info(f"Exact match chunks found: {len(exact_scores)}")

            # ----------------------------------------------------------------
            # Reciprocal Rank Fusion (RRF) across all three channels
            # ----------------------------------------------------------------
            rrf_scores: dict[int, float] = {}

            for rank, idx in enumerate(top_bm25_idx):
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (60 + rank)

            for rank, idx in enumerate(top_vec_idx):
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (60 + rank)

            if exact_scores:
                sorted_exact = sorted(
                    exact_scores.keys(),
                    key=lambda x: exact_scores[x],
                    reverse=True,
                )
                for rank, idx in enumerate(sorted_exact):
                    rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (60 + rank)

            # Expanded candidate pool to accommodate three channels without
            # crowding out legitimate semantic hits (was 60)
            rrf_candidates = sorted(
                rrf_scores, key=rrf_scores.__getitem__, reverse=True
            )[:80]

            logger.info(f"RRF candidates (pre-rerank): {len(rrf_candidates)}")

            # ----------------------------------------------------------------
            # Reranking
            # ----------------------------------------------------------------
            candidate_chunks = [chunks[idx] for idx in rrf_candidates]
            rerank_scores    = await rerank_api(search_query, candidate_chunks)

            sorted_pairs = sorted(
                zip(rrf_candidates, rerank_scores),
                key=lambda x: x[1],
                reverse=True,
            )
            top_20_pairs = sorted_pairs[:20]

            logger.info(f"Top reranked chunk indices: {[idx for idx, _ in top_20_pairs]}")

            # ----------------------------------------------------------------
            # Adaptive Sequential Chunk Consolidation
            # Merges contiguous top-ranked chunks into coherent passages
            # ----------------------------------------------------------------
            retrieved_chunks_map = {idx: chunks[idx] for idx, _ in top_20_pairs}
            target_indices       = list(retrieved_chunks_map.keys())

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
                    current_passage += (
                        f"\n[Contiguous Block Extension]\n{retrieved_chunks_map[next_idx]}"
                    )
                    visited.add(next_idx)
                    next_idx += 1

                consolidated_passages.append(current_passage)

            retrieved_context = "\n\n[... Structural Fragment Break ...]\n\n".join(
                consolidated_passages
            )

            final_prompt = prompt.replace(f"{{{{{primary_key}}}}}", retrieved_context)
            response     = await self._llm.query(final_prompt)
            return ToolOutput(output=response.output_text_str)

        except Exception as e:
            logger.error(f"Retrieval tool error: {e}")
            return ToolOutput(output=str(e), error=str(e))
