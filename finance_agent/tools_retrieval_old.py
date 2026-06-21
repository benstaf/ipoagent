import asyncio
import hashlib
import json
import logging
import math
import os
import re
import pickle
from typing import Any

#import torch
import aiohttp
from bs4 import BeautifulSoup
from model_library.agent import Tool, ToolOutput
from model_library.base import LLM
from tavily import AsyncTavilyClient
from simpleeval import SimpleEval


from dotenv import load_dotenv
load_dotenv()

# Retrieval stack
import numpy as np
from rank_bm25 import BM25Okapi
#from sentence_transformers import SentenceTransformer
#from FlagEmbedding import FlagReranker

from .exceptions import (
    RetryExhaustedError,
    get_retry_policy,
    retry_http_errors,
    retry_with_policy,
)
from .key_rotator import KeyRotator, get_rotator

MAX_END_DATE = "2026-06-01"
CACHE_DIR = ".benchmark_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

VALID_TOOLS = [
    "web_search",
    "retrieve_information",
    "parse_html_page",
    "edgar_search",
    "calculator",
    "price_history",
]


# ── Enrichment constants ──────────────────────────────────────────────────────
ENRICHMENT_BASE_URL = "https://api.deepinfra.com/v1/openai/chat/completions"
ENRICHMENT_MODEL    = "Qwen/Qwen3.6-35B-A3B"
# ──────────────────────────────────────────────────────────────────────────────

EMBED_MODEL   = os.getenv("MODEL_EMBEDDING", "BAAI/bge-m3")
RERANKER_MODEL = os.getenv("MODEL_RERANKER", "Qwen/Qwen3-Reranker-0.6B")

async def get_embeddings_api(texts: list[str], batch_size: int = 96) -> np.ndarray:
    api_key = os.getenv("OPENAI_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    all_vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        payload = {
            "model": EMBED_MODEL,
            "input": batch,
            "encoding_format": "float",
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.deepinfra.com/v1/openai/embeddings",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                all_vectors.extend([item["embedding"] for item in data["data"]])
    return np.array(all_vectors, dtype=np.float32)



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
# [Keep Calculator, SubmitFinalResult, TavilyWebSearch, EDGARSearch, PriceHistory as-is]


class ParseHtmlPage(Tool):
    name = "parse_html_page"
    description = (
        "Parses an HTML document or SEC filing. It automatically extracts key metadata, "
        "converts asset tables to clean markdown, slices sections into optimal word chunks, "
        "and runs concurrent contextual LLM enrichment before updating the disk cache."
    )
    parameters: dict[str, Any] = {
        "url": {"type": "string", "description": "The target URL of the HTML page/filing to process."},
        "key": {"type": "string", "description": "The unique cache storage key for saving and loading the index."},
    }
    required = ["url", "key"]

    def _table_to_markdown(self, table) -> str:
        """
        Preserves the structural and semantic layout of financial statement data sheets
        by compiling HTML tables directly into markdown syntax.
        """
        rows = table.find_all("tr")
        if not rows:
            return ""
        
        md_rows = []
        for i, row in enumerate(rows):
            cells = [re.sub(r"\s+", " ", cell.get_text().strip()) for cell in row.find_all(["td", "th"])]
            if not any(cells): 
                continue
            md_rows.append("| " + " | ".join(cells) + " |")
            if i == 0:
                md_rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
                
        return "\n".join(md_rows)

    def _extract_sec_metadata(self, text: str) -> dict[str, str]:
        """
        Extracts structural filing metadata deterministically via targeted regex patterns.
        """
        meta = {"filing_type": "SEC Filing", "period": "Unknown Period"}
        
        if re.search(r"FORM\s+10-K", text, re.IGNORECASE):
            meta["filing_type"] = "Form 10-K (Annual Report)"
        elif re.search(r"FORM\s+10-Q", text, re.IGNORECASE):
            meta["filing_type"] = "Form 10-Q (Quarterly Report)"
        elif re.search(r"FORM\s+8-K", text, re.IGNORECASE):
            meta["filing_type"] = "Form 8-K (Current Disclosure)"
        elif re.search(r"FORM\s+S-1", text, re.IGNORECASE):
            meta["filing_type"] = "Form S-1 (Registration Statement)"

        period_match = re.search(
            r"For\s+the\s+(?:fiscal\s+year|quarterly\s+period)\s+ended\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})", 
            text, 
            re.IGNORECASE
        )
        if period_match:
            meta["period"] = period_match.group(1)
        
        return meta

    async def _generate_doc_summary(self, full_text: str) -> str:
        """
        Generates a document summary using head + middle + tail sampling.
        """
        api_key = os.getenv("DEEPINFRA_API_KEY")
        if not api_key:
            return full_text[:2000]

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        if len(full_text) <= 12000:
            sampled_text = full_text
        else:
            head = full_text[:4000]
            tail = full_text[-4000:]
            mid_start = max(0, len(full_text) // 2 - 2000)
            mid = full_text[mid_start : mid_start + 4000]
            sampled_text = f"{head}\n\n[... Middle Section ...]\n\n{mid}\n\n[... Tail Section ...]\n\n{tail}"

        prompt = (
            "You are a financial document analyst.\n"
            "Read the following SEC filing text and write a concise 150-200 word summary covering:\n"
            "- Company name and ticker\n"
            "- Core business description\n"
            "- Key financial metrics mentioned\n"
            "- Primary topics or disclosures in this filing\n\n"
            "Output ONLY the summary. No preamble, no labels.\n\n"
            f"Filing text (sampled segments):\n{sampled_text}"
        )

        payload = {
            "model": ENRICHMENT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "extra_body": {"enable_thinking": False},
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    ENRICHMENT_BASE_URL,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 200:
                        res = await resp.json()
                        content = res["choices"][0]["message"]["content"].strip()
                        return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        except Exception:
            pass

        return full_text[:2000]

    async def _generate_contextual_enrichment_batch(
        self, doc_summary: str, metadata: dict[str, str], chunks: list[str]
    ) -> list[str]:
        """
        Enriches multiple chunks concurrently using a strict JSON schema matrix.
        Fault-tolerant tracking ensures partial failures gracefully default back to empty values.
        """
        api_key = os.getenv("DEEPINFRA_API_KEY")
        if not api_key or not chunks:
            return [""] * len(chunks)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        chunks_input = ""
        for idx, chunk in enumerate(chunks):
            chunks_input += f"<chunk id={idx}>\n{chunk}\n</chunk>\n\n"

        prompt = (
            "You are a financial document preprocessing assistant.\n\n"
            "Document Meta-Context:\n"
            f"- Filing Type: {metadata['filing_type']}\n"
            f"- Period Covered: {metadata['period']}\n"
            f"- Global Summary: {doc_summary}\n\n"
            f"Review the following {len(chunks)} text chunks isolated from this document. For each chunk, "
            "write a single descriptive sentence (max 40 words) situating it structurally and contextually "
            "within the filing.\n\n"
            "You MUST respond exclusively with a valid JSON object containing an array of strings matching the input sequence order.\n"
            "Do not include markdown blocks, preamble, or keys outside the standard JSON structure:\n"
            '{\n  "contexts": [\n    "context sentence for chunk 0",\n    "context sentence for chunk 1"\n  ]\n}\n\n'
            f"Chunks to process:\n{chunks_input}"
        )

        payload = {
            "model": ENRICHMENT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "extra_body": {"enable_thinking": False},
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    ENRICHMENT_BASE_URL,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 200:
                        res = await resp.json()
                        raw_content = res["choices"][0]["message"]["content"].strip()
                        raw_content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
                        
                        if "```json" in raw_content:
                            raw_content = raw_content.split("```json")[1].split("```")[0].strip()
                        elif "```" in raw_content:
                            raw_content = raw_content.split("```")[1].split("```")[0].strip()

                        parsed = json.loads(raw_content)
                        contexts = parsed.get("contexts", [])
                        
                        result = []
                        for i in range(len(chunks)):
                            if i < len(contexts):
                                result.append(str(contexts[i]).strip())
                            else:
                                result.append("")
                        return result
        except Exception:
            pass

        return [""] * len(chunks)

    def _chunk_by_structure(self, html_content: str) -> list[str]:
        """
        Processes elements down into continuous data blocks. Tables are compiled into markdown
        matrices, while normal text is split cleanly into optimal 500-word macro chunks.
        """
        soup = BeautifulSoup(html_content, "html.parser")
        for tag in soup(["script", "style"]):
            tag.extract()

        chunks = []
        seen_large_blocks = set()
        
        chunk_words = 500       
        overlap_words = 75      
        last_text = ""

        for element in soup.find_all(["table", "p"]):
            if element.name == "table":
                text = self._table_to_markdown(element)
            else:
                text = element.get_text(separator=" ").strip()
                
            if len(text) < 40:
                continue
                
            if text == last_text:
                continue
                
            if len(text) > 500:
                text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
                if text_hash in seen_large_blocks:
                    continue
                seen_large_blocks.add(text_hash)
            
            last_text = text
            words = text.split()

            if len(words) > chunk_words:
                start = 0
                while start < len(words):
                    end = start + chunk_words
                    chunk_str = " ".join(words[start:end])
                    chunks.append(chunk_str)
                    if end >= len(words):
                        break
                    start += (chunk_words - overlap_words)
            else:
                chunks.append(text)

        return chunks if chunks else [soup.get_text(separator=" ")]

    async def execute(
        self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger
    ) -> ToolOutput:
        try:
            url        = args["url"]
            key        = args["key"]
            cache_path = os.path.join(CACHE_DIR, f"{key}.pkl")

            # ── Cache Hit ─────────────────────────────────────────────────────
            if os.path.exists(cache_path):
                logger.info(f"Cache hit — loading serializable token-assets for: {key}")
                with open(cache_path, "rb") as f:
                    cached_data = pickle.load(f)
                state[key]             = cached_data["raw_text"]
                state[f"{key}_index"]  = cached_data
                return ToolOutput(output=f"SUCCESS: Loaded portable assets for key: {key}")

            # ── Cache Miss with Defensive Retry Processing ─────────────────────
            @retry_with_policy(get_retry_policy(url))
            async def _fetch_raw_page():
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        url,
                        timeout=aiohttp.ClientTimeout(total=60),
                       # headers={"User-Agent": "ValsAI/Evaluation"},
                        headers={"User-Agent": "ValsAI/antoine@vals.ai"}
                    ) as resp:
                        resp.raise_for_status()
                        return await resp.text()

            logger.info(f"Fetching document data with protective retry layer from target: {url}")
           # html_content = await retry_with_policy(_fetch_raw_page, get_retry_policy(url))
            html_content = await _fetch_raw_page()
            raw_chunks = self._chunk_by_structure(html_content)
            if len(raw_chunks) > 2000:
                logger.warning(f"Document too large ({len(raw_chunks)} chunks), capping at 2000.")
                raw_chunks = raw_chunks[:2000]
            soup       = BeautifulSoup(html_content, "html.parser")
            full_text  = soup.get_text()

            metadata = self._extract_sec_metadata(full_text)
            doc_summary = await self._generate_doc_summary(full_text)

            logger.info(f"Running highly concurrent batch enrichment on {len(raw_chunks)} chunks...")

            # Bounded high-performance execution loop to maximize parallel context generation safely
            semaphore = asyncio.Semaphore(5)
            batch_size = 4
            batches = [raw_chunks[i : i + batch_size] for i in range(0, len(raw_chunks), batch_size)]

            async def throttled_worker(b):
                async with semaphore:
                    return await self._generate_contextual_enrichment_batch(doc_summary, metadata, b)

            results = await asyncio.gather(*[throttled_worker(b) for b in batches])

            enriched_chunks: list[str] = []
            for batch, prefixes in zip(batches, results):
                for prefix, original_chunk in zip(prefixes, batch):
                    combined = f"{prefix}\n{original_chunk}" if prefix else original_chunk
                    enriched_chunks.append(combined)

            normalized_embeddings = await get_embeddings_api(enriched_chunks)


            tokenized_corpus = [c.lower().split() for c in enriched_chunks]

            cached_data = {
                "raw_text":         full_text,
                "chunks":           enriched_chunks,
                "embeddings":       normalized_embeddings,
                "tokenized_corpus": tokenized_corpus,
            }
            with open(cache_path, "wb") as f:
                pickle.dump(cached_data, f)

            state[key]             = cached_data["raw_text"]
            state[f"{key}_index"]  = cached_data

            return ToolOutput(output=f"SUCCESS: Indexed {len(enriched_chunks)} layout-preserved chunks.")

        except Exception as e:
            logger.error(f"Ingestion pipeline error: {e}")
            return ToolOutput(output=str(e), error=str(e))


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
            if key not in state:
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
                doc_content  = state[primary_key]
                final_prompt = prompt.replace(f"{{{{{primary_key}}}}}", doc_content[:50000])
                response = await self._llm.query(final_prompt)
                return ToolOutput(output=response.output_text_str)

            index_assets      = state[f"{primary_key}_index"]
            chunks            = index_assets["chunks"]
            cached_embeddings = index_assets["embeddings"]
            tokenized_corpus  = index_assets["tokenized_corpus"]
            
            # Rebuild lightweight BM25 matrix dynamically from cached primitive structures
            bm25 = BM25Okapi(tokenized_corpus)
            bm25_scores  = bm25.get_scores(search_query.lower().split())
            top_bm25_idx = list(np.argsort(bm25_scores)[-40:][::-1])

            # Dense Vector Search
            query_vector  = await get_embeddings_api([search_query])
            query_vector  = query_vector[0]
            vector_scores = np.dot(cached_embeddings, query_vector)
            top_vec_idx   = list(np.argsort(vector_scores)[-40:][::-1])

            # Reciprocal Rank Fusion (RRF)
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

            # Adaptive Sequential Chunk Compression
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
