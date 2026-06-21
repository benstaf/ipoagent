import asyncio
import hashlib
import json
import logging
import os
import re
import pickle
from typing import Any

import aiohttp
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model_library.agent import Tool, ToolOutput
from .exceptions import get_retry_policy, retry_with_policy
from .embeddings import get_embeddings_api

CACHE_DIR = ".benchmark_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

ENRICHMENT_BASE_URL = "https://api.deepinfra.com/v1/openai/chat/completions"
ENRICHMENT_MODEL    = "Qwen/Qwen3.6-35B-A3B"

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
            "Filing text (sampled segments):\n"
            f"{sampled_text}"
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
        soup = BeautifulSoup(html_content, "html.parser")
        for tag in soup(["script", "style"]):
            tag.extract()

        normalized_blocks = []
        seen_large_blocks = set()

        for element in soup.find_all(["table", "p"]):
            if element.name == "table":
                text = self._table_to_markdown(element)
            else:
                text = element.get_text(separator=" ").strip()

            if len(text) < 40:
                continue

            # Check duplication on larger legal copy paragraphs
            if len(text) > 500:
                text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
                if text_hash in seen_large_blocks:
                    continue
                seen_large_blocks.add(text_hash)

            normalized_blocks.append(text)

        # Consolidate structural markdown blocks via paragraph line-breaks
        full_cleaned_text = "\n\n".join(normalized_blocks)

        # Let LangChain split cleanly at structural bounds (\n\n) without dismantling tables
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, 
            chunk_overlap=300,
            separators=["\n\n", "\n", " ", ""]
        )

        return text_splitter.split_text(full_cleaned_text)

    async def execute(
        self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger
    ) -> ToolOutput:
        try:
            url        = args["url"]
            key        = args["key"]
            cache_path = os.path.join(CACHE_DIR, f"{key}.pkl")

            if os.path.exists(cache_path):
                logger.info(f"Cache hit — loading serializable token-assets for: {key}")
                with open(cache_path, "rb") as f:
                    cached_data = pickle.load(f)
              #  state[key]             = cached_data["raw_text"]
                state[f"{key}_index"]  = cached_data
                return ToolOutput(output=f"SUCCESS: Loaded portable assets for key: {key}")

            @retry_with_policy(get_retry_policy(url))
            async def _fetch_raw_page():
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        url,
                        timeout=aiohttp.ClientTimeout(total=60),
                        headers={"User-Agent": "ValsAI/antoine@vals.ai"}
                    ) as resp:
                        resp.raise_for_status()
                        return await resp.text()

            logger.info(f"Fetching document data with protective retry layer from target: {url}")
            html_content = await _fetch_raw_page()


            soup       = BeautifulSoup(html_content, "html.parser")
            full_text  = soup.get_text()


            raw_chunks = self._chunk_by_structure(html_content)
            if len(raw_chunks) > 2000:
                logger.warning(f"Document too large ({len(raw_chunks)} chunks), capping at 2000.")
                raw_chunks = raw_chunks[:2000]

            if len(raw_chunks) < 50:
                logger.warning(f"Too few chunks ({len(raw_chunks)}) from HTML structure, falling back to raw text split.")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300, separators=["\n\n", "\n", " ", ""])
                raw_chunks = text_splitter.split_text(full_text)
                if len(raw_chunks) > 2000:
                    raw_chunks = raw_chunks[:2000]  

            metadata = self._extract_sec_metadata(full_text)
            doc_summary = await self._generate_doc_summary(full_text)

            logger.info(f"Running highly concurrent batch enrichment on {len(raw_chunks)} chunks...")

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
                "chunks":            enriched_chunks,
                "embeddings":        normalized_embeddings,
                "tokenized_corpus": tokenized_corpus,
            }
            with open(cache_path, "wb") as f:
                pickle.dump(cached_data, f)

#            state[key]             = cached_data["raw_text"]
            state[f"{key}_index"]  = cached_data

            return ToolOutput(output=f"SUCCESS: Indexed {len(enriched_chunks)} layout-preserved chunks.")

        except Exception as e:
            logger.error(f"Ingestion pipeline error: {e}")
            return ToolOutput(output=str(e), error=str(e))
