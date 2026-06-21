import os
import aiohttp
import numpy as np

EMBED_MODEL = os.getenv("MODEL_EMBEDDING", "BAAI/bge-m3")

async def get_embeddings_api(texts: list[str], batch_size: int = 96) -> np.ndarray:
    api_key = os.getenv("DEEPINFRA_API_KEY")
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
