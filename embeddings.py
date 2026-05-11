"""
makes embeddings of the product catalog. 
"""
#imports 
import os
import sys
import math
import requests
import torch
import torch.nn.functional as F
import chromadb
from chromadb.config import Settings
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv
import json 

# Configuration

CATALOG_URL = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
COLLECTION_NAME = "product_catalog"
BATCH_SIZE = 50

# Environment

load_dotenv()

CHROMA_HOST = os.getenv("CHROMA_HOST")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")

missing = [k for k, v in {
    "CHROMA_HOST": CHROMA_HOST,
    "CHROMA_API_KEY": CHROMA_API_KEY,
    "CHROMA_TENANT": CHROMA_TENANT,
    "CHROMA_DATABASE": CHROMA_DATABASE,
}.items() if not v]

if missing:
    print(f"[ERROR] Missing required environment variables: {', '.join(missing)}")
    sys.exit(1)

# Device selection: MPS (Apple M1) → CPU fallback

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("[INFO] MPS backend detected — using Apple M1 GPU acceleration.")
        return torch.device("mps")
    print("[INFO] MPS not available — falling back to CPU.")
    return torch.device("cpu")

# Data fetching

def fetch_catalog(url: str) -> list[dict]:
    print(f"[INFO] Fetching product catalog from:\n       {url}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    raw_text = response.text
    try : 
        catalog = json.loads(raw_text, strict = False)
    except json.JSONDecodeError as e :
        print(f"[ERROR] Failed to parse even with strict False : {e}")
        sys.exit(1)
    print(f"[INFO] Fetched {len(catalog)} products")
    return catalog

# Rich text construction

def build_rich_text(item: dict) -> str:
    """
    Combines important info into a flattened string paragraph
    for semantic retrieval with bge-base-en-v1.5.
    Description is included so queries like "patient records" or
    "digital-first selling" match the right assessments semantically.
    """
    name = item.get("name", "")
    job_levels = item.get("job_levels_raw", "").strip(",")
    languages = item.get("languages_raw", "").strip(",")
    remote = item.get("remote", "")
    adaptive = item.get("adaptive", "")
    skills_list = item.get("keys", [])
    skills_str = ", ".join(skills_list) if isinstance(skills_list, list) else str(skills_list)
    description = item.get("description", "").strip()   # fixed typo: "decription" → "description"
    return (
        f"Name: {name}\n"
        f"Description: {description}\n"
        f"Job Levels: {job_levels}\n"
        f"Languages: {languages}\n"
        f"Remote: {remote}\n"
        f"Adaptive: {adaptive}\n"
        f"Skills: {skills_str}\n"
    )


def extract_metadata(item: dict) -> dict:
    """
    Returns entire data of the entity to make sure llm 
    gets all the info of those retrieved entites.
    """
    return {
        "name": str(item.get("name", "")),
        "link": str(item.get("link", "")),
        "remote": str(item.get("remote", "")),
        "adaptive": str(item.get("adaptive", "")),
        "full_entity_json": json.dumps(item)
    }


# Embedding logic (BAAI/bge-base-en-v1.5)

def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Performs mean pooling, ignoring padding tokens."""
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask_expanded, dim=1) / torch.clamp(
        mask_expanded.sum(dim=1), min=1e-9
    )

def embed_texts(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
) -> list[list[float]]:
    """
    Tokenises, runs forward pass, mean-pools, and L2-normalises a batch of texts.
    Returns a list of float lists (one per text).
    """
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    # Move tensors to the target device
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)

    # outputs.last_hidden_state: (batch, seq_len, hidden_dim)
    pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])

    # L2 normalisation (required for cosine similarity with bge models)
    normalised = F.normalize(pooled, p=2, dim=1)

    # Return as plain Python lists for ChromaDB compatibility
    return normalised.cpu().tolist()


# ChromaDB client


def get_local_collection(name: str):
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})

def get_cloud_collection(name: str):
    print("[INFO] Connecting to Chroma Cloud...")
    client = chromadb.CloudClient(
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE,
        api_key=CHROMA_API_KEY
    )
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})

# Main pipeline

def main() -> None:
    # 1. Fetch catalog
    catalog = fetch_catalog(CATALOG_URL)
    if not catalog:
        print("[ERROR] Catalog is empty.")
        sys.exit(1)

    # 2. Device
    device = get_device()

    # 3. Load model & tokeniser
    print(f"[INFO] Loading tokeniser and model: {EMBEDDING_MODEL} …")
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL).to(device)
    model.eval()
    print("[INFO] Model loaded and set to eval mode.")

    # 4. ChromaDB collection
    local_col= get_local_collection(COLLECTION_NAME)
    cloud_col = get_cloud_collection(COLLECTION_NAME)

    # 5. Prepare documents
    ids: list[str] = []
    rich_texts: list[str] = []
    metadatas: list[dict] = []

    for item in catalog:
        entity_id = str(item.get("entity_id", ""))
        if not entity_id:
            print(f"[WARN] Skipping item with missing entity_id: {item.get('name', '<unknown>')}")
            continue
        ids.append(entity_id)
        rich_texts.append(build_rich_text(item))
        metadatas.append(extract_metadata(item))

    total = len(ids)
    num_batches = math.ceil(total / BATCH_SIZE)
    print(f"\n[INFO] Upserting {total} documents in {num_batches} batch(es) of {BATCH_SIZE}.\n")

    # 6. Batch embed & upsert
    for batch_idx in range(num_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, total)

        batch_ids = ids[start:end]
        batch_texts = rich_texts[start:end]
        batch_meta = metadatas[start:end]

        print(f"[BATCH {batch_idx + 1}/{num_batches}] Embedding docs {start + 1}–{end} …", end=" ", flush=True)
        embeddings = embed_texts(batch_texts, tokenizer, model, device)
        print("done. Upserting …", end=" ", flush=True)
        local_col.upsert(ids=batch_ids, embeddings=embeddings, metadatas=batch_meta, documents=batch_texts)
        cloud_col.upsert(ids=batch_ids, embeddings=embeddings, metadatas=batch_meta, documents=batch_texts)

        print("done.")

    print(f"\n[SUCCESS] Pipeline complete. {total} documents upserted into '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()