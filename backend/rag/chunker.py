import json
import os
import tiktoken
from dataclasses import dataclass, asdict
from backend.rag.scraper import ScrapedPage


CHUNK_SIZE = 500

CHUNK_OVERLAP = 50


@dataclass
class DocChunk:
    chunk_id: str           
    source_url: str
    page_title: str
    text: str
    token_count: int


def chunk_page(page: ScrapedPage, enc: tiktoken.Encoding) -> list[DocChunk]:
    """
    Split a single scraped page into overlapping token chunks.
    """
    tokens = enc.encode(page.text)
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(tokens):
        end = min(start + CHUNK_SIZE, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)

        url_slug = page.url.split("/")[-1].replace(".rst", "").replace(".html", "")

        import hashlib
        title_hash = hashlib.md5(page.title.encode()).hexdigest()[:6]
        chunk_id = f"{url_slug}-{title_hash}-{chunk_index:04d}"
      
        

        chunks.append(DocChunk(
            chunk_id=chunk_id,
            source_url=page.url,
            page_title=page.title,
            text=chunk_text,
            token_count=len(chunk_tokens),
        ))

        
        start += CHUNK_SIZE - CHUNK_OVERLAP
        chunk_index += 1

    return chunks


def chunk_all_pages(pages: list[ScrapedPage]) -> list[DocChunk]:
    """
    Chunk all scraped pages and return a flat list of DocChunks.
    """
   
    enc = tiktoken.get_encoding("cl100k_base")
    all_chunks = []

    for page in pages:
        chunks = chunk_page(page, enc)
        all_chunks.extend(chunks)
        print(f"  '{page.title}' → {len(chunks)} chunks")

    return all_chunks


def save_chunks(chunks: list[DocChunk], output_path: str = "data/chunks.json"):
    """
    Save chunks to disk as JSON so we don't have to re-scrape every time.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([asdict(c) for c in chunks], f, indent=2)
    print(f"\nSaved {len(chunks)} chunks to {output_path}")


def load_chunks(input_path: str = "data/chunks.json") -> list[DocChunk]:
    """
    Load chunks from disk.
    """
    with open(input_path, "r") as f:
        data = json.load(f)
    return [DocChunk(**d) for d in data]