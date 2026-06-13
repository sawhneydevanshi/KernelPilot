from backend.rag.scraper import scrape_all_pages, PYTORCH_KNOWLEDGE_BASE
from backend.rag.chunker import chunk_all_pages, save_chunks, load_chunks


def test_scraper_and_chunker():
    print("\nScraping PyTorch documentation...")
    print(f"Targeting {len(PYTORCH_KNOWLEDGE_BASE)} pages\n")


    pages = scrape_all_pages()

    print(f"\nScraped {len(pages)} pages successfully")
    total_chars = sum(p.char_count for p in pages)
    print(f"Total content: {total_chars:,} characters")

    assert len(pages) > 0, "Should scrape at least some pages"

  
    print("\nChunking pages...")
    chunks = chunk_all_pages(pages)

    print(f"\nTotal chunks: {len(chunks)}")
    print(f"Sample chunk:")
    print(f"  ID: {chunks[0].chunk_id}")
    print(f"  Source: {chunks[0].source_url}")
    print(f"  Tokens: {chunks[0].token_count}")
    print(f"  Text preview: {chunks[0].text[:150]}...")

    assert len(chunks) > 10, "Should have substantial chunks"

   
    save_chunks(chunks)

    
    loaded = load_chunks()
    assert len(loaded) == len(chunks), "Loaded chunks should match saved"
    print(f"\nSuccessfully saved and reloaded {len(loaded)} chunks")
    print("\nAll tests passed!")


if __name__ == "__main__":
    test_scraper_and_chunker()