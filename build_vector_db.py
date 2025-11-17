"""
Build vector database with all chunking strategies
"""

from pathlib import Path
from src.fixed_chunker import FixedChunker
from src.semantic_chunker import SemanticChunker
from src.hierarchical_chunker import HierarchicalChunker
from src.vector_store import VectorStore
from rich.console import Console
import json

console = Console()


def build_all_collections(text: str):
    """Build vector DB collections for all chunking strategies"""
    
    console.print("\n" + "="*70)
    console.print("🏗️  BUILDING VECTOR DATABASE")
    console.print("="*70)
    
    # Initialize vector store
    store = VectorStore(persist_directory="chroma_db")
    
    # Define chunking strategies
    strategies = {
        "fixed_300_20": {
            "chunker": FixedChunker(chunk_size=300, overlap_percentage=0.2),
            "description": "Fixed chunking (300 tokens, 20% overlap)"
        },
        "semantic_300": {
            "chunker": SemanticChunker(target_chunk_size=300, min_chunk_size=200, max_chunk_size=500),
            "description": "Semantic chunking (200-500 tokens)"
        },
        "hierarchical_1000_300": {
            "chunker": HierarchicalChunker(parent_size=1000, child_size=300),
            "description": "Hierarchical (1000/300 tokens)"
        }
    }
    
    results = {}
    
    # Process each strategy
    for collection_name, config in strategies.items():
        console.print(f"\n{'='*70}")
        console.print(f"📦 Processing: {config['description']}")
        console.print(f"{'='*70}")
        
        # Chunk the text
        chunker = config['chunker']
        chunks = chunker.chunk(text)
        
        # Add to vector database
        store.add_chunks(chunks, collection_name=collection_name, reset=True)
        
        # Get stats
        stats = store.get_collection_stats(collection_name)
        
        results[collection_name] = {
            "description": config['description'],
            "num_chunks": stats['count'],
            "chunker_stats": chunker.get_chunk_stats(chunks)
        }
    
    # Summary table
    console.print(f"\n{'='*70}")
    console.print("📊 VECTOR DATABASE SUMMARY")
    console.print(f"{'='*70}\n")
    
    from rich.table import Table
    table = Table(title="Collections Created")
    table.add_column("Collection", style="cyan")
    table.add_column("Description", style="yellow")
    table.add_column("Chunks", style="green", justify="right")
    table.add_column("Avg Tokens", style="magenta", justify="right")
    
    for name, data in results.items():
        table.add_row(
            name,
            data['description'],
            f"{data['num_chunks']:,}",
            f"{data['chunker_stats']['avg_tokens_per_chunk']:.1f}"
        )
    
    console.print(table)
    
    # Save summary
    results_dir = Path("results")
    summary_file = results_dir / "vector_db_summary.json"
    
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    console.print(f"\n💾 Summary saved to: {summary_file}")
    
    return results


def main():
    """Build vector database from extracted text"""
    
    # Load extracted text
    results_dir = Path("results")
    text_files = list(results_dir.glob("*_extracted.txt"))
    
    if not text_files:
        console.print("❌ No extracted text found. Run analyze_pdf.py first.")
        return
    
    text_file = text_files[0]
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    console.print(f"\n📄 Document: {text_file.name}")
    console.print(f"   Characters: {len(text):,}")
    
    # Build vector database
    results = build_all_collections(text)
    
    console.print("\n✅ Vector database ready!")
    console.print(f"   Location: chroma_db/")
    console.print(f"   Collections: {len(results)}")
    console.print(f"\n🎯 Ready for Step 5: Query and retrieval experiments!\n")


if __name__ == "__main__":
    main()
