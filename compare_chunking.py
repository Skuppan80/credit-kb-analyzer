"""
Compare all three chunking strategies
"""

from pathlib import Path
from src.fixed_chunker import FixedChunker
from src.semantic_chunker import SemanticChunker
from src.hierarchical_chunker import HierarchicalChunker
from rich.console import Console
from rich.table import Table
import json

console = Console()


def compare_strategies(text: str):
    """Compare all chunking strategies"""
    
    console.print("\n" + "="*70)
    console.print("🔬 CHUNKING STRATEGY COMPARISON")
    console.print("="*70)
    
    # Create all chunkers
    chunkers = {
        "Fixed (300t, 20% overlap)": FixedChunker(chunk_size=300, overlap_percentage=0.2),
        "Semantic (200-500t)": SemanticChunker(target_chunk_size=300, min_chunk_size=200, max_chunk_size=500),
        "Hierarchical (1000/300t)": HierarchicalChunker(parent_size=1000, child_size=300),
    }
    
    results = {}
    
    # Run each strategy
    for name, chunker in chunkers.items():
        console.print(f"\n{'='*70}")
        console.print(f"Running: {name}")
        console.print(f"{'='*70}")
        
        chunks = chunker.chunk(text)
        stats = chunker.get_chunk_stats(chunks)
        
        results[name] = {
            "chunks": chunks,
            "stats": stats
        }
    
    # Create comparison table
    console.print(f"\n{'='*70}")
    console.print("📊 COMPARISON TABLE")
    console.print(f"{'='*70}\n")
    
    table = Table(title="Chunking Strategy Comparison")
    table.add_column("Strategy", style="cyan")
    table.add_column("Chunks", style="green", justify="right")
    table.add_column("Avg Tokens", style="yellow", justify="right")
    table.add_column("Min Tokens", style="magenta", justify="right")
    table.add_column("Max Tokens", style="magenta", justify="right")
    table.add_column("Total Tokens", style="blue", justify="right")
    
    for name, data in results.items():
        stats = data["stats"]
        table.add_row(
            name,
            f"{stats['num_chunks']:,}",
            f"{stats['avg_tokens_per_chunk']:.1f}",
            f"{stats['min_tokens']:,}",
            f"{stats['max_tokens']:,}",
            f"{stats['total_tokens']:,}",
        )
    
    console.print(table)
    
    # Analysis
    console.print(f"\n{'='*70}")
    console.print("💡 KEY INSIGHTS")
    console.print(f"{'='*70}\n")
    
    fixed_chunks = results["Fixed (300t, 20% overlap)"]["stats"]["num_chunks"]
    semantic_chunks = results["Semantic (200-500t)"]["stats"]["num_chunks"]
    hierarchical_chunks = results["Hierarchical (1000/300t)"]["stats"]["num_chunks"]
    
    console.print(f"1. Fixed chunking: {fixed_chunks} uniform chunks")
    console.print(f"   → Predictable, consistent size")
    console.print(f"   → May split sentences mid-thought")
    
    console.print(f"\n2. Semantic chunking: {semantic_chunks} variable chunks")
    console.print(f"   → Respects sentence boundaries")
    console.print(f"   → Better context preservation")
    
    console.print(f"\n3. Hierarchical: {hierarchical_chunks} child chunks")
    console.print(f"   → Each child has access to parent context")
    console.print(f"   → Best for complex queries requiring broad context")
    
    return results


def main():
    """Main comparison"""
    
    # Load text
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
    
    # Compare
    results = compare_strategies(text)
    
    # Save comparison
    comparison_file = results_dir / "chunking_comparison.json"
    
    comparison_data = {
        strategy: {
            "num_chunks": data["stats"]["num_chunks"],
            "avg_tokens": data["stats"]["avg_tokens_per_chunk"],
            "min_tokens": data["stats"]["min_tokens"],
            "max_tokens": data["stats"]["max_tokens"],
            "total_tokens": data["stats"]["total_tokens"],
        }
        for strategy, data in results.items()
    }
    
    with open(comparison_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    console.print(f"\n💾 Comparison saved to: {comparison_file}\n")


if __name__ == "__main__":
    main()
