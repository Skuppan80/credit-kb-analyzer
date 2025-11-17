"""
Fixed-size chunking with overlap (AWS Bedrock default strategy)
"""

from typing import List
from .chunker_base import BaseChunker, Chunk
from rich.console import Console

console = Console()


class FixedChunker(BaseChunker):
    """
    Fixed-size chunking with overlap
    
    This mimics AWS Bedrock's default chunking strategy:
    - Fixed chunk size (e.g., 300 tokens)
    - Fixed overlap (e.g., 20% = 60 tokens)
    """
    
    def __init__(self, chunk_size: int = 300, overlap_percentage: float = 0.2):
        """
        Initialize fixed chunker
        
        Args:
            chunk_size: Target tokens per chunk
            overlap_percentage: Percentage of overlap between chunks (0.0 to 0.5)
        """
        super().__init__()
        self.chunk_size = chunk_size
        self.overlap_percentage = overlap_percentage
        self.overlap_tokens = int(chunk_size * overlap_percentage)
        
        console.print(f"\n🔧 Fixed Chunker Configuration:")
        console.print(f"   Chunk size: {chunk_size} tokens")
        console.print(f"   Overlap: {overlap_percentage*100}% ({self.overlap_tokens} tokens)")
    
    def chunk(self, text: str) -> List[Chunk]:
        """
        Split text into fixed-size chunks with overlap
        
        Process:
        1. Tokenize entire text
        2. Create chunks of fixed size
        3. Add overlap from previous chunk
        """
        console.print(f"\n📊 Starting fixed chunking...")
        
        # Tokenize entire text
        tokens = self.encoding.encode(text)
        total_tokens = len(tokens)
        
        console.print(f"   Total tokens: {total_tokens:,}")
        
        chunks = []
        chunk_id = 0
        position = 0
        
        while position < total_tokens:
            # Calculate chunk boundaries
            chunk_start = max(0, position - self.overlap_tokens)
            chunk_end = min(position + self.chunk_size, total_tokens)
            
            # Extract token slice
            chunk_tokens = tokens[chunk_start:chunk_end]
            
            # Decode back to text
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Calculate character positions (approximate)
            # This is approximate because tokenization isn't 1:1 with characters
            char_ratio = len(text) / total_tokens if total_tokens > 0 else 1
            start_char = int(chunk_start * char_ratio)
            end_char = int(chunk_end * char_ratio)
            
            # Create chunk
            chunk = Chunk(
                text=chunk_text,
                chunk_id=chunk_id,
                start_char=start_char,
                end_char=end_char,
                token_count=len(chunk_tokens),
                metadata={
                    "strategy": "fixed",
                    "chunk_size": self.chunk_size,
                    "overlap_tokens": self.overlap_tokens if chunk_id > 0 else 0,
                    "token_start": chunk_start,
                    "token_end": chunk_end,
                }
            )
            
            chunks.append(chunk)
            chunk_id += 1
            
            # Move to next chunk (accounting for overlap)
            position = chunk_end
            
            # Progress
            if chunk_id % 10 == 0:
                progress = (position / total_tokens) * 100
                console.print(f"   Progress: {chunk_id} chunks created ({progress:.1f}%)")
        
        console.print(f"✅ Created {len(chunks)} fixed-size chunks")
        
        return chunks


def main():
    """Test fixed chunker"""
    from pathlib import Path
    import json
    
    # Load extracted text
    results_dir = Path("results")
    text_files = list(results_dir.glob("*_extracted.txt"))
    
    if not text_files:
        console.print("❌ No extracted text found. Run analyze_pdf.py first.")
        return
    
    # Load text
    text_file = text_files[0]
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    console.print(f"\n📄 Loaded: {text_file.name}")
    console.print(f"   Characters: {len(text):,}")
    
    # Create chunker
    chunker = FixedChunker(chunk_size=300, overlap_percentage=0.2)
    
    # Chunk the text
    chunks = chunker.chunk(text)
    
    # Get statistics
    stats = chunker.get_chunk_stats(chunks)
    
    console.print(f"\n📊 Chunking Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            console.print(f"   {key}: {value:.2f}")
        else:
            console.print(f"   {key}: {value:,}")
    
    # Show first 3 chunks
    console.print(f"\n📝 First 3 chunks:\n")
    for i, chunk in enumerate(chunks[:3]):
        console.print(f"--- Chunk {i} ({chunk.token_count} tokens) ---")
        console.print(chunk.text[:200] + "...")
        console.print()
    
    # Save chunks
    output_file = results_dir / "chunks_fixed.json"
    chunks_data = [
        {
            "chunk_id": c.chunk_id,
            "text": c.text,
            "token_count": c.token_count,
            "metadata": c.metadata
        }
        for c in chunks
    ]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f, indent=2)
    
    console.print(f"💾 Saved chunks to: {output_file}")


if __name__ == "__main__":
    main()
