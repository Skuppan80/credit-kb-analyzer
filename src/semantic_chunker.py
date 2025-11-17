"""
Semantic chunking: Split on sentence/paragraph boundaries
"""

from typing import List
import nltk
from .chunker_base import BaseChunker, Chunk
from rich.console import Console

console = Console()


class SemanticChunker(BaseChunker):
    """
    Semantic chunking that respects sentence boundaries
    
    Strategy:
    - Split text into sentences
    - Group sentences into chunks
    - Target chunk size but don't split sentences
    - Results in variable-sized chunks
    """
    
    def __init__(self, target_chunk_size: int = 300, 
                 min_chunk_size: int = 200, 
                 max_chunk_size: int = 500):
        """
        Initialize semantic chunker
        
        Args:
            target_chunk_size: Ideal tokens per chunk
            min_chunk_size: Minimum tokens (avoid tiny chunks)
            max_chunk_size: Maximum tokens (hard limit)
        """
        super().__init__()
        self.target_chunk_size = target_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        # Ensure NLTK data is downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            console.print("📥 Downloading NLTK sentence tokenizer...")
            nltk.download('punkt', quiet=True)
        
        console.print(f"\n🔧 Semantic Chunker Configuration:")
        console.print(f"   Target size: {target_chunk_size} tokens")
        console.print(f"   Range: {min_chunk_size}-{max_chunk_size} tokens")
    
    def chunk(self, text: str) -> List[Chunk]:
        """
        Split text into semantic chunks
        
        Process:
        1. Split into sentences
        2. Group sentences until target size reached
        3. Respect min/max boundaries
        """
        console.print(f"\n📊 Starting semantic chunking...")
        
        # Split into sentences
        sentences = nltk.sent_tokenize(text)
        console.print(f"   Found {len(sentences):,} sentences")
        
        chunks = []
        current_chunk_sentences = []
        current_tokens = 0
        chunk_id = 0
        char_position = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # Check if adding this sentence would exceed max size
            if current_tokens + sentence_tokens > self.max_chunk_size and current_chunk_sentences:
                # Create chunk from current sentences
                chunk = self._create_chunk(
                    current_chunk_sentences, 
                    chunk_id, 
                    char_position
                )
                chunks.append(chunk)
                chunk_id += 1
                
                # Update char position
                char_position += len(chunk.text)
                
                # Reset for next chunk
                current_chunk_sentences = []
                current_tokens = 0
            
            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            current_tokens += sentence_tokens
            
            # Check if we've reached target size
            if current_tokens >= self.target_chunk_size:
                chunk = self._create_chunk(
                    current_chunk_sentences, 
                    chunk_id, 
                    char_position
                )
                chunks.append(chunk)
                chunk_id += 1
                
                char_position += len(chunk.text)
                
                current_chunk_sentences = []
                current_tokens = 0
            
            # Progress
            if chunk_id % 10 == 0 and chunk_id > 0:
                console.print(f"   Progress: {chunk_id} chunks created")
        
        # Add remaining sentences as final chunk
        if current_chunk_sentences:
            chunk = self._create_chunk(
                current_chunk_sentences, 
                chunk_id, 
                char_position
            )
            chunks.append(chunk)
        
        console.print(f"✅ Created {len(chunks)} semantic chunks")
        
        return chunks
    
    def _create_chunk(self, sentences: List[str], chunk_id: int, 
                     start_char: int) -> Chunk:
        """Create a chunk from sentences"""
        chunk_text = " ".join(sentences)
        token_count = self.count_tokens(chunk_text)
        
        return Chunk(
            text=chunk_text,
            chunk_id=chunk_id,
            start_char=start_char,
            end_char=start_char + len(chunk_text),
            token_count=token_count,
            metadata={
                "strategy": "semantic",
                "sentence_count": len(sentences),
                "target_size": self.target_chunk_size,
            }
        )


def main():
    """Test semantic chunker"""
    from pathlib import Path
    import json
    
    # Load extracted text
    results_dir = Path("results")
    text_files = list(results_dir.glob("*_extracted.txt"))
    
    if not text_files:
        console.print("❌ No extracted text found. Run analyze_pdf.py first.")
        return
    
    text_file = text_files[0]
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    console.print(f"\n📄 Loaded: {text_file.name}")
    
    # Create chunker
    chunker = SemanticChunker(target_chunk_size=300, min_chunk_size=200, max_chunk_size=500)
    
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
        console.print(f"--- Chunk {i} ({chunk.token_count} tokens, {chunk.metadata['sentence_count']} sentences) ---")
        console.print(chunk.text[:200] + "...")
        console.print()
    
    # Save chunks
    output_file = results_dir / "chunks_semantic.json"
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
