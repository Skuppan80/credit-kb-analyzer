"""
Hierarchical (parent-child) chunking strategy
"""

from typing import List, Dict
from .chunker_base import BaseChunker, Chunk
from rich.console import Console

console = Console()


class HierarchicalChunker(BaseChunker):
    """
    Hierarchical chunking with parent-child relationships
    
    Strategy:
    - Create large parent chunks (context)
    - Split parents into smaller child chunks (retrieval)
    - Children maintain reference to parent
    - Best of both worlds: detailed retrieval + broad context
    """
    
    def __init__(self, parent_size: int = 1000, child_size: int = 300):
        """
        Initialize hierarchical chunker
        
        Args:
            parent_size: Tokens per parent chunk (context)
            child_size: Tokens per child chunk (retrieval)
        """
        super().__init__()
        self.parent_size = parent_size
        self.child_size = child_size
        
        console.print(f"\n🔧 Hierarchical Chunker Configuration:")
        console.print(f"   Parent size: {parent_size} tokens (context)")
        console.print(f"   Child size: {child_size} tokens (retrieval)")
    
    def chunk(self, text: str) -> List[Chunk]:
        """
        Create hierarchical chunks
        
        Returns:
            List of child chunks (with parent metadata)
        """
        console.print(f"\n📊 Starting hierarchical chunking...")
        
        # Tokenize entire text
        tokens = self.encoding.encode(text)
        total_tokens = len(tokens)
        
        console.print(f"   Total tokens: {total_tokens:,}")
        
        all_chunks = []
        parent_id = 0
        global_chunk_id = 0
        
        # Create parent chunks
        parent_position = 0
        
        while parent_position < total_tokens:
            # Define parent chunk boundaries
            parent_end = min(parent_position + self.parent_size, total_tokens)
            parent_tokens = tokens[parent_position:parent_end]
            parent_text = self.encoding.decode(parent_tokens)
            
            # Now split parent into children
            child_position = 0
            child_id_in_parent = 0
            
            while child_position < len(parent_tokens):
                child_end = min(child_position + self.child_size, len(parent_tokens))
                child_tokens = parent_tokens[child_position:child_end]
                child_text = self.encoding.decode(child_tokens)
                
                # Calculate character positions
                char_ratio = len(text) / total_tokens if total_tokens > 0 else 1
                start_char = int((parent_position + child_position) * char_ratio)
                end_char = int((parent_position + child_end) * char_ratio)
                
                # Create child chunk with parent reference
                chunk = Chunk(
                    text=child_text,
                    chunk_id=global_chunk_id,
                    start_char=start_char,
                    end_char=end_char,
                    token_count=len(child_tokens),
                    metadata={
                        "strategy": "hierarchical",
                        "parent_id": parent_id,
                        "child_id_in_parent": child_id_in_parent,
                        "parent_text": parent_text,  # Full parent for context
                        "parent_token_count": len(parent_tokens),
                        "child_size": self.child_size,
                    }
                )
                
                all_chunks.append(chunk)
                global_chunk_id += 1
                child_id_in_parent += 1
                child_position = child_end
            
            parent_id += 1
            parent_position = parent_end
            
            # Progress
            if parent_id % 5 == 0:
                progress = (parent_position / total_tokens) * 100
                console.print(f"   Progress: {parent_id} parents, {global_chunk_id} children ({progress:.1f}%)")
        
        console.print(f"✅ Created {parent_id} parents with {len(all_chunks)} total children")
        
        return all_chunks


def main():
    """Test hierarchical chunker"""
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
    chunker = HierarchicalChunker(parent_size=1000, child_size=300)
    
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
    
    # Show hierarchy example
    console.print(f"\n🌳 Hierarchy Example (first parent):\n")
    first_parent_chunks = [c for c in chunks if c.metadata['parent_id'] == 0]
    
    console.print(f"Parent 0 has {len(first_parent_chunks)} children:")
    for chunk in first_parent_chunks[:3]:
        console.print(f"  - Child {chunk.metadata['child_id_in_parent']}: {chunk.token_count} tokens")
        console.print(f"    {chunk.text[:100]}...")
        console.print()
    
    # Save chunks (without full parent_text to reduce file size)
    output_file = results_dir / "chunks_hierarchical.json"
    chunks_data = [
        {
            "chunk_id": c.chunk_id,
            "text": c.text,
            "token_count": c.token_count,
            "metadata": {
                "strategy": c.metadata["strategy"],
                "parent_id": c.metadata["parent_id"],
                "child_id_in_parent": c.metadata["child_id_in_parent"],
                "parent_token_count": c.metadata["parent_token_count"],
                # Exclude parent_text to keep file size reasonable
            }
        }
        for c in chunks
    ]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f, indent=2)
    
    console.print(f"💾 Saved chunks to: {output_file}")


if __name__ == "__main__":
    main()
