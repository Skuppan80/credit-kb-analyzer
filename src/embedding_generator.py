"""
Generate embeddings using Sentence Transformers
"""

from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np
from rich.console import Console
from rich.progress import Progress
from pathlib import Path
import json

console = Console()


class EmbeddingGenerator:
    """
    Generate embeddings for text chunks
    
    Uses Sentence Transformers (free, local alternative to AWS Bedrock embeddings)
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model
        
        Args:
            model_name: Sentence transformer model
                - all-MiniLM-L6-v2: 384 dimensions, fast, good quality (DEFAULT)
                - all-mpnet-base-v2: 768 dimensions, slower, better quality
                - multi-qa-mpnet-base-dot-v1: 768 dimensions, optimized for Q&A
        """
        console.print(f"\n🤖 Loading embedding model: {model_name}")
        console.print("   (This may take a minute on first run...)")
        
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        console.print(f"✅ Model loaded!")
        console.print(f"   Embedding dimensions: {self.embedding_dim}")
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings
            batch_size: Process this many texts at once
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        console.print(f"\n🔢 Generating embeddings for {len(texts)} texts...")
        
        with Progress() as progress:
            task = progress.add_task("[cyan]Embedding...", total=len(texts))
            
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            progress.update(task, completed=len(texts))
        
        console.print(f"✅ Generated embeddings: shape {embeddings.shape}")
        
        return embeddings
    
    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        return self.model.encode(text, convert_to_numpy=True)
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Returns:
            Similarity score (0 to 1, higher = more similar)
        """
        # Cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        similarity = dot_product / (norm1 * norm2)
        
        return float(similarity)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model"""
        return {
            "model_name": self.model_name,
            "embedding_dimensions": self.embedding_dim,
            "max_seq_length": self.model.max_seq_length,
        }


def main():
    """Test embedding generator"""
    
    # Create generator
    generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    
    # Test with sample texts
    console.print("\n" + "="*70)
    console.print("🧪 TESTING EMBEDDINGS")
    console.print("="*70)
    
    test_texts = [
        "The borrower is TALF LLC, a Delaware limited liability company.",
        "The lender is the Federal Reserve Bank of New York.",
        "The loan amount is $200,000,000,000.",
        "This is a completely unrelated sentence about cats and dogs.",
    ]
    
    # Generate embeddings
    embeddings = generator.generate_embeddings(test_texts)
    
    # Compute similarities
    console.print("\n📊 Similarity Matrix:\n")
    
    from rich.table import Table
    table = Table()
    table.add_column("Text Pair", style="cyan")
    table.add_column("Similarity", style="green", justify="right")
    
    for i in range(len(test_texts)):
        for j in range(i + 1, len(test_texts)):
            similarity = generator.compute_similarity(embeddings[i], embeddings[j])
            
            label1 = test_texts[i][:40] + "..."
            label2 = test_texts[j][:40] + "..."
            
            table.add_row(
                f"{i} ↔ {j}",
                f"{similarity:.4f}"
            )
    
    console.print(table)
    
    console.print("\n💡 Key Observations:")
    console.print("   - Text 0-1 (borrower-lender): Related financial terms")
    console.print("   - Text 0-3 (borrower-cats): Unrelated (low similarity)")
    console.print("   - Higher score = more semantically similar")
    
    # Show model info
    info = generator.get_model_info()
    console.print(f"\n📋 Model Info:")
    for key, value in info.items():
        console.print(f"   {key}: {value}")


if __name__ == "__main__":
    main()
