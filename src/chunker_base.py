"""
Base class for all chunking strategies
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass
import tiktoken


@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    text: str
    chunk_id: int
    start_char: int
    end_char: int
    token_count: int
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseChunker(ABC):
    """Base class for all chunking strategies"""
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding = tiktoken.get_encoding(encoding_name)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    @abstractmethod
    def chunk(self, text: str) -> List[Chunk]:
        """
        Split text into chunks
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of Chunk objects
        """
        pass
    
    def get_chunk_stats(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Calculate statistics for chunks"""
        if not chunks:
            return {}
        
        token_counts = [c.token_count for c in chunks]
        char_counts = [len(c.text) for c in chunks]
        
        return {
            "num_chunks": len(chunks),
            "total_tokens": sum(token_counts),
            "avg_tokens_per_chunk": sum(token_counts) / len(chunks),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "avg_chars_per_chunk": sum(char_counts) / len(chunks),
            "total_chars": sum(char_counts),
        }
