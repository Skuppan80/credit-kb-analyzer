"""
Query and retrieve chunks from vector database
"""

from typing import List, Dict, Any
from rich.console import Console
from rich.table import Table
from .vector_store import VectorStore

console = Console()


class Retriever:
    """
    Retrieve relevant chunks based on queries
    """
    
    def __init__(self, vector_store: VectorStore):
        """
        Initialize retriever
        
        Args:
            vector_store: VectorStore instance
        """
        self.vector_store = vector_store
    
    def retrieve(self, query: str, collection_name: str, 
                 top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve top-K most relevant chunks
        
        Args:
            query: Search query
            collection_name: Which collection to search
            top_k: Number of chunks to retrieve
            
        Returns:
            List of retrieved chunks with metadata
        """
        results = self.vector_store.query(query, collection_name, top_k)
        return results['results']
    
    def retrieve_multi_query(self, queries: List[str], collection_name: str,
                            top_k_per_query: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve chunks for multiple queries and deduplicate
        
        Args:
            queries: List of search queries
            collection_name: Collection to search
            top_k_per_query: Chunks per query
            
        Returns:
            Deduplicated list of chunks
        """
        all_chunks = {}
        
        for query in queries:
            results = self.retrieve(query, collection_name, top_k_per_query)
            
            for result in results:
                chunk_id = result['chunk_id']
                if chunk_id not in all_chunks:
                    all_chunks[chunk_id] = result
        
        # Sort by original chunk_id to maintain document order
        sorted_chunks = sorted(
            all_chunks.values(),
            key=lambda x: int(x['metadata'].get('chunk_id', 0))
        )
        
        return sorted_chunks
    
    def display_results(self, query: str, results: List[Dict[str, Any]], 
                       max_text_length: int = 100):
        """Display retrieval results in a nice table"""
        
        console.print(f"\n🔍 Query: '{query}'")
        console.print(f"   Found: {len(results)} chunks\n")
        
        table = Table(title="Retrieved Chunks")
        table.add_column("Rank", style="cyan", width=6)
        table.add_column("Chunk ID", style="yellow", width=10)
        table.add_column("Tokens", style="green", justify="right", width=8)
        table.add_column("Distance", style="magenta", justify="right", width=10)
        table.add_column("Preview", style="white", width=60)
        
        for i, result in enumerate(results, 1):
            preview = result['text'][:max_text_length].replace('\n', ' ')
            if len(result['text']) > max_text_length:
                preview += "..."
            
            distance = result.get('distance', 0.0)
            tokens = result['metadata'].get('token_count', 0)
            
            table.add_row(
                str(i),
                result['chunk_id'],
                str(tokens),
                f"{distance:.4f}",
                preview
            )
        
        console.print(table)
    
    def get_combined_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Combine retrieved chunks into a single context string
        
        Args:
            results: List of retrieved chunks
            
        Returns:
            Combined text from all chunks
        """
        texts = [result['text'] for result in results]
        return "\n\n---\n\n".join(texts)


def main():
    """Test retrieval"""
    from .vector_store import VectorStore
    
    console.print("\n" + "="*70)
    console.print("🔍 TESTING RETRIEVAL")
    console.print("="*70)
    
    # Initialize
    store = VectorStore()
    retriever = Retriever(store)
    
    # List available collections
    collections = store.list_collections()
    console.print(f"\n📚 Available collections: {collections}")
    
    if not collections:
        console.print("\n❌ No collections found. Run build_vector_db.py first.")
        return
    
    # Test queries
    test_queries = [
        "Who is the borrower and what is their entity type?",
        "Who is the lender?",
        "What is the total loan amount?",
        "What are the interest rate terms?",
        "What is the maturity date?",
    ]
    
    # Test on first collection
    collection_name = collections[0]
    
    console.print(f"\n🎯 Testing collection: {collection_name}\n")
    
    for query in test_queries[:2]:  # Just test first 2
        results = retriever.retrieve(query, collection_name, top_k=5)
        retriever.display_results(query, results, max_text_length=80)


if __name__ == "__main__":
    main()
