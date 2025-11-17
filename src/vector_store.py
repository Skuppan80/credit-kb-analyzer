"""
Vector database using ChromaDB
"""

from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from pathlib import Path
from rich.console import Console
from .chunker_base import Chunk
from .embedding_generator import EmbeddingGenerator
import json

console = Console()


class VectorStore:
    """
    Manage vector database with ChromaDB
    
    ChromaDB is a free, local alternative to:
    - AWS OpenSearch Serverless
    - Pinecone
    - Weaviate
    """
    
    def __init__(self, persist_directory: str = "chroma_db"):
        """
        Initialize vector store
        
        Args:
            persist_directory: Where to save the database
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        
        console.print(f"\n🗄️  Initializing ChromaDB...")
        console.print(f"   Directory: {self.persist_directory}")
        
        # Create ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator()
        
        console.print(f"✅ ChromaDB initialized")
    
    def create_collection(self, collection_name: str, 
                         reset: bool = False) -> chromadb.Collection:
        """
        Create or get a collection
        
        Args:
            collection_name: Name for this collection
            reset: If True, delete existing collection first
            
        Returns:
            ChromaDB collection object
        """
        if reset:
            try:
                self.client.delete_collection(collection_name)
                console.print(f"🗑️  Deleted existing collection: {collection_name}")
            except:
                pass
        
        collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        console.print(f"📁 Collection ready: {collection_name}")
        
        return collection
    
    def add_chunks(self, chunks: List[Chunk], collection_name: str, 
                   reset: bool = False) -> None:
        """
        Add chunks to vector database
        
        Args:
            chunks: List of Chunk objects
            collection_name: Name for this collection
            reset: Clear existing collection first
        """
        console.print(f"\n📥 Adding {len(chunks)} chunks to '{collection_name}'...")
        
        # Create/get collection
        collection = self.create_collection(collection_name, reset=reset)
        
        # Extract texts
        texts = [chunk.text for chunk in chunks]
        
        # Generate embeddings
        console.print("   Generating embeddings...")
        embeddings = self.embedding_generator.generate_embeddings(texts)
        
        # Prepare data for ChromaDB
        ids = [f"chunk_{chunk.chunk_id}" for chunk in chunks]
        metadatas = [
            {
                "chunk_id": chunk.chunk_id,
                "token_count": chunk.token_count,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                **chunk.metadata  # Include strategy-specific metadata
            }
            for chunk in chunks
        ]
        
        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            end_idx = min(i + batch_size, len(chunks))
            
            collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx].tolist(),
                documents=texts[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
            
            console.print(f"   Progress: {end_idx}/{len(chunks)} chunks added")
        
        console.print(f"✅ All chunks added to '{collection_name}'")
    
    def query(self, query_text: str, collection_name: str, 
              top_k: int = 10) -> Dict[str, Any]:
        """
        Search for similar chunks
        
        Args:
            query_text: Search query
            collection_name: Collection to search
            top_k: Number of results to return
            
        Returns:
            Dictionary with results
        """
        collection = self.client.get_collection(collection_name)
        
        # Generate query embedding
        query_embedding = self.embedding_generator.embed_single(query_text)
        
        # Search
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        # Format results
        formatted_results = {
            "query": query_text,
            "num_results": len(results['documents'][0]),
            "results": []
        }
        
        for i in range(len(results['documents'][0])):
            formatted_results["results"].append({
                "chunk_id": results['ids'][0][i],
                "text": results['documents'][0][i],
                "distance": results['distances'][0][i] if 'distances' in results else None,
                "metadata": results['metadatas'][0][i]
            })
        
        return formatted_results
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics about a collection"""
        try:
            collection = self.client.get_collection(collection_name)
            count = collection.count()
            
            return {
                "name": collection_name,
                "count": count,
                "exists": True
            }
        except:
            return {
                "name": collection_name,
                "count": 0,
                "exists": False
            }
    
    def list_collections(self) -> List[str]:
        """List all collections"""
        collections = self.client.list_collections()
        return [c.name for c in collections]


def main():
    """Test vector store"""
    
    # Create vector store
    store = VectorStore()
    
    # Test with sample chunks
    from .chunker_base import Chunk
    
    console.print("\n" + "="*70)
    console.print("🧪 TESTING VECTOR STORE")
    console.print("="*70)
    
    # Create sample chunks
    sample_chunks = [
        Chunk(
            text="The borrower is TALF LLC, a Delaware limited liability company.",
            chunk_id=0,
            start_char=0,
            end_char=64,
            token_count=15,
            metadata={"strategy": "test"}
        ),
        Chunk(
            text="The lender is the Federal Reserve Bank of New York.",
            chunk_id=1,
            start_char=64,
            end_char=116,
            token_count=12,
            metadata={"strategy": "test"}
        ),
        Chunk(
            text="The total facility amount is $200 billion.",
            chunk_id=2,
            start_char=116,
            end_char=158,
            token_count=10,
            metadata={"strategy": "test"}
        ),
    ]
    
    # Add to vector store
    store.add_chunks(sample_chunks, collection_name="test_collection", reset=True)
    
    # Test queries
    console.print("\n" + "="*70)
    console.print("🔍 TESTING SEMANTIC SEARCH")
    console.print("="*70)
    
    test_queries = [
        "Who is the borrower?",
        "What is the loan amount?",
        "Who is providing the loan?",
    ]
    
    for query in test_queries:
        console.print(f"\n📊 Query: '{query}'")
        results = store.query(query, "test_collection", top_k=2)
        
        console.print(f"   Top result: {results['results'][0]['text'][:60]}...")
        console.print(f"   Distance: {results['results'][0]['distance']:.4f}")
    
    # Show stats
    stats = store.get_collection_stats("test_collection")
    console.print(f"\n📈 Collection Stats:")
    console.print(f"   Name: {stats['name']}")
    console.print(f"   Chunks: {stats['count']}")


if __name__ == "__main__":
    main()
