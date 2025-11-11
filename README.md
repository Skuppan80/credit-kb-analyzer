# Credit Knowledge Base Analyzer

**Learning Project:** Understanding embeddings, chunking strategies, and vector databases.

## 🎯 Objectives

1. Understand how embeddings work at the token level
2. Compare chunking strategies:
   - Fixed chunking (300 tokens, 20% overlap)
   - Semantic chunking (sentence boundaries)
   - Hierarchical (parent-child) chunking
3. Determine optimal chunking for credit documents
4. Build evaluation framework for RAG systems

## 🏗️ Architecture
```
PDF → Text Extraction → Chunking Strategies → Embeddings → Vector DB
                                                              ↓
                                            Query → Retrieval → Evaluation
```

## 🛠️ Tech Stack

- **Vector DB:** ChromaDB (local, free)
- **Embeddings:** Sentence Transformers (local, free)
- **PDF Processing:** pdfplumber
- **Visualization:** Plotly, Streamlit
- **Testing:** pytest

## 📊 Chunking Strategies

### Strategy 1: Fixed Chunking
- Chunk size: 300 tokens
- Overlap: 20% (60 tokens)
- AWS Bedrock equivalent: Default chunking

### Strategy 2: Semantic Chunking
- Boundary: Sentence/paragraph
- Size: Variable (200-500 tokens)
- Preserves context

### Strategy 3: Hierarchical Chunking
- Parent: 1000 tokens (context)
- Child: 300 tokens (retrieval)
- Best of both worlds

## 📈 Evaluation Metrics

1. **Retrieval accuracy:** Did we find the right chunks?
2. **Context preservation:** Is information complete?
3. **Query performance:** Speed of retrieval
4. **Storage efficiency:** Vector DB size

## 🚀 Getting Started

Coming in next steps...

## 📚 Learning Resources

- AWS Bedrock Knowledge Base docs
- ChromaDB documentation
- Sentence Transformers guide
- Vector database best practices


