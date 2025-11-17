# Credit Knowledge Base Analyzer 🎯

**AI Engineering Portfolio Project:** End-to-end RAG system evaluation for credit document extraction

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Project Overview

This project implements and evaluates a complete **Retrieval-Augmented Generation (RAG)** system for extracting structured data from credit agreements. It compares three chunking strategies against a baseline full-document approach, demonstrating **80% cost savings** while maintaining extraction quality.

### Key Results

✅ **80% API cost reduction** using RAG vs full-document extraction  
✅ **86% token reduction** through targeted retrieval  
✅ **Production-ready** pipeline from PDF to structured JSON  
✅ **Empirical evaluation** of chunking strategies  

---

## 🏗️ Architecture
```
PDF → Text Extraction → Chunking → Embeddings → Vector DB → Query → Retrieval → Claude API → JSON
```

### Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Vector Database | ChromaDB | Local vector storage (free alternative to AWS OpenSearch) |
| Embeddings | Sentence Transformers | 384-dim vectors (free alternative to AWS Bedrock) |
| LLM | Claude Sonnet 4 | Document extraction |
| PDF Processing | pdfplumber | Text extraction |
| Visualization | Plotly | Interactive charts |

---

## 📊 Results Summary

| Strategy | Chunks | Input Tokens | Cost | Savings |
|----------|--------|--------------|------|---------|
| **Semantic Chunking** | 12 | 3,982 | $0.0180 | **80.2%** |
| Fixed Chunking | 15 | 4,521 | $0.0194 | 78.6% |
| Hierarchical | 15 | 4,521 | $0.0195 | 78.5% |
| Baseline (Full Doc) | N/A | 28,212 | $0.0908 | - |

**Winner:** Semantic chunking - best cost savings with complete field extraction

---

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/Skuppan80/credit-kb-analyzer.git
cd credit-kb-analyzer

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key
```bash
# Create .env file
cat > .env << 'ENVEOF'
ANTHROPIC_API_KEY=your-key-here
ENVEOF

# Get key from: https://console.anthropic.com/settings/keys
```

### 3. Run Analysis
```bash
# Step 1: Extract text from PDF
python analyze_pdf.py

# Step 2: Build vector database
python build_vector_db.py

# Step 3: Run RAG evaluation
python evaluate_rag.py

# Step 4: Generate visualizations
python visualize_results.py

# Step 5: Create final report
python generate_report.py
```

---

## 📁 Project Structure
```
credit-kb-analyzer/
├── src/                          # Core modules
│   ├── chunker_base.py          # Base chunking class
│   ├── fixed_chunker.py         # Fixed-size (300t, 20% overlap)
│   ├── semantic_chunker.py      # Sentence-aware (200-500t)
│   ├── hierarchical_chunker.py  # Parent-child (1000/300t)
│   ├── embedding_generator.py   # Sentence Transformers
│   ├── vector_store.py          # ChromaDB integration
│   ├── retriever.py             # Semantic search
│   └── claude_extractor.py      # Claude API extraction
├── data/pdfs/                   # Input documents
├── results/                     # Outputs & visualizations
├── chroma_db/                   # Vector database
├── analyze_pdf.py               # PDF → text extraction
├── build_vector_db.py           # Create vector DB
├── evaluate_rag.py              # Run RAG evaluation
├── visualize_results.py         # Generate charts
└── generate_report.py           # Create final report
```

---

## 🔬 Chunking Strategies

### 1. Fixed Chunking (AWS Bedrock Default)
- **Size:** 300 tokens per chunk
- **Overlap:** 20% (60 tokens)
- **Pros:** Predictable, consistent size
- **Cons:** May split sentences mid-thought

### 2. Semantic Chunking
- **Size:** Variable (200-500 tokens)
- **Boundary:** Sentence-aware
- **Pros:** Preserves context, best quality
- **Cons:** Variable chunk sizes

### 3. Hierarchical Chunking
- **Parent:** 1000 tokens (context)
- **Child:** 300 tokens (retrieval)
- **Pros:** Broad context + detailed retrieval
- **Cons:** More complex, larger storage

---

## 💰 Cost Analysis

### Per-Document Extraction
```python
# Baseline (Full Document)
Input: 28,212 tokens × $0.000003 = $0.0846
Output: 412 tokens × $0.000015 = $0.0062
Total: $0.0908

# RAG (Semantic Chunking)
Input: 3,982 tokens × $0.000003 = $0.0119
Output: 401 tokens × $0.000015 = $0.0060
Total: $0.0180

Savings: $0.0728 (80.2%)
```

### At Scale (10,000 documents/year)

- Baseline: $908/year
- RAG: $180/year
- **Annual Savings: $728**

---

## 📈 Visualizations

All visualizations are generated in `results/visualizations/`:

- `dashboard.html` - Comprehensive comparison dashboard
- `cost_comparison.html` - Cost breakdown by strategy
- `token_usage.html` - Input/output token analysis
- `cost_savings.html` - Savings vs baseline
- `chunks_comparison.html` - Retrieval efficiency

Open `dashboard.html` in your browser for interactive charts!

---

## 🎓 Learning Outcomes

### What This Project Demonstrates

✅ **RAG Implementation:** Complete pipeline from chunking to extraction  
✅ **Cost Optimization:** Token-level cost analysis and reduction  
✅ **Engineering Decisions:** Data-driven strategy selection  
✅ **Production Patterns:** Error handling, modularity, evaluation  

### Interview Talking Points

**Q: "Walk me through a technical project."**

> "I built an end-to-end RAG system to extract credit terms from legal documents. I implemented three chunking strategies—fixed, semantic, and hierarchical—and compared them empirically. Through systematic evaluation, I proved semantic chunking achieves 80% cost savings while maintaining extraction quality. This required deep understanding of embeddings, vector search, and LLM prompt engineering."

**Q: "How do you optimize AI system costs?"**

> "I take a measurement-first approach. In my RAG project, I tracked costs at the token level across strategies. By retrieving only relevant chunks instead of processing full 28K-token documents, I reduced input to 4K tokens—an 86% reduction. At 10K documents/year, this saves $728 annually. The key was proving that targeted retrieval maintains quality while cutting costs."

---

## 🔧 Development

### Run Tests
```bash
# Test individual modules
python -m src.fixed_chunker
python -m src.semantic_chunker
python -m src.retriever
python -m src.claude_extractor

# Compare all strategies
python compare_chunking.py
```

### Add New PDF
```bash
# Copy PDF to data/pdfs/
cp /path/to/document.pdf data/pdfs/

# Re-run analysis
python analyze_pdf.py
python build_vector_db.py
python evaluate_rag.py
```

---

## 📚 Documentation

- **[FINAL_REPORT.md](results/FINAL_REPORT.md)** - Complete project report
- **[Chunking Comparison](results/chunking_comparison.json)** - Strategy metrics
- **[RAG Evaluation](results/rag_evaluation_results.json)** - Full results
- **[Vector DB Summary](results/vector_db_summary.json)** - Database stats

---

## 🚀 Future Enhancements

- [ ] Hybrid search (keyword + semantic)
- [ ] Re-ranking algorithms
- [ ] Batch processing pipeline
- [ ] Cost tracking dashboard
- [ ] A/B testing framework
- [ ] Additional chunking strategies

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details

---

## 🙏 Acknowledgments

- **Anthropic** - Claude API
- **ChromaDB** - Vector database
- **Sentence Transformers** - Embedding models
- **Plotly** - Visualizations

---

## 📧 Contact

**Project by:** Saravanan Kuppan  
**GitHub:** [@Skuppan80](https://github.com/Skuppan80)  
**Purpose:** AI Engineering Portfolio

---

**⭐ If this project helps you, please star it!**
