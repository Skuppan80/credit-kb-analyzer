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



## 🔑 API Key Setup

1. Create a `.env` file in the project root:
```bash
touch .env
```

2. Add your Anthropic API key:
```
ANTHROPIC_API_KEY=your-key-here
```

3. Get your key from: https://console.anthropic.com/settings/keys

4. Verify: `python verify_setup.py`

---

## 🔑 Environment Setup

### Step 1: Create .env file
```bash
# In the project root directory
cat > .env << 'ENVEOF'
ANTHROPIC_API_KEY=your-key-here
LOG_LEVEL=INFO
ENVEOF
```

### Step 2: Add your API key

1. Get your key from: https://console.anthropic.com/settings/keys
2. Open `.env` in a text editor
3. Replace `your-key-here` with your actual API key
4. Save the file

### Step 3: Verify
```bash
python verify_setup.py
```

Expected output:
```
✅ API Key - Configured
🎉 All core dependencies installed and configured!
```

⚠️ **IMPORTANT:** Never commit the `.env` file. It's already in `.gitignore`.

