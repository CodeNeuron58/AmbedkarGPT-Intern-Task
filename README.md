# **AmbedkarGPT – RAG System with Evaluation Framework**

This project implements a Retrieval-Augmented Generation (RAG) system built over **Dr. B.R. Ambedkar’s works**, along with a complete **evaluation pipeline** to measure retrieval quality, answer quality, and semantic consistency across different chunking strategies.

This assignment was developed as part of **Kalpit Pvt Ltd – AI Intern Hiring (Assignment 2)**.

---


# 🧠 **1. RAG Architecture**

### **Step 1: Document Loading**

All 6 provided Ambedkar documents are stored in the `corpus/` directory and loaded using LangChain document loaders.

### **Step 2: Chunking**

Each experiment runs with one of three chunk sizes (small, medium, large).
Chunks are split using `RecursiveCharacterTextSplitter`.

### **Step 3: Embedding**

The system uses:
**sentence-transformers/all-MiniLM-L6-v2**

Embeddings are stored in **ChromaDB** for fast similarity search.

### **Step 4: Retrieval**

Top-K retrieval (`k=3`) is used to fetch context for answering user queries.

### **Step 5: Answer Generation**

The system uses **Ollama + Mistral 7B** for answer generation.

Alternative LLMs can be integrated (e.g., Cohere API, OpenAI).

### **Step 6: Evaluation**

Each generated answer is compared against the ground-truth dataset using:

#### Retrieval Metrics

* Hit Rate
* Mean Reciprocal Rank (MRR)
* Precision@K

#### Answer Quality Metrics

* ROUGE-L
* BLEU
* Answer Relevance
* Faithfulness

#### Semantic Metrics

* Cosine Similarity

---

# 🧩 **3. Chunking Strategies**

The system is tested with **three configurations**:

| Strategy   | Chunk Size     | Purpose                       |
| ---------- | -------------- | ----------------------------- |
| **Small**  | 200–300 chars  | Highest precision, less noise |
| **Medium** | 500–600 chars  | Balanced context              |
| **Large**  | 800–1000 chars | Rich semantic context         |

Results are saved under `test_results.json`.

---

# ▶️ **4. How to Run the System**

To run the full evaluation for all chunking strategies:

```
python evaluation.py
```

The script will:

1. Load documents
2. Create vector store
3. Run evaluation for small, medium, and large chunks
4. Generate test_results.json
5. Generate output logs

---

# 🧪 **5. Evaluation Pipeline**

### **Run Full Evaluation**

To execute the complete evaluation across all chunking strategies:

```
python evaluation.py
```

This will automatically:
* Process all three chunking strategies (small, medium, large)
* Generate embeddings and vector stores
* Perform retrieval and answer generation
* Calculate all metrics
* Save results to `test_results.json`

### **Output**

The evaluation generates:
* `test_results.json` – Quantitative metrics per strategy
* `result_analysis.md` – Detailed analysis and recommendations

---

# 📦 **6. Dependencies & Environment Setup**

### **Install Dependencies**

All required packages are listed in `requirements.txt`.

```
pip install -r requirements.txt
```

### **Requires if you are using ollama:**

* Python 3.8+
* Ollama installed locally
* Mistral 7B pulled in Ollama:

  ```
  ollama pull mistral
  ```
### **Requires if you are using Cohere:**

* A Cohere API Key


# 🏁 **Summary**

This project successfully implements:

✅ A fully functional RAG system
✅ A complete evaluation pipeline
✅ Chunking strategy comparison
✅ Comprehensive metrics: retrieval, semantic, lexical
✅ Final analysis & recommendations

For full interpretation of results, refer to:
📄 **result_analysis.md**

---