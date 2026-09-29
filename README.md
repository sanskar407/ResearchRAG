# ResearchRAG 📚

**A Retrieval-Augmented Generation (RAG) system for querying research papers with source-grounded answers and page-level citations.**

ResearchRAG is a RAG pipeline built from scratch to explore and implement the core components of document retrieval and grounded answer generation. It ingests research paper PDFs, splits them into meaningful chunks, retrieves relevant passages using hybrid search, reranks candidates using a cross-encoder, and generates answers using an LLM with document and page references.

The project is developed incrementally, with each version introducing and testing a specific improvement to the retrieval pipeline.

## ✨ Features

* **PDF ingestion:** Extract text and page information from research papers.
* **Paragraph-aware chunking:** Preserve meaningful text boundaries and attach document metadata.
* **Dense retrieval:** Generate semantic embeddings using Sentence Transformers and search with FAISS.
* **Hybrid retrieval:** Combine dense semantic search with BM25 keyword retrieval.
* **Reciprocal Rank Fusion (RRF):** Merge ranked results from multiple retrieval systems.
* **Cross-encoder reranking:** Reorder retrieved candidates according to query-document relevance.
* **LLM-based generation:** Generate answers using retrieved context.
* **Source citations:** Include document names and page numbers to help trace answers back to the source.

## 🏗️ Architecture

```text
Research Paper PDFs
        |
        v
   PDF Ingestion
        |
        v
 Paragraph-Aware Chunking
        |
        +-------------------------+
        |                         |
        v                         v
  BGE Embeddings              BM25 Index
        |                         |
        v                         v
   FAISS Search             Keyword Search
        |                         |
        +------------+------------+
                     |
                     v
        Reciprocal Rank Fusion
                     |
                     v
          Candidate Documents
                     |
                     v
        Cross-Encoder Reranking
                     |
                     v
             Top-K Chunks
                     |
                     v
            Context Builder
                     |
                     v
              LLM Generation
                     |
                     v
          Answer + Page Citations
```

## 🚀 Development Milestones

### V1 — Semantic Retrieval

* Generated sentence embeddings using `all-MiniLM-L6-v2`.
* Implemented vector similarity search with FAISS.
* Explored cosine similarity and top-k retrieval.

### V2 — Improved Chunking

* Implemented paragraph-aware chunking.
* Preserved source document, page number, and chunk ID metadata.
* Improved the organization of extracted PDF text.

### V3 — Stronger Embeddings

* Adopted `BAAI/bge-base-en-v1.5` for semantic retrieval.
* Compared retrieval rankings against the earlier embedding model.

### V4 — Hybrid Retrieval

* Integrated BM25 keyword-based retrieval.
* Combined dense and sparse search results using Reciprocal Rank Fusion.
* Evaluated retrieval behavior across multiple research papers and queries.

### V5 — Cross-Encoder Reranking

* Added `cross-encoder/ms-marco-MiniLM-L-6-v2`.
* Reranked the hybrid retrieval candidates using query-passage relevance scores.
* Improved the position of relevant passages in tested queries.

### V6 — Grounded Answer Generation

* Constructed context from the top-ranked passages.
* Integrated an LLM through the Groq API.
* Generated answers based on retrieved context.
* Included document and page references in generated answers.

## 🛠️ Tech Stack

| Component        | Technology             |
| ---------------- | ---------------------- |
| Language         | Python                 |
| PDF processing   | PyMuPDF / pypdf        |
| Embeddings       | Sentence Transformers  |
| Dense retrieval  | FAISS                  |
| Sparse retrieval | rank-bm25              |
| Hybrid ranking   | Reciprocal Rank Fusion |
| Reranking        | Cross-Encoder          |
| LLM API          | Groq                   |
| Experimentation  | Jupyter Notebook       |
| Version control  | Git and GitHub         |

## 📂 Project Structure

```text
ResearchRAG/
├── data/                          # Local research PDFs (not tracked)
├── notebooks/
│   ├── 01_embeddings_and_retrieval.ipynb
│   ├── 02_pdf_ingestion_and_chunking.ipynb
│   └── 03_vector_retrieval.ipynb
├── src/
│   ├── __init__.py
│   ├── loader.py
│   ├── chunker.py
│   └── retriever.py
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
```

## ⚙️ Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/sanskar407/ResearchRAG.git
cd ResearchRAG
```

### 2. Create a Conda environment

```bash
conda create -n rag python=3.11
conda activate rag
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

If a dependency used by the latest notebook is missing, install it with:

```bash
python -m pip install sentence-transformers faiss-cpu pypdf rank-bm25 groq python-dotenv jupyter ipykernel
```

### 4. Configure the Groq API key

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_groq_api_key
```

Keep this file private. Never commit API keys or other secrets to GitHub.

### 5. Add research papers

Place your research paper PDFs inside the local `data/` directory. The directory is excluded from version control by default.

### 6. Run the notebook

Launch Jupyter:

```bash
jupyter notebook
```

Start with `01_embeddings_and_retrieval.ipynb`, followed by `02_pdf_ingestion_and_chunking.ipynb` and `03_vector_retrieval.ipynb`.

Execute the notebook cells in order to load the documents, build the retrieval indexes, rerank results, and generate answers.

## 🧪 Current Evaluation

The pipeline has been manually tested with questions about:

* Self-attention
* LoRA (Low-Rank Adaptation)
* Chain-of-thought prompting
* Transformer architecture

In these experiments, hybrid retrieval identified relevant research papers, and cross-encoder reranking promoted relevant passages within the candidate set. The end-to-end pipeline also generated a self-attention explanation with a source document and page reference.

**Evaluation status:** Preliminary qualitative testing. A systematic benchmark using metrics such as Recall@K, Mean Reciprocal Rank (MRR), citation correctness, and answer faithfulness is planned.

## 🔒 Data and Security

* Research PDFs in `data/` are excluded from version control by default.
* API keys are stored in `.env`.
* `.env` and other secret files should never be committed.
* Only add documents to the repository when you have permission to distribute them.

## 🗺️ Roadmap

* [ ] Build a retrieval evaluation framework with Recall@K and MRR.
* [ ] Evaluate answer faithfulness and citation correctness.
* [ ] Extract and retrieve relevant PDF figures and tables.
* [ ] Refactor notebook functions into reusable modules under `src/`.
* [ ] Add automated tests for ingestion, chunking, and retrieval.
* [ ] Build an interactive PDF question-answering interface.
* [ ] Add document management and persistent vector indexes.
* [ ] Explore query rewriting and advanced retrieval strategies.

## 🎯 Project Goal

The goal of ResearchRAG is to understand and implement the fundamental building blocks of a reliable RAG system, from raw document ingestion to source-grounded answer generation, while evaluating how each retrieval improvement affects the final results.

---

**Author:** Sanskar Pal

**Repository:** [ResearchRAG on GitHub](https://github.com/sanskar407/ResearchRAG)
