# ISKCON Guru Q&A Chatbot — RAG Pipeline

An AI-powered Q&A chatbot built on 1,391 transcripts of ISKCON spiritual discourse by Chaitanya Charan das, using Retrieval-Augmented Generation (RAG).

#  Project Overview

This project processes 1,391 real transcripts (.txt + .docx) of ISKCON guru talks and builds a semantic search + AI answer generation system. Instead of manually searching through thousands of pages, users can simply ask a question and get a grounded, summarized answer derived directly from the source transcripts.
Example:

Your question: What does the guru say about moral values?

💬 Answer:
According to the teachings, morality is fundamentally concerned with
justice — acting rightly in our dealings with others. However,
spirituality transcends morality by introducing the principle of mercy,
which operates at a higher level than strict justice...

📚 Based on: Can Morality Be Derived From Science | Row_1488_Can we tell the truth selectively


# Architecture

Transcripts.zip (1,391 files)
        │
        ▼
┌─────────────────────┐
│  Step 1: Extract    │  python-docx, zipfile
│  .txt + .docx files │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Step 2: Clean      │  regex, unicode normalization
│  + Preprocess text  │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Step 3: Chunk      │  400-word chunks, 80-word overlap
│  text into pieces   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Step 4: Embed      │  SentenceTransformers
│  (all-MiniLM-L6-v2) │  384-dimensional vectors
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Step 5: Store      │  ChromaDB (persistent local DB)
│  Vector Database    │
└────────┬────────────┘
         │
    ── Query Time ──
         │
         ▼
┌─────────────────────┐     ┌──────────────────────┐
│  User Question      │────▶│  Embed Question      │
└─────────────────────┘     └──────────┬───────────┘
                                        │
                                        ▼
                             ┌──────────────────────┐
                             │  Cosine Similarity   │
                             │  Search (Top-5)      │
                             └──────────┬───────────┘
                                        │
                                        ▼
                             ┌──────────────────────┐
                             │  Groq LLM            │
                             │  (Llama 3.3 70B)     │
                             └──────────┬───────────┘
                                        │
                                        ▼
                             ┌──────────────────────┐
                             │  Summarized Answer   │
                             │  + Source Citations  │
                             └──────────────────────┘

# Key Concepts used:

1.RAG (Retrieval-Augmented Generation)

Instead of relying on the LLM's pre-trained knowledge, the system first retrieves relevant transcript chunks and passes them as context to the LLM. This ensures answers are grounded in actual source material.

2.Vector Embeddings

Each text chunk is converted into a 384-dimensional numerical vector using the all-MiniLM-L6-v2 model. Similar sentences produce similar vectors.

3.Cosine Similarity

The mathematical method used to compare the query vector against all stored chunk vectors. Returns the top-5 most semantically relevant chunks.

4.Chunking with Overlap

Transcripts are split into 400-word pieces with 80-word overlap to ensure no important context is lost at chunk boundaries.



