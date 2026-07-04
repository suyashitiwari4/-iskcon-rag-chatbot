"""
ISKCON Guru Q&A RAG Pipeline
Install: pip install python-docx chromadb sentence-transformers google-genai groq tqdm
"""

import os, re, sys, json, zipfile
from pathlib import Path
from tqdm import tqdm

# ── API KEYS — paste your keys here ──
GROQ_API_KEY   = "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"   # FREE at console.groq.com
GOOGLE_API_KEY = "AQ.Ab8RN6KItWai5b22KdhSxwwMvC7PxpKAQh5kbtMBD6EhOF8GMg"  # fallback


# ─────────────────────────────────────────
# EXTRACT ZIP
# ─────────────────────────────────────────
def extract_zip(zip_path="Transcripts.zip", output_dir="transcripts_extracted"):
    if not os.path.exists(zip_path):
        print(f"❌ ZIP not found: {zip_path}"); sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)
    print(f"📦 Extracting {zip_path}...")
    count = 0
    with zipfile.ZipFile(zip_path, 'r') as z:
        for name in z.namelist():
            if name.endswith('.txt') or name.endswith('.docx'):
                fname = Path(name).name
                if not fname or fname.startswith('~$'): continue
                with z.open(name) as src, open(os.path.join(output_dir, fname), 'wb') as dst:
                    dst.write(src.read())
                count += 1
    print(f"✅ Extracted {count} files")


# ─────────────────────────────────────────
# READ + CLEAN
# ─────────────────────────────────────────
def read_txt(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def read_docx(path):
    try:
        from docx import Document
        return "\n".join([p.text for p in Document(path).paragraphs if p.text.strip()])
    except Exception as e:
        print(f"⚠️  Skipping {Path(path).name}: {e}"); return ""

def clean_text(text):
    for a, b in [('\u2019',"'"),('\u2018',"'"),('\u201c','"'),('\u201d','"'),('\u2013','-'),('\u00a0',' ')]:
        text = text.replace(a, b)
    return re.sub(r'\s+', ' ', text).strip()

def load_all_transcripts(folder="transcripts_extracted"):
    docs = []
    files = [f for f in Path(folder).iterdir() if f.suffix in ('.txt', '.docx')]
    print(f"\n📂 Loading {len(files)} files...")
    for file in tqdm(files):
        try:
            text = read_txt(str(file)) if file.suffix == '.txt' else read_docx(str(file))
            text = clean_text(text)
            if len(text) > 100:
                docs.append({"title": file.stem, "text": text, "source": file.name})
        except Exception as e:
            print(f"⚠️  {file.name}: {e}")
    print(f"✅ Loaded {len(docs)} transcripts")
    return docs


# ─────────────────────────────────────────
# CHUNK
# ─────────────────────────────────────────
def chunk_text(text, size=400, overlap=80):
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        chunk = " ".join(words[start:start+size])
        if len(chunk) > 50: chunks.append(chunk)
        start += size - overlap
    return chunks

def prepare_chunks(docs):
    all_chunks, all_meta = [], []
    for doc in docs:
        for i, chunk in enumerate(chunk_text(doc["text"])):
            all_chunks.append(chunk)
            all_meta.append({"title": doc["title"][:100], "source": doc["source"][:100], "chunk_index": str(i)})
    print(f"✅ {len(all_chunks)} chunks from {len(docs)} transcripts")
    return all_chunks, all_meta


# ─────────────────────────────────────────
# VECTOR STORE
# ─────────────────────────────────────────
def build_vector_store(chunks, metadata, db_path="iskcon_vectordb"):
    import chromadb
    from chromadb.utils import embedding_functions
    print("\n🔨 Building vector store...")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=db_path)
    try: client.delete_collection("iskcon_transcripts")
    except: pass
    col = client.create_collection(name="iskcon_transcripts", embedding_function=ef)
    for i in tqdm(range(0, len(chunks), 100), desc="Embedding"):
        b = chunks[i:i+100]; m = metadata[i:i+100]
        col.add(documents=b, metadatas=m, ids=[f"chunk_{i+j}" for j in range(len(b))])
    print(f"✅ Vector store saved: {db_path}/")


# ─────────────────────────────────────────
# GENERATE ANSWER
# ─────────────────────────────────────────
def generate_answer(question, context):
    prompt = f"""You are a helpful assistant answering questions based on teachings of ISKCON guru Chaitanya Charan das.

Use ONLY the transcript excerpts below. Write a clear answer in 150-200 words.
Speak naturally. End with "Key takeaway: ..."

QUESTION: {question}

EXCERPTS:
{context}

ANSWER:"""

    # ── Groq (primary) ──
    if GROQ_API_KEY and GROQ_API_KEY != "gsk_your_groq_key_here":
        try:
            from groq import Groq
            client = Groq(api_key=GROQ_API_KEY)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️  Groq error: {e}")

    # ── Gemini (fallback) ──
    if GOOGLE_API_KEY:
        for model_name in ["gemini-2.0-flash", "gemini-2.0-flash-lite"]:
            try:
                from google import genai
                client = genai.Client(api_key=GOOGLE_API_KEY)
                response = client.models.generate_content(model=model_name, contents=prompt)
                return response.text.strip()
            except Exception as e:
                print(f"⚠️  {model_name} error: {e}")

    return context[:800] + "\n\n[No API available — showing raw text]"


# ─────────────────────────────────────────
# QUERY
# ─────────────────────────────────────────
def query_guru(question, db_path="iskcon_vectordb", top_k=5):
    import chromadb
    from chromadb.utils import embedding_functions
    if not os.path.exists(db_path):
        print("❌ Vector DB not found! Run:  python iskcon_rag_pipeline.py build"); return
    ef  = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    col = chromadb.PersistentClient(path=db_path).get_collection(name="iskcon_transcripts", embedding_function=ef)
    results = col.query(query_texts=[question], n_results=top_k)
    chunks  = results["documents"][0]
    sources = [m["title"] for m in results["metadatas"][0]]
    print("\n" + "="*60)
    print(f"🙏 Question: {question}")
    print("="*60)
    print("⏳ Generating answer...\n")
    answer = generate_answer(question, "\n\n".join(chunks))
    print("💬 Answer:")
    print("-"*60)
    print(answer)
    print("-"*60)
    print(f"\n📚 Based on: {' | '.join(list(dict.fromkeys(sources))[:3])}\n")


# ─────────────────────────────────────────
# BUILD PIPELINE
# ─────────────────────────────────────────
def build_pipeline(zip_path="Transcripts.zip"):
    print("\n🚀 Starting Build\n")
    extract_zip(zip_path)
    docs = load_all_transcripts()
    with open("transcripts_dataset.jsonl", "w", encoding="utf-8") as f:
        for doc in docs: f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print("✅ Saved → transcripts_dataset.jsonl")
    chunks, meta = prepare_chunks(docs)
    build_vector_store(chunks, meta)
    print("\n✅ Done! Now run:  python iskcon_rag_pipeline.py\n")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_pipeline()
    else:
        print("\n" + "="*60)
        print("  🙏  ISKCON Guru Q&A Chatbot  🙏")
        print("="*60)
        if GROQ_API_KEY and GROQ_API_KEY != "gsk_your_groq_key_here":
            print(f"✅ Groq API key loaded (starts with: {GROQ_API_KEY[:8]}...)")
        else:
            print("⚠️  Groq key not set — paste your key in line 12")
        print("Ask any spiritual question. Type 'quit' to exit.\n")
        while True:
            try:
                q = input("Your question: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n🙏 Hare Krishna! Goodbye.\n"); break
            if not q: continue
            if q.lower() in ("quit","exit","q","bye"):
                print("\n🙏 Hare Krishna! Goodbye.\n"); break
            query_guru(q)
