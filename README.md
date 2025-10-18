# 🌌 Constellation Notebook (Local)

**A local-first knowledge constellation.**  
Paste a note → get structured summaries, embeddings, semantic search, a living graph, topic clusters, and portable exports.  
Dark UI, neon sapphire highlights. Minimal code. Maximum value.

A neon lab to help you learn from your own thoughts — entirely local by design.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export OPENAI_API_KEY="sk-..."  # or set in config.yaml
streamlit run app.py
```

## Features
- Add notes → **Pydantic + Instructor** structure (title, summary, entities, relations, tags)
- Local semantic search via **ChromaDB**
- **RAG**-style Q&A grounded in your notes
- Entity + note graph with **NetworkX** (auto-linked related notes by similarity)
- Topic clustering with **scikit-learn** (KMeans) and 2D PCA plot
- **Exports**: CSV, Markdown, JSON bundle (+ toggle to exclude embeddings), and a per-note Markdown bundle (.zip)

## Why It Matters
Constellation Notebook gives you a private, AI-assisted workspace that turns raw reflections into an organized knowledge map, helping you rediscover insights faster without sacrificing control over your data.
- **Think locally, stay private** – Everything runs on your machine: SQLite, vector store, and API usage tied to your own account.
- **Structure your ideas effortlessly** – Drop in messy notes and get clean summaries, tags, relations, and embeddings without manual cleanup.
- **Spot hidden links** – The neon graph and clusters surface connections between thoughts that are hard to see in raw notebooks.
- **Ask better questions of yourself** – Retrieval-augmented answers cite the exact notes they used, closing the feedback loop on personal knowledge.
- **Own your archive** – Exports (CSV, Markdown, JSON bundles) mean you can back up or migrate without vendor lock-in.

## File Layout
```
.
├── app.py
├── config.yaml
├── requirements.txt
├── .streamlit/
│   └── config.toml
├── data/              # local persistence (SQLite + ChromaDB dir will appear here)
└── docs/
    ├── USAGE.md
    ├── ARCHITECTURE.md
    └── CHANGELOG.md
```

## Philosophy
> Minimum code, maximum value.  
Everything here augments your thinking — no lock-in, no bloat, and totally local.

---

Licensed under the MIT License © 2025 Markus Applegate
