# USAGE

## 1) Add Notes
- Paste raw text in the **Add** tab.
- Click **Summarize & Save**.
- The app extracts a title, summary, entities, relations, and tags, then embeds and stores it.

## 2) Search
- Use **Semantic Search** to find related notes.
- Distance is cosine distance (lower is closer).

## 3) Ask
- Enter a question in **Ask**. The model answers using only retrieved notes and will cite note IDs.

## 4) Graph
- Visualizes entities, note mentions, and related-note edges.
- Edge thickness reflects note-to-note similarity.

## 5) Clusters
- KMeans clusters your notes (k adjustable).
- Uses PCA (2-D) for the scatter plot.

## 6) Export / Import
- **Bundle (.json)**: all SQLite rows + Chroma vectors (toggle embeddings on/off).
- **CSV**: notes, relations, note_links.
- **Markdown**: single file summary or per-note Markdown bundle (.zip).
- **Recompute links**: rebuilds note-to-note similarity edges from embeddings.

## Environment Tips
- Prefer `OPENAI_API_KEY` env var over storing in `config.yaml`.
- If Chroma burps, delete `./data/chroma/` and restart.
- If SQLite locks, close duplicate Streamlit tabs and rerun.

