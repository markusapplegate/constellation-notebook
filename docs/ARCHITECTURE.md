# ARCHITECTURE

## Layers
- **UI**: Streamlit (single-page, tabs). Custom dark theme + neon sapphire accents.
- **Config**: YAML read into `AppConfig` (Pydantic v2).
- **LLM**: OpenAI chat + Instructor for strict Pydantic model parsing.
- **Embeddings**: OpenAI `text-embedding-3-small` stored in ChromaDB (cosine space).
- **Relational Store**: SQLite (`notes`, `relations`, `note_links`).
- **Graph**: NetworkX; entities and notes as nodes; edges for relations and similarity.
- **Clustering**: scikit-learn KMeans + PCA for 2D visualization.

## Tables
- `notes(id, created_at, title, text, source, tags)`
- `relations(id, note_id, head, relation, tail)`
- `note_links(id, note_id, other_id, score)` where `score ≈ 1 - cosine_distance`

## Similarity Links
On save:
1. Embed the new note
2. Add to Chroma
3. Query nearest neighbors
4. Insert links with similarity scores
Manual rebuild recomputes these for all notes.

## Export Bundles
JSON bundle schema:
```json
{
  "version": "constellation-notebook:1",
  "created_at": "<ISO8601>",
  "sqlite": {
    "notes": [...],
    "relations": [...],
    "note_links": [...]
  },
  "chroma": {
    "ids": [...],
    "documents": [...],
    "metadatas": [...],
    "embeddings": [...],
    "exclude_embeddings": true | false
  }
}
```
