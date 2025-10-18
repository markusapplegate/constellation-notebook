\
import os
import io
import re
import csv
import json
import uuid
import sqlite3
import textwrap
from typing import List, Optional, Dict, Any

import yaml
import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from pyvis.network import Network

from pydantic import BaseModel, Field
from datetime import datetime

# OpenAI + Instructor
import instructor
from openai import OpenAI

# Vector + ML
import chromadb
from chromadb.config import Settings as ChromaSettings
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import zipfile


# ---------------------------
# Styling (neon sapphire)
# ---------------------------
NEON_CSS = """
<style>
:root, .stApp, .stMarkdown, .stTextInput, .stTextArea {
  color: #cfe8ff !important;
}
html, body, .stApp {
  background-color: #0b0b10 !important;
}
.neon {
  color: #7fb3ff !important;
  text-shadow:
    0 0 2px #7fb3ff,
    0 0 6px #2e86ff,
    0 0 12px #2e86ff;
}
.small { font-size: 0.9rem; opacity: 0.8; }
.card {
  padding: 0.8rem 1rem;
  border: 1px solid #1b2747;
  border-radius: 8px;
  background: linear-gradient(180deg, rgba(17,20,40,.5) 0%, rgba(14,14,22,.7) 100%);
  margin-bottom: 0.6rem;
}
.codebox {
  background: #0f1220;
  border: 1px solid #1b2747;
  border-radius: 6px;
  padding: 0.6rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  white-space: pre-wrap;
}
</style>
"""


# Ensure .env values (e.g., OPENAI_API_KEY) are available via os.getenv
load_dotenv()


# ---------------------------
# Config
# ---------------------------
class AppConfig(BaseModel):
    app_title: str = "Constellation Notebook (Local)"
    data_dir: str = "./data"
    sqlite_path: str = "./data/notes.db"
    chroma_path: str = "./data/chroma"
    openai_api_key: Optional[str] = None
    chat_model: str = "gpt-4o-mini"
    embed_model: str = "text-embedding-3-small"
    top_k: int = 5
    kmeans_k: int = 6
    temperature: float = 0.2

def load_config(path: str = "config.yaml") -> AppConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    cfg = AppConfig(**raw)
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        cfg.openai_api_key = env_key
    return cfg


# ---------------------------
# Data models (Pydantic)
# ---------------------------
class Relation(BaseModel):
    head: str = Field(..., description="Entity head")
    relation: str = Field(..., description="Relation verb or label")
    tail: str = Field(..., description="Entity tail")

class NoteSummary(BaseModel):
    title: str
    short_summary: str
    key_points: List[str] = []
    entities: List[str] = []
    suggested_tags: List[str] = []
    relations: List[Relation] = []

class QAAnswer(BaseModel):
    answer: str
    citations: List[str] = Field(default_factory=list, description="IDs of cited notes")


# ---------------------------
# Paths & Persistence
# ---------------------------
def ensure_dirs(cfg: AppConfig):
    os.makedirs(cfg.data_dir, exist_ok=True)
    os.makedirs(cfg.chroma_path, exist_ok=True)

def db_connect(cfg: AppConfig):
    conn = sqlite3.connect(cfg.sqlite_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            title TEXT,
            text TEXT NOT NULL,
            source TEXT,
            tags TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS relations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            note_id TEXT NOT NULL,
            head TEXT NOT NULL,
            relation TEXT NOT NULL,
            tail TEXT NOT NULL
        )
    """)
    # note-to-note similarity links
    conn.execute("""
        CREATE TABLE IF NOT EXISTS note_links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            note_id TEXT NOT NULL,
            other_id TEXT NOT NULL,
            score REAL NOT NULL  -- similarity (1 - cosine distance)
        )
    """)
    conn.commit()
    return conn


# ---------------------------
# Chroma (local) vector store
# ---------------------------
def get_chroma(cfg: AppConfig):
    client = chromadb.PersistentClient(
        path=cfg.chroma_path,
        settings=ChromaSettings(allow_reset=False)
    )
    coll = client.get_or_create_collection(
        name="notes",
        metadata={"hnsw:space": "cosine"}
    )
    return coll

def embed_text(client: OpenAI, cfg: AppConfig, text: str) -> List[float]:
    emb = client.embeddings.create(model=cfg.embed_model, input=text)
    return emb.data[0].embedding


# ---------------------------
# OpenAI + Instructor helpers
# ---------------------------
def get_clients(cfg: AppConfig):
    if not cfg.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY not set (env var preferred or config.yaml).")
    base_client = OpenAI(api_key=cfg.openai_api_key)
    patched = instructor.patch(base_client)
    return base_client, patched

SUMMARY_SYS = """You are a concise research assistant. Extract clean structure from a user note."""
SUMMARY_USER_TMPL = """Extract a structured summary of the note below.

Return fields:
- title (<=8 words)
- short_summary (2-3 sentences)
- key_points (3-7 bullets)
- entities (unique, proper nouns or key terms)
- suggested_tags (short kebab-case)
- relations (triples: head, relation, tail)

Note:
---
{note}
---
"""

def llm_summarize(patched_client: OpenAI, cfg: AppConfig, text: str) -> NoteSummary:
    out: NoteSummary = patched_client.chat.completions.create(
        model=cfg.chat_model,
        temperature=cfg.temperature,
        response_model=NoteSummary,  # type: ignore
        messages=[
            {"role": "system", "content": SUMMARY_SYS},
            {"role": "user", "content": SUMMARY_USER_TMPL.format(note=text)}
        ]
    )
    return out

QA_SYS = "You answer based only on the provided notes. If unsure, say so. Keep it concise and cite note IDs."
QA_USER_TMPL = """Question: {q}

Context notes:
{ctx}

Return a JSON with:
- answer: string
- citations: array of note IDs used
"""

def llm_answer(patched_client: OpenAI, cfg: AppConfig, question: str, ctx_blocks: List[dict]) -> QAAnswer:
    ctx_str = "\n".join(
        [f"- ({c['id']}) {c['title']}\n{textwrap.shorten(c['text'], width=300)}" for c in ctx_blocks]
    )
    out: QAAnswer = patched_client.chat.completions.create(
        model=cfg.chat_model,
        temperature=cfg.temperature,
        response_model=QAAnswer,  # type: ignore
        messages=[
            {"role": "system", "content": QA_SYS},
            {"role": "user", "content": QA_USER_TMPL.format(q=question, ctx=ctx_str)}
        ]
    )
    return out


# ---------------------------
# Graph utilities (NetworkX)
# ---------------------------
def build_graph(conn) -> nx.Graph:
    G = nx.Graph()
    # Notes as nodes
    cur = conn.execute("SELECT id, title FROM notes")
    for row in cur.fetchall():
        G.add_node(row["id"], label=row["title"], kind="note")

    # Entity relations
    cur = conn.execute("SELECT note_id, head, relation, tail FROM relations")
    for r in cur.fetchall():
        head = f"ent::{r['head']}"
        tail = f"ent::{r['tail']}"
        G.add_node(head, label=r["head"], kind="entity")
        G.add_node(tail, label=r["tail"], kind="entity")
        G.add_edge(head, tail, label=r["relation"], note=r["note_id"])
        G.add_edge(r["note_id"], head, label="mentions")
        G.add_edge(r["note_id"], tail, label="mentions")

    # Evolving note-to-note links
    cur = conn.execute("SELECT note_id, other_id, score FROM note_links")
    for r in cur.fetchall():
        G.add_edge(r["note_id"], r["other_id"], label=f"related ({r['score']:.2f})", weight=r["score"])

    return G


def draw_graph(G: nx.Graph):
    net = Network(
        height="620px",
        width="100%",
        bgcolor="#0b0b10",
        font_color="#cfe8ff",
        notebook=False,
        directed=False,
    )

    net.force_atlas_2based(
        gravity=-25,
        central_gravity=0.015,
        spring_length=140,
        spring_strength=0.08,
        damping=0.85,
        overlap=0.6,
    )

    for node, data in G.nodes(data=True):
        kind = data.get("kind", "entity")
        label = data.get("label", node)
        base_title = f"{label}"
        if kind == "note":
            net.add_node(
                node,
                label=label,
                title=f"Note · {base_title}",
                shape="dot",
                size=26,
                color={
                    "background": "#2E86FF",
                    "border": "#7FB3FF",
                    "highlight": {"background": "#7FB3FF", "border": "#cfe8ff"},
                    "hover": {"background": "#7FB3FF", "border": "#cfe8ff"},
                },
            )
        else:
            net.add_node(
                node,
                label=label,
                title=f"Entity · {base_title}",
                shape="dot",
                size=18,
                color={
                    "background": "#0f1220",
                    "border": "#4A6FFF",
                    "highlight": {"background": "#4A6FFF", "border": "#cfe8ff"},
                    "hover": {"background": "#4A6FFF", "border": "#cfe8ff"},
                },
            )

    for u, v, data in G.edges(data=True):
        label = data.get("label", "")
        edge_weight = float(data.get("weight", 0.6))
        tooltip = label or ""
        note_context = data.get("note")
        if note_context:
            tooltip = f"{label} · note {note_context}"
        net.add_edge(
            u,
            v,
            title=tooltip if tooltip else None,
            color="#7FB3FF",
            width=max(1.0, 1.2 + 2.0 * edge_weight),
            smooth=True,
        )

    net_options = {
        "interaction": {
            "hover": True,
            "navigationButtons": True,
            "keyboard": True,
        },
        "nodes": {
            "font": {"size": 12, "face": "Inter"},
            "borderWidth": 1.5,
        },
        "edges": {
            "color": {"color": "#4A6FFF", "highlight": "#7FB3FF"},
            "font": {
                "size": 10,
                "color": "#cfe8ff",
                "strokeWidth": 0,
                "vadjust": -6,
                "align": "horizontal",
            },
            "smooth": {"type": "continuous"},
        },
        "physics": {
            "stabilization": {"iterations": 120},
            "minVelocity": 0.75,
        },
    }
    net.set_options(json.dumps(net_options))

    html = net.generate_html(notebook=False)
    components.html(html, height=640, scrolling=False)


# ---------------------------
# Export / Import helpers
# ---------------------------
def fetch_all_sqlite(conn) -> Dict[str, Any]:
    data: Dict[str, Any] = {"notes": [], "relations": [], "note_links": []}
    for row in conn.execute("SELECT id, created_at, title, text, source, tags FROM notes").fetchall():
        data["notes"].append(dict(row))
    for row in conn.execute("SELECT note_id, head, relation, tail FROM relations").fetchall():
        data["relations"].append(dict(row))
    for row in conn.execute("SELECT note_id, other_id, score FROM note_links").fetchall():
        data["note_links"].append(dict(row))
    return data

def chroma_export(coll, exclude_embeddings: bool = False) -> Dict[str, Any]:
    try:
        total = coll.count()
    except Exception:
        total = None
    include = ["documents", "metadatas", "embeddings"]
    if total is not None:
        got = coll.get(include=include, limit=total)
    else:
        got = coll.get(include=include)
    payload = {
        "ids": got.get("ids", []),
        "documents": got.get("documents", []),
        "metadatas": got.get("metadatas", []),
        "embeddings": [] if exclude_embeddings else got.get("embeddings", []),
        "exclude_embeddings": exclude_embeddings,
    }
    return payload

def chroma_import(coll, bundle: Dict[str, Any]):
    ids = bundle.get("ids", [])
    docs = bundle.get("documents", [])
    metas = bundle.get("metadatas", [])
    embs = bundle.get("embeddings", [])
    if not ids:
        return
    try:
        if embs:
            coll.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
        else:
            coll.upsert(ids=ids, documents=docs, metadatas=metas)
    except Exception:
        if embs:
            coll.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
        else:
            coll.add(ids=ids, documents=docs, metadatas=metas)

def sqlite_import(conn, payload: Dict[str, Any]):
    with conn:
        for n in payload.get("notes", []):
            conn.execute(
                "INSERT OR REPLACE INTO notes (id, created_at, title, text, source, tags) VALUES (?, ?, ?, ?, ?, ?)",
                (n["id"], n["created_at"], n.get("title"), n["text"], n.get("source"), n.get("tags"))
            )
        for r in payload.get("relations", []):
            conn.execute(
                "INSERT INTO relations (note_id, head, relation, tail) VALUES (?, ?, ?, ?)",
                (r["note_id"], r["head"], r["relation"], r["tail"])
            )
        for l in payload.get("note_links", []):
            conn.execute(
                "INSERT INTO note_links (note_id, other_id, score) VALUES (?, ?, ?)",
                (l["note_id"], l["other_id"], float(l["score"]))
            )


# ---------------------------
# Note management helpers
# ---------------------------
def list_notes(conn, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    base_query = "SELECT id, title, created_at FROM notes ORDER BY created_at DESC"
    if limit:
        cur = conn.execute(base_query + " LIMIT ?", (limit,))
    else:
        cur = conn.execute(base_query)
    return [dict(row) for row in cur.fetchall()]


def fetch_note(conn, note_id: str) -> Optional[Dict[str, Any]]:
    cur = conn.execute(
        "SELECT id, created_at, title, text, source, tags FROM notes WHERE id = ?",
        (note_id,),
    )
    row = cur.fetchone()
    return dict(row) if row else None


def delete_note(conn, coll, note_id: str) -> bool:
    with conn:
        conn.execute("DELETE FROM relations WHERE note_id = ?", (note_id,))
        conn.execute("DELETE FROM note_links WHERE note_id = ? OR other_id = ?", (note_id, note_id))
        cur = conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        removed = cur.rowcount
    try:
        coll.delete(ids=[note_id])
    except Exception:
        # Swallow Chroma delete errors; note already gone locally
        pass
    return bool(removed)


# ---------------------------
# Similarity linking (evolving graph)
# ---------------------------
def link_similar_notes(conn, coll, new_id: str, new_vec: List[float], topn: int = 5):
    try:
        res = coll.query(query_embeddings=[new_vec], n_results=topn + 1, include=["ids", "distances"])
    except Exception:
        return
    ids = (res.get("ids") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    with conn:
        for other_id, dist in zip(ids, dists):
            if other_id is None or other_id == new_id:
                continue
            sim = max(0.0, 1.0 - float(dist))
            conn.execute(
                "INSERT INTO note_links (note_id, other_id, score) VALUES (?, ?, ?)",
                (new_id, other_id, sim)
            )

def rebuild_all_links(conn, coll, topn: int = 5):
    try:
        with conn:
            conn.execute("DELETE FROM note_links")
        got = coll.get(include=["embeddings", "ids"])
        ids = got.get("ids", [])
        embs = got.get("embeddings", [])
        if not ids or not embs:
            return 0
        for nid, vec in zip(ids, embs):
            res = coll.query(query_embeddings=[vec], n_results=topn + 1, include=["ids", "distances"])
            cids = (res.get("ids") or [[]])[0]
            cdists = (res.get("distances") or [[]])[0]
            for other_id, dist in zip(cids, cdists):
                if other_id == nid:
                    continue
                sim = max(0.0, 1.0 - float(dist))
                conn.execute(
                    "INSERT INTO note_links (note_id, other_id, score) VALUES (?, ?, ?)",
                    (nid, other_id, sim)
                )
        conn.commit()
        return len(ids)
    except Exception:
        return 0


# ---------------------------
# CSV / Markdown export builders
# ---------------------------
def safe_slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r'[^a-z0-9]+', '-', s).strip('-')
    return s or "untitled"

def build_notes_csv(conn) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "created_at", "title", "source", "tags", "text"])
    for row in conn.execute("SELECT id, created_at, title, source, tags, text FROM notes"):
        w.writerow([row["id"], row["created_at"], row["title"] or "", row["source"] or "", row["tags"] or "", row["text"]])
    return buf.getvalue().encode("utf-8")

def build_relations_csv(conn) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["note_id", "head", "relation", "tail"])
    for row in conn.execute("SELECT note_id, head, relation, tail FROM relations"):
        w.writerow([row["note_id"], row["head"], row["relation"], row["tail"]])
    return buf.getvalue().encode("utf-8")

def build_links_csv(conn) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["note_id", "other_id", "score"])
    for row in conn.execute("SELECT note_id, other_id, score FROM note_links"):
        w.writerow([row["note_id"], row["other_id"], row["score"]])
    return buf.getvalue().encode("utf-8")

def build_markdown(conn) -> bytes:
    lines = ["# Constellation Notebook Export", "", f"_Generated: {datetime.utcnow().isoformat()}_", ""]
    # Notes
    lines.append("## Notes")
    for row in conn.execute("SELECT id, created_at, title, source, tags, text FROM notes"):
        title = row["title"] or "(untitled)"
        lines.append(f"### {title}  `[{row['id']}]`")
        lines.append(f"- Created: {row['created_at']}")
        if row["source"]:
            lines.append(f"- Source: {row['source']}")
        if row["tags"]:
            lines.append(f"- Tags: {row['tags']}")
        lines.append("")
        lines.append("```")
        lines.append(row["text"] or "")
        lines.append("```")
        lines.append("")
    # Relations
    lines.append("## Relations (Triples)")
    for row in conn.execute("SELECT note_id, head, relation, tail FROM relations"):
        lines.append(f"- `{row['note_id']}`: **{row['head']}** — _{row['relation']}_ → **{row['tail']}**")
    # Links
    lines.append("")
    lines.append("## Note Links (Similarity)")
    for row in conn.execute("SELECT note_id, other_id, score FROM note_links"):
        lines.append(f"- `{row['note_id']}` ⇄ `{row['other_id']}` · score={row['score']:.3f}")
    return "\n".join(lines).encode("utf-8")

def build_markdown_bundle_zip(conn) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        # Index README
        index_lines = ["# Constellation Notes", ""]
        notes = list(conn.execute("SELECT id, created_at, title, source, tags, text FROM notes"))
        # Map for relations and links
        rels = list(conn.execute("SELECT note_id, head, relation, tail FROM relations"))
        links = list(conn.execute("SELECT note_id, other_id, score FROM note_links"))

        rel_by_note = {}
        for r in rels:
            rel_by_note.setdefault(r["note_id"], []).append(r)
        links_by_note = {}
        for l in links:
            links_by_note.setdefault(l["note_id"], []).append(l)

        for row in notes:
            nid = row["id"]
            title = row["title"] or "(untitled)"
            slug = safe_slug(title)
            md_lines = [
                f"# {title}  `[{nid}]`",
                "",
                f"- Created: {row['created_at']}",
                f"- Source: {row['source'] or ''}",
                f"- Tags: {row['tags'] or ''}",
                "",
                "## Content",
                "```",
                row["text"] or "",
                "```",
            ]
            # Relations for this note
            rs = rel_by_note.get(nid, [])
            if rs:
                md_lines.append("")
                md_lines.append("## Relations")
                for r in rs:
                    md_lines.append(f"- **{r['head']}** — _{r['relation']}_ → **{r['tail']}**")
            # Similar links for this note
            ls = links_by_note.get(nid, [])
            if ls:
                md_lines.append("")
                md_lines.append("## Related Notes")
                for l in ls:
                    md_lines.append(f"- `{l['other_id']}` · score={float(l['score']):.3f}")
            content = "\n".join(md_lines).encode("utf-8")
            z.writestr(f"notes/{slug}-{nid}.md", content)
            index_lines.append(f"- [{title}](notes/{slug}-{nid}.md)  `{nid}`")

        # Relations CSV
        rel_csv = "note_id,head,relation,tail\n" + "\n".join(
            [f"{r['note_id']},{r['head']},{r['relation']},{r['tail']}" for r in rels]
        )
        z.writestr("relations.csv", rel_csv.encode("utf-8"))
        # Links CSV
        link_csv = "note_id,other_id,score\n" + "\n".join(
            [f"{l['note_id']},{l['other_id']},{float(l['score']):.6f}" for l in links]
        )
        z.writestr("note_links.csv", link_csv.encode("utf-8"))

        z.writestr("README.md", "\n".join(index_lines).encode("utf-8"))
    buf.seek(0)
    return buf.getvalue()


# ---------------------------
# Streamlit App
# ---------------------------
def main():
    st.set_page_config(page_title="Constellation Notebook", page_icon="⭐", layout="wide")
    st.markdown(NEON_CSS, unsafe_allow_html=True)

    cfg = load_config()
    ensure_dirs(cfg)

    st.title(f"🔭  {cfg.app_title}")
    st.caption("A neon lab to help you learn from your own thoughts — entirely local by design.")

    with st.sidebar:
        st.subheader("Config")
        st.write(f"Chroma path: `{cfg.chroma_path}`")
        st.write(f"SQLite: `{cfg.sqlite_path}`")
        st.write(f"Chat model: `{cfg.chat_model}`")
        st.write(f"Embed model: `{cfg.embed_model}`")
        st.write(f"Top-k: {cfg.top_k}")
        st.write(f"KMeans k: {cfg.kmeans_k}")
        key_status = "✅ detected" if cfg.openai_api_key else "❌ missing"
        st.write(f"OpenAI key: {key_status}")

    try:
        base_client, patched_client = get_clients(cfg)
    except Exception as e:
        st.error(f"OpenAI client error: {e}")
        return

    conn = db_connect(cfg)
    coll = get_chroma(cfg)

    tab_add, tab_search, tab_ask, tab_graph, tab_clusters, tab_manage, tab_backup = st.tabs(
        ["Add", "Search", "Ask", "Graph", "Clusters", "Manage", "Export/Import"]
    )

    # ---------------- Add
    with tab_add:
        st.subheader("Add a Note")
        text = st.text_area("Paste text / notes", height=180, placeholder="Drop in a snippet, thought, or paragraph…")
        col1, col2 = st.columns(2)
        with col1:
            source = st.text_input("Source (optional)", placeholder="e.g., book/article/url/personal")
        with col2:
            tags_raw = st.text_input("Tags (comma-separated)", placeholder="e.g., ai, lego, ethics")
        if st.button("Summarize & Save", disabled=not bool(text.strip())):
            with st.spinner("Summarizing, embedding, and saving…"):
                nid = f"note-{uuid.uuid4().hex[:8]}"
                summary = llm_summarize(patched_client, cfg, text)
                vec = embed_text(base_client, cfg, text)
                # Save Chroma first (so similarity search can see it) then link
                coll.add(
                    ids=[nid],
                    documents=[text],
                    metadatas=[{
                        "title": summary.title,
                        "source": source or "",
                        "tags": ",".join(summary.suggested_tags) if summary.suggested_tags else tags_raw
                    }],
                    embeddings=[vec]
                )
                # Persist note + relations
                conn.execute(
                    "INSERT INTO notes (id, created_at, title, text, source, tags) VALUES (?, ?, ?, ?, ?, ?)",
                    (nid, datetime.utcnow().isoformat(), summary.title, text, source, ",".join(summary.suggested_tags) if summary.suggested_tags else tags_raw)
                )
                for rel in summary.relations:
                    conn.execute(
                        "INSERT INTO relations (note_id, head, relation, tail) VALUES (?, ?, ?, ?)",
                        (nid, rel.head, rel.relation, rel.tail)
                    )
                conn.commit()
                # Evolving links
                link_similar_notes(conn, coll, nid, vec, topn=5)

                st.success(f"Saved note `{nid}` and linked to similar notes.")
                with st.expander("Structured summary"):
                    st.markdown(f"**Title:** {summary.title}")
                    st.markdown(f"**Short summary:** {summary.short_summary}")
                    if summary.key_points:
                        st.markdown("**Key points:**")
                        for k in summary.key_points:
                            st.markdown(f"- {k}")
                    if summary.entities:
                        st.markdown(f"**Entities:** {', '.join(summary.entities)}")
                    if summary.suggested_tags:
                        st.markdown(f"**Suggested tags:** {', '.join(summary.suggested_tags)}")
                    if summary.relations:
                        st.markdown("**Relations (triples):**")
                        for r in summary.relations:
                            st.markdown(f"- ({r.head}) — *{r.relation}* → ({r.tail})")

    # ---------------- Search
    with tab_search:
        st.subheader("Semantic Search")
        q = st.text_input("Query", placeholder="e.g., 'How do my ideas relate to Carl Jung?'")
        topk = st.slider("Top-k", 1, 20, cfg.top_k)
        if st.button("Search", disabled=not q.strip()):
            with st.spinner("Searching…"):
                qvec = embed_text(base_client, cfg, q)
                res = coll.query(query_embeddings=[qvec], n_results=topk, include=["metadatas", "documents", "distances"])
                if not res["ids"]:
                    st.info("No results yet. Try adding notes.")
                else:
                    for i in range(len(res["ids"][0])):
                        nid = res["ids"][0][i]
                        dist = res["distances"][0][i]
                        meta = res["metadatas"][0][i] or {}
                        doc = res["documents"][0][i]
                        st.markdown(f"**{meta.get('title','(untitled)')}**  \n`{nid}`  · distance: {dist:.3f}")
                        st.markdown(f"<div class='card'><div class='small'>{meta.get('source','')}</div><div class='codebox'>{doc}</div><div class='small'>tags: {meta.get('tags','')}</div></div>", unsafe_allow_html=True)

    # ---------------- Ask
    with tab_ask:
        st.subheader("Ask a Question (RAG)")
        q2 = st.text_input("Your question", placeholder="Ask based on your saved notes…")
        topk2 = st.slider("Retrieve top-k", 1, 10, cfg.top_k, key="ask_k")
        if st.button("Answer", disabled=not q2.strip()):
            with st.spinner("Retrieving and composing…"):
                qvec = embed_text(base_client, cfg, q2)
                res = coll.query(query_embeddings=[qvec], n_results=topk2, include=["metadatas", "documents"])
                ctx = []
                if res["ids"]:
                    for i in range(len(res["ids"][0])):
                        ctx.append({
                            "id": res["ids"][0][i],
                            "title": (res["metadatas"][0][i] or {}).get("title","(untitled)"),
                            "text": res["documents"][0][i],
                        })
                if not ctx:
                    st.info("No notes to answer from yet.")
                else:
                    ans = llm_answer(patched_client, cfg, q2, ctx)
                    st.markdown("**Answer**")
                    st.write(ans.answer)
                    if ans.citations:
                        st.caption("Citations: " + ", ".join(ans.citations))
                    with st.expander("Retrieved context"):
                        for c in ctx:
                            st.markdown(f"- **{c['title']}** (`{c['id']}`)")

    # ---------------- Graph
    with tab_graph:
        st.subheader("Knowledge Graph")
        G = build_graph(conn)
        if G.number_of_nodes() == 0:
            st.info("Add a few notes with relations to see the graph.")
        else:
            draw_graph(G)

    # ---------------- Clusters
    with tab_clusters:
        st.subheader("Topic Clusters")
        got = coll.get(include=["embeddings", "metadatas"])
        embs = got.get("embeddings")
        if embs is None:
            embs = []
        ids = got.get("ids", [])
        metas = got.get("metadatas", [])
        if len(embs) == 0:
            st.info("No embeddings yet. Add notes first.")
        elif len(embs) < 2:
            st.info("Need at least two notes to form clusters.")
        else:
            max_k = min(20, len(embs))
            default_k = min(max(2, cfg.kmeans_k), max_k)
            if max_k == 2:
                k = 2
                st.caption("Exactly two notes found; clustering uses k=2.")
            else:
                k = st.slider("KMeans (k)", 2, max_k, default_k)
            km = KMeans(n_clusters=k, n_init="auto", random_state=42)
            labels = km.fit_predict(embs)
            pca = PCA(n_components=2, random_state=42)
            pts = pca.fit_transform(embs)

            palette = [
                "#2E86FF",
                "#7FB3FF",
                "#9D5CFF",
                "#F39C12",
                "#FF6B81",
                "#1ABC9C",
                "#8E44AD",
                "#16A085",
            ]
            point_colors = [palette[int(lab) % len(palette)] for lab in labels]

            fig, ax = plt.subplots(figsize=(7, 5))
            fig.patch.set_facecolor("#0b0b10")
            ax.set_facecolor("#0f1220")

            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=70,
                c=point_colors,
                alpha=0.9,
                edgecolors="#cfe8ff",
                linewidths=0.6,
            )
            for i, nid in enumerate(ids):
                ax.text(
                    pts[i, 0],
                    pts[i, 1],
                    f" {labels[i]}",
                    fontsize=8,
                    alpha=0.75,
                    color="#cfe8ff",
                )
            ax.set_title("PCA of embeddings · neon clusters", color="#7FB3FF", fontsize=11, pad=16)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.margins(0.1)
            st.pyplot(fig)

            with st.expander("Cluster members"):
                groups = {}
                for i, lab in enumerate(labels):
                    groups.setdefault(lab, []).append((ids[i], (metas[i] or {}).get("title","(untitled)")))
                for lab, items in sorted(groups.items()):
                    st.markdown(f"**Cluster {lab}**")
                    for nid, title in items:
                        st.markdown(f"- `{nid}` — {title}")

    # ---------------- Manage
    with tab_manage:
        st.subheader("Manage Notes")
        notes = list_notes(conn)
        if not notes:
            st.info("No notes available yet.")
        else:
            option_map = {
                f"{(n.get('title') or '(untitled)')[:60]} · {n['id']}": n["id"] for n in notes
            }
            labels = list(option_map.keys())
            choice = st.selectbox("Select a note to review or delete", labels)
            if choice:
                target_id = option_map[choice]
                note = fetch_note(conn, target_id)
                if not note:
                    st.warning("Note could not be loaded. Try refreshing the app.")
                else:
                    st.markdown(f"**Title:** {note.get('title') or '(untitled)'}")
                    st.markdown(f"**Created:** {note.get('created_at', 'unknown')}")
                    if note.get("source"):
                        st.markdown(f"**Source:** {note['source']}")
                    tags = (note.get("tags") or "").strip()
                    if tags:
                        st.markdown(f"**Tags:** {tags}")
                    st.markdown("**Note text:**")
                    st.text_area(
                        "Note text",
                        note.get("text", ""),
                        height=220,
                        disabled=True,
                        label_visibility="collapsed",
                        key=f"preview_{target_id}",
                    )
                    st.divider()
                    st.warning("Deleting a note removes its text, embeddings, graph links, and exports. This action cannot be undone.")
                    confirm = st.toggle(
                        "I understand and want to delete this note.",
                        key=f"confirm_delete_{target_id}",
                    )
                    if st.button("Delete note", type="primary", disabled=not confirm, key=f"delete_{target_id}"):
                        deleted = delete_note(conn, coll, target_id)
                        if deleted:
                            st.success(f"Deleted note `{target_id}`.")
                            st.rerun()
                        else:
                            st.error("Note was not found. It may have been removed already.")

    # ---------------- Export / Import
    with tab_backup:
        st.subheader("Export / Import")
        st.caption("Bundle includes SQLite notes/relations/links + Chroma vectors. All local.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Export current bundle**")
            exclude_emb = st.checkbox("Exclude embeddings from bundle (lighter)", value=True)
            if st.button("Create Export Bundle (.json)"):
                with st.spinner("Collecting data…"):
                    sql_blob = fetch_all_sqlite(conn)
                    vec_blob = chroma_export(coll, exclude_embeddings=exclude_emb)
                    bundle = {
                        "version": "constellation-notebook:1",
                        "created_at": datetime.utcnow().isoformat(),
                        "sqlite": sql_blob,
                        "chroma": vec_blob
                    }
                    j = json.dumps(bundle)
                    st.download_button(
                        "Download bundle.json",
                        data=j,
                        file_name="constellation_bundle.json",
                        mime="application/json"
                    )

            st.markdown("---")
            st.markdown("**CSV exports**")
            if st.button("Build CSVs"):
                notes_csv = build_notes_csv(conn)
                rel_csv = build_relations_csv(conn)
                links_csv = build_links_csv(conn)
                st.download_button("Download notes.csv", data=notes_csv, file_name="notes.csv", mime="text/csv")
                st.download_button("Download relations.csv", data=rel_csv, file_name="relations.csv", mime="text/csv")
                st.download_button("Download note_links.csv", data=links_csv, file_name="note_links.csv", mime="text/csv")

            st.markdown("---")
            st.markdown("**Markdown export**")
            if st.button("Build Markdown"):
                md = build_markdown(conn)
                st.download_button("Download constellation_notes.md", data=md, file_name="constellation_notes.md", mime="text/markdown")
            if st.button("Build Markdown bundle (.zip)"):
                bundle_bytes = build_markdown_bundle_zip(conn)
                st.download_button("Download notes_markdown_bundle.zip", data=bundle_bytes, file_name="notes_markdown_bundle.zip", mime="application/zip")

        with c2:
            st.markdown("**Import bundle**")
            up = st.file_uploader("Upload a bundle.json", type=["json"])
            if up is not None:
                try:
                    payload = json.load(up)
                    with st.spinner("Importing bundle…"):
                        sqlite_import(conn, payload.get("sqlite", {}))
                        chroma_import(coll, payload.get("chroma", {}))
                    st.success("Import complete.")
                except Exception as e:
                    st.error(f"Import failed: {e}")

        st.divider()
        st.markdown("**Rebuild related links**")
        if st.button("Recompute all note-to-note links"):
            with st.spinner("Recomputing similarity links…"):
                n = rebuild_all_links(conn, coll, topn=5)
            st.success("Rebuild complete." if n else "No notes/embeddings available yet.")

if __name__ == "__main__":
    main()
