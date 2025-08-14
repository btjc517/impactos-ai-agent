import os
import json
import sqlite3
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from embedding_registry import get_embedding_model
from rapidfuzz import process, fuzz

from vector_search import FAISSVectorSearch


logger = logging.getLogger(__name__)


class ConceptGraph:
    """Lightweight concept graph backed by SQLite + optional embeddings.

    Provides lookup of concepts, aliases, and relations. Embeddings can be
    computed on demand for names/aliases and cached externally if needed.
    """

    def __init__(self, db_path: str = "db/impactos.db"):
        env_db = os.getenv('IMPACTOS_DB_PATH')
        self.db_path = env_db if (env_db and db_path == "db/impactos.db") else db_path
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Ensure concept graph schema exists by executing migration if needed."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='concept'")
                if cur.fetchone() is None:
                    # Attempt to run migration file if present
                    mig = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'db', 'migrations', '20250813T060000Z__concept_graph.sql')
                    if os.path.exists(mig):
                        with open(mig, 'r') as f:
                            conn.executescript(f.read())
        except Exception as e:
            logger.warning(f"Concept graph schema ensure failed: {e}")

    def list_concepts(self, concept_type: Optional[str] = None) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            if concept_type:
                cur.execute("SELECT * FROM concept WHERE type=? ORDER BY key", (concept_type,))
            else:
                cur.execute("SELECT * FROM concept ORDER BY type, key")
            return [dict(r) for r in cur.fetchall()]

    def get_concept_by_key(self, concept_type: str, key: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM concept WHERE type=? AND key=?", (concept_type, key))
            row = cur.fetchone()
            return dict(row) if row else None

    def list_aliases(self, concept_id: int) -> List[str]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT alias FROM concept_alias WHERE concept_id=?", (concept_id,))
            return [r[0] for r in cur.fetchall()]

    def upsert_alias(self, concept_id: int, alias: str, lang: str = 'en', source: str = 'runtime', confidence: float = 0.7) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            try:
                cur.execute(
                    """
                    INSERT OR IGNORE INTO concept_alias (concept_id, alias, lang, source, confidence)
                    VALUES (?,?,?,?,?)
                    """,
                    (concept_id, alias.strip(), lang, source, confidence)
                )
                conn.commit()
            except Exception as e:
                logger.warning(f"Failed to upsert alias '{alias}' for concept {concept_id}: {e}")

    def record_resolution_event(self, resolved_type: str, input_text: str, context: Optional[Dict[str, Any]], decided_concept_id: Optional[int], score: Optional[float], outcome: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            ctx_json = json.dumps(context or {})
            cur.execute(
                """
                INSERT INTO resolution_event (resolved_type, input_text, context_json, decided_concept_id, score, outcome)
                VALUES (?,?,?,?,?,?)
                """,
                (resolved_type, input_text, ctx_json, decided_concept_id, score, outcome)
            )
            conn.commit()


class SemanticResolver:
    """Semantic resolver that fuses fuzzy string match and embeddings over the concept graph.

    - Avoids hardcoding by resolving to concepts stored in the DB.
    - Returns canonical keys and scores with robust abstention behavior.
    """

    def __init__(self, db_path: str = "db/impactos.db"):
        self.graph = ConceptGraph(db_path)
        self.embed = self._init_embedding()

    def _init_embedding(self) -> Optional["SentenceTransformer"]:
        try:
            return get_embedding_model()
        except Exception as e:
            logger.warning(f"Failed to init embedding model: {e}")
            return None

    @staticmethod
    def _normalize_text(text: str) -> str:
        return (text or '').strip().lower()

    def _fuzzy_candidates(self, text: str, concepts: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], float]]:
        """Generate candidates via fuzzy string against names + aliases."""
        text_n = self._normalize_text(text)
        choices: List[Tuple[str, Dict[str, Any]]] = []
        for c in concepts:
            base = self._normalize_text(c.get('name') or c.get('key') or '')
            choices.append((base, c))
            for a in self.graph.list_aliases(c['id']):
                choices.append((self._normalize_text(a), c))
        if not choices:
            return []
        labels = [lbl for lbl, _ in choices]
        matches = process.extract(text_n, labels, scorer=fuzz.WRatio, limit=10)
        out: List[Tuple[Dict[str, Any], float]] = []
        for cand_label, score, idx in matches:
            out.append((choices[idx][1], float(score)/100.0))
        # Deduplicate by concept id keep max score
        best: Dict[int, Tuple[Dict[str, Any], float]] = {}
        for c, s in out:
            prev = best.get(c['id'])
            if not prev or s > prev[1]:
                best[c['id']] = (c, s)
        return sorted(best.values(), key=lambda x: x[1], reverse=True)

    def _embed_candidates(self, text: str, concepts: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], float]]:
        if not self.embed:
            return []
        try:
            qv = self.embed.encode([text])[0]
            qv = qv / np.linalg.norm(qv)
        except Exception:
            return []
        cand_vectors = []
        cand_meta: List[Dict[str, Any]] = []
        for c in concepts:
            # Use name + description + aliases as text
            parts = [c.get('name') or '', c.get('description') or '']
            aliases = self.graph.list_aliases(c['id'])
            if aliases:
                parts.append(" | ".join(aliases))
            text_c = " \n ".join(parts)
            try:
                vec = self.embed.encode([text_c])[0]
                vec = vec / np.linalg.norm(vec)
                cand_vectors.append(vec)
                cand_meta.append(c)
            except Exception:
                continue
        if not cand_vectors:
            return []
        sims = np.dot(np.stack(cand_vectors), qv)
        paired = list(zip(cand_meta, sims.tolist()))
        paired.sort(key=lambda x: x[1], reverse=True)
        return paired[:10]

    def resolve(self, concept_type: str, text: str, context: Optional[Dict[str, Any]] = None, min_accept: float = 0.78, min_suggest: float = 0.6) -> Dict[str, Any]:
        """Resolve arbitrary text to a concept of given type.

        Returns: { outcome: 'accepted'|'suggestions'|'abstained', key, name, score, suggestions: [...] }
        """
        concepts = self.graph.list_concepts(concept_type)
        if not concepts:
            return {"outcome": "abstained", "reason": "no_concepts"}

        # Fuzzy + embedding fusion
        fuzzy = self._fuzzy_candidates(text, concepts)
        embed = self._embed_candidates(text, concepts)
        # Merge: take top fuzzy and embedding, weighted average (fuzzy 0.4, embed 0.6)
        scores: Dict[int, float] = {}
        for c, s in fuzzy[:5]:
            scores[c['id']] = max(scores.get(c['id'], 0.0), 0.4 * s)
        for c, s in embed[:5]:
            scores[c['id']] = scores.get(c['id'], 0.0) + 0.6 * float(s)
        ranked: List[Tuple[Dict[str, Any], float]] = []
        for c in concepts:
            if c['id'] in scores:
                ranked.append((c, scores[c['id']]))
        ranked.sort(key=lambda x: x[1], reverse=True)

        if ranked and ranked[0][1] >= min_accept:
            top, score = ranked[0]
            self.graph.record_resolution_event(concept_type, text, context, top['id'], score, 'accepted')
            return {"outcome": "accepted", "key": top['key'], "name": top['name'], "score": float(score)}

        if ranked and ranked[0][1] >= min_suggest:
            suggestions = [{"key": c['key'], "name": c['name'], "score": float(s)} for c, s in ranked[:5]]
            self.graph.record_resolution_event(concept_type, text, context, ranked[0][0]['id'], ranked[0][1], 'suggestions')
            return {"outcome": "suggestions", "suggestions": suggestions}

        self.graph.record_resolution_event(concept_type, text, context, None, None, 'abstained')
        return {"outcome": "abstained"}

    def resolve_file_type(self, filename: str, explicit_hint: Optional[str] = None) -> Dict[str, Any]:
        # Prefer explicit hint when available
        tokens = [explicit_hint or "", os.path.splitext(filename)[1].lstrip('.'), filename]
        text = " ".join([t for t in tokens if t])
        return self.resolve('file_type', text, context={"filename": filename, "hint": explicit_hint})


