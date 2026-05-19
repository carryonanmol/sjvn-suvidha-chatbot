"""
Feedback & Self-Learning Module — v4 (unchanged from v3 fixes)
"""

import os
import re
import sqlite3
from typing import Optional

_BASE   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DB_PATH = os.path.join(_BASE, 'data', 'index.db')


def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.execute('PRAGMA journal_mode=WAL')
    return c


def init_feedback_tables():
    c = _conn()
    c.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            query       TEXT NOT NULL,
            bad_answer  TEXT,
            correction  TEXT,
            rating      INTEGER DEFAULT 0,
            source_hint TEXT,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS correction_memory (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            query_pat   TEXT UNIQUE NOT NULL,
            correct_ans TEXT NOT NULL,
            raw_query   TEXT,
            hit_count   INTEGER DEFAULT 0,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.commit()
    c.close()
    print("[FEEDBACK] feedback + correction_memory tables ready.")


_STOP = {
    'a','an','the','and','or','but','is','are','was','were','in','on','at',
    'to','for','of','with','what','how','tell','me','give','about','does',
    'do','please','can','could','would','should','its','it','this','that',
    'these','those','has','have','had','be','been','being','will','shall',
    'say','says','said','per','as','by','from','up','if','so','no','not',
}

_DOC_ALIASES = {
    'dop': 'dop', 'delegation': 'dop',
    'leave': '5889', '5889': '5889',
    'procurement': 'procurement', 'works': 'procurement',
    'grievance': '1578', '1578': '1578',
    'deputation': '2290', '2290': '2290',
    'kpi': 'kpi', 'appraisal': 'kpi', 'par': 'kpi', 'performance': 'kpi',
}


def _normalize(text: str) -> str:
    t = text.lower().strip()
    tokens = set()
    for m in re.finditer(r'[a-z]*\d+\.\d+(?:\.\d+)*', t):
        tokens.add(m.group())
    consumed = set(m.span() for m in re.finditer(r'[a-z]*\d+\.\d+(?:\.\d+)*', t))
    for m in re.finditer(r'[a-z0-9]+', t):
        if any(s <= m.start() < e for s, e in consumed):
            continue
        w = m.group()
        if w not in _STOP and len(w) >= 2:
            tokens.add(_DOC_ALIASES.get(w, w))
    return ' '.join(sorted(tokens))


def _key_terms(text: str) -> set:
    t = text.lower()
    terms = set()
    for m in re.finditer(r'[a-z]*\d+(?:\.\d+)+', t):
        terms.add(m.group())
    for m in re.finditer(r'\bc(\d+)\b', t):
        terms.add(f"c{m.group(1)}")
    for m in re.finditer(r'\b(\d{1,3})\b', t):
        terms.add(f"num:{m.group(1)}")
    for alias, canon in _DOC_ALIASES.items():
        if re.search(r'\b' + re.escape(alias) + r'\b', t):
            terms.add(canon)
    for m in re.finditer(r'[a-z]{4,}', t):
        w = m.group()
        if w not in _STOP:
            terms.add(w)
    return terms


def _extract_numbers(text: str) -> set:
    nums = set()
    t = text.lower()
    dotted_spans = []
    for m in re.finditer(r'\d+\.\d+(?:\.\d+)*', t):
        nums.add(m.group())
        dotted_spans.append(m.span())
    for m in re.finditer(r'\bc(\d+)\b', t):
        if not any(s <= m.start() < e for s, e in dotted_spans):
            nums.add(m.group(1))
    for m in re.finditer(r'\b(\d{1,3})\b', t):
        if not any(s <= m.start() < e for s, e in dotted_spans):
            nums.add(m.group(1))
    return nums


def log_feedback(query: str, rating: int,
                 bad_answer: Optional[str] = None,
                 correction: Optional[str] = None,
                 source_hint: Optional[str] = None):
    c = _conn()
    c.execute(
        "INSERT INTO feedback(query, bad_answer, correction, rating, source_hint) "
        "VALUES (?, ?, ?, ?, ?)",
        (query, bad_answer or '', correction or '', rating, source_hint or '')
    )
    c.commit()

    if correction and correction.strip() and rating < 0:
        pat = _normalize(query)
        try:
            c.execute("""
                INSERT INTO correction_memory(query_pat, correct_ans, raw_query, hit_count)
                VALUES (?, ?, ?, 0)
                ON CONFLICT(query_pat)
                DO UPDATE SET
                    correct_ans = excluded.correct_ans,
                    raw_query   = excluded.raw_query,
                    hit_count   = hit_count + 1
            """, (pat, correction.strip(), query))
            c.commit()
        except Exception as ex:
            import traceback
            print(f"[FEEDBACK] SAVE FAILED: {ex}")
            traceback.print_exc()
    c.close()


def get_direct_override(query: str) -> Optional[str]:
    c    = _conn()
    rows = c.execute(
        "SELECT query_pat, correct_ans, raw_query, hit_count "
        "FROM correction_memory ORDER BY hit_count DESC"
    ).fetchall()
    c.close()

    if not rows:
        return None

    current_pat   = _normalize(query)
    current_terms = _key_terms(query)
    current_words = set(current_pat.split())

    for row_pat, correct_ans, raw_query, hit_count in rows:
        if row_pat == current_pat:
            _bump_hit(row_pat)
            return correct_ans

    current_numbers = _extract_numbers(query)
    best_score = 0
    best_ans   = None
    best_pat   = None

    for row_pat, correct_ans, raw_query, hit_count in rows:
        row_terms   = _key_terms(raw_query or row_pat)
        row_words   = set(row_pat.split())
        row_numbers = _extract_numbers(raw_query or row_pat)

        if current_numbers and row_numbers and not (current_numbers & row_numbers):
            continue

        shared_terms = current_terms & row_terms
        shared_words = current_words & row_words

        if not shared_terms:
            continue

        coverage = len(shared_words) / max(len(row_words), len(current_words), 1)
        score    = len(shared_terms) * 3 + len(shared_words)

        if coverage >= 0.35 and score > best_score:
            best_score = score
            best_ans   = correct_ans
            best_pat   = row_pat

    if best_ans:
        _bump_hit(best_pat)
        return best_ans

    return None


def _bump_hit(pat: str):
    try:
        c = _conn()
        c.execute(
            "UPDATE correction_memory SET hit_count = hit_count + 1 WHERE query_pat = ?",
            (pat,)
        )
        c.commit()
        c.close()
    except Exception:
        pass


def get_relevant_corrections(query: str, top_k: int = 5) -> list:
    c    = _conn()
    rows = c.execute(
        "SELECT query_pat, correct_ans, hit_count FROM correction_memory "
        "ORDER BY hit_count DESC"
    ).fetchall()
    c.close()
    current_terms = _key_terms(query)
    results = []
    for pat, correct_ans, hits in rows:
        if current_terms & _key_terms(pat):
            results.append({'correct_ans': correct_ans, 'hit_count': hits})
    return results[:top_k]


def build_correction_context(query: str) -> str:
    corrections = get_relevant_corrections(query)
    if not corrections:
        return ''
    lines = ["### VERIFIED CORRECTIONS (user-confirmed — always use these)"]
    for cx in corrections:
        lines.append(f"- CORRECT ANSWER: {cx['correct_ans']}")
    return '\n'.join(lines)


def get_feedback_stats() -> dict:
    c    = _conn()
    good = c.execute("SELECT COUNT(*) FROM feedback WHERE rating > 0").fetchone()[0]
    bad  = c.execute("SELECT COUNT(*) FROM feedback WHERE rating < 0").fetchone()[0]
    mems = c.execute("SELECT COUNT(*) FROM correction_memory").fetchone()[0]
    c.close()
    return {'good': good, 'bad': bad, 'corrections_stored': mems}


def get_all_corrections_debug() -> list:
    c    = _conn()
    rows = c.execute(
        "SELECT id, query_pat, raw_query, correct_ans, hit_count, created_at "
        "FROM correction_memory ORDER BY created_at DESC"
    ).fetchall()
    c.close()
    return [
        {
            'id':          r[0],
            'query_pat':   r[1],
            'raw_query':   r[2],
            'correct_ans': r[3][:200],
            'hit_count':   r[4],
            'created_at':  r[5],
        }
        for r in rows
    ]
