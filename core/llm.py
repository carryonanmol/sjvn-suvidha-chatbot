"""
LLM interface — Ollama-only, 100% free, lifetime use.
Optimized for qwen2.5:3b (High Speed, True Vector Multi-Doc RAG, Few-Shot Memory)

v5 Fixes:
  - MAX_CTX_CHARS was set to 000 (zero). Fixed to 12000.
  - Hindi/Hinglish answers now properly translated into Devanagari via a dedicated
    post-processing step instead of relying on the LLM to do it inline.
  - Contextual memory window expanded from 3 to 6 turns for better follow-up accuracy.
  - Clause links now include PDF page anchor (#page=N) for direct navigation.
  - vector_doc_router threshold lowered from 0.5 to 0.4 to include more relevant docs.
  - Added is_hindi() helper used by stream_chat to trigger post-translation.
  - Greeting detection expanded to cover more common phrases.
"""

import re, json, os, urllib.request, urllib.error
from typing import Optional, Generator, Tuple

from core.store    import search, get_clause_text
from core.feedback import build_correction_context, get_direct_override

# ── Config ─────────────────────────────────────────────────────────────────────
OLLAMA_BASE   = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
OLLAMA_GEN    = f"{OLLAMA_BASE}/api/generate"
OLLAMA_TAGS   = f"{OLLAMA_BASE}/api/tags"

MODEL_PRIORITY = ['qwen2.5:3b', 'qwen2.5:7b', 'llama3.2', 'phi3']

MAX_CHUNKS    = 12        # slightly fewer chunks → tighter, more relevant context
MAX_CTX_CHARS = 10000     # fits comfortably inside num_ctx=8192 after prompt overhead
NUM_CTX       = 8192      # ← CRITICAL FIX: 3072 was too small; 12000-char context was silently truncated
TEMPERATURE   = 0.0       # zero creativity = no hallucination drift

_selected_model: Optional[str] = None

# ── System prompt: hard-blocking, not soft-suggesting ─────────────────────────
# Small models (3B) ignore polite instructions. Every rule must be stated as an
# absolute prohibition with an explicit fallback phrase to use when blocked.
SYSTEM_PROMPT = """You are SJVN Suvidha, the official HR Document Assistant for SJVN Limited.

ABSOLUTE RULES — NO EXCEPTIONS:
1. YOU HAVE NO KNOWLEDGE OF YOUR OWN. You are a document reader only.
2. EVERY fact, number, date, and policy detail you state MUST come word-for-word from the DOCUMENT CONTEXT in the prompt. If it is not in the context, it does not exist.
3. If the answer is not in the document context, you MUST write exactly: "⚠️ This information was not found in the provided documents."
4. DO NOT complete the answer from memory, training data, or general HR knowledge. STOP at the boundary of the documents.
5. NEVER say things like "typically", "usually", "in general", "standard practice", or "commonly" — these are signs you are hallucinating outside the documents.
6. Cite EVERY statement: [Source: filename.pdf | Page X]

LANGUAGE: Always reply in ENGLISH. A Hindi translation will be appended automatically below your answer.
SYNONYM MAPPING: "male employee" / "father" → look for Paternity Leave. "female employee" / "mother" → look for Maternity Leave."""

_ROMAN_MAP = {'i': 1, 'ii': 2, 'iii': 3, 'iv': 4, 'v': 5,
              'vi': 6, 'vii': 7, 'viii': 8, 'ix': 9, 'x': 10}

_GREETINGS = {
    'hi', 'hey', 'hello', 'namaste', 'namaskar', 'good morning',
    'good afternoon', 'good evening', 'hii', 'helo', 'sup',
    'नमस्ते', 'नमस्कार', 'हेलो',
}

# Hindi character detection — used to decide whether to run post-translation
_HINDI_RE = re.compile(r'[\u0900-\u097F]')
_HINGLISH_KEYWORDS = re.compile(
    r'\b(kya|hai|mujhe|meri|mera|batao|bata|do|dena|kitne|kab|kaise|'
    r'chahiye|hoga|leav|chutti|salari|niyam|policy|nahi|nai)\b', re.I
)


def is_hindi(text: str) -> bool:
    """Returns True if the query is in Hindi (Devanagari) or Hinglish (Roman Hindi)."""
    if _HINDI_RE.search(text):
        return True
    words = text.lower().split()
    hinglish_hits = sum(1 for w in words if _HINGLISH_KEYWORDS.search(w))
    return hinglish_hits >= 2


def _roman_to_int(s: str) -> Optional[int]:
    if not s: return None
    s = s.strip().lower()
    return _ROMAN_MAP.get(s, int(s) if s.isdigit() else None)


def get_available_models() -> list:
    try:
        req = urllib.request.Request(OLLAMA_TAGS, method='GET')
        with urllib.request.urlopen(req, timeout=5) as resp:
            return [m['name'] for m in json.loads(resp.read()).get('models', [])]
    except Exception:
        return []


def select_best_model() -> Optional[str]:
    global _selected_model
    if _selected_model: return _selected_model
    available = get_available_models()
    if not available: return None
    for preferred in MODEL_PRIORITY:
        if preferred in available:
            _selected_model = preferred
            return preferred
    _selected_model = available[0]
    return available[0]


def get_model_name() -> str:
    return os.environ.get('OLLAMA_MODEL') or select_best_model() or 'qwen2.5:3b'


def detect_clause(query: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    section_int = None
    sec_m = re.search(
        r'\bsection\s*[–\-]?\s*(i{1,3}v?|iv|vi{0,3}|vii|viii|ix|x|\d{1,2})\b',
        query, re.I
    )
    if sec_m:
        section_int = _roman_to_int(sec_m.group(1))

    doc_hint = None

    if m := re.search(r'\bclause\s+(\d+(?:\.\d+)?)', query, re.I):
        return m.group(1), section_int, doc_hint
    if m := re.search(r'(?:^|\s)[Cc](\d+(?:\.\d+)?)\b', query):
        return f"C{m.group(1)}", None, doc_hint
    if m := re.search(r'\b(\d{1,2}\.\d{1,3})\b', query):
        return m.group(1), section_int, doc_hint
    if m := re.search(r'\bpara\s+(\d+(?:\.\d+)?)', query, re.I):
        return m.group(1), section_int, doc_hint
    if section_int is not None:
        if m := re.search(r'\b(\d{1,3})\b(?!.*section)', query, re.I):
            if int(m.group(1)) not in range(1, 11):
                return m.group(1), section_int, doc_hint
    return None, None, None


def _resolve_clause(clause_id: str, section_int: Optional[int],
                    filename_filter: Optional[str]) -> Optional[str]:
    if clause_id.upper().startswith('C') and clause_id[1:].isdigit():
        return get_clause_text(clause_id, filename_filter) or \
               get_clause_text(clause_id, None)
    attempts = []
    if section_int is not None:
        attempts.extend([
            (f"S{section_int}:{clause_id}", filename_filter),
            (f"S{section_int}:{clause_id}", None),
        ])
    attempts.extend([(clause_id, filename_filter), (clause_id, None)])
    for key, ff in attempts:
        if text := get_clause_text(key, ff):
            return text
    return None


def _build_context(hits: list) -> str:
    ctx = ''
    seen = set()
    for h in hits:
        text = h['text'].strip()
        if text[:80] in seen:
            continue
        seen.add(text[:80])
        block = (
            f"[{h.get('filename', '?')} | Page {h.get('page', '?')}]\n"
            f"{text}\n\n"
        )
        if len(ctx) + len(block) > MAX_CTX_CHARS:
            break
        ctx += block
    return ctx.strip()


def _build_prompt(query: str, context: str, history: list = None) -> str:
    hist_str = ''
    if history:
        for turn in history[-6:]:
            role = turn.get('role', 'user').upper()
            hist_str += f"{role}: {turn['content'][:400]}\n"
        hist_str = f"### PRIOR CONVERSATION\n{hist_str}\n"

    correction_ctx = build_correction_context(query)
    correction_block = f"\n{correction_ctx}\n" if correction_ctx else ''

    # ── Sandwich pattern ──────────────────────────────────────────────────────
    # Small models need the constraint stated BEFORE the context (so they enter
    # reading mode) AND AFTER the context (so it's in their recency window when
    # they start generating).  The FORBIDDEN block makes refusal explicit rather
    # than relying on the model to self-censor.
    return (
        f"{hist_str}{correction_block}"
        # ── PRE-CONTEXT INSTRUCTION ──
        f"⛔ FORBIDDEN: Do not use any knowledge from your training data.\n"
        f"✅ ALLOWED: Only facts found verbatim in the DOCUMENT CONTEXT below.\n"
        f"If the answer is absent from the context, say: "
        f"\"⚠️ This information was not found in the provided documents.\"\n\n"
        # ── CONTEXT ──
        f"=== DOCUMENT CONTEXT START ===\n"
        f"{context}\n"
        f"=== DOCUMENT CONTEXT END ===\n\n"
        # ── POST-CONTEXT INSTRUCTION (recency anchor) ──
        f"REMINDER: Answer ONLY from the document context above. "
        f"Do NOT add facts from outside the documents.\n"
        f"- Format the answer clearly using bullet points, or a Markdown table if the user asks for one.\n"
        f"- Copy numbers and limits EXACTLY as written.\n"
        f"- Cite every bullet or table row using a clickable Markdown link exactly like this: [Source: filename.pdf (Page X)](/api/files/filename.pdf#page=X)\n"
        f"- If not in context → \"⚠️ This information was not found in the provided documents.\"\n\n"
        f"QUESTION: {query}\n\nANSWER:\n"
    )


def _verify_response(response: str, context: str) -> str:
    """
    Two checks:
    1. Number hallucination: figures in the response not present in the context.
    2. Citation missing: response has no [Source:...] citation at all, which means
       the model answered from training data rather than the documents.
    """
    extra = ''

    # Check 1: unverifiable numbers
    resp_nums = set(re.findall(
        r'\b\d[\d,]*(?:\.\d+)?\s*(?:days?|weeks?|months?|years?|crore|lakh|%)?',
        response, re.I
    ))
    ctx_lower = context.lower()
    suspicious = [
        num.strip() for num in resp_nums
        if re.search(r'\d[\d,]*(?:\.\d+)?', num).group() not in ctx_lower
    ]
    if suspicious:
        extra += (
            f"\n\n---\n⚠️ **Verification Notice:** The following figure(s) could not be "
            f"confirmed in the loaded document context and may be inaccurate: "
            f"**{', '.join(set(suspicious))}**."
        )

    # Check 2: no citations at all → model likely answered from training data
    has_citation = bool(re.search(r'Source:', response, re.I))
    is_not_found_msg = '⚠️ This information was not found' in response
    if not has_citation and not is_not_found_msg and len(response.strip()) > 80:
        extra += (
            "\n\n---\n⚠️ **Warning:** This answer contains no document citations. "
            "It may be based on general knowledge rather than your loaded documents. "
            "Please verify against the source PDFs."
        )

    return response + extra if extra else response


def _translate_to_hindi(english_text: str, model: str) -> str:
    """
    Post-process: translate an English answer into Hindi (Devanagari).
    Called only when the user's query was detected as Hindi/Hinglish.
    """
    # Strip citation footers before translating so they stay in English
    citation_footer_match = re.search(r'\n\n---\n\*\*🔗', english_text)
    main_text = english_text[:citation_footer_match.start()] if citation_footer_match else english_text
    footer    = english_text[citation_footer_match.start():] if citation_footer_match else ''

    # Simplified prompt: 3B models respond better to direct, strict translation commands
    prompt = (
        "You are a professional English to Hindi translator. "
        "Translate the following text into fluent Hindi (Devanagari script).\n\n"
        "RULES:\n"
        "1. DO NOT repeat the same sentence.\n"
        "2. Keep numbers and technical terms in English.\n"
        "3. Output ONLY the Hindi translation. No introductions.\n\n"
        f"ENGLISH TEXT:\n{main_text}\n\n"
        "HINDI TRANSLATION:\n"
    )

    url = f"{OLLAMA_BASE}/api/generate"
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1, 
            "num_ctx": 4096, 
            "num_predict": 1200,
            "top_p": 0.5,
            "repeat_penalty": 1.15  # <--- CRITICAL FIX: Stops the endless repeating loop
        }
    }).encode('utf-8')

    try:
        req = urllib.request.Request(
            url, data=payload, headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            translated = json.loads(resp.read()).get('response', '').strip()
            if translated:
                return translated + footer
    except Exception as e:
        print(f"[LLM] Hindi translation failed: {e}")

    return english_text  # fallback: return original if translation fails


def _stream_ollama(
    prompt: str, model: str, context: str = '', translate_hindi: bool = False
) -> Generator[str, None, None]:
    payload = json.dumps({
        'model':   model,
        'prompt':  prompt,
        'stream':  True,
        'system':  SYSTEM_PROMPT,
        'options': {
            'temperature': 0.0,    # zero = no hallucination drift
            'num_ctx':     8192,   # ← CRITICAL: must fit context(~10k chars) + prompt overhead
            'num_predict': 800,
            'top_p':       0.5,    # low → model stays close to retrieved text
            'repeat_penalty': 1.1, # discourages looping / padding
        }
    }).encode('utf-8')

    req = urllib.request.Request(
        OLLAMA_GEN, data=payload, headers={'Content-Type': 'application/json'}
    )
    full_response = ''
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            for raw_line in resp:
                if not (line := raw_line.decode('utf-8').strip()):
                    continue
                try:
                    obj = json.loads(line)
                    if token := obj.get('response', ''):
                        full_response += token
                        yield token
                    if obj.get('done', False):
                        break
                except json.JSONDecodeError:
                    continue

        # Verification notice (numbers check)
        if context:
            if extra := _verify_response(full_response, context)[len(full_response):]:
                yield extra
                full_response += extra

        # ── Hindi post-translation ──────────────────────────────────────────
        # ALWAYS append a Hindi translation below the English answer so the
        # user sees BOTH versions (English first, then Hindi).
        # We buffer the full English response first (done above via streaming),
        # then call the LLM to translate and append it.
        # Skip translation only for very short answers (greetings / not-found)
        # to avoid unnecessary latency.
        skip_translation = (
            len(full_response.strip()) < 60
            or '⚠️ This information was not found' in full_response
        )
        if not skip_translation:
            hindi = _translate_to_hindi(full_response, model)
            if hindi and hindi.strip() and hindi != full_response:
                yield f"\n\n---\n### 🇮🇳 हिंदी अनुवाद / Hindi Translation\n{hindi}"

    except Exception as e:
        yield f"\n\n⚠️ **Ollama error:** {e}"


# ── TRUE VECTOR ROUTING & MEMORY ───────────────────────────────────────────────

def rewrite_query_with_memory(query: str, history: list) -> str:
    """
    Uses structured JSON output from Qwen to produce a standalone search phrase
    from a follow-up question.  Falls back to the raw query on any failure.
    """
    if not history:
        return query

    # Use last 8 turns for better context
    last_turns = history[-8:]
    hist_text = "".join([
        f"{t.get('role', 'user').upper()}: {t.get('content', '')[:300]}\n"
        for t in last_turns
    ])

    prompt = (
        "You are a search query optimizer for an HR policy chatbot.\n"
        "Analyze the conversation and rewrite the follow-up question into a "
        "standalone English search phrase.\n"
        "CRITICAL: Convert pronouns and gender terms to policy names "
        "(e.g., 'male'→'paternity leave', 'she'→the topic from context).\n\n"
        f"CONVERSATION:\n{hist_text.strip()}\n"
        f"FOLLOW-UP: {query}\n\n"
        'Output ONLY a JSON object: {"search_query": "..."}'
    )

    url = f"{OLLAMA_BASE}/api/generate"
    payload = json.dumps({
        "model":   "qwen2.5:3b",
        "prompt":  prompt,
        "format":  "json",
        "stream":  False,
        "options": {"temperature": 0.0, "num_predict": 60, "num_ctx": 1500}
    }).encode('utf-8')

    try:
        req = urllib.request.Request(
            url, data=payload, headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read()).get('response', '{}')
            sq   = json.loads(data).get('search_query', '').strip()
            print(f"[MEMORY] Rewritten query: '{sq}'")
            return sq if sq and 5 < len(sq) < 150 else query
    except Exception as e:
        print(f"[LLM] Memory rewrite failed: {e}")
        return query


def vector_doc_router(standalone_query: str) -> str:
    """
    True Vector Routing: semantic cosine math to pick which document(s) to
    search, eliminating AI hallucination in routing.
    Secondary files included if their score ≥ 40% of the top file (was 50%).
    """
    hits = search(standalone_query, top_k=20, filename_filter="ALL")
    if not hits:
        return "ALL"

    file_scores: dict = {}
    for h in hits:
        fname = h['filename']
        file_scores[fname] = file_scores.get(fname, 0) + h['score']

    sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
    if not sorted_files:
        return "ALL"

    top_file, top_score = sorted_files[0]
    target_files = [top_file]

    for fname, score in sorted_files[1:]:
        # ← Lowered threshold: 0.4 instead of 0.5 to catch more cross-doc hits
        if score >= (top_score * 0.4):
            target_files.append(fname)

    return ','.join(target_files)


def _build_clause_footer(hits: list) -> str:
    seen_files: dict = {}  
    # hits are already sorted by relevance (highest score first)
    for h in hits:
        fn   = h.get('filename', '')
        page = h.get('page', 0)
        # Save the page of the most relevant hit for this file
        if fn and fn not in seen_files:
            seen_files[fn] = page 

    if not seen_files:
        return ''

    lines = ["\n\n---\n**🔗 Reference Documents:**"]
    for fn, best_page in seen_files.items():
        encoded = fn.replace(' ', '%20')
        if best_page:
            lines.append(f"- [{fn} (pg. {best_page})](/api/files/{encoded}#page={best_page})")
        else:
            lines.append(f"- [{fn}](/api/files/{encoded})")

    return '\n'.join(lines) + '\n'

def stream_chat(
    query: str,
    history: list = None,
    filename_filter: str = None,
    api_key: str = None,
    force_hindi: bool = False   # set True when UI Hindi-mode toggle is ON
) -> Generator[str, None, None]:

    # ── Greeting shortcut ──────────────────────────────────────────────────────
    if query.lower().strip() in _GREETINGS:
        msg = (
            "नमस्ते! (Hello!) I am **SJVN Suvidha**, your AI Document Assistant. "
            "How can I help you with company policies today?\n\n"
            "आप मुझसे कंपनी की नीतियों के बारे में हिंदी या अंग्रेज़ी में पूछ सकते हैं।"
        )
        for i in range(0, len(msg), 40):
            yield msg[i:i + 40]
        return

    # ── Direct correction override ─────────────────────────────────────────────
    if override := get_direct_override(query):
        notice = (
            f"✅ **Showing your saved correction for this question.**\n\n"
            f"{override}\n\n---\n"
            f"*This answer was corrected by you. If this is also wrong, use 👎 to update it.*"
        )
        for i in range(0, len(notice), 80):
            yield notice[i:i + 80]
        return

    print(f"\n[LLM] 🗣️  Raw query   : {query}")

    # ── Language detection (done once, used for post-translation) ─────────────
    query_is_hindi = force_hindi or is_hindi(query)

    # ── Contextual memory rewrite ─────────────────────────────────────────────
    has_history = (
        history and isinstance(history, list)
        and any(isinstance(t, dict) and t.get('content') for t in history)
    )
    standalone_query = rewrite_query_with_memory(query, history) if has_history else query
    print(f"[LLM] 🧠  Standalone  : {standalone_query}")

    # ── Vector document routing ───────────────────────────────────────────────
    effective_file = filename_filter
    if not effective_file:
        effective_file = vector_doc_router(standalone_query)

    if effective_file and effective_file != "ALL":
        print(f"[LLM] 🔀  Routing to  : {effective_file}")

    # ── Exact clause resolution ───────────────────────────────────────────────
    clause_id, section_int, doc_hint = detect_clause(standalone_query)
    final_file = effective_file or doc_hint

    if clause_id:
        clause_text = (
            _resolve_clause(clause_id, section_int, final_file)
            or _resolve_clause(clause_id, section_int, None)
        )
        if clause_text:
            for i in range(0, len(clause_text), 80):
                yield clause_text[i:i + 80]
            return

    # ── Vector search ──────────────────────────────────────────────────────────
    is_broad = any(
        t in standalone_query.lower()
        for t in ['summarize', 'summary', 'overview', 'all types', 'list everything']
    )
    top_k = 8 if is_broad else MAX_CHUNKS
    hits  = search(standalone_query, top_k=top_k, filename_filter=final_file)

    if not hits:
        msg = (
            "⚠️ No relevant content was found in the loaded documents for this query.\n\n"
            "This means the answer is **not available** in the documents currently indexed. "
            "Please try:\n"
            "- Rephrasing with different keywords\n"
            "- Selecting a specific document from the sidebar\n"
            "- Checking if the correct PDF has been uploaded"
        )
        yield msg
        return

    context = _build_context(hits)

    # ── Debug: print context window ────────────────────────────────────────────
    print("\n======= 📄 CONTEXT WINDOW =======")
    print(context[:1500] if context.strip() else "⚠️ EMPTY CONTEXT")
    if len(context) > 1500:
        print("... [truncated] ...")
    print("=================================\n")

    # ── Build prompt & stream answer ──────────────────────────────────────────
    # Use raw query (not standalone_query) so the LLM sees the user's actual words
    prompt = _build_prompt(query, context, history)
    model  = get_model_name()

    yield from _stream_ollama(
        prompt, model, context, translate_hindi=query_is_hindi
    )

    # ── Deep-link reference footer ────────────────────────────────────────────
    footer = _build_clause_footer(hits)
    if footer:
        for i in range(0, len(footer), 40):
            yield footer[i:i + 40]


def get_model_info() -> dict:
    return {
        'backend':        'Ollama (local, free)',
        'selected_model': get_model_name(),
        'available':      get_available_models(),
        'ollama_host':    OLLAMA_BASE,
    }