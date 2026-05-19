"""
SJVN Suvidha — Document Chatbot (Ollama Edition)
100% free, runs locally, no API keys needed.

v6: Hindi translation is appended below the English answer (both versions shown).
    Also exposes /api/files/<filename>#page=N passthrough for deep PDF links.
"""

import os
import sys
import json
import time
import threading
import traceback
from flask import (
    Flask, request, jsonify, Response, render_template,
    stream_with_context, send_from_directory, session,
    redirect, url_for, render_template_string
)

from core.ingestor import ingest, ALLOWED, file_hash
from core.store    import (
    init_db, index_chunks, list_files, remove_file,
    get_file_hash, get_conn, get_all_faqs, add_faq, delete_faq
)
from core.llm      import stream_chat, get_model_info, get_available_models, select_best_model
from core.feedback import (
    log_feedback, get_feedback_stats, init_feedback_tables, get_all_corrections_debug
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, 'documents')
PORT     = int(os.environ.get('PORT', 5000))
WATCH_INTERVAL = 10

os.makedirs(DOCS_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key    = os.environ.get('FLASK_SECRET_KEY', 'super_secure_fallback_key_change_this')
ADMIN_PASSWORD    = os.environ.get('ADMIN_PASSWORD', '1245678')

init_db()
init_feedback_tables()

_index_lock = threading.Lock()

# ── Audit Logging ──────────────────────────────────────────────────────────────

@app.before_request
def log_request_info():
    if request.path == '/api/chat' and request.method == 'POST':
        try:
            body  = request.get_json(force=True, silent=True) or {}
            query = body.get('query', '').strip()
            if query:
                c = get_conn()
                c.execute(
                    "INSERT INTO audit_logs (ip_address, query) VALUES (?, ?)",
                    (request.remote_addr, query)
                )
                c.commit()
                c.close()
        except Exception as e:
            print(f"[AUDIT] Failed: {e}")


# ── Indexing & File Watcher ────────────────────────────────────────────────────

def _index_one(fname: str, path: str, force: bool = False) -> bool:
    try:
        fh = file_hash(path)
        if not force and get_file_hash(fname) == fh:
            return False
        print(f"[INDEXER] ⟳  Indexing: {fname}")
        data = ingest(path)
        index_chunks(data['chunks'], path, data['file_hash'], data.get('clause_map'))
        print(f"[INDEXER] ✓  {fname} — {len(data['chunks'])} chunks, "
              f"{len(data.get('clause_map', {}))} clauses")
        return True
    except Exception as e:
        print(f"[INDEXER] ✗  {fname} — {e}")
        traceback.print_exc()
        return False


def index_documents(force: bool = False):
    files = sorted(
        f for f in os.listdir(DOCS_DIR)
        if os.path.splitext(f)[1].lower() in ALLOWED
    )
    if not files:
        print(f"\n[INDEXER] No documents in '{DOCS_DIR}/' — add your PDFs.\n")
        return
    print(f"\n[INDEXER] Scanning {len(files)} file(s) in documents/ ...")
    changed = 0
    with _index_lock:
        for fname in files:
            if _index_one(fname, os.path.join(DOCS_DIR, fname), force):
                changed += 1
        indexed = {f['filename'] for f in list_files()}
        on_disk = set(files)
        for stale in indexed - on_disk:
            remove_file(stale)
            print(f"[INDEXER] Removed stale entry: {stale}")
    total = len(list_files())
    if changed:
        print(f"[INDEXER] Done — {changed} file(s) (re)indexed. {total} ready.\n")
    else:
        print(f"[INDEXER] Done — all {total} file(s) already up-to-date.\n")


def _watcher_loop():
    print(f"[WATCHER] Watching documents/ every {WATCH_INTERVAL}s …")
    while True:
        time.sleep(WATCH_INTERVAL)
        try:
            disk_files = {
                f for f in os.listdir(DOCS_DIR)
                if os.path.splitext(f)[1].lower() in ALLOWED
            }
            db_files = {f['filename'] for f in list_files()}
            with _index_lock:
                for fname in sorted(disk_files):
                    path = os.path.join(DOCS_DIR, fname)
                    try:
                        if get_file_hash(fname) != file_hash(path):
                            print(f"[WATCHER] Changed: {fname}")
                            _index_one(fname, path, force=True)
                    except Exception as ex:
                        print(f"[WATCHER] Error checking {fname}: {ex}")
                for stale in db_files - disk_files:
                    remove_file(stale)
                    print(f"[WATCHER] Removed deleted file: {stale}")
        except Exception as ex:
            print(f"[WATCHER] Error: {ex}")


def start_watcher():
    threading.Thread(target=_watcher_loop, daemon=True).start()


# ── Admin Dashboard ────────────────────────────────────────────────────────────

@app.route('/admin', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        if request.form.get('password') == ADMIN_PASSWORD:
            session['is_admin'] = True
            if (
                request.headers.get('X-Requested-With') == 'XMLHttpRequest'
                or 'application/json' in request.headers.get('Accept', '')
            ):
                return jsonify({"status": "success"}), 200
            return redirect(url_for('admin_dashboard'))
        return "Unauthorized: Incorrect Password", 401
    return render_template_string("""
    <div style="max-width:400px;margin:100px auto;font-family:sans-serif;text-align:center;">
        <h2>Admin Authentication</h2>
        <form method="post" style="display:flex;flex-direction:column;gap:10px;">
            <input type="password" name="password" placeholder="Enter Admin Password"
                   required style="padding:10px;font-size:16px;">
            <button type="submit"
                    style="padding:10px;font-size:16px;background:#007bff;color:white;border:none;cursor:pointer;">
                Access Logs
            </button>
        </form>
    </div>""")


@app.route('/admin/dashboard')
def admin_dashboard():
    if not session.get('is_admin'):
        return redirect(url_for('admin_login'))
    c    = get_conn()
    logs = c.execute(
        "SELECT timestamp, ip_address, query FROM audit_logs ORDER BY timestamp DESC LIMIT 200"
    ).fetchall()
    c.close()
    return render_template_string("""
    <div style="padding:20px;font-family:sans-serif;background:#f4f4f9;min-height:100vh;">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <h2>🛡️ System Audit Logs</h2>
            <a href="{{ url_for('admin_logout') }}"
               style="padding:8px 16px;background:#dc3545;color:white;text-decoration:none;border-radius:4px;">
                Logout
            </a>
        </div>
        <table style="width:100%;border-collapse:collapse;background:white;
                      box-shadow:0 1px 3px rgba(0,0,0,0.2);">
            <tr style="background:#343a40;color:white;text-align:left;">
                <th style="padding:12px;border:1px solid #ddd;">Timestamp</th>
                <th style="padding:12px;border:1px solid #ddd;">IP Address</th>
                <th style="padding:12px;border:1px solid #ddd;">Query</th>
            </tr>
            {% for log in logs %}
            <tr style="border-bottom:1px solid #ddd;">
                <td style="padding:12px;">{{ log[0] }}</td>
                <td style="padding:12px;font-family:monospace;">{{ log[1] }}</td>
                <td style="padding:12px;">{{ log[2] }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>""", logs=logs)


@app.route('/admin/logout')
def admin_logout():
    session.pop('is_admin', None)
    return redirect(url_for('index'))


# ── API Routes ─────────────────────────────────────────────────────────────────

@app.route('/api/faqs', methods=['GET'])
def api_get_faqs():
    return jsonify(get_all_faqs())


@app.route('/api/admin/faqs', methods=['POST', 'DELETE'])
def api_admin_faqs():
    if not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 401
    body = request.get_json(force=True)
    if request.method == 'POST':
        q = (body.get('q') or '').strip()
        a = (body.get('a') or '').strip()
        if not q:
            return jsonify({'error': 'Question is required'}), 400
        if add_faq(q, a):
            return jsonify({'status': 'ok'})
        return jsonify({'error': 'FAQ already exists'}), 400
    if request.method == 'DELETE':
        faq_id = body.get('id')
        if faq_id:
            delete_faq(faq_id)
            return jsonify({'status': 'ok'})
        return jsonify({'error': 'ID required'}), 400


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/files')
def api_files():
    return jsonify(list_files())


@app.route('/api/files/<path:filename>')
def serve_file(filename):
    """
    Serves PDF files.  The #page=N anchor is handled entirely by the browser's
    PDF viewer — Flask just needs to return the file with the correct MIME type.
    """
    return send_from_directory(DOCS_DIR, filename, mimetype='application/pdf')


@app.route('/api/status')
def api_status():
    info  = get_model_info()
    files = list_files()
    stats = get_feedback_stats()
    return jsonify({
        'documents':          len(files),
        'files':              [f['filename'] for f in files],
        'llm_backend':        info['backend'],
        'model':              info['selected_model'],
        'models_avail':       info['available'],
        'ollama_host':        info['ollama_host'],
        'feedback_good':      stats['good'],
        'feedback_bad':       stats['bad'],
        'corrections_stored': stats['corrections_stored'],
        'voice_supported':    True,
    })


@app.route('/api/models')
def api_models():
    return jsonify({
        'available': get_available_models(),
        'selected':  get_model_info()['selected_model'],
    })


@app.route('/api/refresh', methods=['POST'])
def api_refresh():
    threading.Thread(target=index_documents, kwargs={'force': False}, daemon=True).start()
    return jsonify({'status': 'indexing started'})


@app.route('/api/chat', methods=['POST'])
def api_chat():
    body     = request.get_json(force=True)
    query    = body.get('query', '').strip()
    history  = body.get('history', [])
    filename = body.get('filename') or None
    force_hindi = bool(body.get('force_hindi', False))

    if not query:
        return jsonify({'error': 'Empty query'}), 400
    if filename in ('__all__', ''):
        filename = None

    def generate():
        try:
            for chunk in stream_chat(query, history=history,
                                     filename_filter=filename, api_key=None,
                                     force_hindi=force_hindi):
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps(f'Error: {e}')}\n\n"
        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    body        = request.get_json(force=True)
    query       = (body.get('query') or '').strip()
    rating      = int(body.get('rating', 0))
    bad_answer  = (body.get('bad_answer') or '').strip() or None
    correction  = (body.get('correction') or '').strip() or None
    source_hint = (body.get('source_hint') or '').strip() or None

    if not query:
        return jsonify({'error': 'query is required'}), 400
    if rating not in (-1, 0, 1):
        return jsonify({'error': 'rating must be -1, 0, or +1'}), 400

    log_feedback(query, rating, bad_answer, correction, source_hint)
    msg = 'Feedback recorded.' + (' Correction saved.' if correction and rating < 0 else '')
    return jsonify({'status': 'ok', 'message': msg})


@app.route('/api/feedback/stats')
def api_feedback_stats():
    return jsonify(get_feedback_stats())


@app.route('/api/feedback/debug')
def api_feedback_debug():
    return jsonify(get_all_corrections_debug())


if __name__ == '__main__':
    force  = '--reindex' in sys.argv
    banner = f"""
============================================================
  SJVN Suvidha — Document Assistant (Ollama Edition) v5
  http://127.0.0.1:{PORT}
============================================================"""
    print(banner)
    models = get_available_models()
    if models:
        print(f"  Ollama ✓   Model : {select_best_model()}")
    else:
        print("  ⚠  Ollama not running. Start with : ollama serve")
        print("  ⚠  Then pull              : ollama pull qwen2.5:3b")
        print("  ⚠  And                    : ollama pull nomic-embed-text")
    print("============================================================")
    print("  Features: Hindi translation · Deep-link clauses · Memory")
    print("  Security: Behavioral logging · Admin dashboard")
    print("============================================================\n")
    index_documents(force=force)
    start_watcher()
    app.run(host="0.0.0.0", debug=False, port=PORT, threaded=True)
