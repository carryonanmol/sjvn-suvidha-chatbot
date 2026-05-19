"""
Ingestor — Parses PDFs into searchable chunks and a precise clause map.

v4 (Section-III Table Fix):
  - Section III (and all numbered sections) clauses are extracted from TABLES,
    not just regex on plain text. Each table row with a clause number becomes
    its own clause_map entry, including ALL officer/power sub-rows.
  - Clause keys stored as: "S3:23", "S3:28" etc. AND bare "23", "28" so both
    "clause 23 section 3 dop" and bare "clause 23 dop" resolve correctly.
  - C-style guideline clauses (C23, C28 in Guidelines block) stored only under
    C-prefixed keys so they NEVER collide with Section-III row clauses.
  - Table rows are assembled into rich Markdown tables showing Subject | Officer | Power.
  - Synthetic summary chunks for HR policy PDFs unchanged.
"""

"""
Ingestor — Parses PDFs into searchable chunks and a precise clause map.

v4.1: Upgraded _build_chunks to use Semantic Chunking (paragraph-based).
"""

import os, re, hashlib
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    raise ImportError("pip install pdfplumber")

ALLOWED = {'.pdf', '.docx', '.txt'}

# Note: CHUNK_SIZE/CHUNK_OVERLAP are less strict now since we use semantic splitting,
# but remain as fallbacks for the synthetic HR summaries if needed.
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 60

ROMAN_TO_INT = {'I':1,'II':2,'III':3,'IV':4,'V':5,'VI':6,'VII':7,'VIII':8,'IX':9,'X':10}
INT_TO_ROMAN = {v:k for k,v in ROMAN_TO_INT.items()}


def file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(65536), b''):
            h.update(blk)
    return h.hexdigest()


def ingest(path: str) -> dict:
    ext = Path(path).suffix.lower()
    if ext == '.pdf':
        pages, clause_map = _parse_pdf(path)
    elif ext == '.txt':
        pages, clause_map = _parse_txt(path)
    else:
        try:
            with open(path, 'r', errors='replace') as f:
                content = f.read()
            pages = [{'page': 1, 'text': content, 'type': 'text'}]
            clause_map = {}
            _extract_clauses_from_text(content, 1, clause_map, '', section_num=0)
        except Exception:
            pages, clause_map = [], {}

    fname  = os.path.basename(path)
    chunks = _build_chunks(pages, fname)
    return {'chunks': chunks, 'file_hash': file_hash(path), 'clause_map': clause_map}


def _build_page_section_map(pdf) -> dict:
    result = {}
    current_section = 0
    for i, page in enumerate(pdf.pages):
        text = page.extract_text() or ''
        m = re.search(
            r'(?:^|\n)\s*Section\s*[–\-]?\s*(I{1,3}V?|IV|VI{0,3}|VII|VIII|IX|X)\b',
            text, re.I | re.MULTILINE
        )
        if m:
            s = m.group(1).strip().upper()
            current_section = ROMAN_TO_INT.get(s, current_section)
        result[i + 1] = current_section
    return result


def _is_outer_border_table(found_table, page_width: float, page_height: float) -> bool:
    bbox    = found_table.bbox
    t_w     = bbox[2] - bbox[0]
    t_h     = bbox[3] - bbox[1]
    n_cells = len(found_table.cells)
    return (t_w > page_width * 0.85 and t_h > page_height * 0.85 and n_cells <= 4)


def _is_dop_section_table(table: list) -> bool:
    if not table:
        return False

    header = [str(c or '').strip().lower() for c in table[0]]
    header_text = ' '.join(header)
    if ('subject matter' in header_text or 'delegation' in header_text) and \
       ('extent' in header_text or 'power' in header_text or 'officer' in header_text):
        return True

    if not table or len(table[0]) < 3:
        return False

    num_cols = len(table[0])
    if num_cols not in (3, 4):
        return False

    has_clause_row = False
    for row in table[:6]:
        col0 = str(row[0] or '').strip()
        if re.match(r'^\d+\.?\d*\s*\.?\s*$', col0) or re.match(r'^\d+\.\n?\(', col0):
            has_clause_row = True
            break
        last = str(row[-1] or '').strip() if row else ''
        if col0 == '' or col0 is None:
            if re.search(r'(Director|CMD|ED|CGM|GM|DGM|Manager|Head|HOP|Full Power)', last, re.I):
                has_clause_row = True
                break

    return has_clause_row


def _forward_fill(table: list) -> list:
    if not table:
        return table
    num_cols = max(len(row) for row in table)
    norm = []
    for row in table:
        padded = list(row) + [None] * (num_cols - len(row))
        norm.append([(c or '').strip().replace('\n', ' ') for c in padded])

    last = [''] * num_cols
    result = []
    for row_idx, cells in enumerate(norm):
        filled = []
        for col_idx, cell in enumerate(cells):
            if cell:
                last[col_idx] = cell
                filled.append(cell)
            elif row_idx > 0 and last[col_idx]:
                filled.append(last[col_idx])
            else:
                filled.append('')
        result.append(filled)
    return result


def _extract_section_clauses_from_table(table: list, page_num: int,
                                         clause_map: dict, fname: str,
                                         section_num: int):
    sec_prefix = f"S{section_num}:" if section_num > 0 else ""
    filled = _forward_fill(table)
    if not filled:
        return

    no_col = subj_col = off_col = pow_col = -1
    header_text = ' '.join(str(c or '') for c in filled[0]).lower()
    is_real_header = ('subject matter' in header_text or 'officers to whom' in header_text)

    if is_real_header:
        for i, h in enumerate(filled[0]):
            hl = h.lower()
            if re.search(r'\bno\.?\b', hl) and no_col < 0:
                no_col = i
            if 'subject matter' in hl and subj_col < 0:
                subj_col = i
            if 'officer' in hl and off_col < 0:
                off_col = i
            if 'extent' in hl:
                pow_col = i

    if subj_col < 0:
        ncols = len(filled[0])
        if ncols >= 4:
            no_col, subj_col, off_col, pow_col = 0, 1, 2, 3
        elif ncols == 3:
            no_col, subj_col, off_col, pow_col = 0, 1, 2, 2
        else:
            return

    data_start = 1 if is_real_header else 0

    current_no    = None
    current_subj  = None
    current_rows  = [] 

    def _flush(cno, csubj, crows, cpagenum):
        if not cno or not crows:
            return
        lines = [
            f"## Clause {cno} — {csubj}  [Section {INT_TO_ROMAN.get(section_num, section_num)}]\n",
            "| Officers to whom Powers Delegated | Extent of Power |",
            "|---|---|",
        ]
        for officer, power in crows:
            officer_clean = officer.replace('\n', ' ').strip()
            power_clean   = power.replace('\n', ' ').strip()
            lines.append(f"| {officer_clean} | {power_clean} |")

        snippet = '\n'.join(lines)

        for key_variant in _clause_key_variants(cno, sec_prefix):
            existing = clause_map.get(key_variant)
            if not existing or len(snippet) > len(existing.get('text', '')):
                clause_map[key_variant] = {'text': snippet, 'page': cpagenum,
                                            'section': section_num}

    for row in filled[data_start:]: 
        no_val   = row[no_col].strip()   if no_col   >= 0 else ''
        subj_val = row[subj_col].strip() if subj_col >= 0 else ''
        off_val  = row[off_col].strip()  if off_col  >= 0 else ''
        pow_val  = row[pow_col].strip()  if pow_col  >= 0 else ''

        if no_col >= 0 and subj_col >= 0 and off_col >= 0 and pow_col >= 0:
            non_empty = sum(1 for v in row if v.strip())
            if non_empty == 1 and row[no_col].strip():
                if current_no:
                    current_rows.append(('_Remarks_', row[no_col].strip()))
                continue

        m_no = re.match(r'^(\d+(?:\.\d+)?)\s*\.?\s*(?:\(.*\))?$', no_val)
        if m_no:
            cn = m_no.group(1).rstrip('.')
            if cn != current_no: 
                _flush(current_no, current_subj, current_rows, page_num)
                current_no   = cn
                current_subj = subj_val or ''
                current_rows = []
            else:
                if subj_val and not current_subj:
                    current_subj = subj_val
            if off_val or pow_val:
                current_rows.append((off_val, pow_val))
        elif current_no:
            if subj_val and not current_subj:
                current_subj = subj_val
            if off_val or pow_val:
                current_rows.append((off_val, pow_val))

    _flush(current_no, current_subj, current_rows, page_num)


def _clause_key_variants(clause_no: str, sec_prefix: str) -> list:
    variants = set()
    if sec_prefix:
        variants.add(f"{sec_prefix}{clause_no}")
        parts = clause_no.split('.')
        if parts[0].isdigit():
            norm = str(int(parts[0]))
            if len(parts) > 1:
                norm += '.' + '.'.join(parts[1:])
            variants.add(f"{sec_prefix}{norm}")

    variants.add(clause_no)
    parts = clause_no.split('.')
    if parts[0].isdigit():
        norm = str(int(parts[0]))
        if len(parts) > 1:
            norm += '.' + '.'.join(parts[1:])
        variants.add(norm)

    return list(variants)


def _table_to_markdown(table: list) -> str:
    """
    Convert a pdfplumber table (list of rows) to proper GFM Markdown table format.
    Outputs pipe-delimited rows with a separator line after the header so that
    marked.js / any Markdown renderer renders it as an HTML <table>.
    """
    if not table:
        return ''
    num_cols = max(len(row) for row in table)
    norm = []
    for row in table:
        padded = list(row) + [None] * (num_cols - len(row))
        # Clean: strip whitespace, collapse internal newlines to space, escape pipes
        cells = [
            (c or '').strip().replace('\n', ' ').replace('|', '\\|')
            for c in padded
        ]
        norm.append(cells)

    if not norm:
        return ''

    # Find the first non-empty row to use as the header
    header_idx = 0
    for i, row in enumerate(norm):
        if any(c for c in row):
            header_idx = i
            break

    # Forward-fill empty cells in body rows (merged cells in PDFs)
    last_value = [''] * num_cols
    output_rows = []

    for row_idx, cells in enumerate(norm):
        is_header = (row_idx <= header_idx)
        filled = []
        for col_idx, cell in enumerate(cells):
            if cell:
                last_value[col_idx] = cell
                filled.append(cell)
            elif not is_header and last_value[col_idx]:
                # Forward-fill only in body, not header
                filled.append(last_value[col_idx])
            else:
                filled.append('')
        # Skip rows that are entirely empty
        if not any(c.strip() for c in filled):
            continue
        output_rows.append((is_header, filled))

    if not output_rows:
        return ''

    lines = []
    header_emitted = False

    for is_header, cells in output_rows:
        row_md = '| ' + ' | '.join(cells) + ' |'
        lines.append(row_md)
        if is_header and not header_emitted:
            # GFM separator row required for proper table rendering
            sep = '| ' + ' | '.join('---' for _ in cells) + ' |'
            lines.append(sep)
            header_emitted = True

    # If we never had a header row, insert a dummy separator after first row
    if not header_emitted and lines:
        sep = '| ' + ' | '.join('---' for _ in range(num_cols)) + ' |'
        lines.insert(1, sep)

    return '\n'.join(lines)


def _parse_pdf(path: str):
    pages      = []
    clause_map = {}

    with pdfplumber.open(path) as pdf:
        total = len(pdf.pages)
        raw_texts = []

        page_section = _build_page_section_map(pdf)

        for pnum in range(total):
            page    = pdf.pages[pnum]
            body    = (page.extract_text() or '').strip()
            sec_num = page_section.get(pnum + 1, 0)
            raw_texts.append((body, sec_num))

            try:
                found_tables = page.find_tables() or []
            except Exception:
                found_tables = []

            page_w = page.width
            page_h = page.height

            for found_tbl in found_tables:
                if _is_outer_border_table(found_tbl, page_w, page_h):
                    continue
                try:
                    tbl = found_tbl.extract()
                except Exception:
                    continue
                if not tbl:
                    continue

                if sec_num > 0 and _is_dop_section_table(tbl):
                    _extract_section_clauses_from_table(
                        tbl, pnum + 1, clause_map,
                        os.path.basename(path), sec_num
                    )

                tt = _table_to_markdown(tbl)
                if tt.strip():
                    pages.append({
                        'page':    pnum + 1,
                        'text':    tt,
                        'type':    'table',
                        'section': sec_num,
                    })

            if body:
                pages.append({
                    'page':    pnum + 1,
                    'text':    body,
                    'type':    'text',
                    'section': sec_num,
                })

        for i in range(total):
            _, sec_num = raw_texts[i]
            if sec_num != 0:
                continue   
            window = raw_texts[i:min(i + 3, total)]
            combined = '\n'.join(t for t, _ in window)
            combined = re.sub(r'\nPage \d+ of \d+\n?', '\n', combined)
            combined = re.sub(r'Guidelines for exercising the DoP\n?', '', combined)
            _extract_guideline_clauses(combined, i + 1, clause_map,
                                        os.path.basename(path))

        full_doc_text = '\n'.join(t for t, _ in raw_texts)
        synthetic = _build_policy_summary_chunks(full_doc_text, os.path.basename(path))
        pages.extend(synthetic)

    return pages, clause_map


def _extract_guideline_clauses(text: str, page_num: int, clause_map: dict, fname: str):
    c_positions = list(re.finditer(r'(?:^|\n)(C(\d+)\.)', text, re.MULTILINE))
    for idx, m in enumerate(c_positions):
        c_num = m.group(2)
        start = m.start()
        if text[start] == '\n':
            start += 1
        end     = c_positions[idx + 1].start() if idx + 1 < len(c_positions) else len(text)
        snippet = text[start:end].strip()
        snippet = re.sub(r'\nPage \d+ of \d+\n?', '\n', snippet).strip()
        if not snippet:
            continue

        for key in [f"C{c_num}", f"c{c_num}"]:
            existing = clause_map.get(key)
            if not existing or len(snippet) > len(existing.get('text', '')):
                clause_map[key] = {'text': snippet, 'page': page_num, 'section': 0}


def _extract_clauses_from_text(text: str, page_num: int, clause_map: dict,
                                fname: str, section_num: int = 0):
    if not text:
        return

    sec_prefix = f"S{section_num}:" if section_num > 0 else ""

    sub_iter = re.finditer(
        r'^(\d{1,3}\.\d{1,3}(?:\.\d{1,3})?)\s+([\w₹\(\[].+)',
        text, re.MULTILINE
    )
    for m in sub_iter:
        cn    = m.group(1)
        start = m.start()
        next_m = re.search(r'\n(?:C\d+\.|\d{1,3}\.\d)', text[start + 1:])
        end     = start + 1 + next_m.start() if next_m else len(text)
        snippet = text[start:end].strip()
        snippet = re.sub(r'\nPage \d+ of \d+\n?', '\n', snippet).strip()
        if not snippet:
            continue

        parts = cn.split('.')
        alt   = f"{int(parts[0])}.{parts[1]}" if parts[0].isdigit() else cn

        for key in [cn, alt]:
            if key not in clause_map:
                clause_map[key] = {'text': snippet, 'page': page_num}

        if sec_prefix:
            for key in [f"{sec_prefix}{cn}", f"{sec_prefix}{alt}"]:
                existing = clause_map.get(key)
                if not existing or len(snippet) > len(existing.get('text', '')):
                    clause_map[key] = {'text': snippet, 'page': page_num,
                                       'section': section_num}


def _build_policy_summary_chunks(full_text: str, filename: str) -> list:
    if "Capital Sum Assured" not in full_text and "Insurance Cover" not in full_text:
        return []

    chunks = []

    csa_match = re.search(
        r"4\.6\s+Capital Sum Assured.*?(?=\n5\.0|\n6\.0|\Z)",
        full_text, re.DOTALL | re.IGNORECASE
    )
    csa_text = csa_match.group(0).strip() if csa_match else ""

    benefits_match = re.search(
        r"5\.0\s+Benefits.*?(?=\n6\.0|\n7\.0|\Z)",
        full_text, re.DOTALL | re.IGNORECASE
    )
    benefits_text = benefits_match.group(0).strip() if benefits_match else ""

    ann1_match = re.search(
        r"ANNEXURE[\s-]*I\s*\n.*?(?=ANNEXURE[\s-]*II|\Z)",
        full_text, re.DOTALL | re.IGNORECASE
    )
    ann1_text = ann1_match.group(0).strip() if ann1_match else ""

    gis_match = re.search(
        r"GROUP INSURANCE SCHEME\n.*?(?=3\.0\s+The procedure|\Z)",
        full_text, re.DOTALL | re.IGNORECASE
    )
    gis_text = gis_match.group(0).strip() if gis_match else ""

    if benefits_text:
        combined_parts = []
        if csa_text:
            combined_parts.append(
                "=== CAPITAL SUM ASSURED: Insurance Cover Amount Per Employee Level ===\n"
                "The compensation/amount payable to the nominee or employee under GPAIS "
                "is calculated as a percentage of the Capital Sum Assured (CSA). "
                "The CSA for each category is:\n" + csa_text
            )
        combined_parts.append(
            "=== GPAIS BENEFITS SUMMARY: Compensation Amount Payable to Nominee or Employee ===\n"
            "GPAIS Group Personal Accident Insurance Scheme pays the following amounts:\n"
            "- On DEATH: nominee receives 100% of Capital Sum Assured (minimum Rs.15 lacs)\n"
            "- On PERMANENT DISABLEMENT: employee receives percentage of Capital Sum Assured as per Annexure-I/II\n"
            "- On TEMPORARY DISABLEMENT: employee receives 1% of Capital Sum Assured per week (max Rs.10,000/week, max 104 weeks)\n"
            "Full details:\n" + benefits_text
        )
        if ann1_text:
            combined_parts.append("=== PERMANENT DISABLEMENT COMPENSATION ===\n" + ann1_text)
        
        # Applying semantic splitting to synthetic summaries as well
        combined = "\n\n".join(combined_parts)
        paragraphs = re.split(r'\n\s*\n', combined)
        for i, para in enumerate(paragraphs):
            if len(para.strip()) > 10:
                chunks.append({"filename": filename, "text": para.strip(), "page": 0, "type": "summary", "section": 0})

    if gis_text and not benefits_text:
        paragraphs = re.split(r'\n\s*\n', gis_text)
        for i, para in enumerate(paragraphs):
            if len(para.strip()) > 10:
                chunks.append({"filename": filename, "text": para.strip(), "page": 0, "type": "summary", "section": 0})

    return chunks


def _parse_txt(path: str):
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    pages      = [{'page': 1, 'text': content, 'type': 'text', 'section': 0}]
    clause_map = {}
    _extract_clauses_from_text(content, 1, clause_map, os.path.basename(path), section_num=0)
    return pages, clause_map


def _build_chunks(pages: list, filename: str) -> list:
    """
    Reverted to fixed-size overlapping chunks. 
    This handles messy PDF extraction much more reliably than strict paragraph splitting,
    ensuring the LLM context window doesn't get overflowed by missing line breaks.
    """
    chunks      = []
    chunk_index = 0

    for page_info in pages:
        text = page_info['text'].strip()
        if not text:
            continue
        
        words = text.split()
        start = 0

        while start < len(words):
            end   = min(start + CHUNK_SIZE, len(words))
            chunk = ' '.join(words[start:end])
            
            chunks.append({
                'id':          f"{filename}_p{page_info['page']}_c{chunk_index}",
                'filename':    filename,
                'text':        chunk,
                'page':        page_info['page'],
                'chunk_index': chunk_index,
                'type':        page_info['type'],
                'section':     page_info.get('section', 0),
            })
            
            chunk_index += 1
            if end == len(words):
                break
            
            # The overlap is critical for PDF tables/clauses that span across chunks
            start = end - CHUNK_OVERLAP

    return chunks