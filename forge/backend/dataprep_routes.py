"""
dataprep_routes.py
Add these imports and routes to your existing main.py.

IMPORTS TO ADD at top of main.py:
    import tiktoken
    from openai import OpenAI  # pip install openai

INSTALL:
    pip install openai tiktoken

Then paste the route functions below into main.py before the `if __name__ == "__main__":` line.
"""

# ─── DataPrep State ───────────────────────────────────────────────────────────
# Add this near the top of main.py alongside the `jobs: dict = {}` line:
#   dataprep_jobs: dict = {}

# ─── Pydantic models to add ───────────────────────────────────────────────────

import json
import logging
import uuid
import math
from datetime import datetime
from pathlib import Path
import re
from fastapi import BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional, List


logger = logging.getLogger("llm-finetune")
dataprep_jobs: dict = {}


def _sanitize_jsonable(value):
    """Recursively replace NaN/Inf with None for strict JSON compliance."""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: _sanitize_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_jsonable(v) for v in value]
    return value


def _normalize_key(k: str) -> str:
    """Normalize Excel column headers to clean snake_case keys."""
    import re
    k = str(k).strip()
    # Remove non-breaking spaces and other unicode whitespace
    k = k.replace('\xa0', ' ').replace('\u00a0', ' ')
    k = re.sub(r'[\s]+', ' ', k).strip()
    # Replace spaces and special chars with underscore
    k = re.sub(r'[^\w]+', '_', k.lower())
    k = re.sub(r'_+', '_', k).strip('_')
    return k


def _records_from_df(df):
    """Convert DataFrame rows to dicts with JSON-safe nulls and normalized keys."""
    import pandas as pd

    # Normalize column headers first
    df.columns = [_normalize_key(c) for c in df.columns]
    cleaned = df.astype(object).where(pd.notna(df), None)
    return _sanitize_jsonable(cleaned.to_dict(orient='records'))


class DataPrepConfig(BaseModel):
    job_id: str
    mode: str                        # "golden" | "gpt"
    doc_paths: List[str]             # uploaded doc paths (extracted text already saved)
    golden_path: Optional[str]       # path to Excel/JSON/JSONL/CSV ground truth (mode=golden)
    openai_key: Optional[str]        # for mode=gpt
    generation_instruction: Optional[str]  # freeform GPT instruction
    system_prompt: str = "You are a helpful assistant that extracts structured information from documents."
    chunk_size_chars: int = 6000     # ~1500 tokens
    chunk_overlap_chars: int = 800   # ~200 tokens overlap
    max_chunks_per_doc: int = 0      # 0 = unlimited
    max_seq_length: int = 2048       # warn threshold
    output_filename: str = "dataset"


# ─── Chunking helpers ─────────────────────────────────────────────────────────

def _chunk_document(text: str, filename: str, chunk_size: int, overlap: int, max_chunks: int) -> list[dict]:
    """
    Split a document into overlapping chunks.
    Strategy:
      1. Split on Azure DI page breaks (<!-- PageBreak -->)
      2. Within each page, split on section headers (## or \n\n)
      3. Merge small pages, split large ones
      4. Inject doc header into every chunk so model knows context
    Returns list of {chunk_text, page_range, chunk_index, total_chunks}
    """
    import re

    # Clean up excessive whitespace but preserve page breaks
    text = re.sub(r'\n{4,}', '\n\n\n', text)

    # Split on page breaks first
    page_break_pattern = re.compile(r'<!--\s*PageBreak\s*-->', re.IGNORECASE)
    pages = page_break_pattern.split(text)
    pages = [p.strip() for p in pages if p.strip()]

    if not pages:
        return []

    # Build chunks by merging/splitting pages to hit target chunk_size
    chunks = []
    current_chunk = []
    current_len = 0
    current_start_page = 1

    def flush_chunk(start_page, end_page, content):
        if not content.strip():
            return
        header = f"[Document: {filename} | Pages: {start_page}-{end_page}]\n\n"
        chunks.append({
            "chunk_text": header + content,
            "page_range": f"{start_page}-{end_page}",
            "chunk_index": len(chunks),
        })

    for page_idx, page_text in enumerate(pages):
        page_num = page_idx + 1

        # If this single page exceeds chunk_size, split it further
        if len(page_text) > chunk_size:
            # Flush any pending chunk first
            if current_chunk:
                flush_chunk(current_start_page, page_num - 1, "\n\n".join(current_chunk))
                current_chunk = []
                current_len = 0

            # Split large page into sub-chunks with overlap
            start = 0
            while start < len(page_text):
                end = start + chunk_size
                sub = page_text[start:end]

                # Try to break at a natural boundary (double newline or sentence)
                if end < len(page_text):
                    break_at = sub.rfind('\n\n')
                    if break_at > chunk_size * 0.6:
                        sub = page_text[start:start + break_at]
                        end = start + break_at

                header = f"[Document: {filename} | Page: {page_num} | Segment: {len(chunks)+1}]\n\n"
                chunks.append({
                    "chunk_text": header + sub.strip(),
                    "page_range": str(page_num),
                    "chunk_index": len(chunks),
                })
                start = end - overlap  # overlap
                if start <= 0:
                    break
            current_start_page = page_num + 1
        else:
            # Normal page — accumulate
            if current_len + len(page_text) > chunk_size and current_chunk:
                flush_chunk(current_start_page, page_num - 1, "\n\n".join(current_chunk))
                # Keep last bit for overlap
                overlap_text = current_chunk[-1] if current_chunk else ""
                current_chunk = [overlap_text, page_text] if overlap_text else [page_text]
                current_len = sum(len(c) for c in current_chunk)
                current_start_page = page_num
            else:
                current_chunk.append(page_text)
                current_len += len(page_text)
                if len(current_chunk) == 1:
                    current_start_page = page_num

    # Flush remaining
    if current_chunk:
        flush_chunk(current_start_page, len(pages), "\n\n".join(current_chunk))

    # Tag with total
    total = len(chunks)
    for c in chunks:
        c["total_chunks"] = total

    # Apply cap
    if max_chunks > 0:
        chunks = chunks[:max_chunks]

    return chunks


def _estimate_tokens(text: str) -> int:
    """Fast token estimate: ~4 chars per token (no tiktoken dependency required)."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4


def _build_chat_example(system_prompt: str, user_content: str, assistant_content: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


# ─── Golden Set joining logic ─────────────────────────────────────────────────

def _load_golden_records(golden_path: str) -> list[dict]:
    """Load ground truth from Excel, JSON, JSONL, or CSV."""
    import csv
    import pandas as pd
    from pathlib import Path as P

    def _read_text_with_fallback(path: P) -> str:
        encodings = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
        last_err = None
        for enc in encodings:
            try:
                return path.read_text(encoding=enc)
            except UnicodeDecodeError as e:
                last_err = e
        raise ValueError(
            f"Could not decode '{path.name}' with supported encodings {encodings}. "
            f"Last error: {last_err}"
        )

    p = P(golden_path)
    suffix = p.suffix.lower()
    raw_head = p.read_bytes()[:8]

    # Some uploads are mislabeled (e.g., .csv file that is actually .xlsx).
    is_zip_xlsx = raw_head.startswith(b"PK\x03\x04")
    is_ole_xls = raw_head.startswith(b"\xD0\xCF\x11\xE0")

    if suffix == ".csv" and (is_zip_xlsx or is_ole_xls):
        logger.warning(
            "Golden set '%s' has CSV extension but Excel file signature; loading via read_excel.",
            p.name,
        )
        df = pd.read_excel(p, dtype=str)
        return _records_from_df(df)

    if suffix in ('.xlsx', '.xls'):
        df = pd.read_excel(p, dtype=str)
        return _records_from_df(df)
    elif suffix == '.csv':
        encodings = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
        parser_errors = []
        for enc in encodings:
            try:
                # Read a sample to sniff delimiter; if sniff fails fallback to pandas auto.
                sample = p.read_text(encoding=enc, errors="replace")[:8000]
                detected_sep = None
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
                    detected_sep = dialect.delimiter
                except Exception:
                    detected_sep = None

                seps_to_try = [detected_sep] if detected_sep else [None, ",", ";", "\t", "|"]
                for sep in seps_to_try:
                    try:
                        df = pd.read_csv(
                            p,
                            dtype=str,
                            encoding=enc,
                            sep=sep,
                            engine="python",
                        )
                        return _records_from_df(df)
                    except pd.errors.ParserError as pe:
                        parser_errors.append(f"encoding={enc}, sep={sep!r}: {pe}")
                        continue
            except UnicodeDecodeError:
                continue

        # Last resort: keep job moving while surfacing malformed rows.
        for enc in encodings:
            try:
                logger.warning(
                    "Golden CSV appears malformed; retrying with on_bad_lines='skip'. File: %s",
                    p.name,
                )
                df = pd.read_csv(
                    p,
                    dtype=str,
                    encoding=enc,
                    sep=None,
                    engine="python",
                    on_bad_lines="skip",
                )
                return _records_from_df(df)
            except Exception:
                continue
        raise ValueError(
            f"Could not decode CSV golden set '{p.name}'. "
            f"Tried encodings: {encodings}. Parser errors: {' | '.join(parser_errors[-3:])}"
        )
    elif suffix == '.jsonl':
        lines = _read_text_with_fallback(p).strip().splitlines()
        return [json.loads(l) for l in lines if l.strip()]
    elif suffix == '.json':
        data = json.loads(_read_text_with_fallback(p))
        return data if isinstance(data, list) else [data]
    else:
        raise ValueError(f"Unsupported golden set format: {suffix}")


def _match_golden(filename: str, records: list[dict]) -> dict | None:
    """
    Find the golden record for a given doc filename.
    Handles uploaded names like:
      <uuid>_<original-name>.<ext>.extracted.txt
    and matches them against common filename columns in the golden set.
    """
    from pathlib import Path as P

    def _normalize_name(value: str) -> tuple[str, str]:
        s = str(value or "").strip()
        if not s:
            return "", ""

        # Keep only leaf name if a path is present.
        s = s.replace("\\", "/").split("/")[-1]

        # Strip upload UUID prefix if present.
        s = re.sub(
            r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_',
            "",
            s,
            flags=re.IGNORECASE,
        )

        # Strip extraction suffixes.
        s = re.sub(r'\.extracted\.txt$', '', s, flags=re.IGNORECASE)
        s = re.sub(r'\.extracted$', '', s, flags=re.IGNORECASE)

        base = s.lower().strip()
        stem = P(base).stem.lower().strip()

        # Canonical forms for fuzzy matching.
        base_canon = re.sub(r'[^a-z0-9]+', '', base)
        stem_canon = re.sub(r'[^a-z0-9]+', '', stem)
        return base_canon, stem_canon

    target_base, target_stem = _normalize_name(filename)
    if not target_base and not target_stem:
        return None

    candidate_keys = (
        "file_name", "filename", "File Name", "file", "document", "document_name",
        "doc_name", "source_file", "source_filename", "name", "title", "path",
    )

    for rec in records:
        values = []
        for k in candidate_keys:
            if k in rec and rec.get(k):
                values.append(rec.get(k))
        if not values:
            # Fallback: inspect any key containing 'file' or 'name'
            for k, v in rec.items():
                lk = str(k).lower()
                if v and ("file" in lk or "name" in lk):
                    values.append(v)

        for v in values:
            rec_base, rec_stem = _normalize_name(str(v))
            if not rec_base and not rec_stem:
                continue

            # Exact base or stem matches.
            if rec_base == target_base or rec_stem == target_stem:
                return rec

            # Prefix/contains fallback for minor naming differences.
            if target_stem and rec_stem and (
                target_stem in rec_stem
                or rec_stem in target_stem
                or target_stem[:24] == rec_stem[:24]
            ):
                return rec
    return None


# ─── GPT generation logic ─────────────────────────────────────────────────────

def _gpt_generate_pairs(
    openai_key: str,
    chunk_text: str,
    generation_instruction: str,
    system_prompt: str,
) -> list[dict]:
    """
    Send a chunk to GPT-4o-mini with the user's freeform instruction.
    Returns list of {user, assistant} pairs.
    GPT is instructed to return JSON array of pairs.
    """
    from openai import OpenAI

    client = OpenAI(api_key=openai_key)

    meta_prompt = f"""You are a dataset builder for LLM fine-tuning.

Given the document chunk below, generate training examples following this instruction:
{generation_instruction}

RULES:
- Return ONLY a valid JSON array. No preamble, no markdown fences.
- Each element must be: {{"user": "<question or instruction>", "assistant": "<response>"}}
- The document content must be preserved verbatim in the user turn as context.
- Generate 3-5 high quality, diverse examples per chunk.
- Assistant responses must be grounded in the document — no hallucination.

Document chunk:
{chunk_text}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": meta_prompt}],
        temperature=0.7,
        max_tokens=2000,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rstrip("`").strip()

    pairs = json.loads(raw)
    if not isinstance(pairs, list):
        pairs = [pairs]

    return pairs


# ─── Dataset quality analysis ─────────────────────────────────────────────────

def _analyze_dataset(jsonl_path: str, max_seq_length: int = 2048) -> dict:
    """
    Comprehensive dataset quality analysis.
    Returns structured report for frontend rendering.
    """
    import re
    from collections import Counter, defaultdict
    from math import log

    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {jsonl_path}")

    lines = [l for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]
    total = len(lines)

    # ── Parse records ──
    records = []
    parse_errors = []
    for i, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except Exception as e:
            parse_errors.append({"line": i + 1, "error": str(e)})

    # ── Format validity ──
    format_issues = []
    valid_records = []
    for i, rec in enumerate(records):
        msgs = rec.get("messages", [])
        if not isinstance(msgs, list) or len(msgs) < 2:
            format_issues.append({"index": i + 1, "issue": "Missing or empty messages array"})
            continue
        roles = [m.get("role", "") for m in msgs]
        if "assistant" not in roles:
            format_issues.append({"index": i + 1, "issue": "No assistant message"})
            continue
        if "user" not in roles and "human" not in roles:
            format_issues.append({"index": i + 1, "issue": "No user message"})
            continue
        for m in msgs:
            if not m.get("content", "").strip():
                format_issues.append({"index": i + 1, "issue": f"Empty {m.get('role')} content"})
                break
        else:
            valid_records.append(rec)

    # ── Token length analysis ──
    token_counts = []
    truncation_warnings = []
    for i, rec in enumerate(valid_records):
        msgs = rec.get("messages", [])
        full_text = " ".join(m.get("content", "") for m in msgs)
        tokens = _estimate_tokens(full_text)
        token_counts.append(tokens)
        if tokens > max_seq_length:
            truncation_warnings.append({
                "index": i + 1,
                "tokens": tokens,
                "excess": tokens - max_seq_length,
            })

    if token_counts:
        token_counts_sorted = sorted(token_counts)
        n = len(token_counts_sorted)
        avg_tokens = sum(token_counts) / n
        median_tokens = token_counts_sorted[n // 2]
        p90_tokens = token_counts_sorted[int(n * 0.9)]
        p99_tokens = token_counts_sorted[min(int(n * 0.99), n - 1)]
        min_tokens = token_counts_sorted[0]
        max_tokens = token_counts_sorted[-1]

        # Histogram buckets
        buckets = [0, 256, 512, 1024, 2048, 4096, 99999]
        bucket_labels = ["<256", "256-512", "512-1024", "1024-2048", "2048-4096", ">4096"]
        hist = [0] * len(bucket_labels)
        for t in token_counts:
            for bi in range(len(buckets) - 1):
                if buckets[bi] <= t < buckets[bi + 1]:
                    hist[bi] += 1
                    break
        token_histogram = [{"range": bucket_labels[i], "count": hist[i]} for i in range(len(bucket_labels))]
    else:
        avg_tokens = median_tokens = p90_tokens = p99_tokens = min_tokens = max_tokens = 0
        token_histogram = []

    # ── Output quality ──
    assistant_contents = []
    empty_responses = 0
    short_responses = 0
    json_valid = 0
    json_invalid = 0

    for rec in valid_records:
        msgs = rec.get("messages", [])
        asst = next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), "")
        assistant_contents.append(asst)
        if not asst.strip():
            empty_responses += 1
            continue
        if len(asst.strip()) < 20:
            short_responses += 1

        # Try JSON parse
        try:
            parsed = json.loads(asst.strip())
            if isinstance(parsed, (dict, list)):
                json_valid += 1
            else:
                json_invalid += 1
        except Exception:
            json_invalid += 1

    json_validity_rate = json_valid / max(len(valid_records), 1)

    # ── Field balance (for JSON outputs) ──
    field_stats = {}
    if json_valid > len(valid_records) * 0.5:  # majority JSON outputs
        field_counts = Counter()
        field_null_counts = Counter()
        field_values = defaultdict(list)

        for asst in assistant_contents:
            try:
                obj = json.loads(asst.strip())
                if not isinstance(obj, dict):
                    continue
                for k, v in obj.items():
                    field_counts[k] += 1
                    if v is None or str(v).strip() == "" or str(v).strip().lower() == "null":
                        field_null_counts[k] += 1
                    else:
                        field_values[k].append(str(v))
            except Exception:
                continue

        n_valid = max(json_valid, 1)
        for field, count in field_counts.items():
            vals = field_values.get(field, [])
            # Check if always same value
            unique_vals = set(vals)
            always_same = len(unique_vals) == 1 and len(vals) > 2
            always_null = field_null_counts[field] == count

            field_stats[field] = {
                "present_in": count,
                "present_pct": round(count / n_valid * 100, 1),
                "null_count": field_null_counts[field],
                "null_pct": round(field_null_counts[field] / count * 100, 1) if count else 0,
                "unique_values": len(unique_vals),
                "always_null": always_null,
                "always_same": always_same,
                "warning": (
                    "Always null — useless for training" if always_null
                    else "Always same value — no learning signal" if always_same
                    else "Rarely populated (<20%)" if count / n_valid < 0.2
                    else None
                ),
            }

    # ── Near-duplicate detection (TF-IDF cosine on user messages) ──
    user_messages = []
    for rec in valid_records:
        msgs = rec.get("messages", [])
        user_msg = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
        user_messages.append(user_msg)

    duplicates = []
    if len(user_messages) > 1:
        # Simple TF-IDF without sklearn
        def tokenize(text):
            return re.findall(r'\w+', text.lower())

        def tfidf_vector(tokens, idf):
            tf = Counter(tokens)
            total = max(len(tokens), 1)
            return {t: (tf[t] / total) * idf.get(t, 1.0) for t in tf}

        def cosine_sim(v1, v2):
            keys = set(v1) & set(v2)
            if not keys:
                return 0.0
            dot = sum(v1[k] * v2[k] for k in keys)
            mag1 = sum(x**2 for x in v1.values()) ** 0.5
            mag2 = sum(x**2 for x in v2.values()) ** 0.5
            return dot / (mag1 * mag2 + 1e-10)

        # Build IDF
        N = len(user_messages)
        doc_freq = Counter()
        tokenized = [tokenize(m[:500]) for m in user_messages]  # truncate for speed
        for toks in tokenized:
            for t in set(toks):
                doc_freq[t] += 1
        idf = {t: log(N / (df + 1)) for t, df in doc_freq.items()}

        vectors = [tfidf_vector(toks, idf) for toks in tokenized]

        # Check first 200 pairs max for speed
        checked = 0
        dup_pairs = set()
        for i in range(min(len(vectors), 200)):
            for j in range(i + 1, min(len(vectors), 200)):
                sim = cosine_sim(vectors[i], vectors[j])
                if sim > 0.92:
                    key = (min(i, j), max(i, j))
                    if key not in dup_pairs:
                        dup_pairs.add(key)
                        duplicates.append({
                            "index_a": i + 1,
                            "index_b": j + 1,
                            "similarity": round(sim, 3),
                        })
                checked += 1
                if checked > 5000:
                    break
            if checked > 5000:
                break

    # ── Verdict ──
    issues = []
    warnings_list = []

    if parse_errors:
        issues.append(f"{len(parse_errors)} lines failed JSON parsing")
    if format_issues:
        issues.append(f"{len(format_issues)} records have format issues")
    if empty_responses > 0:
        issues.append(f"{empty_responses} records have empty assistant responses")
    if truncation_warnings:
        pct = len(truncation_warnings) / max(len(valid_records), 1) * 100
        issues.append(f"{len(truncation_warnings)} records ({pct:.0f}%) will be silently truncated at {max_seq_length} tokens")
    if duplicates:
        warnings_list.append(f"{len(duplicates)} near-duplicate user messages detected")
    if short_responses > len(valid_records) * 0.3:
        warnings_list.append(f"{short_responses} responses are very short (<20 chars) — may hurt training")

    for field, fstat in field_stats.items():
        if fstat.get("warning"):
            warnings_list.append(f"Field '{field}': {fstat['warning']}")

    if not issues and not warnings_list:
        verdict = "GO"
        verdict_msg = "Dataset looks healthy. Good to train."
    elif issues:
        verdict = "STOP"
        verdict_msg = f"{len(issues)} critical issue(s) found. Fix before training."
    else:
        verdict = "WARN"
        verdict_msg = f"{len(warnings_list)} warning(s). Review before training."

    return {
        "total_records": total,
        "valid_records": len(valid_records),
        "parse_errors": parse_errors,
        "format_issues": format_issues[:20],  # cap for UI
        "token_stats": {
            "avg": round(avg_tokens),
            "median": round(median_tokens),
            "p90": round(p90_tokens),
            "p99": round(p99_tokens),
            "min": round(min_tokens),
            "max": round(max_tokens),
            "histogram": token_histogram,
        },
        "truncation_warnings": truncation_warnings[:20],
        "truncation_count": len(truncation_warnings),
        "output_quality": {
            "empty_responses": empty_responses,
            "short_responses": short_responses,
            "json_valid": json_valid,
            "json_invalid": json_invalid,
            "json_validity_rate": round(json_validity_rate * 100, 1),
        },
        "field_stats": field_stats,
        "duplicates": duplicates[:20],
        "duplicate_count": len(duplicates),
        "verdict": verdict,
        "verdict_msg": verdict_msg,
        "issues": issues,
        "warnings": warnings_list,
        "max_seq_length": max_seq_length,
    }


# ─── Background job runner ────────────────────────────────────────────────────

def _run_dataprep_job(job_id: str, config_dict: dict):
    """
    Runs in a background thread (via FastAPI BackgroundTasks).
    Supports mode="golden" and mode="gpt".
    """
    job = dataprep_jobs[job_id]  # noqa: F821 — added to main.py globals

    def log(msg, level="INFO"):
        entry = {"ts": datetime.utcnow().isoformat(), "level": level, "msg": msg}
        job["logs"].append(entry)
        logger.info(f"[DataPrep {job_id}] {msg}")  # noqa: F821

    def update_progress(pct, status_msg=""):
        job["progress"] = pct
        if status_msg:
            job["status_msg"] = status_msg

    try:
        config = DataPrepConfig(**config_dict)
        mode = config.mode
        doc_paths = config.doc_paths
        output_dir = Path("uploads")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{config.output_filename}_{job_id}.jsonl"

        log(f"Mode: {mode} | Docs: {len(doc_paths)} | Chunk size: {config.chunk_size_chars} chars")
        update_progress(2, "Initializing")

        golden_records = []
        if mode == "golden" and config.golden_path:
            log(f"Loading golden set from {config.golden_path}")
            golden_records = _load_golden_records(config.golden_path)
            log(f"Loaded {len(golden_records)} golden records")

        total_examples = 0
        total_truncation_warnings = 0

        with open(output_path, "w", encoding="utf-8") as out_file:
            for doc_idx, doc_path in enumerate(doc_paths):
                doc_path_obj = Path(doc_path)
                if not doc_path_obj.exists():
                    log(f"Doc not found, skipping: {doc_path}", "WARN")
                    continue

                filename = doc_path_obj.name
                log(f"Processing [{doc_idx+1}/{len(doc_paths)}]: {filename}")
                update_progress(
                    5 + int((doc_idx / len(doc_paths)) * 85),
                    f"Processing {filename}"
                )

                # Read extracted text (already extracted by /api/extract)
                doc_text = doc_path_obj.read_text(encoding="utf-8", errors="replace")

                # Chunk the document
                chunks = _chunk_document(
                    doc_text,
                    filename,
                    config.chunk_size_chars,
                    config.chunk_overlap_chars,
                    config.max_chunks_per_doc,
                )
                log(f"  → {len(chunks)} chunks generated")

                if mode == "golden":
                    # Join all chunks with the same golden record
                    golden = _match_golden(filename, golden_records)
                    if not golden:
                        log(f"  No golden match for {filename} — skipping", "WARN")
                        continue

                    assistant_content = json.dumps(
                        {k: v for k, v in golden.items() if k not in ("sl_no", "sl_no_", "file_name", "filename", "file_name_")},
                        ensure_ascii=False
                    )

                    for chunk in chunks:
                        user_content = f"Extract the data from the following document section:\n\n{chunk['chunk_text']}"
                        example = _build_chat_example(config.system_prompt, user_content, assistant_content)

                        # Token check
                        full_text = " ".join(m["content"] for m in example["messages"])
                        tokens = _estimate_tokens(full_text)
                        if tokens > config.max_seq_length:
                            total_truncation_warnings += 1
                            log(
                                f"  WARN: chunk {chunk['chunk_index']+1} is {tokens} tokens "
                                f"(>{config.max_seq_length} max_seq_length — will be truncated during training)",
                                "WARN"
                            )

                        out_file.write(json.dumps(example, ensure_ascii=False) + "\n")
                        total_examples += 1

                elif mode == "gpt":
                    if not config.openai_key:
                        log("GPT mode requires openai_key", "ERROR")
                        break
                    if not config.generation_instruction:
                        log("GPT mode requires generation_instruction", "ERROR")
                        break

                    for chunk in chunks:
                        try:
                            pairs = _gpt_generate_pairs(
                                config.openai_key,
                                chunk["chunk_text"],
                                config.generation_instruction,
                                config.system_prompt,
                            )
                            for pair in pairs:
                                user_content = pair.get("user", "")
                                assistant_content = pair.get("assistant", "")
                                if not user_content or not assistant_content:
                                    continue

                                example = _build_chat_example(config.system_prompt, user_content, assistant_content)
                                tokens = _estimate_tokens(" ".join(m["content"] for m in example["messages"]))
                                if tokens > config.max_seq_length:
                                    total_truncation_warnings += 1
                                    log(
                                        f"  WARN: GPT example is {tokens} tokens "
                                        f"(>{config.max_seq_length} — will be truncated)",
                                        "WARN"
                                    )

                                out_file.write(json.dumps(example, ensure_ascii=False) + "\n")
                                total_examples += 1

                            log(f"  Chunk {chunk['chunk_index']+1}/{chunk['total_chunks']}: {len(pairs)} pairs generated")
                        except Exception as e:
                            log(f"  GPT failed for chunk {chunk['chunk_index']+1}: {e}", "ERROR")

        log(f"Dataset complete: {total_examples} examples written to {output_path}")
        if total_truncation_warnings:
            log(
                f"⚠ {total_truncation_warnings} examples exceed max_seq_length={config.max_seq_length}. "
                "Increase max_seq_length in TrainConfig or reduce chunk_size.",
                "WARN"
            )

        update_progress(100, "Complete")
        job["status"] = "completed"
        job["output_path"] = str(output_path)
        job["total_examples"] = total_examples
        job["truncation_warnings"] = total_truncation_warnings
        job["completed_at"] = datetime.utcnow().isoformat()  # noqa: F821

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        log(f"DataPrep failed: {e}", "ERROR")
        logger.error(f"[DataPrep {job_id}] {e}", exc_info=True)  # noqa: F821


# ─── FastAPI routes to add to main.py ────────────────────────────────────────

"""
Paste these route functions into main.py. They reference `dataprep_jobs`,
`logger`, `datetime`, `uuid`, `Path`, `json`, `BackgroundTasks` which are
already imported/defined in main.py.

Also add near the top of main.py:
    dataprep_jobs: dict = {}
"""


# @app.post("/api/dataprep/start")
async def start_dataprep(config: DataPrepConfig, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())[:8]
    dataprep_jobs[job_id] = {
        "job_id": job_id,
        "status": "running",
        "progress": 0,
        "status_msg": "Queued",
        "logs": [],
        "output_path": None,
        "total_examples": 0,
        "truncation_warnings": 0,
        "error": None,
        "started_at": datetime.utcnow().isoformat(),
        "completed_at": None,
    }
    background_tasks.add_task(_run_dataprep_job, job_id, config.dict())
    return {"job_id": job_id, "status": "running"}


# @app.get("/api/dataprep/job/{job_id}")
async def get_dataprep_job(job_id: str):
    if job_id not in dataprep_jobs:
        raise HTTPException(404, f"DataPrep job {job_id} not found")
    return dataprep_jobs[job_id]


# @app.get("/api/dataprep/list")
async def list_dataprep_jobs():
    return list(dataprep_jobs.values())


# @app.post("/api/dataprep/analyze")
async def analyze_dataset(payload: dict):
    """
    POST body: {"dataset_path": "...", "max_seq_length": 2048}
    """
    dataset_path = payload.get("dataset_path")
    max_seq_length = payload.get("max_seq_length", 2048)
    if not dataset_path:
        raise HTTPException(400, "dataset_path required")
    try:
        result = _analyze_dataset(dataset_path, max_seq_length)
        return result
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))