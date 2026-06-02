"""
Dataset Analysis + Gemini Recommendation Endpoint
Add this router to your existing FastAPI app.

POST /analyze-dataset
Body: { dataset_path, model_id, gpu_type, quantization }
Returns: { stats: {...}, recommendation: { params, reasoning, oom_warning } }

The structural checks (row count, token lengths, field density) are done
entirely in Python — dataset contents are NOT sent to any LLM.
Only the aggregate statistics are forwarded to Gemini.
"""

import json
import os
import math
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

# ── GPU VRAM limits (GB) ──────────────────────────────────────────────────────
GPU_VRAM = {
    "T4":   15,
    "A10G": 24,
    "A100": 80,
}

# ── Approx model base VRAM at full precision (GB) ────────────────────────────
MODEL_BASE_VRAM = {
    "Qwen/Qwen2.5-1.5B-Instruct": 3.0,
    "Qwen/Qwen2.5-3B-Instruct":   6.0,
}

QUANT_FACTOR = {
    "none": 1.0,
    "8bit": 0.5,
    "4bit": 0.28,
}

DEMO_SAFE_LIMITS = {
    "T4": {"max_seq_length": 512, "grad_accumulation": 16},
    "A10G": {"max_seq_length": 1024, "grad_accumulation": 8},
    "A100": {"max_seq_length": 2048, "grad_accumulation": 8},
}

# ── Request / Response schemas ────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    dataset_path: str
    model_id: Optional[str] = "Qwen/Qwen2.5-1.5B-Instruct"
    gpu_type: Optional[str] = "A10G"
    quantization: Optional[str] = "4bit"


class DatasetStats(BaseModel):
    row_count: int
    avg_tokens: float
    max_tokens: int
    p95_tokens: int
    avg_chars: float
    max_chars: int
    field_count: Optional[int] = None
    null_ratio: Optional[float] = None
    train_rows: int
    val_rows: int


class Recommendation(BaseModel):
    params: dict
    reasoning: str
    oom_warning: Optional[str] = None


class AnalyzeResponse(BaseModel):
    stats: DatasetStats
    recommendation: Optional[Recommendation] = None


# ── Helpers ───────────────────────────────────────────────────────────────────
def resolve_dataset_path(dataset_path: str) -> Path:
    """Resolve uploaded dataset paths and built-in sample dataset aliases."""
    cleaned = str(dataset_path or "").strip()
    project_root = Path(__file__).resolve().parents[2]

    sample_paths = {
        "__sample_train__": project_root / "training_data.jsonl",
        "__sample_eval__": project_root / "eval_data.jsonl",
    }
    if cleaned in sample_paths:
        return sample_paths[cleaned]

    raw = Path(cleaned)
    if raw.is_absolute():
        return raw

    candidates = [
        Path.cwd() / raw,
        project_root / raw,
        Path(__file__).resolve().parent / raw,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def rough_token_count(text: str) -> int:
    """~4 chars per token estimate — no tokeniser dependency."""
    return max(1, len(text) // 4)


def compute_dataset_stats(path: str, split_ratio: float = 0.9) -> DatasetStats:
    """
    Reads the dataset file from disk and computes structural statistics.
    Supports .jsonl, .json (list), .csv.
    No data content is stored — only aggregate numbers.
    """
    p = resolve_dataset_path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    suffix = p.suffix.lower()
    rows = []

    if suffix in (".jsonl",):
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    elif suffix == ".json":
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        rows = data if isinstance(data, list) else [data]

    elif suffix == ".csv":
        import csv
        with open(p, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

    else:
        # Treat as plain text — each non-empty line is one sample
        with open(p, encoding="utf-8") as f:
            rows = [{"text": line.strip()} for line in f if line.strip()]

    if not rows:
        raise ValueError("Dataset is empty or could not be parsed")

    # Serialise each row to a string and measure length
    char_lengths = []
    token_counts = []
    total_fields = 0
    null_fields = 0

    for row in rows:
        text = json.dumps(row, ensure_ascii=False) if isinstance(row, dict) else str(row)
        char_lengths.append(len(text))
        token_counts.append(rough_token_count(text))

        if isinstance(row, dict):
            for v in row.values():
                # For chat format: recurse into messages
                if isinstance(v, list):
                    for msg in v:
                        if isinstance(msg, dict) and "content" in msg:
                            try:
                                content_obj = json.loads(msg["content"])
                                if isinstance(content_obj, dict):
                                    for cv in content_obj.values():
                                        total_fields += 1
                                        if cv is None:
                                            null_fields += 1
                            except (json.JSONDecodeError, TypeError):
                                pass
                else:
                    total_fields += 1
                    if v is None:
                        null_fields += 1

    token_counts_sorted = sorted(token_counts)
    n = len(token_counts_sorted)
    p95_idx = min(n - 1, int(math.ceil(n * 0.95)) - 1)

    train_rows = int(len(rows) * split_ratio)
    val_rows = len(rows) - train_rows

    return DatasetStats(
        row_count=len(rows),
        avg_tokens=round(sum(token_counts) / n, 1),
        max_tokens=max(token_counts),
        p95_tokens=token_counts_sorted[p95_idx],
        avg_chars=round(sum(char_lengths) / n, 1),
        max_chars=max(char_lengths),
        field_count=total_fields if total_fields else None,
        null_ratio=round(null_fields / total_fields, 3) if total_fields else None,
        train_rows=train_rows,
        val_rows=val_rows,
    )


def available_vram_for_training(model_id: str, gpu_type: str, quantization: str) -> float:
    """Returns approximate GB available for activations/gradients after model load."""
    total_vram = GPU_VRAM.get(gpu_type, 16)
    base_vram  = MODEL_BASE_VRAM.get(model_id, 3.0)
    quant_mult = QUANT_FACTOR.get(quantization, 0.28)
    model_vram = base_vram * quant_mult
    # LoRA adapters add ~5–10% extra; leave ~2GB headroom for OS/CUDA
    available = total_vram - model_vram * 1.1 - 2.0
    return max(0.5, available)


def _round_down_to_multiple(value: int, multiple: int = 64) -> int:
    return max(multiple, (int(value) // multiple) * multiple)


def make_demo_safe_recommendation(
    stats: DatasetStats,
    model_id: str,
    gpu_type: str,
    quantization: str,
    base: Optional[Recommendation] = None,
) -> Recommendation:
    """
    Return settings biased toward a reliable live demo instead of max quality.
    The key safety choice is to truncate long rows rather than chase P95 context.
    """
    gpu = (gpu_type or "A10G").upper()
    limits = DEMO_SAFE_LIMITS.get(gpu, DEMO_SAFE_LIMITS["A10G"])
    seq_cap = limits["max_seq_length"]
    if "3B" in (model_id or "").upper() and gpu in {"T4", "A10G"}:
        seq_cap = min(seq_cap, 768)

    # Keep context modest. Covering every token is useful for production quality,
    # but risky for a live demo because eval loss computes full vocab logits.
    target_seq = min(seq_cap, max(512, _round_down_to_multiple(int(stats.avg_tokens * 1.25))))

    is_small_dataset = stats.row_count < 100
    recommended_epochs = 3 if is_small_dataset else 4
    params = {
        "epochs": recommended_epochs,
        "learning_rate": 0.0001 if is_small_dataset else 0.00005,
        "batch_size": 1,
        "grad_accumulation": limits["grad_accumulation"],
        "max_seq_length": target_seq,
        "lora_r": 4,
        "lora_alpha": 8,
        "warmup_ratio": 0.03,
        "quantization": "4bit" if quantization in {"none", "8bit", "4bit", None} else quantization,
        "max_steps": -1,
        "merge_weights": False,
    }

    if base and base.params:
        # Let Gemini choose only fields that do not increase demo OOM risk.
        for key in ("learning_rate", "warmup_ratio"):
            if key in base.params:
                params[key] = base.params[key]

    reasoning = (
        f"Demo-safe mode intentionally uses max_seq_length={params['max_seq_length']} instead of the "
        f"P95 length ({stats.p95_tokens}) so long rows are truncated rather than causing CUDA OOM. "
        f"Batch size is fixed at 1 with small LoRA rank, at least 3 epochs, and 4-bit quantization to keep the run reliable on {gpu_type}. "
        "Production reference: 5 epochs, learning_rate=2e-4, batch_size=2, grad_accumulation=8, "
        "max_seq_length near P95, lora_r=8, lora_alpha=16. If that OOMs, lower batch_size to 1 first, "
        "then reduce max_seq_length before increasing gradient accumulation."
    )
    if base and base.reasoning:
        reasoning = f"{reasoning} Gemini note: {base.reasoning}"

    return Recommendation(
        params=params,
        reasoning=reasoning,
        oom_warning=(
            "Demo-safe settings applied: long samples may be truncated, but this is far less likely to OOM. "
            "For production quality, increase max_seq_length later after the demo."
        ),
    )


def ask_gemini(stats: DatasetStats, model_id: str, gpu_type: str, quantization: str) -> Recommendation:
    """
    Sends only aggregate statistics (not data content) to Gemini Flash
    and asks for safe hyperparameter recommendations.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in environment")

    try:
        import google.generativeai as genai
    except ImportError as e:
        raise ValueError("google-generativeai is not installed") from e

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-3.5-flash")

    avail_vram = available_vram_for_training(model_id, gpu_type, quantization)

    prompt = f"""You are an expert in LLM fine-tuning. Given the dataset statistics and hardware below,
recommend conservative DEMO-SAFE training hyperparameters that avoid OOM errors even if that means truncating long rows.

Dataset statistics (computed by Python — no data was sent to you):
- Total rows: {stats.row_count}
- Avg tokens per row: {stats.avg_tokens}
- Max tokens per row: {stats.max_tokens}
- P95 tokens per row: {stats.p95_tokens}
- Training rows: {stats.train_rows}
- Validation rows: {stats.val_rows}
- Field count (total): {stats.field_count or 'unknown'}
- Null ratio: {stats.null_ratio or 'unknown'}

Hardware:
- GPU: {gpu_type} ({GPU_VRAM.get(gpu_type, '?')}GB VRAM)
- Model: {model_id}
- Quantization: {quantization}
- Approx VRAM available after model load: {avail_vram:.1f}GB

Rules:
1. This is a live demo profile. Prioritize "definitely runs" over quality.
2. Do NOT require max_seq_length to cover P95. Long rows may be truncated for safety.
3. For A10G, prefer max_seq_length <= 1024. For T4, <= 512. For A100, <= 2048.
4. batch_size must be 1 unless the sequence length is tiny.
5. batch_size * grad_accumulation should be 8–16 for this demo.
6. For datasets under 100 rows: epochs=3, learning_rate=1e-4, lora_r=4.
7. For 100–500 rows: epochs=4, learning_rate=5e-5, lora_r=4 or 8.
8. Use 4bit quantization for demo safety.
5. lora_alpha = 2 * lora_r always.
9. Set max_steps=-1 so the selected epoch count controls the cached sample run.

Respond ONLY with a JSON object — no markdown, no explanation outside the JSON:
{{
  "params": {{
    "epochs": <int>,
    "learning_rate": <float>,
    "batch_size": <int>,
    "grad_accumulation": <int>,
    "max_seq_length": <int>,
    "lora_r": <int>,
    "lora_alpha": <int>,
    "warmup_ratio": <float>,
    "quantization": "<none|8bit|4bit>",
    "max_steps": <int>,
    "merge_weights": <boolean>
  }},
  "reasoning": "<2-3 sentence plain-English explanation of the key decisions>",
  "oom_warning": "<short warning if any settings are close to VRAM limit, or null>"
}}"""

    response = model.generate_content(prompt)
    text = response.text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    parsed = json.loads(text)
    raw_recommendation = Recommendation(
        params=parsed.get("params", {}),
        reasoning=parsed.get("reasoning", ""),
        oom_warning=parsed.get("oom_warning") or None,
    )
    return make_demo_safe_recommendation(stats, model_id, gpu_type, quantization, raw_recommendation)


# ── Route ─────────────────────────────────────────────────────────────────────
@router.post("/analyze-dataset", response_model=AnalyzeResponse)
async def analyze_dataset(req: AnalyzeRequest):
    """
    1. Reads dataset from disk and computes structural stats in Python.
    2. Sends only the aggregate stats to Gemini for hyperparameter recommendations.
    3. Returns stats + recommendations with OOM-safe limits.
    """
    try:
        stats = compute_dataset_stats(req.dataset_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dataset parsing error: {str(e)}")

    recommendation = None
    try:
        recommendation = ask_gemini(stats, req.model_id, req.gpu_type, req.quantization)
    except ValueError as e:
        # Missing API key — return stats without recommendation
        recommendation = make_demo_safe_recommendation(
            stats,
            req.model_id,
            req.gpu_type,
            req.quantization,
            Recommendation(
                params={},
                reasoning=f"Gemini recommendation unavailable: {str(e)}. Using local demo-safe defaults instead.",
                oom_warning=None,
            ),
        )
    except Exception as e:
        recommendation = make_demo_safe_recommendation(
            stats,
            req.model_id,
            req.gpu_type,
            req.quantization,
            Recommendation(
                params={},
                reasoning=f"Gemini call failed: {str(e)}. Using local demo-safe defaults instead.",
                oom_warning=None,
            ),
        )

    return AnalyzeResponse(stats=stats, recommendation=recommendation)


# ── Wire up to your existing app ──────────────────────────────────────────────
# In your main.py / app.py, add:
#
#   from analyze_dataset_route import router as analyze_router
#   app.include_router(analyze_router)
#
# And add to your api.js:
#
#   analyzeDataset: (body) => post('/analyze-dataset', body),
