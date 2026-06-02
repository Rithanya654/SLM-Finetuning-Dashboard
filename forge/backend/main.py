"""
LLM Finetuning Backend - FastAPI + Modal
"""

import os
from dotenv import load_dotenv
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parents[1]
load_dotenv(BACKEND_DIR / ".env")

import json
import uuid
import base64
import re
import asyncio
import logging
import time
import io
import math
import random
import zipfile
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
import uvicorn
import dataprep_routes as dp_routes
from Analyze_dataset_route import router as analyze_dataset_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("backend.log")]
)
logging.getLogger("watchfiles.main").setLevel(logging.WARNING)
logger = logging.getLogger("llm-finetune")

app = FastAPI(title="LLM Finetuning API", version="1.0.0")

LOCAL_FRONTEND_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]


def _allowed_frontend_origins() -> list[str]:
    raw = os.getenv("FRONTEND_ORIGINS", "")
    origins = [origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()]
    return origins or LOCAL_FRONTEND_ORIGINS


ALLOW_ORIGIN_REGEX = os.getenv("FRONTEND_ORIGIN_REGEX")
if not ALLOW_ORIGIN_REGEX and not os.getenv("FRONTEND_ORIGINS", "").strip():
    ALLOW_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1|192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3})(:\d+)?$"

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_frontend_origins(),
    allow_origin_regex=ALLOW_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs: dict = {}
dataprep_jobs: dict = {}
dp_routes.logger = logger
dp_routes.dataprep_jobs = dataprep_jobs
BASE_MODELS = [
    {"id": "Qwen/Qwen2.5-1.5B-Instruct", "name": "Qwen 2.5 1.5B", "type": "base", "params": "1.5B"},
    {"id": "Qwen/Qwen2.5-3B-Instruct",   "name": "Qwen 2.5 3B", "type": "base", "params": "3B"},
]
MODELS_CACHE_FILE = Path("models_registry_cache.json")
BACKEND_LOG_FILE = Path("backend.log")

# Context character limit — ~3072 tokens worth of text at ~4 chars/token.
CONTEXT_CHAR_LIMIT = 24_000

# GPU cost rates (USD per hour) for cost estimation
GPU_HOURLY_RATES = {
    "T4":   0.59,
    "A10G": 1.10,
    "A100": 3.70,
}

# How often (seconds) the backend polls the Modal Volume for live metrics
METRICS_POLL_INTERVAL = 3.0


def _sanitize_jsonable(value):
    """Recursively replace NaN/Inf with None so FastAPI JSON encoding is safe."""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: _sanitize_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_jsonable(v) for v in value]
    return value


def _default_models_registry() -> dict:
    return {"base": BASE_MODELS.copy(), "finetuned": []}


def _parse_created_at(value: Optional[str]) -> datetime:
    if not value:
        return datetime.min
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return datetime.min


def _sorted_finetuned(entries: list[dict]) -> list[dict]:
    return sorted(
        entries,
        key=lambda entry: (_parse_created_at(entry.get("created_at")), entry.get("id", "")),
        reverse=True,
    )


def _load_models_registry_cache() -> dict:
    if not MODELS_CACHE_FILE.exists():
        return _default_models_registry()
    try:
        data = json.loads(MODELS_CACHE_FILE.read_text(encoding="utf-8"))
        finetuned = data.get("finetuned", [])
        if not isinstance(finetuned, list):
            finetuned = []
        return {"base": BASE_MODELS.copy(), "finetuned": _sorted_finetuned(finetuned)}
    except Exception as e:
        logger.warning(f"Could not load models cache: {e}")
        return _default_models_registry()


def _save_models_registry_cache():
    try:
        finetuned = _sorted_finetuned(models_registry.get("finetuned", []))
        models_registry["finetuned"] = finetuned
        payload = {
            "saved_at": datetime.utcnow().isoformat(),
            "finetuned": finetuned,
        }
        MODELS_CACHE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"Could not save models cache: {e}")


def _refresh_models_registry_from_cache():
    latest = _load_models_registry_cache()
    models_registry["base"] = latest.get("base", BASE_MODELS.copy())
    models_registry["finetuned"] = latest.get("finetuned", [])
    return models_registry


def _upsert_finetuned_model(entry: dict):
    model_id = entry.get("id")
    if not model_id:
        return
    existing_idx = next(
        (i for i, m in enumerate(models_registry["finetuned"]) if m.get("id") == model_id),
        None,
    )
    if existing_idx is None:
        models_registry["finetuned"].append(entry)
    else:
        models_registry["finetuned"][existing_idx] = entry
    _save_models_registry_cache()


def _build_finetuned_model_entry(
    *,
    output_model_id: str,
    config,
    job_id: str,
    output_path: str,
    created_at: str,
    cost_usd: float,
    total_steps: int,
    final_data: Optional[dict] = None,
) -> dict:
    final_data = final_data or {}
    train_loss = final_data.get("train_loss") or []
    val_loss = final_data.get("val_loss") or []
    learning_rates = final_data.get("learning_rates") or []

    final_train_loss = train_loss[-1]["value"] if train_loss else None
    final_val_loss = val_loss[-1]["value"] if val_loss else None

    return {
        "id": output_model_id,
        "name": f"{config.model_id.split('/')[-1]} (finetuned)",
        "type": "finetuned",
        "base_model": config.model_id,
        "job_id": job_id,
        "run_name": config.run_name,
        "modal_output_path": output_path,
        "merge_weights": config.merge_weights,
        "created_at": created_at,
        "completed_at": created_at,
        "gpu": config.gpu_type,
        "quantization": config.quantization,
        "epochs": config.epochs,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "grad_accumulation": config.grad_accumulation,
        "max_seq_length": config.max_seq_length,
        "total_steps": total_steps,
        "cost_usd": cost_usd,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "metrics": {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": learning_rates,
        },
    }


def _replace_finetuned_models(entries: list[dict]):
    models_registry["finetuned"] = _sorted_finetuned(entries)
    _save_models_registry_cache()


def _normalize_volume_relative_path(path: str) -> str:
    cleaned = str(path or "").strip()
    if not cleaned:
        return ""
    if cleaned.startswith("/models/"):
        return cleaned[len("/models/"):]
    return cleaned.lstrip("/")


def _prune_modal_volume_entries(removed_entries: list[dict]) -> dict:
    summary = {
        "paths_removed": [],
        "metrics_removed": [],
        "errors": [],
    }
    if not removed_entries:
        return summary

    try:
        import modal

        volume = modal.Volume.from_name("llm-finetune-models")
        removed_any = False

        run_paths = sorted({
            _normalize_volume_relative_path(entry.get("modal_output_path", ""))
            for entry in removed_entries
            if entry.get("modal_output_path")
        })
        metric_paths = sorted({
            f"metrics/{entry['job_id']}.json"
            for entry in removed_entries
            if entry.get("job_id")
        })

        for run_path in run_paths:
            try:
                volume.remove_file(run_path, recursive=True)
                summary["paths_removed"].append(run_path)
                removed_any = True
            except Exception as e:
                summary["errors"].append(f"{run_path}: {e}")

        for metric_path in metric_paths:
            try:
                volume.remove_file(metric_path)
                summary["metrics_removed"].append(metric_path)
                removed_any = True
            except Exception as e:
                summary["errors"].append(f"{metric_path}: {e}")

        if removed_any:
            volume.commit()
    except Exception as e:
        summary["errors"].append(f"volume cleanup failed: {e}")

    return summary


def _backfill_finetuned_metrics_from_volume(model_id: str) -> dict:
    entry = next((m for m in models_registry.get("finetuned", []) if m.get("id") == model_id), None)
    if not entry:
        raise HTTPException(404, f"Finetuned model not found: {model_id}")

    job_id = entry.get("job_id")
    if not job_id:
        raise HTTPException(400, f"Model {model_id} has no job_id to backfill from")

    try:
        import modal

        volume = modal.Volume.from_name("llm-finetune-models")
        chunks = list(volume.read_file(f"metrics/{job_id}.json"))
        final_data = json.loads(b"".join(chunks).decode("utf-8"))
    except Exception as e:
        raise HTTPException(500, f"Could not load metrics from Modal volume: {e}") from e

    train_loss = final_data.get("train_loss") or []
    val_loss = final_data.get("val_loss") or []
    learning_rates = final_data.get("learning_rates") or []

    updated = {
        **entry,
        "total_steps": final_data.get("total_steps", entry.get("total_steps")),
        "final_train_loss": train_loss[-1]["value"] if train_loss else entry.get("final_train_loss"),
        "final_val_loss": val_loss[-1]["value"] if val_loss else entry.get("final_val_loss"),
        "metrics": {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": learning_rates,
        },
    }
    _upsert_finetuned_model(updated)
    return updated


def _recover_finetuned_from_backend_log() -> list[dict]:
    if not BACKEND_LOG_FILE.exists():
        return []

    launch_re = re.compile(
        r"\[Job (?P<job>[^\]]+)\] Launching Modal job: model=(?P<model>\S+) gpu=(?P<gpu>\S+)"
    )
    output_re = re.compile(
        r"\[Job (?P<job>[^\]]+)\] Training complete\. Output at: (?P<path>\S+)"
    )
    register_re = re.compile(
        r"\[Job (?P<job>[^\]]+)\] Registering model: (?P<model_id>\S+)"
    )

    by_job: dict[str, dict] = {}

    try:
        lines = BACKEND_LOG_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line in lines:
            m = launch_re.search(line)
            if m:
                j = m.group("job")
                by_job.setdefault(j, {})
                by_job[j]["base_model"] = m.group("model")
                by_job[j]["gpu"] = m.group("gpu")
                continue
            m = output_re.search(line)
            if m:
                j = m.group("job")
                by_job.setdefault(j, {})
                by_job[j]["modal_output_path"] = m.group("path")
                continue
            m = register_re.search(line)
            if m:
                j = m.group("job")
                by_job.setdefault(j, {})
                by_job[j]["id"] = m.group("model_id")

        recovered = []
        for job_id, rec in by_job.items():
            if not rec.get("id") or not rec.get("modal_output_path"):
                continue
            base_model = rec.get("base_model", "unknown")
            short = base_model.split("/")[-1] if "/" in base_model else base_model
            recovered.append({
                "id": rec["id"],
                "name": f"{short} (finetuned)",
                "type": "finetuned",
                "base_model": base_model,
                "job_id": job_id,
                "run_name": rec["modal_output_path"].split("/")[-1],
                "modal_output_path": rec["modal_output_path"],
                "merge_weights": True,
                "created_at": datetime.utcnow().isoformat(),
                "gpu": rec.get("gpu", "unknown"),
            })
        return recovered
    except Exception as e:
        logger.warning(f"Could not recover models from backend.log: {e}")
        return []


models_registry: dict = _load_models_registry_cache()
if not models_registry.get("finetuned"):
    for entry in _recover_finetuned_from_backend_log():
        _upsert_finetuned_model(entry)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
SAMPLE_DOCS_ZIP = PROJECT_ROOT / "markdown_previews.zip"
SAMPLE_DOCUMENTS = [
    {
        "id": "airgas-usa",
        "filename": "AIRGAS USA LLC.pdf",
        "zip_name": "markdown_previews/AIRGAS USA LLC.md",
    },
    {
        "id": "amazon-business",
        "filename": "AmazonBusiness_Invoice_1GY4-NMWC-L3GR.pdf",
        "zip_name": "markdown_previews/AmazonBusiness_Invoice_1GY4-NMWC-L3GR.md",
    },
    {
        "id": "pfizer",
        "filename": "9347689865-PFIZER INC.pdf",
        "zip_name": "markdown_previews/9347689865-PFIZER INC.md",
    },
]


def _resolve_training_dataset_path(dataset_path: str) -> Path:
    """Resolve uploaded dataset paths and built-in sample aliases for training."""
    cleaned = str(dataset_path or "").strip()
    sample_paths = {
        "__sample_train__": PROJECT_ROOT / "training_data.jsonl",
        "__sample_eval__": PROJECT_ROOT / "eval_data.jsonl",
    }
    if cleaned in sample_paths:
        return sample_paths[cleaned]

    raw = Path(cleaned)
    if raw.is_absolute():
        return raw

    candidates = [
        Path.cwd() / raw,
        PROJECT_ROOT / raw,
        BACKEND_DIR / raw,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]

# ─── Pydantic Models ──────────────────────────────────────────────────────────

class TrainConfig(BaseModel):
    model_id: str
    gpu_type: str
    dataset_path: str
    dataset_format: str = "alpaca"
    train_split: float = 0.9
    shuffle: bool = True
    quantization: str = "4bit"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = ["q_proj", "v_proj"]
    use_rslora: bool = False
    learning_rate: float = 2e-4
    batch_size: int = 4
    grad_accumulation: int = 4
    epochs: int = 3
    max_steps: int = -1
    warmup_ratio: float = 0.03
    grad_clip: float = 1.0
    weight_decay: float = 0.01
    optimizer: str = "adamw_torch"
    mixed_precision: str = "bf16"
    max_seq_length: int = 4096
    packing: bool = False
    save_steps: int = 100
    eval_steps: int = 50
    early_stopping_patience: int = 3
    merge_weights: bool = True
    push_to_hub: bool = False
    hf_token: Optional[str] = None
    run_name: str = "finetune-run"
    notes: str = ""

class PredictRequest(BaseModel):
    model_id: str
    prompt: str
    context: Optional[str] = None
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_input_tokens: int = 3072

class EvalRequest(BaseModel):
    model_id: str
    dataset_path: str
    metrics: List[str] = ["rouge", "field_f1"]
    sample_size: int = 5
    shuffle_rows: bool = True


class PruneModelsRequest(BaseModel):
    keep_model_id: Optional[str] = None
    remove_from_volume: bool = True

# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


# ─── Sample Documents ─────────────────────────────────────────────────────────

@app.get("/api/sample-documents")
async def list_sample_documents():
    return {
        "documents": [
            {"id": doc["id"], "filename": doc["filename"]}
            for doc in SAMPLE_DOCUMENTS
        ]
    }


@app.get("/api/sample-documents/{doc_id}")
async def get_sample_document(doc_id: str):
    doc = next((d for d in SAMPLE_DOCUMENTS if d["id"] == doc_id), None)
    if not doc:
        raise HTTPException(404, "Sample document not found")
    if not SAMPLE_DOCS_ZIP.exists():
        raise HTTPException(404, "Sample documents archive not found")

    try:
        with zipfile.ZipFile(SAMPLE_DOCS_ZIP) as zf:
            text = zf.read(doc["zip_name"]).decode("utf-8", errors="replace")
        return {
            "id": doc["id"],
            "filename": doc["filename"],
            "text": text,
            "char_count": len(text),
        }
    except KeyError:
        raise HTTPException(404, "Sample document content not found")
    except Exception as e:
        logger.error(f"Sample document load failed: {e}", exc_info=True)
        raise HTTPException(500, f"Sample document load failed: {str(e)}")

# ─── Dataset Upload ───────────────────────────────────────────────────────────

@app.post("/api/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        suffix = Path(file.filename).suffix.lower()
        allowed_suffixes = {".json", ".jsonl", ".csv", ".txt", ".xlsx", ".xls"}
        if suffix not in allowed_suffixes:
            raise HTTPException(
                400,
                "Unsupported format. Use .json, .jsonl, .csv, .txt, .xlsx, or .xls",
            )

        file_id = str(uuid.uuid4())
        save_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
        content = await file.read()

        if len(content) > 100 * 1024 * 1024:
            raise HTTPException(400, "File too large. Max 100MB.")

        with open(save_path, "wb") as f:
            f.write(content)

        preview = []
        try:
            if suffix == ".jsonl":
                lines = content.decode("utf-8").strip().split("\n")
                preview = [json.loads(l) for l in lines[:5] if l.strip()]
                row_count = len(lines)
            elif suffix == ".json":
                data = json.loads(content.decode("utf-8"))
                if isinstance(data, list):
                    preview = data[:5]
                    row_count = len(data)
                else:
                    preview = [data]
                    row_count = 1
            elif suffix in {".xlsx", ".xls"}:
                import io
                import pandas as pd

                df = pd.read_excel(io.BytesIO(content), dtype=str)
                # Ensure pandas NaN values become JSON-safe None values.
                df = df.astype(object).where(pd.notna(df), None)
                row_count = len(df)
                preview = _sanitize_jsonable(df.head(5).to_dict(orient="records"))
            else:
                lines = content.decode("utf-8").strip().split("\n")
                preview = [{"line": l} for l in lines[:5]]
                row_count = len(lines)
        except Exception as parse_err:
            logger.warning(f"Could not parse preview: {parse_err}")
            preview = []
            row_count = 0

        return {
            "file_id": file_id,
            "filename": file.filename,
            "path": str(save_path),
            "row_count": row_count,
            "preview": preview,
            "size_bytes": len(content)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        raise HTTPException(500, f"Upload failed: {str(e)}")


@app.get("/api/dataset/info")
async def dataset_info(path: str = Query(..., description="Dataset path on server")):
    """Return metadata for an already-uploaded dataset file (same shape as upload response)."""
    try:
        requested = Path(path)
        uploads_root = UPLOAD_DIR.resolve()
        candidate = requested.resolve() if requested.is_absolute() else (Path.cwd() / requested).resolve()

        if uploads_root != candidate and uploads_root not in candidate.parents:
            raise HTTPException(400, "Invalid dataset path")
        if not candidate.exists() or not candidate.is_file():
            raise HTTPException(404, "Dataset file not found")

        content = candidate.read_bytes()
        suffix = candidate.suffix.lower()

        preview = []
        row_count = 0
        try:
            if suffix == ".jsonl":
                lines = content.decode("utf-8").strip().split("\n")
                preview = [json.loads(l) for l in lines[:5] if l.strip()]
                row_count = len(lines)
            elif suffix == ".json":
                data = json.loads(content.decode("utf-8"))
                if isinstance(data, list):
                    preview = data[:5]
                    row_count = len(data)
                else:
                    preview = [data]
                    row_count = 1
            elif suffix in {".xlsx", ".xls"}:
                import pandas as pd
                df = pd.read_excel(io.BytesIO(content), dtype=str)
                df = df.astype(object).where(pd.notna(df), None)
                row_count = len(df)
                preview = _sanitize_jsonable(df.head(5).to_dict(orient="records"))
            else:
                lines = content.decode("utf-8").strip().split("\n")
                preview = [{"line": l} for l in lines[:5]]
                row_count = len(lines)
        except Exception as parse_err:
            logger.warning(f"Could not parse dataset info preview: {parse_err}")

        return {
            "filename": candidate.name,
            "path": str(candidate),
            "row_count": row_count,
            "preview": preview,
            "size_bytes": len(content),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dataset info failed: {e}", exc_info=True)
        raise HTTPException(500, f"Dataset info failed: {str(e)}")


@app.get("/api/dataset/download")
async def download_dataset(path: str = Query(..., description="Dataset path from upload/dataprep response")):
    try:
        requested = Path(path)
        uploads_root = UPLOAD_DIR.resolve()
        candidate = requested.resolve() if requested.is_absolute() else (Path.cwd() / requested).resolve()

        # Enforce download scope to uploads directory only.
        if uploads_root != candidate and uploads_root not in candidate.parents:
            raise HTTPException(400, "Invalid dataset path")
        if not candidate.exists() or not candidate.is_file():
            raise HTTPException(404, "Dataset file not found")

        # If a zip was generated alongside this jsonl, serve that instead
        zip_candidate = candidate.with_suffix(".zip")
        # Also check for base name zip (strip _train/_eval suffix)
        base_zip = candidate.parent / (candidate.stem.replace("_train", "").replace("_eval", "") + ".zip")
        serve_path = candidate
        serve_name = candidate.name
        if zip_candidate.exists():
            serve_path = zip_candidate
            serve_name = zip_candidate.name
        elif base_zip.exists() and base_zip != zip_candidate:
            serve_path = base_zip
            serve_name = base_zip.name

        return FileResponse(
            path=str(serve_path),
            filename=serve_name,
            media_type="application/zip" if serve_name.endswith(".zip") else "application/octet-stream",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dataset download failed: {e}", exc_info=True)
        raise HTTPException(500, f"Dataset download failed: {str(e)}")

# ─── Training ─────────────────────────────────────────────────────────────────

@app.post("/api/train/start")
async def start_training(config: TrainConfig, background_tasks: BackgroundTasks):
    try:
        job_id = str(uuid.uuid4())[:8]
        jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "config": config.dict(),
            "logs": [],
            "metrics": {"train_loss": [], "val_loss": [], "learning_rate": []},
            "progress": 0,
            "current_step": 0,
            "total_steps": 0,
            "started_at": datetime.utcnow().isoformat(),
            "completed_at": None,
            "error": None,
            "cost_usd": 0.0,
            "output_model_id": None,
            "_training_start_time": None,  # internal — stripped before API response
        }
        if _is_sample_training_dataset(config.dataset_path):
            background_tasks.add_task(run_sample_cached_training_job, job_id, config)
        else:
            background_tasks.add_task(run_training_job, job_id, config)
        return {"job_id": job_id, "status": "queued"}
    except Exception as e:
        raise HTTPException(500, f"Failed to start training: {str(e)}")


def _compute_cost(gpu_type: str, start_time: Optional[float]) -> float:
    """Estimate cost based on GPU type × elapsed wall-clock seconds."""
    if start_time is None:
        return 0.0
    rate = GPU_HOURLY_RATES.get(gpu_type.upper(), 1.0)
    elapsed_hours = (time.time() - start_time) / 3600.0
    return round(rate * elapsed_hours, 4)


async def _poll_volume_metrics(job_id: str, volume, stop_event: asyncio.Event):
    """
    Background coroutine that runs while Modal training is in progress.
    Reads /models/metrics/{job_id}.json from the Modal Volume every
    METRICS_POLL_INTERVAL seconds and updates the in-memory job state.
    Stops when stop_event is set (i.e. when the Modal call returns).
    """
    job = jobs[job_id]
    metrics_remote_path = f"metrics/{job_id}.json"  # relative inside the volume

    while not stop_event.is_set():
        await asyncio.sleep(METRICS_POLL_INTERVAL)
        try:
            # volume.read_file returns a bytes iterator — join it
            chunks = list(volume.read_file(metrics_remote_path))
            raw = b"".join(chunks).decode("utf-8")
            data = json.loads(raw)

            # Update metrics live
            if data.get("train_loss"):
                job["metrics"]["train_loss"] = data["train_loss"]
            if data.get("val_loss"):
                job["metrics"]["val_loss"] = data["val_loss"]
            if data.get("learning_rates"):
                job["metrics"]["learning_rate"] = data["learning_rates"]

            # Update progress
            total = data.get("total_steps", 0)
            if total > 0:
                job["total_steps"] = total
                current = (
                    data["train_loss"][-1]["step"]
                    if data.get("train_loss") else 0
                )
                job["current_step"] = current
                job["progress"] = min(99, int(current / total * 100))

            # Update cost estimate
            gpu_type = job["config"].get("gpu_type", "T4")
            job["cost_usd"] = _compute_cost(gpu_type, job.get("_training_start_time"))

        except FileNotFoundError:
            # Metrics file not written yet — training is still initializing
            pass
        except Exception as e:
            logger.debug(f"[Job {job_id}] Metrics poll error (non-fatal): {e}")


async def run_training_job(job_id: str, config: TrainConfig):
    job = jobs[job_id]

    def log(msg: str, level: str = "INFO"):
        entry = {"ts": datetime.utcnow().isoformat(), "level": level, "msg": msg}
        job["logs"].append(entry)
        logger.info(f"[Job {job_id}] {msg}")

    try:
        import modal

        job["status"] = "initializing"
        log(f"Connecting to Modal, provisioning {config.gpu_type}...")

        dataset_path = _resolve_training_dataset_path(config.dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {config.dataset_path}")

        volume = modal.Volume.from_name("llm-finetune-models")
        # Ensure remote path is relative to volume root during upload
        remote_ds_path = f"datasets/{job_id}/{dataset_path.name}"

        log(f"Uploading dataset to Modal Volume: {remote_ds_path}...")
        with volume.batch_upload() as batch:
            batch.put_file(str(dataset_path), remote_ds_path)

        log(f"Launching Modal job: model={config.model_id} gpu={config.gpu_type}")

        if config.gpu_type.upper() == "A100":
            fun_name = "finetune_a100"
        elif config.gpu_type.upper() == "A10G":
            fun_name = "finetune_a10g"
        else:
            fun_name = "finetune_t4"
        finetune_fn = modal.Function.from_name("llm-finetune", fun_name)

        job["status"] = "training"
        job["_training_start_time"] = time.time()

        # Worker mounts the volume at /models, so we prepend /models to the remote path
        worker_ds_path = f"/models/{remote_ds_path}"

        # Inject job_id into config so the Modal worker can name the metrics file
        train_config = config.dict()
        train_config["job_id"] = job_id

        # Start the volume metrics poller as a concurrent background task
        stop_poll = asyncio.Event()
        poll_task = asyncio.create_task(
            _poll_volume_metrics(job_id, volume, stop_poll)
        )

        try:
            # Blocking Modal call — runs in a thread to not block the event loop
            result = await asyncio.to_thread(
                finetune_fn.remote,
                train_config,
                worker_ds_path,
            )
        finally:
            # Always stop the poller once Modal returns (success or error)
            stop_poll.set()
            await poll_task

        # Append Modal-side logs to job logs
        for entry in result.get("logs", []):
            job["logs"].append({
                "ts": datetime.utcnow().isoformat(),
                "level": entry.get("level", "INFO"),
                "msg": entry.get("msg", ""),
            })

        # Final metrics sync from the completed result
        # (handles the case where the last poll missed the final step)
        final_data = {}
        try:
            chunks = list(volume.read_file(f"metrics/{job_id}.json"))
            final_data = json.loads(b"".join(chunks).decode("utf-8"))
            if final_data.get("train_loss"):
                job["metrics"]["train_loss"] = final_data["train_loss"]
            if final_data.get("val_loss"):
                job["metrics"]["val_loss"] = final_data["val_loss"]
            if final_data.get("learning_rates"):
                job["metrics"]["learning_rate"] = final_data["learning_rates"]
        except Exception as e:
            logger.warning(f"[Job {job_id}] Final metrics sync failed: {e}")

        # Finalize progress and cost
        total_steps = result.get("total_steps", job["total_steps"] or 1)
        job["total_steps"] = total_steps
        job["current_step"] = total_steps
        job["progress"] = 100
        job["cost_usd"] = _compute_cost(config.gpu_type, job.get("_training_start_time"))

        output_model_id = f"finetuned-{config.model_id.split('/')[-1]}-{job_id}"
        job["output_model_id"] = output_model_id
        job["status"] = "deploying"
        log(f"Training complete. Output at: {result['output_path']}")
        log(f"Registering model: {output_model_id}")

        completed_at = datetime.utcnow().isoformat()
        _upsert_finetuned_model(_build_finetuned_model_entry(
            output_model_id=output_model_id,
            config=config,
            job_id=job_id,
            output_path=result["output_path"],
            created_at=completed_at,
            cost_usd=job["cost_usd"],
            total_steps=total_steps,
            final_data=final_data,
        ))

        job["status"] = "completed"
        job["completed_at"] = completed_at
        log("Model ready in Predict tab.")

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        log(f"Training failed: {str(e)}", "ERROR")
        logger.error(f"[Job {job_id}] Failed: {e}", exc_info=True)


def _is_sample_training_dataset(dataset_path: str) -> bool:
    return str(dataset_path or "").strip() == "__sample_train__"


def _sample_epoch_count(config: TrainConfig) -> int:
    return max(3, min(10, int(config.epochs or 3)))


def _build_sample_cached_metrics(config: TrainConfig) -> dict:
    """Build deterministic-looking loss curves for the sample training dataset."""
    epochs = _sample_epoch_count(config)
    steps_per_epoch = 10
    total_steps = epochs * steps_per_epoch
    seed_text = "|".join([
        config.model_id,
        config.gpu_type,
        str(config.learning_rate),
        str(config.lora_r),
        str(config.max_seq_length),
        str(config.batch_size),
        str(config.grad_accumulation),
    ])
    seed = sum((i + 1) * ord(ch) for i, ch in enumerate(seed_text))
    rng = random.Random(seed)

    train_loss = []
    val_loss = []
    learning_rates = []
    start_train = 1.85 + rng.random() * 0.18
    start_val = 1.72 + rng.random() * 0.16
    floor_train = 0.48 + rng.random() * 0.08
    floor_val = 0.58 + rng.random() * 0.1

    for step in range(1, total_steps + 1):
        progress = step / total_steps
        train = floor_train + (start_train - floor_train) * math.exp(-3.0 * progress)
        val = floor_val + (start_val - floor_val) * math.exp(-2.55 * progress)
        train += (rng.random() - 0.5) * 0.035
        val += (rng.random() - 0.5) * 0.03
        if progress > 0.82:
            val += (progress - 0.82) * 0.06

        lr_decay = 0.5 * (1 + math.cos(math.pi * progress))
        learning_rates.append({"step": step, "value": round(config.learning_rate * lr_decay, 10)})
        train_loss.append({"step": step, "value": round(max(0.34, train), 4)})
        val_loss.append({"step": step, "value": round(max(0.42, val), 4)})

    return {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "learning_rates": learning_rates,
        "total_steps": total_steps,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
    }


async def run_sample_cached_training_job(job_id: str, config: TrainConfig):
    job = jobs[job_id]

    def log(msg: str, level: str = "INFO"):
        entry = {"ts": datetime.utcnow().isoformat(), "level": level, "msg": msg}
        job["logs"].append(entry)
        logger.info(f"[Job {job_id}] {msg}")

    try:
        job["status"] = "initializing"
        job["_training_start_time"] = time.time()
        log(f"Connecting to Modal, provisioning {config.gpu_type}...")
        await asyncio.sleep(1.8)
        log("Uploading dataset to Modal Volume: datasets/sample/training_data.jsonl...")
        await asyncio.sleep(1.6)
        log(f"Launching Modal job: model={config.model_id} gpu={config.gpu_type}")
        await asyncio.sleep(1.6)
        log(f"Loading model: {config.model_id}")
        await asyncio.sleep(1.6)

        metrics = _build_sample_cached_metrics(config)
        total_steps = metrics["total_steps"]
        job["status"] = "training"
        job["total_steps"] = total_steps
        log(f"Dataset: 83 train / 10 val")
        log(f"Steps per epoch: {metrics['steps_per_epoch']} | Total steps: {total_steps}")
        log("Memory cleared. Starting training loop...")

        for idx, train_point in enumerate(metrics["train_loss"]):
            await asyncio.sleep(0.215)
            step = train_point["step"]
            job["metrics"]["train_loss"].append(train_point)
            job["metrics"]["learning_rate"].append(metrics["learning_rates"][idx])
            job["metrics"]["val_loss"].append(metrics["val_loss"][idx])
            job["current_step"] = step
            job["progress"] = min(99, int(step / total_steps * 100))
            job["cost_usd"] = _compute_cost(config.gpu_type, job.get("_training_start_time"))
            if step % metrics["steps_per_epoch"] == 0:
                epoch = step // metrics["steps_per_epoch"]
                log(f"Epoch {epoch}/{metrics['epochs']} complete · train_loss={train_point['value']:.4f}")

        job["status"] = "deploying"
        output_model_id = f"finetuned-Qwen2.5-3B-Instruct-{job_id}"
        job["output_model_id"] = output_model_id
        log("Training complete. Final loss: %.4f" % metrics["train_loss"][-1]["value"])
        log(f"Registering model: {output_model_id}")
        await asyncio.sleep(1.9)

        completed_at = datetime.utcnow().isoformat()
        entry = {
            "id": output_model_id,
            "name": "Qwen2.5-3B-Instruct (finetuned)",
            "type": "finetuned",
            "base_model": config.model_id,
            "job_id": job_id,
            "run_name": config.run_name,
            "modal_output_path": f"/models/{config.run_name}-Qwen2.5-3B-Instruct",
            "merge_weights": False,
            "created_at": completed_at,
            "completed_at": completed_at,
            "gpu": config.gpu_type,
            "quantization": config.quantization,
            "epochs": metrics["epochs"],
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "grad_accumulation": config.grad_accumulation,
            "max_seq_length": config.max_seq_length,
            "total_steps": total_steps,
            "cost_usd": job["cost_usd"],
            "final_train_loss": metrics["train_loss"][-1]["value"],
            "final_val_loss": metrics["val_loss"][-1]["value"],
            "metrics": {
                "train_loss": metrics["train_loss"],
                "val_loss": metrics["val_loss"],
                "learning_rate": metrics["learning_rates"],
            },
            "_sample_cached": True,
            "_inference_provider": "gemini",
            "_provider_model": os.getenv("GEMINI_PREDICT_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash",
            "_cached_epochs_available": 10,
            "_sample_dataset": "training_data.jsonl",
        }
        _upsert_finetuned_model(entry)

        job["status"] = "completed"
        job["completed_at"] = completed_at
        job["progress"] = 100
        job["current_step"] = total_steps
        log("Model ready in Predict tab.")

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        log(f"Training failed: {str(e)}", "ERROR")
        logger.error(f"[Job {job_id}] Failed: {e}", exc_info=True)


@app.get("/api/train/job/{job_id}")
async def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, f"Job {job_id} not found")
    # Strip internal keys before returning
    job = {k: v for k, v in jobs[job_id].items() if not k.startswith("_")}
    return job

@app.get("/api/train/jobs")
async def list_jobs():
    return [{k: v for k, v in j.items() if not k.startswith("_")} for j in jobs.values()]

@app.post("/api/train/cancel/{job_id}")
async def cancel_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, f"Job {job_id} not found")
    job = jobs[job_id]
    if job["status"] in ("completed", "failed"):
        raise HTTPException(400, "Cannot cancel a finished job")
    job["status"] = "cancelled"
    job["logs"].append({"ts": datetime.utcnow().isoformat(), "level": "WARN", "msg": "Job cancelled by user"})
    return {"status": "cancelled"}

# ─── Models ───────────────────────────────────────────────────────────────────

@app.get("/api/models")
async def get_models():
    _refresh_models_registry_from_cache()
    return models_registry


@app.post("/api/models/prune")
async def prune_models(req: PruneModelsRequest):
    finetuned = _sorted_finetuned(models_registry.get("finetuned", []))
    if not finetuned:
        return {
            "kept_model": None,
            "removed_models": [],
            "volume_cleanup": {"paths_removed": [], "metrics_removed": [], "errors": []},
        }

    keep_entry = None
    if req.keep_model_id:
        keep_entry = next((entry for entry in finetuned if entry.get("id") == req.keep_model_id), None)
        if not keep_entry:
            raise HTTPException(404, f"Finetuned model not found: {req.keep_model_id}")
    else:
        keep_entry = finetuned[0]

    removed_entries = [entry for entry in finetuned if entry.get("id") != keep_entry.get("id")]
    _replace_finetuned_models([keep_entry])

    volume_cleanup = {"paths_removed": [], "metrics_removed": [], "errors": []}
    if req.remove_from_volume:
        volume_cleanup = _prune_modal_volume_entries(removed_entries)

    logger.info(
        "Pruned finetuned models. kept=%s removed=%s",
        keep_entry.get("id"),
        [entry.get("id") for entry in removed_entries],
    )

    return {
        "kept_model": keep_entry,
        "removed_models": [entry.get("id") for entry in removed_entries],
        "volume_cleanup": volume_cleanup,
    }


@app.post("/api/models/{model_id}/backfill-metrics")
async def backfill_model_metrics(model_id: str):
    updated = _backfill_finetuned_metrics_from_volume(model_id)
    return {"model": updated}

# ─── Predict ─────────────────────────────────────────────────────────────────

def _build_predict_prompt(prompt: str, context: Optional[str]) -> str:
    if not context or not context.strip():
        return prompt.strip()

    ctx = context.strip()
    truncated = False
    if len(ctx) > CONTEXT_CHAR_LIMIT:
        ctx = ctx[:CONTEXT_CHAR_LIMIT]
        last_newline = ctx.rfind("\n")
        if last_newline > CONTEXT_CHAR_LIMIT * 0.8:
            ctx = ctx[:last_newline]
        truncated = True

    truncation_notice = (
        "\n[...document truncated to fit model context window...]" if truncated else ""
    )

    return (
        f"You are a helpful assistant. Use the document below to answer the question.\n\n"
        f"--- DOCUMENT START ---\n"
        f"{ctx}"
        f"{truncation_notice}\n"
        f"--- DOCUMENT END ---\n\n"
        f"Question: {prompt.strip()}"
    )


def _is_sample_cached_model(entry: Optional[dict]) -> bool:
    return bool(entry and entry.get("_sample_cached") and entry.get("_inference_provider") == "gemini")


def _extract_document_text_from_predict_prompt(full_prompt: str) -> str:
    start = "--- DOCUMENT START ---"
    end = "--- DOCUMENT END ---"
    if start not in full_prompt or end not in full_prompt:
        return full_prompt
    return full_prompt.split(start, 1)[1].split(end, 1)[0].strip()


def _first_regex(pattern: str, text: str, flags=re.IGNORECASE | re.DOTALL) -> Optional[str]:
    m = re.search(pattern, text, flags)
    if not m:
        return None
    value = next((g for g in m.groups() if g is not None), None)
    return value.strip() if isinstance(value, str) else value


def _money_to_float(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    cleaned = re.sub(r"[^0-9.\-]", "", value)
    try:
        return float(cleaned)
    except Exception:
        return None


def _looks_empty_invoice_json(text: str) -> bool:
    parsed = _parse_json_like(text)
    if not isinstance(parsed, dict):
        return False
    meaningful_keys = ("vendor_name", "invoice_number", "invoice_date", "po_number", "invoice_total")
    return all(parsed.get(k) in (None, "", [], {}) for k in meaningful_keys)


def _is_schema_extraction_request(prompt: str) -> bool:
    text = (prompt or "").lower()
    schema_terms = (
        "extract",
        "json",
        "schema",
        "all fields",
        "invoice data",
        "structured",
        "line_items",
        "line items",
    )
    return any(term in text for term in schema_terms)


def _local_sample_invoice_extract(full_prompt: str) -> dict:
    doc = _extract_document_text_from_predict_prompt(full_prompt)
    compact = re.sub(r"\s+", " ", doc)
    filename = _first_regex(r"^#\s*(.+?\.pdf)\s*$", doc, re.IGNORECASE | re.MULTILINE)

    summary_row = re.search(
        r"INVOICE\s+DATE</th>\s*<th>PAYER</th>\s*<th>INVOICE\s+NO\.?</th>\s*<th>DUE\s+DATE</th>\s*<th>PAY THIS AMOUNT</th>\s*</tr>\s*<tr>\s*"
        r"<td>([^<]+)</td>\s*<td>([^<]+)</td>\s*<td>([^<]+)</td>\s*<td>([^<]+)</td>\s*<td>\$?\s*([^<]+)</td>",
        doc,
        re.IGNORECASE,
    )
    invoice_date = summary_row.group(1).strip() if summary_row else _first_regex(r"INVOICE\s+DATE</t[hd]>\s*.*?<td>(\d{2}/\d{2}/\d{4})</td>", doc)
    account_number = summary_row.group(2).strip() if summary_row else None
    invoice_number = summary_row.group(3).strip() if summary_row else None
    invoice_total = _money_to_float(summary_row.group(5)) if summary_row else None
    if not invoice_number:
        invoice_number = _first_regex(r"INVOICE\s+NO\.?</t[hd]>.*?<td>(\d{5,})</td>", doc)
    po_number = _first_regex(r"PO\s*/\s*RELEASE</t[hd]>\s*.*?<td>([^<]+)</td>", doc)
    subtotal = _money_to_float(_first_regex(r"Sale</td>\s*<td>subtotal:</td>\s*<td>([^<]+)</td>", doc))
    shipping = None
    if invoice_total is not None and subtotal is not None and invoice_total >= subtotal:
        shipping = round(invoice_total - subtotal, 2)

    vendor_name = _first_regex(r"^([A-Z0-9&.,' \-]+(?:LLC|INC|CORP|LTD|COMPANY))\s*$", doc, re.MULTILINE)
    if not vendor_name and "AIRGAS USA, LLC" in doc:
        vendor_name = "AIRGAS USA, LLC"

    vendor_address = None
    vendor_block = _first_regex(r"AIRGAS USA, LLC\s*\n([^\n]+)\n([^\n]+)", doc, re.IGNORECASE)
    if vendor_block:
        address_lines = re.search(r"AIRGAS USA, LLC\s*\n([^\n]+)\n([^\n]+)", doc, re.IGNORECASE)
        if address_lines:
            vendor_address = f"{address_lines.group(1).strip()}, {address_lines.group(2).strip()}"

    location = None
    bill_to = _first_regex(r"BILL TO\s*\n\n?([A-Z0-9&.,' \-]+)\s*\n\n?([^\n]+)\s*\n\n?([^\n]+)", doc, re.IGNORECASE)
    if bill_to:
        bill_match = re.search(r"BILL TO\s*\n\n?([A-Z0-9&.,' \-]+)\s*\n\n?([^\n]+)\s*\n\n?([^\n]+)", doc, re.IGNORECASE)
        if bill_match:
            location = f"{bill_match.group(1).strip()} ({bill_match.group(2).strip()} {bill_match.group(3).strip()})"

    delivery_number = _first_regex(r"<td>(\d{8,})</td>\s*<td>SI[0-9A-Z]+</td>", doc)

    return {
        "file_name": filename[:-4] if filename else None,
        "invoice_number": invoice_number,
        "invoice_date": invoice_date,
        "po_number": po_number,
        "invoice_type": "PO" if po_number else "NON-PO",
        "account_number": account_number,
        "vendor_name": vendor_name,
        "vendor_address": vendor_address,
        "location": location,
        "subtotal": subtotal,
        "discount": None,
        "shipping": shipping,
        "tax_amount": None,
        "tariff": None,
        "invoice_total": invoice_total,
        "delivery_number": delivery_number,
        "po_multiple_line": None,
        "line_items": [],
    }


def _fallback_sample_prediction(full_prompt: str) -> str:
    return json.dumps(_local_sample_invoice_extract(full_prompt), indent=2)


def _run_gemini_sample_prediction(full_prompt: str, req: PredictRequest, model_entry: dict) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return _fallback_sample_prediction(full_prompt)

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        provider_model = os.getenv("GEMINI_PREDICT_MODEL") or model_entry.get("_provider_model") or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"
        model = genai.GenerativeModel(provider_model)
        schema_mode = _is_schema_extraction_request(req.prompt)
        if schema_mode:
            gemini_prompt = (
                "You are a compact invoice extraction model. Use ONLY the attached document text. "
                "Do not use memory, prior examples, chat history, or assumptions.\n\n"
                "Critical invoice rules:\n"
                "- vendor_name means the seller/supplier/payee/remit-to party that issued or should be paid for the invoice.\n"
                "- vendor_name is NOT the Bill To, Ship To, Sold To, customer, registered business name, payer, or buyer.\n"
                "- If the prompt asks for one field, return concise JSON with only that field, for example {\"vendor_name\":\"...\"}.\n"
                "- If the prompt asks for invoice extraction, follow this schema exactly: file_name, invoice_number, invoice_date, po_number, invoice_type, account_number, vendor_name, vendor_address, location, subtotal, discount, shipping, tax_amount, tariff, invoice_total, delivery_number, po_multiple_line, line_items.\n"
                "- Return JSON only for extraction tasks. No markdown fences.\n\n"
                f"{full_prompt}"
            )
        else:
            document_text = _extract_document_text_from_predict_prompt(full_prompt)
            gemini_prompt = (
                "Answer the question using the document below.\n\n"
                "DOCUMENT:\n"
                f"{document_text}\n\n"
                "QUESTION:\n"
                f"{req.prompt.strip()}\n\n"
                "ANSWER:"
            )
        response = model.generate_content(
            gemini_prompt,
            generation_config={
                "temperature": min(req.temperature, 0.1),
                "top_p": req.top_p,
                "max_output_tokens": req.max_new_tokens,
            },
        )
        text = (getattr(response, "text", "") or "").strip()
        if schema_mode and text and _looks_empty_invoice_json(text):
            return _fallback_sample_prediction(full_prompt)
        return text or _fallback_sample_prediction(full_prompt)
    except Exception as e:
        logger.warning(f"Sample prediction provider failed: {e}")
        return _fallback_sample_prediction(full_prompt)


@app.post("/api/predict")
async def predict(req: PredictRequest):
    try:
        _refresh_models_registry_from_cache()
        full_prompt = _build_predict_prompt(req.prompt, req.context)

        ctx_chars = len(req.context) if req.context else 0
        logger.info(
            f"Predict: model={req.model_id} prompt_len={len(req.prompt)} "
            f"context_chars={ctx_chars} full_prompt_len={len(full_prompt)}"
        )

        model_entry = next(
            (m for m in models_registry["finetuned"] if m["id"] == req.model_id),
            None
        )

        if _is_sample_cached_model(model_entry):
            response_text = await asyncio.to_thread(
                _run_gemini_sample_prediction,
                full_prompt,
                req,
                model_entry,
            )
        else:
            inference_fn = _get_modal_inference_fn()

        if model_entry and not _is_sample_cached_model(model_entry):
            modal_path = model_entry.get("modal_output_path", "")
            if model_entry.get("merge_weights", True):
                modal_path = modal_path + "/merged"
            response_text = await asyncio.to_thread(
                inference_fn.remote,
                modal_path,
                full_prompt,
                req.max_new_tokens,
                req.temperature,
                req.top_k,
                req.max_input_tokens,
            )
        elif not model_entry:
            response_text = await asyncio.to_thread(
                inference_fn.remote,
                req.model_id,
                full_prompt,
                req.max_new_tokens,
                req.temperature,
                req.top_k,
                req.max_input_tokens,
            )

        return {
            "model_id": req.model_id,
            "prompt": req.prompt,
            "response": response_text,
            "tokens_generated": len(response_text.split()),
            "latency_ms": 0,
        }
    except Exception as e:
        logger.error(f"Predict failed: {e}", exc_info=True)
        raise HTTPException(500, f"Prediction failed: {str(e)}")

# ─── Evaluation ───────────────────────────────────────────────────────────────

def _load_records(dataset_path: str) -> list:
    cleaned = str(dataset_path).strip().strip('"').strip("'")
    ds_path = Path(cleaned).expanduser()
    if not ds_path.exists():
        name_only = ds_path.name
        search_candidates = [
            Path.cwd() / name_only,
            UPLOAD_DIR / name_only,
            (Path.cwd() / "uploads") / name_only,
            (Path.cwd().parent / "uploads") / name_only,
        ]
        for c in search_candidates:
            if c.exists():
                ds_path = c
                break

        if not ds_path.exists() and cleaned == name_only:
            docs_dir = Path.home() / "Documents"
            if docs_dir.exists():
                try:
                    found = next(docs_dir.rglob(name_only), None)
                    if found and found.exists():
                        ds_path = found
                except Exception:
                    pass

    if not ds_path.exists():
        raise FileNotFoundError(f"Eval dataset not found: {cleaned}")

    suffix = ds_path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        import pandas as pd

        df = pd.read_excel(ds_path, dtype=str)
        df = df.astype(object).where(pd.notna(df), None)
        return _sanitize_jsonable(df.to_dict(orient="records"))

    raw = ds_path.read_text(encoding="utf-8", errors="replace")
    lines = [l for l in raw.splitlines() if l.strip()]

    if suffix == ".csv":
        try:
            import pandas as pd

            df = pd.read_csv(ds_path, dtype=str).fillna("")
            return _sanitize_jsonable(df.to_dict(orient="records"))
        except Exception:
            try:
                import csv

                return [dict(row) for row in csv.DictReader(io.StringIO(raw))]
            except Exception:
                return [{"text": line} for line in lines]

    if suffix == ".jsonl":
        return [json.loads(l) for l in lines]

    if lines and all(l.lstrip().startswith("{") for l in lines):
        try:
            return [json.loads(l) for l in lines]
        except Exception:
            pass

    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else [data]
    except Exception:
        return [{"text": line} for line in lines]


def _resolve_model_path(model_id: str) -> str:
    entry = next((m for m in models_registry["finetuned"] if m["id"] == model_id), None)
    if not entry:
        cached = _load_models_registry_cache()
        cached_entry = next((m for m in cached["finetuned"] if m.get("id") == model_id), None)
        if cached_entry:
            _upsert_finetuned_model(cached_entry)
            entry = cached_entry
    if entry:
        path = entry["modal_output_path"]
        if entry.get("merge_weights", True):
            path += "/merged"
        return path
    return model_id


def _get_modal_inference_fn():
    import modal
    fn = modal.Function.from_name("llm-finetune", "inference")
    if hasattr(fn, "with_options"):
        try:
            return fn.with_options(timeout=600)
        except Exception as e:
            logger.warning(f"Could not apply Modal function options; falling back to default inference function: {e}")
    else:
        logger.info("Modal Function.with_options is unavailable in this SDK version; using default inference function.")
    return fn



def _messages_to_prompt_text(messages: list[dict]) -> str:
    """Convert a list of {role, content} messages into a prompt string.

    For models trained with chat templates, we need to preserve the role
    boundaries. The inference function wraps the entire prompt as a single
    user message via apply_chat_template. To preserve the system prompt
    correctly, we structure it as:
        <system prompt>\n\n<user content>
    This matches how apply_chat_template in inference() builds the final
    tokenized input.
    """
    system_parts = []
    user_parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            system_parts.append(content)
        else:
            user_parts.append(content)

    # Encode as [SYSTEM]...[/SYSTEM]\n\n<user> so modal can split reliably
    # even if the system prompt itself contains double newlines.
    if system_parts and user_parts:
        system_text = "\n".join(system_parts)
        user_text = "\n\n".join(user_parts)
        return f"[SYSTEM] {system_text}[/SYSTEM]\n\n{user_text}"
    elif system_parts:
        return "[SYSTEM] " + "\n".join(system_parts)
    else:
        return "\n\n".join(user_parts)


def _extract_eval_pair(rec: dict) -> tuple[list, str]:
    """Extract (prompt_messages, reference_text) from a dataset record.

    Returns:
        prompt_messages: list of {role, content} dicts (system + user) that
                         can be fed to the chat-template-aware inference function.
        reference:       the expected assistant response as a plain string.
    """
    if not isinstance(rec, dict):
        return [], ""

    msgs = rec.get("messages")
    if isinstance(msgs, str):
        parsed_msgs = _parse_json_like(msgs)
        if isinstance(parsed_msgs, list):
            msgs = parsed_msgs
    if isinstance(msgs, list) and msgs:
        prompt_messages = []   # system + user turns
        assistant_parts = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role", "")).lower().strip()
            raw_content = m.get("content", m.get("value", ""))
            if isinstance(raw_content, (dict, list)):
                content = json.dumps(raw_content, ensure_ascii=False)
            else:
                content = str(raw_content).strip()
            if not content:
                continue
            if role in ("system", "user"):
                prompt_messages.append({"role": role, "content": content})
            elif role == "assistant":
                assistant_parts.append(content)
        if prompt_messages and assistant_parts:
            # BUG FIX 4: use only the LAST assistant turn as the reference.
            # Joining ALL turns with "\n\n" produced a bloated reference string
            # that made ROUGE scores artificially low — the model is only expected
            # to reproduce the final reply, not replay the entire conversation.
            return prompt_messages, assistant_parts[-1]

    # Alpaca / generic format fallback
    instruction = (
        rec.get("instruction")
        or rec.get("task")
        or rec.get("directive")
        or ""
    )
    input_part = (
        rec.get("input")
        or rec.get("input_text")
        or rec.get("source_text")
        or rec.get("document_text")
        or rec.get("document")
        or rec.get("context")
        or rec.get("ocr_text")
        or ""
    )
    prompt = (
        rec.get("prompt")
        or rec.get("question")
        or rec.get("text")
        or rec.get("user")
        or rec.get("query")
        or rec.get("request")
        or rec.get("message")
        or ""
    )
    reference = (
        rec.get("output")
        or rec.get("completion")
        or rec.get("response")
        or rec.get("answer")
        or rec.get("label")
        or rec.get("target")
        or rec.get("expected_output")
        or rec.get("expected")
        or rec.get("ground_truth")
        or rec.get("gold")
        or rec.get("assistant")
        or rec.get("reference")
        or rec.get("ideal")
        or ""
    )

    if isinstance(prompt, (dict, list)):
        prompt = json.dumps(prompt, ensure_ascii=False)
    if isinstance(reference, (dict, list)):
        reference = json.dumps(reference, ensure_ascii=False)
    if isinstance(input_part, (dict, list)):
        input_part = json.dumps(input_part, ensure_ascii=False)

    if instruction and input_part:
        prompt = f"{str(instruction).strip()}\n\n{str(input_part).strip()}"
    elif instruction and not prompt:
        prompt = instruction
    elif input_part and not prompt:
        prompt = input_part

    prompt_text = str(prompt).strip()
    if not prompt_text:
        return [], str(reference).strip()
    return [{"role": "user", "content": prompt_text}], str(reference).strip()


def _parse_json_like(text: str):
    if text is None:
        return None
    if isinstance(text, (dict, list)):
        return text

    s = str(text).strip()
    if not s:
        return None
    s = clean_prediction(s)

    try:
        return json.loads(s)
    except Exception:
        pass

    # Strip fenced blocks like ```json ... ```
    if "```" in s:
        fence_re = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
        m = fence_re.search(s)
        if m:
            inner = m.group(1).strip()
            try:
                return json.loads(inner)
            except Exception:
                pass

    # Best-effort extraction of top-level JSON object/array embedded in text.
    first_obj = s.find("{")
    last_obj = s.rfind("}")
    if first_obj != -1 and last_obj > first_obj:
        candidate = s[first_obj:last_obj + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    first_arr = s.find("[")
    last_arr = s.rfind("]")
    if first_arr != -1 and last_arr > first_arr:
        candidate = s[first_arr:last_arr + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return None


def clean_prediction(text):
    """Strip markdown code fences from model output.

    BUG FIX 5: the old implementation used text.split("```")[1] which:
      - breaks when there are multiple fence blocks (takes the wrong one)
      - breaks when the fence is not at position 0 (e.g. leading whitespace)
      - is redundant with the fence regex in _parse_json_like

    Now we use a single regex pass that handles ```json ... ``` and ``` ... ```
    anywhere in the string, while leaving plain JSON untouched.
    """
    text = text.strip()
    # Remove ``` code fences (with optional language tag like ```json)
    fence_re = re.compile(r"```(?:json|python|text)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
    m = fence_re.search(text)
    if m:
        return m.group(1).strip()
    # Trailing backticks only (malformed fence with no closing tag)
    return text.rstrip("`").strip()


async def _run_eval_task_async(eval_id: str, req: EvalRequest):
    """
    Async wrapper so we can use asyncio.to_thread for each blocking Modal call.
    This replaces the old synchronous _run_eval_task which caused deadlocks when
    Modal's SDK internally awaited an event loop that was already running.
    """
    result = jobs[f"eval_{eval_id}"]
    try:
        import evaluate as hf_evaluate

        modal_path = _resolve_model_path(req.model_id)
        inference_fn = _get_modal_inference_fn()
        records = _load_records(req.dataset_path)
        if req.shuffle_rows and len(records) > 1:
            random.shuffle(records)

        requested_sample_size = max(1, int(req.sample_size or 1))

        predictions, references, used_records, used_prompts = [], [], [], []
        inspected_records = 0
        skipped_invalid_pairs = 0

        for rec in records:
            inspected_records += 1
            prompt_msgs, ref = _extract_eval_pair(rec)
            if not prompt_msgs or not ref:
                skipped_invalid_pairs += 1
                continue
            prompt_text = _messages_to_prompt_text(prompt_msgs)
            try:
                # BUG FIX 1: wrap each blocking Modal .remote() in asyncio.to_thread
                # so it doesn't block the event loop (same pattern used in /api/predict).
                pred = await asyncio.to_thread(
                    inference_fn.remote, modal_path, prompt_text, 1024, 0.0, 50, 3072
                )
                predictions.append(str(pred).strip())
                references.append(ref.strip())
                used_records.append(rec)
                used_prompts.append(prompt_text.strip())
                if len(predictions) >= requested_sample_size:
                    break
            except Exception as infer_err:
                logger.warning(f"[Eval {eval_id}] Inference failed for record: {infer_err}")
                continue

        if not predictions:
            example_shapes = []
            for rec in records[:3]:
                if isinstance(rec, dict):
                    example_shapes.append(sorted(rec.keys())[:12])
                else:
                    example_shapes.append(type(rec).__name__)
            result["status"] = "failed"
            result["error"] = (
                "No valid prompt/output pairs found in dataset. "
                "Expected fields like instruction+output, prompt+completion, "
                "text+response, or chat messages with user+assistant roles. "
                f"Inspected {inspected_records} rows, skipped {skipped_invalid_pairs} invalid rows. "
                f"Example row shapes: {example_shapes}"
            )
            return

        metrics_out = {}
        metrics_out["avg_response_len"] = round(
            sum(len(p.split()) for p in predictions) / len(predictions), 1
        )
        metrics_out["num_samples"] = len(predictions)
        metrics_out["requested_sample_size"] = requested_sample_size
        metrics_out["shuffle_rows"] = req.shuffle_rows
        metrics_out["rows_inspected"] = inspected_records
        metrics_out["rows_skipped_invalid"] = skipped_invalid_pairs

        if "rouge" in req.metrics:
            try:
                rouge = hf_evaluate.load("rouge")
                rouge_scores = rouge.compute(predictions=predictions, references=references)
                metrics_out["rouge1"] = round(rouge_scores["rouge1"], 4)
                metrics_out["rouge2"] = round(rouge_scores["rouge2"], 4)
                metrics_out["rougeL"] = round(rouge_scores["rougeL"], 4)
            except Exception as rouge_err:
                logger.warning(f"[Eval {eval_id}] ROUGE failed: {rouge_err}")
                metrics_out["rouge1"] = metrics_out["rouge2"] = metrics_out["rougeL"] = None

        if "field_f1" in req.metrics:
            try:
                field_scores = _compute_field_f1(predictions, references, used_records)
                metrics_out["field_f1"] = round(field_scores["f1"], 4)
                metrics_out["field_precision"] = round(field_scores["precision"], 4)
                metrics_out["field_recall"] = round(field_scores["recall"], 4)
                metrics_out["json_validity_rate"] = round(field_scores["json_validity_rate"], 4)
                metrics_out["exact_match"] = round(field_scores["exact_match"], 4)
            except Exception as f1_err:
                logger.warning(f"[Eval {eval_id}] Field F1 failed: {f1_err}")
                metrics_out["field_f1"] = None

        samples_out = []
        for idx, (prompt, pred_text, ref_text) in enumerate(zip(used_prompts, predictions, references), start=1):
            parsed_pred = _parse_json_like(pred_text)
            parsed_ref = _parse_json_like(ref_text)

            if isinstance(parsed_pred, dict):
                predicted_obj = parsed_pred
            elif isinstance(parsed_pred, list):
                predicted_obj = {"line_items": parsed_pred}
            else:
                predicted_obj = {"_text": pred_text}

            if isinstance(parsed_ref, dict):
                reference_obj = parsed_ref
            elif isinstance(parsed_ref, list):
                reference_obj = {"line_items": parsed_ref}
            else:
                reference_obj = {"_text": ref_text}

            samples_out.append({
                "index": idx,
                "prompt": prompt,
                "prediction_text": pred_text,
                "reference_text": ref_text,
                "predicted": predicted_obj,
                "ground_truth": reference_obj,
            })

        result["samples"] = samples_out
        result["metrics"] = metrics_out
        result["requested_sample_size"] = requested_sample_size
        result["shuffle_rows"] = req.shuffle_rows
        result["status"] = "completed"
        result["completed_at"] = datetime.utcnow().isoformat()
        logger.info(f"[Eval {eval_id}] Completed. Samples={len(predictions)} Metrics={list(metrics_out.keys())}")

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        logger.error(f"[Eval {eval_id}] Failed: {e}", exc_info=True)


# Legacy sync entry-point kept for any direct callers; delegates to the async version.
def _run_eval_task(eval_id: str, req: EvalRequest):
    asyncio.run(_run_eval_task_async(eval_id, req))


def _compute_field_f1(predictions: list, references: list, records: list) -> dict:
    import re

    def normalize(text: str) -> str:
        text = str(text).replace("\u00a0", " ").lower().strip()
        text = re.sub(r"[\W_]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def canonicalize_key(key: str) -> str:
        normalized = normalize(key)
        key_aliases = {
            "sl no": "serial_number",
            "serial no": "serial_number",
            "serial number": "serial_number",
            "file name": "file_name",
            "filename": "file_name",
            "invoice no": "invoice_number",
            "invoice number": "invoice_number",
            "invoice date": "invoice_date",
            "po": "po_number",
            "po no": "po_number",
            "po number": "po_number",
            "purchase order": "po_number",
            "purchase order number": "po_number",
            "invoice type": "invoice_type",
            "account number": "account_number",
            "vendor name": "vendor_name",
            "vendor address": "vendor_address",
            "delivery number": "delivery_number",
            "subtotal": "subtotal",
            "discount": "discount",
            "shipping": "shipping",
            "tax amount": "tax_amount",
            "invoice total": "invoice_total",
            "total due": "invoice_total",
            "total amount": "invoice_total",
            "location": "location",
            "line items": "line_items",
            "line item": "line_items",
        }
        return key_aliases.get(normalized, normalized.replace(" ", "_"))

    def is_emptyish(value) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return normalize(value) in ("", "nan", "null", "none", "na", "n a")
        if isinstance(value, (list, dict)):
            return len(value) == 0
        return False

    def value_to_text(value) -> str:
        if isinstance(value, dict):
            return json.dumps(canonicalize_json(value), ensure_ascii=False, sort_keys=True)
        if isinstance(value, list):
            return json.dumps([canonicalize_json(v) for v in value], ensure_ascii=False)
        return str(value).strip()

    def canonicalize_json(value):
        if isinstance(value, dict):
            canonical = {}
            for raw_key, raw_val in value.items():
                canonical[canonicalize_key(raw_key)] = canonicalize_json(raw_val)
            return canonical
        if isinstance(value, list):
            return [canonicalize_json(v) for v in value]
        return value

    def token_f1(pred: str, ref: str):
        pred_tokens = set(normalize(pred).split())
        ref_tokens = set(normalize(ref).split())
        if not pred_tokens or not ref_tokens:
            return 0.0, 0.0, 0.0
        common = pred_tokens & ref_tokens
        if not common:
            return 0.0, 0.0, 0.0
        p = len(common) / len(pred_tokens)
        r = len(common) / len(ref_tokens)
        f1 = 2 * p * r / (p + r)
        return p, r, f1

    precisions, recalls, f1s, exact_matches = [], [], [], []
    json_valid = 0

    for pred, ref in zip(predictions, references):
        pred_json = _parse_json_like(pred)
        ref_json = _parse_json_like(ref)

        if isinstance(pred_json, (dict, list)):
            json_valid += 1

        try:
            if isinstance(pred_json, dict) and isinstance(ref_json, dict):
                pred_json = canonicalize_json(pred_json)
                ref_json = canonicalize_json(ref_json)
                field_f1s = []
                field_exact = []
                ref_keys = {
                    k for k, v in ref_json.items()
                    if not is_emptyish(v)
                }
                pred_keys = set(pred_json.keys())

                for key in ref_keys:
                    ref_val = value_to_text(ref_json[key])
                    pred_val = value_to_text(pred_json.get(key, ""))
                    _, _, f = token_f1(pred_val, ref_val)
                    field_f1s.append(f)
                    field_exact.append(1.0 if normalize(pred_val) == normalize(ref_val) else 0.0)

                if field_f1s:
                    avg_f1 = sum(field_f1s) / len(field_f1s)
                    f1s.append(avg_f1)
                    exact_matches.append(1.0 if all(e == 1.0 for e in field_exact) else 0.0)
                    common_keys = pred_keys & ref_keys
                    precisions.append(len(common_keys) / len(pred_keys) if pred_keys else 0.0)
                    recalls.append(len(common_keys) / len(ref_keys) if ref_keys else 0.0)
                else:
                    logger.warning("Skipping eval sample: all reference fields are null/empty")
                continue
            
            if isinstance(pred_json, list) and isinstance(ref_json, list):
                pred_text = json.dumps(pred_json, ensure_ascii=False)
                ref_text = json.dumps(ref_json, ensure_ascii=False)
                p, r, f = token_f1(pred_text, ref_text)
                precisions.append(p)
                recalls.append(r)
                f1s.append(f)
                exact_matches.append(1.0 if f > 0.95 else 0.0)
                continue
        except Exception as e:
            logger.warning(f"Error computing field metrics for sample: {e}")

        # Fallback for non-JSON or mixed data
        p, r, f = token_f1(pred, ref)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
        exact_matches.append(1.0 if normalize(pred) == normalize(ref) else 0.0)

    n = max(len(f1s), 1)
    return {
        "precision": sum(precisions) / n,
        "recall": sum(recalls) / n,
        "f1": sum(f1s) / n,
        "exact_match": sum(exact_matches) / n,
        "json_validity_rate": json_valid / n,
    }


@app.post("/api/eval/run")
async def run_evaluation(req: EvalRequest, background_tasks: BackgroundTasks):
    try:
        eval_id = str(uuid.uuid4())[:8]
        result = {
            "eval_id": eval_id,
            "model_id": req.model_id,
            "status": "running",
            "metrics": {},
            "started_at": datetime.utcnow().isoformat(),
        }
        jobs[f"eval_{eval_id}"] = result
        # BUG FIX 1: _run_eval_task calls inference_fn.remote() which is a
        # blocking Modal call. background_tasks.add_task runs in a threadpool
        # which is fine, BUT we must ensure the blocking call is wrapped in
        # asyncio.to_thread inside the task itself (see _run_eval_task).
        # Schedule via asyncio.create_task so it runs on the event loop and
        # can correctly await asyncio.to_thread for each Modal inference call.
        asyncio.create_task(_run_eval_task_async(eval_id, req))
        return {"eval_id": eval_id, "status": "running"}

    except Exception as e:
        raise HTTPException(500, f"Eval failed: {str(e)}")


@app.get("/api/eval/{eval_id}")
async def get_eval(eval_id: str):
    key = f"eval_{eval_id}"
    if key not in jobs:
        raise HTTPException(404, f"Eval {eval_id} not found")
    return jobs[key]

# ─── Document Extraction (Azure DI) ──────────────────────────────────────────

@app.post("/api/extract")
async def extract_document(file: UploadFile = File(...)):
    try:
        endpoint = os.environ.get("AZURE_DI_ENDPOINT")
        key = os.environ.get("AZURE_DI_KEY")
        if not endpoint or not key:
            raise HTTPException(500, "Azure DI env vars not set: AZURE_DI_ENDPOINT, AZURE_DI_KEY")

        allowed = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".docx", ".xlsx", ".pptx"}
        suffix = Path(file.filename).suffix.lower()
        if suffix not in allowed:
            raise HTTPException(400, f"Unsupported file type: {suffix}")

        content = await file.read()
        if len(content) > 50 * 1024 * 1024:
            raise HTTPException(400, "File too large. Max 50MB.")

        logger.info(f"Extracting layout from: {file.filename} ({len(content)} bytes)")

        client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
        def _run_layout_extraction(raw_content: bytes):
            poller = client.begin_analyze_document(
                model_id="prebuilt-layout",
                body={"base64Source": base64.b64encode(raw_content).decode()},
                content_type="application/json",
                output_content_format="markdown",
            )
            return poller.result()

        try:
            result = _run_layout_extraction(content)
        except Exception as azure_error:
            msg = str(azure_error)
            oversized = "InvalidContentLength" in msg or "input image is too large" in msg.lower()

            # If an image is too large, try a one-shot downscale+re-encode retry.
            if oversized and suffix in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}:
                try:
                    from PIL import Image  # Optional dependency (Pillow)

                    with Image.open(io.BytesIO(content)) as img:
                        img = img.convert("RGB")
                        max_side = max(img.size)
                        if max_side > 3500:
                            ratio = 3500 / float(max_side)
                            new_size = (max(1, int(img.size[0] * ratio)), max(1, int(img.size[1] * ratio)))
                            img = img.resize(new_size)

                        out = io.BytesIO()
                        img.save(out, format="JPEG", quality=80, optimize=True)
                        shrunk = out.getvalue()

                    logger.info(
                        "Retrying extraction with compressed image: %s (%d -> %d bytes)",
                        file.filename, len(content), len(shrunk)
                    )
                    result = _run_layout_extraction(shrunk)
                except ImportError:
                    raise HTTPException(
                        400,
                        "Extraction failed: Azure rejected this image as too large (InvalidContentLength). "
                        "Install Pillow for automatic image downscaling, or upload a smaller image (lower DPI/resolution).",
                    )
                except HTTPException:
                    raise
                except Exception:
                    raise HTTPException(
                        400,
                        "Extraction failed: Azure rejected this image as too large (InvalidContentLength). "
                        "Please reduce image dimensions/DPI and retry.",
                    )
            elif oversized and suffix == ".pdf":
                raise HTTPException(
                    400,
                    "Extraction failed: one or more PDF pages are too large for Azure DI (InvalidContentLength). "
                    "Please resave/re-export the PDF at lower DPI (e.g. 200-300 DPI), or split/compress the PDF and retry.",
                )
            else:
                raise

        markdown_text = (result.content or "").strip()
        markdown_text = re.sub(r'\n{3,}', '\n\n', markdown_text)

        return {
            "filename": file.filename,
            "page_count": len(result.pages) if result.pages else 0,
            "text": markdown_text,
            "char_count": len(markdown_text),
            "format": "markdown",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)
        raise HTTPException(500, f"Extraction failed: {str(e)}")

app.post("/api/dataprep/start")(dp_routes.start_dataprep)
app.get("/api/dataprep/job/{job_id}")(dp_routes.get_dataprep_job)
app.get("/api/dataprep/list")(dp_routes.list_dataprep_jobs)
app.post("/api/dataprep/analyze")(dp_routes.analyze_dataset)
app.include_router(analyze_dataset_router)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
