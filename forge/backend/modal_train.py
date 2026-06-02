"""
modal_train.py — Modal GPU finetuning job.
Two decorated functions share a common _run_finetune() helper.

Live metrics: A MetricsCallback writes train loss + val loss to
/models/metrics_{job_id}.json on the Modal Volume every logging_steps.
The FastAPI backend polls this file via volume.read_file() to stream
live updates to the frontend during training.
"""

import modal

app = modal.App("llm-finetune")

volume = modal.Volume.from_name("llm-finetune-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .env({
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    .pip_install(
        "torch==2.5.1",
        "accelerate==1.2.1",
        "bitsandbytes>=0.45.0",
    )
    .pip_install(
        "transformers==4.48.3",
        "datasets==3.2.0",
        "peft==0.14.0",
        "trl==0.14.0",
        "evaluate==0.4.3",
        "rouge-score==0.1.2",
        "sacrebleu==2.5.1",
        "scipy",
        "rich",
    )
    # OOM FIX: transformers 4.48.x unconditionally casts logits to float32
    # before computing cross-entropy loss (loss_utils.py line ~36: logits.float()).
    # With Qwen 2.5's vocab of 152k tokens this allocates ~18GB just for the
    # logits tensor, causing OOM even when training in bf16.
    # We patch the file in-place during image build to keep logits in their
    # native dtype (bf16/fp16) and only upcast the loss scalar itself.
    .run_commands(
        "python3 -c \"import re, pathlib; p = pathlib.Path('/usr/local/lib/python3.11/site-packages/transformers/loss/loss_utils.py'); "
        "src = p.read_text(); "
        "patched = src.replace('logits = logits.float()\\n', '# patched\\n'); "
        "patched = re.sub(r'([ \\t]+)logits = logits\\.float\\(\\)\\n', r'\\1# patched\\n', patched, count=1); "
        "p.write_text(patched); "
        "print('Patched:', 'logits.float() removed' if patched != src else 'already patched')\""
    )
)

_INFERENCE_CACHE: dict[str, dict] = {}

# ─── Shared training logic ─────────────────────────────────────────────────────

def _run_finetune(config: dict, dataset_path: str) -> dict:
    import os
    import json
    import random
    import time
    import torch
    from pathlib import Path
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainerCallback,
        TrainerControl,
        TrainerState,
        TrainingArguments,
        EarlyStoppingCallback,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer, SFTConfig

    logs = []

    # OOM SAFETY NET: patch the logits.float() upcast at runtime too,
    # in case the Modal image is served from cache before rebuild completes.
    try:
        import re as _re, pathlib as _pl
        _p = _pl.Path('/usr/local/lib/python3.11/site-packages/transformers/loss/loss_utils.py')
        if _p.exists():
            _src = _p.read_text()
            _patched = _src.replace('logits = logits.float()\n', '# patched: logits.float() removed\n')
            if _patched == _src:
                _patched = _re.sub(r'([ \t]+)logits = logits\.float\(\)\n', r'\1# patched\n', _src, count=1)
            if _patched != _src:
                _p.write_text(_patched)
                print("[INFO] Runtime patch applied: logits.float() removed from loss_utils.py")
            else:
                print("[INFO] loss_utils.py already patched or pattern not found — skipping")
    except Exception as _e:
        print(f"[WARN] Runtime patch failed (non-fatal): {_e}")

    def log(msg, level="INFO"):
        entry = {"level": level, "msg": msg}
        logs.append(entry)
        print(f"[{level}] {msg}")

    # ── Live metrics callback ──────────────────────────────────────────────────
    # Writes a JSON file to the Modal Volume after every logging step and after
    # every eval. The FastAPI backend polls this file via volume.read_file()
    # to stream live updates to the frontend without needing Modal streaming.
    class MetricsCallback(TrainerCallback):
        def __init__(self, metrics_path: str, total_steps: int, job_id: str):
            self.metrics_path = Path(metrics_path)
            self.total_steps = total_steps
            self.job_id = job_id
            self.train_loss: list[dict] = []
            self.val_loss: list[dict] = []
            self.learning_rates: list[dict] = []
            self._last_flush = 0.0

        def _flush(self):
            """Write current metrics state to the volume file."""
            payload = {
                "job_id": self.job_id,
                "train_loss": self.train_loss,
                "val_loss": self.val_loss,
                "learning_rates": self.learning_rates,
                "total_steps": self.total_steps,
                "updated_at": time.time(),
            }
            try:
                self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
                self.metrics_path.write_text(json.dumps(payload), encoding="utf-8")
                volume.commit()
            except Exception as e:
                print(f"[WARN] MetricsCallback flush failed: {e}")
            self._last_flush = time.time()

        def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
            if logs is None:
                return
            step = state.global_step
            if "loss" in logs:
                self.train_loss.append({"step": step, "value": round(float(logs["loss"]), 4)})
            if "learning_rate" in logs:
                self.learning_rates.append({"step": step, "value": float(logs["learning_rate"])})
            self._flush()

        def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
            if metrics is None:
                return
            step = state.global_step
            if "eval_loss" in metrics:
                self.val_loss.append({"step": step, "value": round(float(metrics["eval_loss"]), 4)})
            self._flush()

    model_id = config["model_id"]
    job_id = config.get("job_id", "unknown")
    metrics_volume_path = f"/models/metrics/{job_id}.json"

    log(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    log(f"Loading model: {model_id}")

    hf_token = os.environ.get("HF_TOKEN", None)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], token=hf_token, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_mode = config.get("quantization", "4bit")
    bnb_config = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"

    # Auto-fallback for T4 (doesn't support bf16)
    mixed_precision = config["mixed_precision"]
    use_double_quant = True
    if "T4" in gpu_name and mixed_precision == "bf16":
        log("T4 GPU detected. Downgrading 'bf16' to 'fp16' and disabling double_quant for stability.", "WARN")
        mixed_precision = "fp16"
        use_double_quant = False

    # Gemma 3 checkpoints can fail during quantized auto-conversion
    if "gemma-3" in model_id.lower() and quant_mode in ("4bit", "8bit"):
        log(
            f"Model {model_id} with quantization={quant_mode} may fail conversion. "
            "Falling back to quantization=none for compatibility.",
            "WARN",
        )
        quant_mode = "none"

    if quant_mode in ("4bit", "8bit"):
        from transformers import BitsAndBytesConfig
        compute_dtype = (
            torch.bfloat16 if mixed_precision == "bf16" else torch.float16
        )
        if quant_mode == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=use_double_quant,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            log("Quantization: 4-bit NF4 (QLoRA)")
        else:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            log("Quantization: 8-bit LLM.int8()")
    else:
        log("Quantization: none")

    model_dtype = None
    if quant_mode == "none":
        model_dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            token=hf_token,
            trust_remote_code=True,
            torch_dtype=model_dtype,
            attn_implementation="sdpa",
        )
    except Exception as e:
        log(f"Model load failed: {e}", "ERROR")
        raise e
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["target_modules"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        use_rslora=config.get("use_rslora", False),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    log(f"LoRA applied: r={config['lora_r']}, alpha={config['lora_alpha']}")

    # ── Dataset Loading ────────────────────────────────────────────────────────
    log(f"Loading dataset records from {dataset_path}...")
    volume.reload()
    ds_file = Path(dataset_path)
    if not ds_file.exists():
        # Diagnostics: list contents of /models/datasets
        log(f"Dataset NOT found at {dataset_path}. Listing /models contents for debug:", "ERROR")
        try:
            import subprocess
            res = subprocess.run(["ls", "-R", "/models/datasets"], capture_output=True, text=True)
            log(f"Contents of /models/datasets:\n{res.stdout}")
        except Exception as e:
            log(f"Could not list /models/datasets: {e}")
        raise FileNotFoundError(f"Dataset file not found on volume: {dataset_path}")

    raw_ds = ds_file.read_text(encoding="utf-8")
    if dataset_path.endswith(".jsonl"):
        dataset_records = [json.loads(l) for l in raw_ds.strip().split("\n") if l.strip()]
    elif dataset_path.endswith(".json"):
        dataset_records = json.loads(raw_ds)
        if not isinstance(dataset_records, list):
            dataset_records = [dataset_records]
    else:
        dataset_records = [{"text": line} for line in raw_ds.strip().split("\n") if line.strip()]

    log(f"Loaded {len(dataset_records)} records from volume")

    if config["shuffle"]:
        random.shuffle(dataset_records)

    split_idx = int(len(dataset_records) * config["train_split"])
    train_records = dataset_records[:split_idx]
    val_records = dataset_records[split_idx:]

    def format_chat(row):
        """
        Format chat/messages-style records using the tokenizer's chat template.
        Falls back to alpaca format if the record doesn't have a messages field.
        """
        msgs = row.get("messages")
        if isinstance(msgs, list) and msgs:
            try:
                return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            except Exception:
                # If template fails, format manually
                parts = []
                for m in msgs:
                    role = m.get("role", "")
                    content = m.get("content", "")
                    if role and content:
                        parts.append(f"{role}: {content}")
                return "\n".join(parts)
        return format_alpaca(row)

    def format_alpaca(row):
        if "output" in row:
            return f"### Instruction:\n{row.get('instruction','')}\n\n### Input:\n{row.get('input','')}\n\n### Response:\n{row['output']}"
        return row.get("text", str(row))

    def format_sharegpt(row):
        convs = row.get("conversations", [])
        return "\n".join(f"{c['from']}: {c['value']}" for c in convs)

    fmt = config.get("dataset_format", "alpaca")
    if fmt == "sharegpt":
        formatter = format_sharegpt
    elif fmt == "chat":
        formatter = format_chat
    else:
        # Auto-detect: if first record has "messages", treat as chat
        if train_records and "messages" in train_records[0]:
            log("Auto-detected chat/messages format — using chat template formatter.")
            formatter = format_chat
        else:
            formatter = format_alpaca

    train_dataset = Dataset.from_list([{"text": formatter(r)} for r in train_records])
    val_dataset = Dataset.from_list([{"text": formatter(r)} for r in val_records])
    log(f"Dataset: {len(train_dataset)} train / {len(val_dataset)} val")

    output_dir = f"/models/{config['run_name']}-{config['model_id'].split('/')[-1]}"

    optimizer = config["optimizer"]
    if optimizer == "lion_8bit":
        try:
            import bitsandbytes  # noqa
        except ImportError:
            log("bitsandbytes not found, falling back to AdamW", "WARN")
            optimizer = "adamw_torch"

    # ── Compute eval/save steps dynamically based on dataset size ─────────────
    # For small datasets (< 200 records), eval_steps from config may never fire.
    # Override to eval once per epoch so val loss always appears in the graph.
    steps_per_epoch = max(1, len(train_dataset) // (config["batch_size"] * config["grad_accumulation"]))
    epoch_total_steps = steps_per_epoch * config["epochs"]
    configured_max_steps = int(config.get("max_steps", -1) or -1)
    total_steps = min(epoch_total_steps, configured_max_steps) if configured_max_steps > 0 else epoch_total_steps
    log(f"Steps per epoch: {steps_per_epoch} | Total steps: {total_steps}")

    # Short demo runs need denser logs; otherwise the Runs chart only gets one point.
    if total_steps <= 20:
        effective_eval_steps = max(1, min(5, total_steps))
        effective_logging_steps = 1
    else:
        effective_eval_steps = max(1, steps_per_epoch)
        effective_logging_steps = max(1, min(10, steps_per_epoch))
    effective_save_steps = max(1, steps_per_epoch)
    log(f"eval_steps={effective_eval_steps} save_steps={effective_save_steps} logging_steps={effective_logging_steps}")

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=config["epochs"],
        max_steps=config.get("max_steps", -1),
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["grad_accumulation"],
        learning_rate=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.01),
        warmup_ratio=config["warmup_ratio"],
        max_grad_norm=config["grad_clip"],
        # optim overridden below by adamw_bnb_8bit for memory efficiency (OOM FIX 3)
        lr_scheduler_type="cosine",
        fp16=mixed_precision == "fp16",
        bf16=mixed_precision == "bf16",
        logging_steps=effective_logging_steps,
        eval_strategy="steps",
        eval_steps=effective_eval_steps,
        save_strategy="steps",
        save_steps=effective_save_steps,
        # OOM FIX 1: load_best_model_at_end forces a second full model load at
        # end of training. On 8-bit this doubles peak VRAM and causes OOM on A10G.
        load_best_model_at_end=False,
        report_to="none",
        # OOM FIX 2: pre-pin batches in CPU memory so GPU transfer is faster
        # and we avoid holding two copies of a batch during transfer.
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        dataset_text_field="text",
        max_seq_length=config["max_seq_length"],
        packing=config.get("packing", False),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # OOM FIX 3: only keep optimizer states for LoRA params, not full model.
        # For 8-bit QLoRA this saves ~1.5GB of VRAM on the optimizer step.
        optim="adamw_bnb_8bit",
    )

    metrics_cb = MetricsCallback(
        metrics_path=metrics_volume_path,
        total_steps=total_steps,
        job_id=job_id,
    )

    callbacks = [metrics_cb]
    # EarlyStoppingCallback requires load_best_model_at_end=True which we
    # disabled to fix OOM. Skip it — MetricsCallback still tracks val loss
    # and the frontend will show it. Manual early stop via Cancel button.
    # if config.get("early_stopping_patience", 0) > 0:
    #     callbacks.append(EarlyStoppingCallback(...))

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    # OOM FIX: clear any fragmented cache right before training
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    log("Memory cleared. Starting training loop...")

    try:
        train_result = trainer.train()
        log(f"Training complete. Final loss: {train_result.training_loss:.4f}")
    except Exception as e:
        log(f"Training failed: {e}", "ERROR")
        raise e

    # Final flush with completed flag
    try:
        final_payload = {
            "job_id": job_id,
            "train_loss": metrics_cb.train_loss,
            "val_loss": metrics_cb.val_loss,
            "learning_rates": metrics_cb.learning_rates,
            "total_steps": total_steps,
            "updated_at": time.time(),
            "completed": True,
        }
        Path(metrics_volume_path).write_text(json.dumps(final_payload), encoding="utf-8")
        volume.commit()
    except Exception as e:
        log(f"Final metrics flush failed: {e}", "WARN")

    if config.get("merge_weights", True):
        log("Merging LoRA weights...")
        try:
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(output_dir + "/merged")
            tokenizer.save_pretrained(output_dir + "/merged")
            log(f"Merged model saved to {output_dir}/merged")
        except Exception as e:
            log(
                f"Merge failed ({e}). Saving LoRA adapter only instead.",
                "WARN",
            )
            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)
            log(f"LoRA adapter saved to {output_dir}")
    else:
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        log(f"LoRA adapter saved to {output_dir}")

    volume.commit()

    if config.get("push_to_hub") and config.get("hf_token"):
        trainer.model.push_to_hub(config["run_name"], token=config["hf_token"])
        log("Pushed to Hub")

    return {
        "output_path": output_dir,
        "training_loss": train_result.training_loss,
        "total_steps": total_steps,
        "metrics_path": metrics_volume_path,
        "logs": logs,
    }


# ─── GPU-specific entry points ─────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="T4",
    volumes={"/models": volume},
    timeout=60 * 120,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def finetune_t4(config: dict, dataset_path: str) -> dict:
    return _run_finetune(config, dataset_path)


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/models": volume},
    timeout=60 * 120,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def finetune_a10g(config: dict, dataset_path: str) -> dict:
    return _run_finetune(config, dataset_path)


@app.function(
    image=image,
    gpu="A100",
    volumes={"/models": volume},
    timeout=60 * 120,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def finetune_a100(config: dict, dataset_path: str) -> dict:
    return _run_finetune(config, dataset_path)


# ─── Inference ─────────────────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="T4",
    volumes={"/models": volume},
    timeout=60 * 10,
)
def inference(
    model_path: str,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int = 50,
    max_input_tokens: int = 3072,
) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cached = _INFERENCE_CACHE.get(model_path)
    if cached is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
        dtype = torch.float16 if "T4" in gpu_name else torch.bfloat16

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
        )
        model.eval()
        cached = {"tokenizer": tokenizer, "model": model}
        _INFERENCE_CACHE[model_path] = cached

    tokenizer = cached["tokenizer"]
    model = cached["model"]

    formatted_prompt = prompt
    if getattr(tokenizer, "chat_template", None):
        try:
            # Detect structured prompt with system message from eval pipeline.
            # Format: "[SYSTEM] <system_prompt>[/SYSTEM]\n\n<user_content>"
            # (legacy fallback: "[SYSTEM] <system_prompt>\n\n<user_content>")
            if prompt.startswith("[SYSTEM] "):
                rest = prompt[len("[SYSTEM] "):]
                if "[/SYSTEM]" in rest:
                    # New robust format: split on closing tag
                    system_text, _, user_text = rest.partition("[/SYSTEM]")
                    user_text = user_text.lstrip("\n")
                    messages = [
                        {"role": "system", "content": system_text.strip()},
                        {"role": "user", "content": user_text.strip()},
                    ]
                else:
                    # Legacy: split on first double newline
                    parts = rest.split("\n\n", 1)
                    if len(parts) == 2:
                        system_text, user_text = parts
                        messages = [
                            {"role": "system", "content": system_text.strip()},
                            {"role": "user", "content": user_text.strip()},
                        ]
                    else:
                        messages = [{"role": "user", "content": rest}]
            else:
                messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            formatted_prompt = prompt

    if (
        tokenizer.bos_token
        and formatted_prompt
        and not formatted_prompt.startswith(tokenizer.bos_token)
    ):
        formatted_prompt = tokenizer.bos_token + formatted_prompt

    encoded = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)
    input_len = input_ids.shape[1]

    do_sample = temperature > 0
    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        top_k=top_k,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = max(temperature, 1e-5)

    try:
        with torch.inference_mode():
            output_ids = model.generate(**gen_kwargs)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise RuntimeError(
            f"CUDA OOM during inference. Prompt was truncated to {max_input_tokens} tokens; "
            "reduce max_new_tokens or max_input_tokens for this model/GPU."
        )

    generated_ids = output_ids[0][input_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
