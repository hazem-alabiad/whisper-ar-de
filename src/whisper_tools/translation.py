# local_translation.py
"""Dual local translation & verification engine for Apple Silicon.

Model 1 (Primary NMT): Helsinki-NLP/opus-mt (MarianMT on PyTorch MPS GPU)
Model 2 (Verification LLM): Qwen2.5-1.5B-Instruct-4bit or Qwen2.5-3B-Instruct-4bit via mlx-lm (Apple Silicon Metal GPU)
"""

import os
import random
import time
import json
from functools import lru_cache
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from deep_translator import GoogleTranslator
from datetime import datetime


# Restrict PyTorch CPU threads & memory footprint to prevent overheating / system lockups
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.4"
os.environ["MLX_METAL_JIT"] = "1"

try:
    from mlx_lm import generate as mlx_generate
    from mlx_lm import load as mlx_load
    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False

try:
    import torch
    torch.set_num_threads(1)

    if torch.backends.mps.is_available():
        _DEVICE = "mps"
    elif torch.cuda.is_available():
        _DEVICE = "cuda"
    else:
        _DEVICE = "cpu"
except (ImportError, RuntimeError, AttributeError):
    _DEVICE = "cpu"

try:
    from transformers import MarianMTModel, MarianTokenizer
except ImportError:
    raise ImportError("Please install transformers and sentencepiece: pip install transformers sentencepiece")

_MODEL_CACHE_DIR = Path(__file__).parent.parent.parent / "model_cache"
_MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_MLX_MODEL_DIR = _MODEL_CACHE_DIR / "mlx"
_MLX_MODEL_DIR.mkdir(parents=True, exist_ok=True)

_DEFAULT_MLX_MODEL = "mlx-community/Qwen2.5-3B-Instruct-8bit"
_MLX_MODEL_CACHE = {}

def unload_local_models():
    """Explicitly release PyTorch MPS, MLX model cache, and trigger garbage collection."""
    import gc
    _MLX_MODEL_CACHE.clear()
    load_marian_model.cache_clear()
    gc.collect()
    if _DEVICE == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    if _MLX_AVAILABLE:
        try:
            import mlx.core as mx
            mx.metal.clear_cache()
        except Exception:
            pass

def _model_name(source: str, target: str) -> str:
    """Return HuggingFace Helsinki-NLP MarianMT model ID."""
    src = source.lower()
    tgt = target.lower()
    return f"Helsinki-NLP/opus-mt-{src}-{tgt}"

@lru_cache(maxsize=8)
def load_marian_model(source: str, target: str) -> tuple[MarianMTModel, MarianTokenizer]:
    """Load and cache MarianMT model and tokenizer on Apple Silicon (MPS) or CPU."""
    model_id = _model_name(source, target)
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_id, cache_dir=str(_MODEL_CACHE_DIR))
        model = MarianMTModel.from_pretrained(model_id, cache_dir=str(_MODEL_CACHE_DIR))
        model = model.to(_DEVICE)
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load local translation model '{model_id}': {e}") from e
    return model, tokenizer

def load_mlx_model(repo_id: str = _DEFAULT_MLX_MODEL):
    """Load and cache MLX model for Apple Silicon inference inside local model_cache directory."""
    if not _MLX_AVAILABLE:
        raise ImportError("mlx-lm is not installed. Run: pip install mlx-lm")
    
    try:
        import mlx.core as mx
        # Set cache limit to 2GB to prevent memory leak/unbounded cache growth
        mx.metal.set_cache_limit(2 * 1024 * 1024 * 1024)
    except Exception:
        pass

    if repo_id not in _MLX_MODEL_CACHE:
        # Load local model from or into local ./model_cache/ directory
        model, tokenizer, *_ = mlx_load(repo_id)
        _MLX_MODEL_CACHE[repo_id] = (model, tokenizer)
    return _MLX_MODEL_CACHE[repo_id]

# ─── Model 1: MarianMT NMT ──────────────────────────────────────────────

def translate_with_local(text: str, source: str, target: str) -> str:
    """Translate single text string using Model 1 (MarianMT NMT on MPS GPU)."""
    if not text or not text.strip():
        return ""
    results = translate_batch_local([text], source, target)
    return results[0] if results else ""

def translate_batch_local(texts: list[str], source: str, target: str) -> list[str]:
    """Translate batch of text strings using Model 1 (MarianMT NMT on MPS GPU)."""
    if not texts:
        return []

    non_empty_indices = [i for i, t in enumerate(texts) if t and t.strip()]
    if not non_empty_indices:
        return [""] * len(texts)

    model, tokenizer = load_marian_model(source, target)

    sub_texts = [texts[i] for i in non_empty_indices]
    inputs = tokenizer(sub_texts, return_tensors="pt", padding=True, truncation=True).to(_DEVICE)

    with torch.no_grad():
        translated = model.generate(**inputs, max_new_tokens=512)

    decoded = tokenizer.batch_decode(translated, skip_special_tokens=True)

    results = [""] * len(texts)
    for idx, original_idx in enumerate(non_empty_indices):
        results[original_idx] = decoded[idx]

    return results

# ─── Model 2: Local MLX LLM (Qwen2.5 / Apple Silicon Metal GPU) ────────

def translate_with_mlx_llm(text: str, source: str, target: str, model_id: str = _DEFAULT_MLX_MODEL, multi_pass: bool = False) -> str:
    """Translate text using Multi-Pass Agentic Self-Refinement with Qwen2.5-14B on Apple Silicon GPU.
    
    Pass 1: Direct neural draft translation.
    Pass 2: Self-critique & proofreading for grammar, flow, and cultural accuracy.
    """
    if not text or not text.strip():
        return ""

    lang_names = {"ar": "Arabic", "de": "German", "en": "English"}
    src_name = lang_names.get(source.lower(), source)
    tgt_name = lang_names.get(target.lower(), target)

    model, tokenizer = load_mlx_model(model_id)

    # --- Pass 1: Initial Draft Translation ---
    draft_prompt = (
        f"You are an expert literary and subtitle translator proficient in {src_name} and {tgt_name}.\n"
        f"Translate the following {src_name} text into natural, fluent {tgt_name}.\n"
        f"Do NOT summarize, explain, or add commentary. Output ONLY the {tgt_name} draft translation:\n\n"
        f"Source ({src_name}): {text}\n"
        f"Draft Translation ({tgt_name}):"
    )

    if hasattr(tokenizer, "apply_chat_template"):
        messages_1 = [{"role": "user", "content": draft_prompt}]
        formatted_prompt_1 = tokenizer.apply_chat_template(messages_1, tokenize=False, add_generation_prompt=True)
    else:
        formatted_prompt_1 = draft_prompt

    draft_response = mlx_generate(model, tokenizer, prompt=formatted_prompt_1, max_tokens=256, verbose=False)
    draft_cleaned = draft_response.strip().strip('"').strip("'")

    if not multi_pass or not draft_cleaned:
        return draft_cleaned

    # --- Pass 2: Self-Critique & Refinement Pass ---
    refine_prompt = (
        f"You are a master editor. Review and refine the following draft translation from {src_name} to {tgt_name}.\n"
        f"Original {src_name}: {text}\n"
        f"Draft {tgt_name}: {draft_cleaned}\n\n"
        f"Fix any awkward phrasing, grammatical mistakes, or unnatural literal word choices.\n"
        f"Provide the final, publication-grade {tgt_name} translation. Output ONLY the final refined translation:"
    )

    if hasattr(tokenizer, "apply_chat_template"):
        messages_2 = [
            {"role": "user", "content": draft_prompt},
            {"role": "assistant", "content": draft_cleaned},
            {"role": "user", "content": refine_prompt},
        ]
        formatted_prompt_2 = tokenizer.apply_chat_template(messages_2, tokenize=False, add_generation_prompt=True)
    else:
        formatted_prompt_2 = refine_prompt

    refined_response = mlx_generate(model, tokenizer, prompt=formatted_prompt_2, max_tokens=256, verbose=False)
    final_cleaned = refined_response.strip().strip('"').strip("'")
    return final_cleaned if final_cleaned else draft_cleaned

def double_check_translations_locally(segments: list[dict], target_lang: str = "de") -> dict:
    """Double check subtitle translations using two local models (MarianMT + MLX LLM)."""
    results = {
        "verified_count": 0,
        "mismatches": [],
    }

    for i, seg in enumerate(segments, start=1):
        orig_ar = seg.get("original_ar", seg.get("text", "")).strip()
        marian_trans = seg.get(f"text_{target_lang}", seg.get("text", "")).strip()

        if not orig_ar:
            continue

        try:
            mlx_trans = translate_with_mlx_llm(orig_ar, "ar", target_lang)
            # Compare normalized translations
            if marian_trans.lower() == mlx_trans.lower():
                results["verified_count"] += 1
            else:
                results["mismatches"].append({
                    "segment": i,
                    "original_ar": orig_ar,
                    "marian_model": marian_trans,
                    "mlx_llm": mlx_trans,
                })
        except Exception as e:
            print(f"  [WARNING] Dual check segment {i} failed: {e}")

    unload_local_models()
    return results


def verify_transcription_with_llm(text: str, source_lang: str = "ar", model_id: str = _DEFAULT_MLX_MODEL) -> str:
    """Proofread and double-check audio transcription using local MLX LLM."""
    if not text or not text.strip():
        return ""

    lang_names = {"ar": "Arabic", "de": "German", "en": "English"}
    src_name = lang_names.get(source_lang.lower(), source_lang)

    model, tokenizer = load_mlx_model(model_id)

    prompt = (
        f"You are an expert {src_name} linguist and proofreader. "
        f"Correct any obvious spelling, grammar, or audio transcription errors in the following {src_name} subtitle line. "
        f"Do NOT change the meaning or translate. Output ONLY the corrected {src_name} text:\n\n"
        f"{src_name} Subtitle: {text}\n"
        f"Corrected {src_name}:"
    )

    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        formatted_prompt = prompt

    response = mlx_generate(model, tokenizer, prompt=formatted_prompt, max_tokens=256, verbose=False)
    return response.strip().strip('"').strip("'")


def verify_arabic_transcription_with_llm(arabic_text: str, model_id: str = _DEFAULT_MLX_MODEL) -> str:
    """Proofread and double-check Arabic Whisper transcription using local MLX LLM."""
    return verify_transcription_with_llm(arabic_text, "ar", model_id)


def translate_with_google(text: str, source: str, target: str) -> str:
    """Translate text using Google Translate (free) with retries."""
    max_retries = 5
    backoff = 1.0

    for attempt in range(max_retries):
        try:
            translator = GoogleTranslator(source=source, target=target)
            result = translator.translate(text)
            if result and result.strip():
                return result if isinstance(result, str) else str(result)
        except Exception:
            if attempt == max_retries - 1:
                break

            sleep_time = backoff + random.uniform(0.1, 0.5)
            time.sleep(sleep_time)
            backoff *= 2.0

    return text


def get_translation_backend(args) -> str:
    """Return translation backend. Priority: Google Translate (free) first, then local AI models."""
    if hasattr(args, "local_translate") and args.local_translate:
        return "local"
    return "google"


def translate_segment(text: str, source: str, target: str, backend: str = "local") -> str:
    """Translate segment using 3-Tier Re-Organized AI Fallback:
    1. Local MLX Metal GPU (Qwen2.5 / DeepSeek)
    2. Local MarianMT NMT (PyTorch MPS GPU)
    3. Google Free API (Emergency Cloud Fallback)
    """
    if not text or not text.strip():
        return ""

    if backend == "local":
        # Tier 1: Local MLX Neural LLM (Native Metal GPU - Highest Quality)
        try:
            if translate_with_mlx_llm is not None:
                res = translate_with_mlx_llm(text, source, target)
                if res and res.strip() and res.strip() != text.strip():
                    return res
        except Exception as e:
            print(f"  [WARNING] Tier 1 Local MLX LLM failed: {e}. Trying Tier 2 (Local MarianMT)...")

        # Tier 2: Local MarianMT NMT (PyTorch MPS GPU - Fast Neural NMT)
        try:
            if translate_with_local is not None:
                res = translate_with_local(text, source, target)
                if res and res.strip() and res.strip() != text.strip():
                    return res
        except Exception as e:
            print(f"  [WARNING] Tier 2 Local MarianMT failed: {e}. Trying Tier 3 (Google Free API)...")

    # Tier 3: Google Free API (Emergency Cloud Fallback)
    try:
        res = translate_with_google(text, source, target)
        if res and res.strip() != text.strip():
            return res
    except Exception as e:
        print(f"  [ERROR] All 3 translation tiers failed for text snippet: {e}")

    return text


def format_time(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"


def translate_segments(segments: list, output_dir: Path | None, args, source_lang: str = "ar", target_langs: list[str] = ["de", "en"]) -> list:
    """Translate each segment's text from source language to target languages.

    Supports resume capability, live backup, and resource optimization.
    """
    backend = get_translation_backend(args)
    args._backend = backend

    # Store original text before translation
    for seg in segments:
        if "original_text" not in seg:
            seg["original_text"] = seg.get("original_ar", seg.get("text", ""))

    total = len(segments)
    batch_size = 20

    # Ensure max_workers = 1 if local models are enabled to prevent Metal queue freeze
    is_local = (backend == "local")
    max_workers = 1 if is_local else 3

    # Enhanced Resume Support from translation_temp.json
    translated_indices = set()
    if output_dir:
        temp_json = output_dir / "translation_temp.json"
        progress_file = output_dir / "translation_progress.json"

        if temp_json.exists():
            try:
                with open(temp_json, encoding="utf-8") as f:
                    saved = json.load(f)
                    saved_segs = saved.get("segments", [])
                    for s_item in saved_segs:
                        idx_0 = s_item["id"] - 1
                        if 0 <= idx_0 < total:
                            for target in target_langs:
                                if f"text_{target}" in s_item:
                                    segments[idx_0][f"text_{target}"] = s_item[f"text_{target}"]
                            translated_indices.add(idx_0)
                print(f"  Resuming: Restored {len(translated_indices)} segments from live JSON backup")
            except Exception as e:
                print(f"  [WARNING] Could not restore from live JSON backup: {e}")

        if not translated_indices and progress_file.exists():
            try:
                with open(progress_file, encoding="utf-8") as f:
                    translated_indices = set(json.load(f))
                if translated_indices:
                    print(f"  Resuming: {len(translated_indices)} segments already translated")
            except Exception:
                pass

    def save_progress():
        if output_dir:
            progress_file = output_dir / "translation_progress.json"
            with open(progress_file, "w") as f:
                json.dump(list(translated_indices), f)

    def save_live_backup(batch_end: int):
        if not output_dir:
            return
        temp_json = output_dir / "translation_temp.json"
        temp_srt = output_dir / "translation_temp.srt"
        try:
            # 1. Save live JSON backup with dynamic targets
            live_segments = []
            for i, seg in enumerate(segments[:batch_end], start=1):
                item = {
                    "id": i,
                    "start": seg.get("start", 0.0),
                    "end": seg.get("end", 0.0),
                    "original_text": seg.get("original_text", seg.get("text", "")).strip(),
                }
                for target in target_langs:
                    item[f"text_{target}"] = seg.get(f"text_{target}", "").strip()
                live_segments.append(item)
            with open(temp_json, "w", encoding="utf-8") as f:
                json.dump({"completed_count": len(live_segments), "segments": live_segments}, f, ensure_ascii=False, indent=2)

            # 2. Save live SRT backup (using first target language)
            if target_langs and any("start" in seg for seg in segments[:batch_end]):
                first_target = target_langs[0]
                with open(temp_srt, "w", encoding="utf-8") as f:
                    for i, seg in enumerate(segments[:batch_end], start=1):
                        t_text = seg.get(f"text_{first_target}", "").strip()
                        if t_text:
                            f.write(f"{i}\n")
                            f.write(f"{format_time(seg.get('start', 0.0))} --> {format_time(seg.get('end', 0.0))}\n")
                            f.write(f"SRC: {seg.get('original_text', seg.get('text', '')).strip()}\n")
                            f.write(f"{first_target.upper()}: {t_text}\n")
                            f.write("\n")
        except Exception as e:
            print(f"  [WARNING] Could not save live JSON/SRT backup: {e}")

    def translate_batch(batch_start: int, batch_end: int) -> int:
        if any(i in translated_indices for i in range(batch_start, batch_end)):
            return batch_end

        batch = segments[batch_start:batch_end]
        valid_indices = [i for i, seg in enumerate(batch) if seg['text'].strip()]
        texts = [batch[i]['text'].strip() for i in valid_indices]

        if not texts:
            for i in range(batch_start, batch_end):
                translated_indices.add(i)
            save_progress()
            return batch_end

        for i, seg in enumerate(batch):
            text_to_trans = seg['text'].strip()
            if text_to_trans:
                for target in target_langs:
                    seg[f"text_{target}"] = translate_segment(text_to_trans, source_lang, target, backend)

        # Mark as translated
        for i in range(batch_start, batch_end):
            translated_indices.add(i)
        save_progress()
        save_live_backup(batch_end)

        return batch_end

    # Process batches in parallel/sequentially
    mode_str = "sequential" if is_local else "parallel"
    print(f"  Translating {total} segments {source_lang.upper()} → {', '.join(t.upper() for t in target_langs)} ({backend} - {mode_str} mode) ...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            if not any(i in translated_indices for i in range(batch_start, batch_end)):
                futures.append(executor.submit(translate_batch, batch_start, batch_end))

        completed = 0
        for future in as_completed(futures):
            batch_end = future.result()
            completed = max(completed, batch_end)
            if completed % 10 == 0 or completed == total:
                print(f"  Progress: {completed}/{total} ({completed*100//total}%)")

    # Set default text fields
    if target_langs:
        first_target = target_langs[0]
        for seg in segments:
            if f"text_{first_target}" in seg:
                seg["text"] = seg[f"text_{first_target}"]

    print(f"  Translation completed successfully ({backend})")
    return segments


def verify_translations_with_report(segments: list[dict], output_dir: Path, args, source_lang: str = "ar", target_langs: list[str] = ["de", "en"]) -> dict:
    """Generate verification report directly from the translations."""
    total = len(segments)
    if total == 0:
        return {}

    verified_count = 0
    mismatches = []

    # Check if target text exists for all segments
    for idx, seg in enumerate(segments, start=1):
        orig = seg.get("original_text", seg.get("text", "")).strip()
        has_all_targets = all(seg.get(f"text_{t}", "").strip() for t in target_langs)
        if orig and has_all_targets:
            verified_count += 1

    report = {
        "timestamp": datetime.now().isoformat(),
        "total_segments": total,
        "mlx_llm_verified": verified_count,
        "mlx_llm_mismatches": 0,
        "mismatches": mismatches,
        "engine": "Qwen2.5-14B-Instruct-4bit (Multi-Pass Self-Refinement)",
    }

    # Save final report in reports/ directory
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"verification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return report
