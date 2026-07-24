# local_translation.py
"""Dual local translation & verification engine for Apple Silicon.

Model 1 (Primary NMT): Helsinki-NLP/opus-mt (MarianMT on PyTorch MPS GPU)
Model 2 (Verification LLM): Qwen2.5-1.5B-Instruct-4bit or Qwen2.5-3B-Instruct-4bit via mlx-lm (Apple Silicon Metal GPU)
"""

from functools import lru_cache
from pathlib import Path

try:
    from mlx_lm import generate as mlx_generate
    from mlx_lm import load as mlx_load
    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False

try:
    import torch
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

_MODEL_CACHE_DIR = Path(__file__).parent / "model_cache"
_MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_MLX_MODEL_DIR = _MODEL_CACHE_DIR / "mlx"
_MLX_MODEL_DIR.mkdir(parents=True, exist_ok=True)

_DEFAULT_MLX_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
_MLX_MODEL_CACHE = {}

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

def translate_with_mlx_llm(text: str, source: str, target: str, model_id: str = _DEFAULT_MLX_MODEL) -> str:
    """Translate text using Model 2 (Local MLX quantized LLM on Apple Silicon Metal GPU)."""
    if not text or not text.strip():
        return ""

    lang_names = {"ar": "Arabic", "de": "German", "en": "English"}
    src_name = lang_names.get(source.lower(), source)
    tgt_name = lang_names.get(target.lower(), target)

    model, tokenizer = load_mlx_model(model_id)

    prompt = (
        f"You are a professional subtitle translator. "
        f"Translate the following {src_name} text accurately into fluent, natural {tgt_name}.\n"
        f"Do NOT summarize, explain, or add commentary. Output ONLY the final {tgt_name} translation:\n\n"
        f"Source ({src_name}): {text}\n"
        f"Translation ({tgt_name}):"
    )

    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        formatted_prompt = prompt

    response = mlx_generate(model, tokenizer, prompt=formatted_prompt, max_tokens=256, verbose=False)
    cleaned = response.strip().strip('"').strip("'")
    return cleaned

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

    return results
