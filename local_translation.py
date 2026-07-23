# local_translation.py
"""Local LLM translation backend.
Provides a simple wrapper around HuggingFace MarianMT models for
source → target language translation without external API calls.
"""

from functools import lru_cache
from typing import Tuple

try:
    from transformers import MarianMTModel, MarianTokenizer
except ImportError:
    raise ImportError("Please install transformers: pip install transformers sentencepiece")

# Optional torch for accelerated inference on Apple Silicon (MPS)
try:
    import torch
    _DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
except Exception:
    _DEVICE = "cpu"

# Directory to store downloaded models locally
from pathlib import Path
_MODEL_CACHE_DIR = Path(__file__).parent / "model_cache"
_MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Cache loaded models per language pair to avoid re‑loading.
_MODEL_CACHE = {}

def _model_name(source: str, target: str) -> str:
    """Return the HuggingFace model identifier for a given language pair.
    Uses Helsinki-NLP/opus‑mt models when available.
    """
    src = source.lower()
    tgt = target.lower()
    return f"Helsinki-NLP/opus-mt-{src}-{tgt}"

@lru_cache(maxsize=32)
def load_model(source: str, target: str) -> Tuple[MarianMTModel, MarianTokenizer]:
    """Load (and cache) the MarianMT model and tokenizer for the pair.
    Raises RuntimeError if the model cannot be downloaded.
    """
    model_id = _model_name(source, target)
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_id)
        model = MarianMTModel.from_pretrained(model_id)
    except Exception as e:
        raise RuntimeError(f"Failed to load translation model {model_id}: {e}")
    return model, tokenizer

def translate_with_local(text: str, source: str, target: str) -> str:
    """Translate *text* from *source* language to *target* language.
    This function loads the appropriate MarianMT model, tokenises the input,
    runs generation, and returns the decoded string.
    Errors are propagated to the caller for fallback handling.
    """
    model, tokenizer = load_model(source, target)
    inputs = tokenizer([text], return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    result = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return result[0] if result else ""
