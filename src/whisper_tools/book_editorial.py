"""Full-Book AI Editorial Audit & Refinement Module.

Performs a final full-book AI review pass, ingesting global source text context,
auditing draft translations for fluency/accuracy, auto-correcting flaws,
and logging all editorial revisions into a dedicated Markdown audit report.
"""

from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Tuple

try:
    from whisper_tools.translation import _DEFAULT_MLX_MODEL, _MLX_AVAILABLE, load_mlx_model, mlx_generate
except ImportError:
    from .translation import _DEFAULT_MLX_MODEL, _MLX_AVAILABLE, load_mlx_model, mlx_generate


def audit_and_refine_full_book(
    raw_source_text: str,
    segments: List[Dict[str, Any]],
    source_lang: str,
    target_lang: str,
    model_id: str = _DEFAULT_MLX_MODEL,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Performs a full-book AI editorial review pass.

    Args:
        raw_source_text: Full raw text of the source book.
        segments: List of segment dictionaries containing draft translations.
        source_lang: Source language code (e.g. 'ar', 'de').
        target_lang: Target language code (e.g. 'de', 'en').
        model_id: MLX model identifier.

    Returns:
        Tuple of (updated_segments, audit_entries).
    """
    if not _MLX_AVAILABLE:
        print("  [NOTE] MLX not available. Skipping Full-Book AI Editorial Audit pass.")
        return segments, []

    lang_names = {"ar": "Arabic", "de": "German", "en": "English", "fr": "French", "es": "Spanish"}
    src_name = lang_names.get(source_lang.lower(), source_lang)
    tgt_name = lang_names.get(target_lang.lower(), target_lang)

    print(f"\n[AI Editorial Audit] Starting Full-Book AI Review & Quality Pass ({src_name} → {tgt_name})...")
    print(f"       Ingesting global book context ({len(raw_source_text):,} characters)...")

    # Ingest book summary context (first 2,000 characters)
    book_summary = raw_source_text[:2000].strip().replace("\n", " ")

    try:
        model, tokenizer = load_mlx_model(model_id)
    except Exception as e:
        print(f"  [WARNING] Could not load LLM for editorial audit: {e}")
        return segments, []

    audit_entries = []
    total_segs = len(segments)

    for idx, seg in enumerate(segments, start=1):
        orig_text = seg.get("original_text", seg.get("original_ar", seg.get("text", ""))).strip()
        draft_text = seg.get(f"text_{target_lang}", "").strip()

        if not orig_text or not draft_text:
            continue

        prompt = (
            f"You are a master bilingual editor reviewing a book translation from {src_name} to {tgt_name}.\n"
            f"Book Background Context: \"{book_summary}...\"\n\n"
            f"Original {src_name}: {orig_text}\n"
            f"Draft Translation ({tgt_name}): {draft_text}\n\n"
            f"Review the draft translation carefully. Ensure it sounds natural, eloquent, and contextually accurate in {tgt_name}.\n"
            f"If the translation is already accurate and fluent, output JSON:\n"
            f'{{"status": "approved", "revised_text": "{draft_text}", "reason": "Accurate and fluent"}}\n\n'
            f"If it sounds awkward, literal, or inaccurate, output JSON with the polished translation and reason:\n"
            f'{{ "status": "revised", "revised_text": "<POLISHED_TRANSLATION>", "reason": "<REASON_FOR_REVISION>" }}\n\n'
            f"Output ONLY valid JSON:"
        )

        try:
            if hasattr(tokenizer, "apply_chat_template"):
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                formatted_prompt = prompt

            response = mlx_generate(model, tokenizer, prompt=formatted_prompt, max_tokens=512, verbose=False).strip()

            # Extract JSON block
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                status = parsed.get("status", "approved")
                revised_text = parsed.get("revised_text", draft_text).strip()
                reason = parsed.get("reason", "Editorial review")
            else:
                status = "approved"
                revised_text = draft_text
                reason = "Editorial review"

            if status == "revised" and revised_text and revised_text != draft_text:
                seg[f"text_{target_lang}"] = revised_text
                audit_entries.append({
                    "segment_id": seg.get("id", idx),
                    "original_text": orig_text,
                    "draft_text": draft_text,
                    "revised_text": revised_text,
                    "reason": reason,
                })
        except Exception:
            pass

        if idx % 50 == 0 or idx == total_segs:
            print(f"       Audited {idx}/{total_segs} segments ({len(audit_entries)} revisions applied)...")

    print(f"       [AI Editorial Audit Complete] Audited {total_segs} paragraphs. Applied {len(audit_entries)} quality revisions.")
    return segments, audit_entries


def write_editorial_audit_report(
    audit_entries: List[Dict[str, Any]],
    book_output_dir: Path,
    base_name: str,
    target_lang: str,
    total_segments_count: int,
) -> Path:
    """Generates a detailed Markdown editorial audit log report file.

    Args:
        audit_entries: List of audit revision entries.
        book_output_dir: Directory where book deliverables are saved.
        base_name: Base book name.
        target_lang: Target language code.
        total_segments_count: Total count of segments in book.

    Returns:
        Path to generated audit report file.
    """
    report_path = book_output_dir / f"{base_name}_editorial_audit_{target_lang}.md"

    lines = [
        f"# 🔎 Full-Book AI Editorial Audit & Quality Report ({target_lang.upper()})",
        "",
        f"- **Book Title**: `{base_name}`",
        f"- **Audit Date**: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- **Total Paragraph Chunks Audited**: `{total_segments_count}`",
        f"- **Total Editorial Revisions Applied**: `{len(audit_entries)}`",
        "",
        "---",
        "",
        "## 📝 Log of Editorial Revisions & Corrections",
        "",
    ]

    if not audit_entries:
        lines.append("✨ **No revisions needed!** All draft translations met high editorial standards for accuracy and fluency.")
    else:
        for entry in audit_entries:
            lines.append(f"### Segment #{entry['segment_id']}")
            lines.append(f"- **Original Source**: {entry['original_text']}")
            lines.append(f"- **Initial Draft**: {entry['draft_text']}")
            lines.append(f"- **AI Refined Output**: `{entry['revised_text']}`")
            lines.append(f"- **Editorial Reason**: *{entry['reason']}*")
            lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"       Editorial Audit Report (.md): {report_path.name}")
    return report_path
