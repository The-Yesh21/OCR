"""Run OCR validation on 10 real dataset images.

Run from the project root:
    python tests/test_ocr_engine.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ocr_engine import OCRConfig, PaddleOCREngine  # noqa: E402


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "ocr_test_results.json"


@dataclass(frozen=True)
class OCRSample:
    category: str
    path: Path


def find_images(directory: Path, limit: int) -> list[Path]:
    """Return a stable sample of images from a dataset directory."""
    if not directory.exists():
        return []
    images = [
        path
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return images[:limit]


def collect_samples() -> list[OCRSample]:
    """Collect at least 10 images across invoice, FUNSD-like, and Kannada sets."""
    sample_groups = [
        ("invoice", PROJECT_ROOT / "datasets" / "part2", 4),
        ("funsd_like", PROJECT_ROOT / "datasets" / "part3", 3),
        ("kannada_custom", PROJECT_ROOT / "datasets" / "kannada", 3),
    ]

    samples: list[OCRSample] = []
    for category, directory, limit in sample_groups:
        samples.extend(OCRSample(category, path) for path in find_images(directory, limit))

    if len(samples) < 10:
        raise AssertionError(f"Expected at least 10 OCR samples, found {len(samples)}")
    return samples[:10]


def count_kannada_chars(text: str) -> int:
    """Count Kannada Unicode block characters."""
    return sum(1 for char in text if "\u0c80" <= char <= "\u0cff")


def count_english_chars(text: str) -> int:
    """Count ASCII English alphabet characters."""
    return sum(1 for char in text if ("A" <= char <= "Z") or ("a" <= char <= "z"))


def detect_language_from_text(text: str) -> str:
    """Detect English/Kannada/mixed from OCR output using Unicode ranges."""
    kannada_count = count_kannada_chars(text)
    english_count = count_english_chars(text)
    if kannada_count > 0 and english_count > 0:
        return "mixed"
    if kannada_count > 0:
        return "kannada"
    if english_count > 0:
        return "english"
    return "unknown"


def quality_metrics(text: str, category: str) -> dict[str, float | int | str]:
    """Compute lightweight OCR quality indicators without ground-truth labels."""
    kannada_chars = count_kannada_chars(text)
    english_chars = count_english_chars(text)
    total_language_chars = kannada_chars + english_chars
    kannada_ratio = kannada_chars / total_language_chars if total_language_chars else 0.0
    english_ratio = english_chars / total_language_chars if total_language_chars else 0.0

    if category == "kannada_custom":
        kannada_quality = "good" if kannada_chars >= 20 else "poor"
    else:
        kannada_quality = "not_applicable"

    english_quality = "good" if english_chars >= 20 else "poor"
    mixed_quality = "mixed_detected" if kannada_chars > 0 and english_chars > 0 else "not_detected"

    return {
        "kannada_char_count": kannada_chars,
        "english_char_count": english_chars,
        "kannada_char_ratio": round(kannada_ratio, 4),
        "english_char_ratio": round(english_ratio, 4),
        "kannada_extraction_quality": kannada_quality,
        "english_extraction_quality": english_quality,
        "mixed_language_extraction_quality": mixed_quality,
    }


def run_ocr_batch() -> dict[str, object]:
    """Run OCR over the selected image batch and return JSON-ready results."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    samples = collect_samples()
    engine = PaddleOCREngine(
        OCRConfig(language="en", use_angle_cls=False, min_confidence=0.0)
    )

    results = []
    summary_rows = []
    for index, sample in enumerate(samples, start=1):
        print(f"[{index}/{len(samples)}] OCR: {sample.category} -> {sample.path.name}")
        started_at = time.perf_counter()
        ocr_result = engine.recognize(sample.path)
        processing_time = time.perf_counter() - started_at

        confidences = [block.confidence for block in ocr_result.text_blocks]
        average_confidence = mean(confidences) if confidences else 0.0
        detected_language = detect_language_from_text(ocr_result.full_text)
        metrics = quality_metrics(ocr_result.full_text, sample.category)

        image_result = {
            "image_name": sample.path.name,
            "image_path": str(sample.path),
            "category": sample.category,
            "ocr_model_language": ocr_result.language,
            "detected_language": detected_language,
            "number_of_text_regions": len(ocr_result.text_blocks),
            "average_confidence": round(average_confidence, 4),
            "processing_time_seconds": round(processing_time, 4),
            "ocr_text_output": ocr_result.full_text,
            "confidence_scores": [round(value, 4) for value in confidences],
            "quality_metrics": metrics,
            "text_blocks": [block.to_dict() for block in ocr_result.text_blocks],
        }
        results.append(image_result)
        summary_rows.append(
            {
                "Image Name": sample.path.name,
                "Detected Language": detected_language,
                "Number of Text Regions": len(ocr_result.text_blocks),
                "Average Confidence": round(average_confidence, 4),
                "Processing Time": round(processing_time, 4),
            }
        )

    payload = {
        "test_metadata": {
            "total_images": len(samples),
            "ocr_model": "PaddleOCR",
            "ocr_model_language": "en",
            "note": (
                "Installed PaddleOCR 3.7.0 does not provide a Kannada lang alias; "
                "Kannada quality is measured from Kannada Unicode characters present "
                "in OCR output."
            ),
        },
        "summary_table": summary_rows,
        "results": results,
    }

    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def print_summary_table(rows: list[dict[str, object]]) -> None:
    """Print a compact Markdown summary table."""
    print("\nOCR Summary Table")
    print("| Image Name | Detected Language | Number of Text Regions | Average Confidence | Processing Time |")
    print("|---|---:|---:|---:|---:|")
    for row in rows:
        print(
            "| {Image Name} | {Detected Language} | {Number of Text Regions} | "
            "{Average Confidence:.4f} | {Processing Time:.4f}s |".format(**row)
        )


def main() -> None:
    payload = run_ocr_batch()
    print_summary_table(payload["summary_table"])
    print(f"\nSaved OCR test results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
