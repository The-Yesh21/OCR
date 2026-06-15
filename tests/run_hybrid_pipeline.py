"""E2E hybrid pipeline runner for document-type-aware extraction.

Runs preprocessing → English OCR → document classification.
If it is a market register, runs Kannada OCR and hybrid extraction.
Saves and prints the final structured JSON.
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path
from typing import Any

# Force UTF-8 stdout/stderr on Windows to prevent print crashes
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing import PreprocessingConfig, preprocess_image_file, save_image
from ocr_engine import OCRConfig, PaddleOCREngine
from document_classifier import classify_document
from information_extractor import InvoiceInformationExtractor
from market_register_extractor import MarketRegisterExtractor


def run_pipeline(image_path: str | Path) -> None:
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"Error: Image not found at {img_path}")
        sys.exit(1)

    print(f"=== Starting pipeline for: {img_path.name} ===")

    # 1. Preprocessing
    print("\n[1] Preprocessing image...")
    preproc_config = PreprocessingConfig(
        denoise_kernel_size=3,
        adaptive_threshold_block_size=31,
        adaptive_threshold_c=15,
        deskew_max_angle=15.0,
    )
    t0 = time.perf_counter()
    preproc_res = preprocess_image_file(img_path, config=preproc_config)
    preproc_time = time.perf_counter() - t0
    print(f"    Preprocessing finished in {preproc_time:.3f}s")

    # Save preprocessed image for validation
    preproc_out = PROJECT_ROOT / "outputs" / "test1_preprocessed.png"
    preproc_out.parent.mkdir(parents=True, exist_ok=True)
    save_image(preproc_res.image, preproc_out)
    print(f"    Saved preprocessed image to {preproc_out}")

    # 2. English OCR (for classification and numbers)
    print("\n[2] Running English OCR...")
    en_config = OCRConfig(
        language="en",
        use_angle_cls=True,  # Enable rotation classification!
        use_gpu=False,
        min_confidence=0.0,
        sort_reading_order=True,
    )
    en_engine = PaddleOCREngine(en_config)
    t0 = time.perf_counter()
    en_res = en_engine.recognize(preproc_res.image, image_path=str(img_path))
    en_ocr_time = time.perf_counter() - t0
    print(f"    English OCR finished in {en_ocr_time:.3f}s")
    print(f"    Detected {len(en_res.text_blocks)} regions with English OCR")

    # 3. Document Classification
    print("\n[3] Classifying document type...")
    doc_type = classify_document(en_res.full_text)
    print(f"    Detected Document Type: {doc_type.upper()}")

    # 4. Extracting based on document type
    output_data: dict[str, Any] = {}
    if doc_type == "market_register":
        # Run Kannada OCR
        print("\n[4] Running Kannada OCR for market register...")
        ka_config = OCRConfig(
            language="ka",
            use_angle_cls=True,
            use_gpu=False,
            min_confidence=0.0,
            sort_reading_order=True,
        )
        ka_engine = PaddleOCREngine(ka_config)
        t0 = time.perf_counter()
        ka_res = ka_engine.recognize(preproc_res.image, image_path=str(img_path))
        ka_ocr_time = time.perf_counter() - t0
        print(f"    Kannada OCR finished in {ka_ocr_time:.3f}s")
        print(f"    Detected {len(ka_res.text_blocks)} regions with Kannada OCR")

        # Extract using MarketRegisterExtractor
        print("\n[5] Extracting market register fields...")
        extractor = MarketRegisterExtractor(min_confidence=0.0)
        output_data = extractor.extract(en_res.to_dict(), ka_res.to_dict())
        
        output_file = PROJECT_ROOT / "outputs" / "test1_market_register_v2.json"
        
    else:
        # Extract using InvoiceInformationExtractor
        print("\n[4] Extracting invoice fields...")
        extractor = InvoiceInformationExtractor(min_confidence=0.0)
        output_data = extractor.extract(en_res.to_dict())
        
        output_file = PROJECT_ROOT / "outputs" / "test1_structured.json"

    # Save output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n[5] Saved final structured JSON to: {output_file}")

    # Print final structured JSON to console
    print("\n=== FINAL STRUCTURED JSON OUTPUT ===")
    print(json.dumps(output_data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    image = "datasets/test1.jpeg"
    if len(sys.argv) > 1:
        image = sys.argv[1]
    run_pipeline(image)
