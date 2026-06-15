"""End-to-end pipeline validation script.

Runs the complete OCR pipeline on a real document image:
    image → preprocessing → OCR → information extraction → structured JSON

Usage (from project root):
    python tests/run_pipeline_validation.py [image_path]

If no image_path is given the script scans datasets/ for test1.* automatically.
"""

from __future__ import annotations

import io
import json
import sys
import time
import textwrap
from pathlib import Path
from statistics import mean, stdev
from typing import Any

# Force UTF-8 output on Windows consoles that default to cp1252
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing import PreprocessingConfig, preprocess_image_file, save_image
from ocr_engine import OCRConfig, PaddleOCREngine, OCRResult
from information_extractor import InvoiceInformationExtractor

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OCR_OUTPUT    = OUTPUTS_DIR / "test1_ocr.json"
STRUCT_OUTPUT = OUTPUTS_DIR / "test1_structured.json"
REPORT_OUTPUT = OUTPUTS_DIR / "test1_validation_report.md"
PREPROC_OUTPUT = OUTPUTS_DIR / "test1_preprocessed.png"

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# ---------------------------------------------------------------------------
# Step 1 – Locate image
# ---------------------------------------------------------------------------

def find_test_image(hint: str | None = None) -> Path:
    if hint:
        p = Path(hint)
        if p.exists():
            return p
        raise FileNotFoundError(f"Provided image path not found: {hint}")

    datasets_dir = PROJECT_ROOT / "datasets"
    # Priority: test1.* at root of datasets/
    for ext in [".jpeg", ".jpg", ".png", ".tif", ".tiff"]:
        candidate = datasets_dir / f"test1{ext}"
        if candidate.exists():
            return candidate

    # Fallback: any supported file anywhere under datasets/
    for path in sorted(datasets_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            return path

    raise FileNotFoundError("No suitable image found under datasets/")


# ---------------------------------------------------------------------------
# Step 2 – Preprocessing
# ---------------------------------------------------------------------------

def run_preprocessing(image_path: Path) -> tuple[Any, float]:
    print("\n[1/3] PREPROCESSING")
    print(f"  Input : {image_path}")
    t0 = time.perf_counter()
    config = PreprocessingConfig(
        denoise_kernel_size=3,
        adaptive_threshold_block_size=31,
        adaptive_threshold_c=15,
        deskew_max_angle=15.0,
    )
    result = preprocess_image_file(image_path, config=config)
    elapsed = time.perf_counter() - t0

    PREPROC_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_image(result.image, PREPROC_OUTPUT)

    print(f"  Original shape : {result.metadata['original_shape']}")
    print(f"  Processed shape: {result.metadata['processed_shape']}")
    print(f"  Skew angle     : {result.skew_angle:.2f}°")
    print(f"  Steps          : {', '.join(result.metadata['preprocessing_steps'])}")
    print(f"  Preprocessed   : {PREPROC_OUTPUT}")
    print(f"  Time           : {elapsed:.3f}s")
    return result, elapsed


# ---------------------------------------------------------------------------
# Step 3 – OCR
# ---------------------------------------------------------------------------

def run_ocr(image_path: Path) -> tuple[OCRResult, float]:
    print("\n[2/3] OCR ENGINE")
    config = OCRConfig(
        language="en",
        use_angle_cls=False,
        use_gpu=False,
        min_confidence=0.0,
        sort_reading_order=True,
    )
    engine = PaddleOCREngine(config)

    t0 = time.perf_counter()
    result = engine.recognize(image_path)
    elapsed = time.perf_counter() - t0

    confidences = [b.confidence for b in result.text_blocks]
    avg_conf = mean(confidences) if confidences else 0.0
    std_conf = stdev(confidences) if len(confidences) > 1 else 0.0
    low_conf_blocks = [b for b in result.text_blocks if b.confidence < 0.80]

    print(f"  Language       : {result.language}")
    print(f"  Text regions   : {len(result.text_blocks)}")
    print(f"  Avg confidence : {avg_conf:.4f}")
    print(f"  Std confidence : {std_conf:.4f}")
    print(f"  Min confidence : {min(confidences, default=0):.4f}")
    print(f"  Max confidence : {max(confidences, default=0):.4f}")
    print(f"  Low-conf (<0.8): {len(low_conf_blocks)} blocks")
    print(f"  Time           : {elapsed:.3f}s")

    if low_conf_blocks:
        print("  Low-confidence blocks:")
        for b in low_conf_blocks:
            safe_text = b.text.encode('utf-8', errors='replace').decode('utf-8')
            print(f"    [{b.confidence:.3f}] {safe_text!r}")

    print(f"\n  Full OCR text:\n  {'─'*60}")
    for line in result.full_text.split("\n"):
        safe_line = line.encode('utf-8', errors='replace').decode('utf-8')
        print(f"  {safe_line}")
    print(f"  {'─'*60}")

    return result, elapsed


# ---------------------------------------------------------------------------
# Step 4 – Save OCR JSON
# ---------------------------------------------------------------------------

def save_ocr_json(ocr_result: OCRResult, image_path: Path, ocr_elapsed: float) -> dict[str, Any]:
    confidences = [b.confidence for b in ocr_result.text_blocks]
    payload = {
        "image_name": image_path.name,
        "image_path": str(image_path),
        "category": "invoice",
        "ocr_model_language": ocr_result.language,
        "number_of_text_regions": len(ocr_result.text_blocks),
        "average_confidence": round(mean(confidences), 4) if confidences else 0.0,
        "processing_time_seconds": round(ocr_elapsed, 4),
        "ocr_text_output": ocr_result.full_text,
        "confidence_scores": [round(c, 4) for c in confidences],
        "text_blocks": [b.to_dict() for b in ocr_result.text_blocks],
    }
    OCR_OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n  OCR JSON saved → {OCR_OUTPUT}")
    return payload


# ---------------------------------------------------------------------------
# Step 5 – Information Extraction
# ---------------------------------------------------------------------------

def run_extraction(ocr_payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
    print("\n[3/3] INFORMATION EXTRACTION")
    extractor = InvoiceInformationExtractor(min_confidence=0.0)
    t0 = time.perf_counter()
    structured = extractor.extract(ocr_payload)
    elapsed = time.perf_counter() - t0

    STRUCT_OUTPUT.write_text(
        json.dumps(structured, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  Extraction time: {elapsed:.3f}s")
    print(f"  Structured JSON → {STRUCT_OUTPUT}")
    return structured, elapsed


# ---------------------------------------------------------------------------
# Step 6 – Validation
# ---------------------------------------------------------------------------

def validate_extraction(
    structured: dict[str, Any],
    ocr_result: OCRResult,
    ocr_elapsed: float,
    preproc_elapsed: float,
    extraction_elapsed: float,
    image_path: Path,
) -> dict[str, Any]:
    """Validate extraction quality and build a report."""
    ok: list[str] = []
    missing: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []
    fixes: list[str] = []

    confidences = [b.confidence for b in ocr_result.text_blocks]

    # --- Invoice number ---
    inv_no = structured.get("invoice_number")
    if inv_no:
        ok.append(f"invoice_number extracted: **{inv_no}**")
    else:
        missing.append("invoice_number — regex did not match any 'Invoice no:' pattern")
        fixes.append(
            "Check OCR text for invoice number keywords. "
            "The regex expects 'Invoice no:' or 'Invoice number:' followed by alphanumeric."
        )

    # --- Date ---
    inv_date = structured.get("invoice_date")
    if inv_date:
        ok.append(f"invoice_date extracted: **{inv_date}**")
    else:
        missing.append("invoice_date — no DD/MM/YYYY or MM-DD-YYYY pattern found")
        fixes.append("Check for non-standard date formats; consider expanding DATE_RE.")

    # --- Seller ---
    seller = structured.get("seller", {})
    if seller.get("name"):
        ok.append(f"seller.name: **{seller['name']}**")
    else:
        missing.append("seller.name — 'Seller:' label may be absent or OCR'd incorrectly")
    if seller.get("tax_id"):
        ok.append(f"seller.tax_id: **{seller['tax_id']}**")
    else:
        missing.append("seller.tax_id")
    if seller.get("address"):
        ok.append(f"seller.address: {len(seller['address'])} line(s)")
    else:
        missing.append("seller.address")

    # --- Client ---
    client = structured.get("client", {})
    if client.get("name"):
        ok.append(f"client.name: **{client['name']}**")
    else:
        missing.append("client.name — 'Client:' label may be absent or OCR'd incorrectly")
    if client.get("tax_id"):
        ok.append(f"client.tax_id: **{client['tax_id']}**")
    else:
        missing.append("client.tax_id")
    if client.get("address"):
        ok.append(f"client.address: {len(client['address'])} line(s)")
    else:
        missing.append("client.address")

    # --- Line items ---
    items = structured.get("line_items", [])
    if items:
        ok.append(f"line_items: **{len(items)} item(s)** extracted")
        for item in items:
            if item.get("unit_price") and item["unit_price"].get("value") is None:
                warnings.append(
                    f"Item {item['item_number']}: unit_price parse failed "
                    f"(raw={item['unit_price']['raw']!r})"
                )
            if item.get("gross_worth") and item["gross_worth"].get("value") is None:
                warnings.append(
                    f"Item {item['item_number']}: gross_worth parse failed "
                    f"(raw={item['gross_worth']['raw']!r})"
                )
            if not item.get("description"):
                warnings.append(
                    f"Item {item['item_number']}: description is empty"
                )
    else:
        missing.append(
            "line_items — ITEMS/SUMMARY section labels not found, "
            "or item-number anchors ('^N.$') not detected in correct x-column"
        )
        fixes.append(
            "Dump text_blocks sorted by x_min to verify item-number column position. "
            "Adjust FALLBACK_COLUMNS['no'] x-range if needed."
        )

    # --- Summary / Totals ---
    summary = structured.get("summary", {})
    subtotal = summary.get("subtotal")
    tax      = summary.get("tax")
    total    = summary.get("total_amount")

    if total and total.get("value") is not None:
        ok.append(f"summary.total_amount: **{total['raw']}** → {total['value']}")
    else:
        missing.append("summary.total_amount")
        fixes.append("Check 'Total' keyword row in OCR output; may be spelled differently.")

    if subtotal and subtotal.get("value") is not None:
        ok.append(f"summary.subtotal: {subtotal['raw']} → {subtotal['value']}")
    else:
        missing.append("summary.subtotal")

    if tax and tax.get("value") is not None:
        ok.append(f"summary.tax: {tax['raw']} → {tax['value']}")
    else:
        missing.append("summary.tax")

    # --- OCR quality ---
    low_conf = [b for b in ocr_result.text_blocks if b.confidence < 0.80]
    if low_conf:
        for b in low_conf:
            warnings.append(
                f"Low OCR confidence [{b.confidence:.3f}]: {b.text!r} "
                f"at bbox xyxy=({b.bbox_xyxy})"
            )
    else:
        ok.append("All OCR blocks have confidence ≥ 0.80")

    return {
        "ok": ok,
        "missing": missing,
        "warnings": warnings,
        "errors": errors,
        "fixes": fixes,
        "confidences": confidences,
        "ocr_elapsed": ocr_elapsed,
        "preproc_elapsed": preproc_elapsed,
        "extraction_elapsed": extraction_elapsed,
        "image_path": image_path,
        "structured": structured,
        "ocr_result": ocr_result,
    }


# ---------------------------------------------------------------------------
# Step 7 – Validation report
# ---------------------------------------------------------------------------

def write_report(report: dict[str, Any]) -> None:
    structured    = report["structured"]
    ocr_result    = report["ocr_result"]
    image_path    = report["image_path"]
    confidences   = report["confidences"]
    ok            = report["ok"]
    missing       = report["missing"]
    warnings      = report["warnings"]
    errors        = report["errors"]
    fixes         = report["fixes"]

    items = structured.get("line_items", [])
    summary = structured.get("summary", {})
    avg_c = mean(confidences) if confidences else 0.0

    lines: list[str] = []

    lines += [
        "# Pipeline Validation Report — test1",
        "",
        f"**Image**: `{image_path.name}`  ",
        f"**Path**: `{image_path}`  ",
        f"**Processed**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## Pipeline Timings",
        "",
        "| Stage | Time |",
        "|---|---|",
        f"| Preprocessing | {report['preproc_elapsed']:.3f}s |",
        f"| OCR Engine    | {report['ocr_elapsed']:.3f}s |",
        f"| Extraction    | {report['extraction_elapsed']:.3f}s |",
        f"| **Total**     | **{report['preproc_elapsed']+report['ocr_elapsed']+report['extraction_elapsed']:.3f}s** |",
        "",
        "---",
        "",
        "## OCR Statistics",
        "",
        f"- **Engine**: PaddleOCR  ",
        f"- **Language model**: {ocr_result.language}  ",
        f"- **Text regions detected**: {len(ocr_result.text_blocks)}  ",
        f"- **Average confidence**: {avg_c:.4f}  ",
        f"- **Min confidence**: {min(confidences, default=0):.4f}  ",
        f"- **Max confidence**: {max(confidences, default=0):.4f}  ",
        f"- **Low-confidence blocks (<0.80)**: {len([c for c in confidences if c < 0.80])}",
        "",
        "### Raw OCR Text",
        "",
        "```",
        ocr_result.full_text,
        "```",
        "",
        "---",
        "",
        "## Extracted Fields",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| document_type | {structured.get('document_type')} |",
        f"| invoice_number | {structured.get('invoice_number')} |",
        f"| invoice_date | {structured.get('invoice_date')} |",
        f"| seller.name | {structured.get('seller', {}).get('name')} |",
        f"| seller.tax_id | {structured.get('seller', {}).get('tax_id')} |",
        f"| seller.address | {'; '.join(structured.get('seller', {}).get('address', []))} |",
        f"| client.name | {structured.get('client', {}).get('name')} |",
        f"| client.tax_id | {structured.get('client', {}).get('tax_id')} |",
        f"| client.address | {'; '.join(structured.get('client', {}).get('address', []))} |",
        f"| line_items count | {len(items)} |",
        f"| subtotal | {summary.get('subtotal', {}).get('raw') if summary.get('subtotal') else 'N/A'} |",
        f"| tax | {summary.get('tax', {}).get('raw') if summary.get('tax') else 'N/A'} |",
        f"| total_amount | {summary.get('total_amount', {}).get('raw') if summary.get('total_amount') else 'N/A'} |",
        "",
        "### Line Items",
        "",
    ]

    if items:
        lines += [
            "| # | Description | Qty | Unit | Unit Price | Net Worth | VAT% | Gross |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for item in items:
            lines.append(
                f"| {item.get('item_number')} "
                f"| {(item.get('description') or '')[:60]} "
                f"| {item.get('quantity')} "
                f"| {item.get('unit')} "
                f"| {item.get('unit_price', {}).get('raw') if item.get('unit_price') else ''} "
                f"| {item.get('net_worth', {}).get('raw') if item.get('net_worth') else ''} "
                f"| {item.get('vat_percent', {}).get('raw') if item.get('vat_percent') else ''} "
                f"| {item.get('gross_worth', {}).get('raw') if item.get('gross_worth') else ''} |"
            )
    else:
        lines.append("_No line items extracted._")

    lines += [
        "",
        "---",
        "",
        "## Validation Results",
        "",
        f"**✅ Successes ({len(ok)})**",
        "",
    ]
    for item in ok:
        lines.append(f"- {item}")

    lines += [
        "",
        f"**⚠️ Warnings ({len(warnings)})**",
        "",
    ]
    if warnings:
        for item in warnings:
            lines.append(f"- {item}")
    else:
        lines.append("- None")

    lines += [
        "",
        f"**❌ Missing Fields ({len(missing)})**",
        "",
    ]
    if missing:
        for item in missing:
            lines.append(f"- {item}")
    else:
        lines.append("- None — all fields extracted successfully!")

    lines += [
        "",
        f"**🔧 Recommended Fixes ({len(fixes)})**",
        "",
    ]
    if fixes:
        for item in fixes:
            lines.append(f"- {item}")
    else:
        lines.append("- No fixes required.")

    # --- Detailed OCR blocks ---
    lines += [
        "",
        "---",
        "",
        "## OCR Block Detail (sorted by reading order)",
        "",
        "| # | Text | Conf | x_min | y_min | x_max | y_max |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, block in enumerate(ocr_result.text_blocks, 1):
        bxy = block.bbox_xyxy
        conf_flag = " ⚠️" if block.confidence < 0.80 else ""
        lines.append(
            f"| {i} | {block.text[:50]!r} | {block.confidence:.3f}{conf_flag} "
            f"| {bxy['x_min']:.0f} | {bxy['y_min']:.0f} "
            f"| {bxy['x_max']:.0f} | {bxy['y_max']:.0f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Output Files",
        "",
        f"| File | Description |",
        f"|---|---|",
        f"| `outputs/test1_preprocessed.png` | Preprocessed (denoised + binarized + deskewed) image |",
        f"| `outputs/test1_ocr.json` | Raw OCR output with all text blocks and bounding boxes |",
        f"| `outputs/test1_structured.json` | Structured invoice record |",
        f"| `outputs/test1_validation_report.md` | This report |",
        "",
    ]

    report_text = "\n".join(lines)
    REPORT_OUTPUT.write_text(report_text, encoding="utf-8")
    print(f"\n  Validation report → {REPORT_OUTPUT}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    image_hint = sys.argv[1] if len(sys.argv) > 1 else None

    print("=" * 65)
    print("  OCR Pipeline Validation — End-to-End")
    print("=" * 65)

    # Locate image
    image_path = find_test_image(image_hint)
    print(f"\n  Image found: {image_path}")
    print(f"  Size       : {image_path.stat().st_size:,} bytes")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Preprocessing ---
    preproc_result, preproc_elapsed = run_preprocessing(image_path)

    # --- OCR (run on original image, preprocessing result used for display) ---
    ocr_result, ocr_elapsed = run_ocr(image_path)

    # --- Save OCR JSON ---
    ocr_payload = save_ocr_json(ocr_result, image_path, ocr_elapsed)

    # --- Extraction ---
    structured, extraction_elapsed = run_extraction(ocr_payload)

    # --- Validation ---
    report = validate_extraction(
        structured, ocr_result, ocr_elapsed,
        preproc_elapsed, extraction_elapsed, image_path,
    )

    # --- Write report ---
    write_report(report)

    # --- Print structured JSON to console ---
    print("\n" + "=" * 65)
    print("  FINAL STRUCTURED JSON OUTPUT")
    print("=" * 65)
    print(json.dumps(structured, ensure_ascii=False, indent=2))

    # --- Final summary ---
    ok      = report["ok"]
    missing = report["missing"]
    warnings = report["warnings"]
    total_t = preproc_elapsed + ocr_elapsed + extraction_elapsed

    print("\n" + "=" * 65)
    print("  VALIDATION SUMMARY")
    print("=" * 65)
    print(f"  ✅ Successes   : {len(ok)}")
    print(f"  ⚠️  Warnings    : {len(warnings)}")
    print(f"  ❌ Missing     : {len(missing)}")
    print(f"  ⏱️  Total time  : {total_t:.2f}s")
    print(f"  📄 Report      : {REPORT_OUTPUT}")
    print("=" * 65)

    if missing:
        print("\n  Missing fields:")
        for m in missing:
            print(f"    - {m}")


if __name__ == "__main__":
    main()
