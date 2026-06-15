"""Trace script to perform evidence-based diagnostics on the OCR extraction pipeline.

Collects and analyzes block-by-block transitions to identify exactly where
information is lost or corrupted.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from statistics import mean

# Force UTF-8 console output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ocr_engine import OCRConfig, PaddleOCREngine
from ocr_fusion import OCRResultFusion, calculate_iou, extract_numeric
from market_register_extractor import MarketRegisterExtractor
from commodity_normalizer import is_valid_commodity, normalize_commodity
from post_processor import MarketRegisterPostProcessor


def run_diagnostics():
    v4_file = PROJECT_ROOT / "outputs" / "test1_market_register_v4.json"
    
    # 1. Load the fused result from v4 to trace
    with open(v4_file, encoding="utf-8") as f:
        v4_data = json.load(f)
        
    print("=== OCR DETECTIONS INVENTORY ===")
    # Load raw OCR results
    with open(PROJECT_ROOT / "outputs" / "test1_ocr.json", encoding="utf-8") as f:
        raw_en = json.load(f)
    with open(PROJECT_ROOT / "outputs" / "test1_ocr_kannada.json", encoding="utf-8") as f:
        raw_ka = json.load(f)
        
    print(f"Original English OCR Blocks: {len(raw_en['text_blocks'])}")
    print(f"Original Kannada OCR Blocks: {len(raw_ka['text_blocks'])}")
    
    # Re-run mapping and fusion steps to trace blocks
    fusion = OCRResultFusion()
    
    # Map raw blocks
    mapped_en_full = fusion.map_coordinates(raw_en["text_blocks"], "original")
    mapped_ka_full = fusion.map_coordinates(raw_ka["text_blocks"], "original")
    
    # We will trace the coordinates of all Kannada blocks
    print("\n--- Track top-of-page Kannada OCR Blocks (y < 500) ---")
    for b in mapped_ka_full:
        if b["bbox_xyxy"]["y_min"] < 500:
            print(f"  Text: {b['text']!r} | y_range: [{b['bbox_xyxy']['y_min']:.1f}, {b['bbox_xyxy']['y_max']:.1f}] | x_center: {b['bbox_xyxy']['x_min'] + (b['bbox_xyxy']['x_max'] - b['bbox_xyxy']['x_min'])/2:.1f}")

    print("\n--- Track table-region Kannada OCR Blocks (500 <= y < 1300) ---")
    table_ka_count = 0
    for b in mapped_ka_full:
        if 500 <= b["bbox_xyxy"]["y_min"] < 1300:
            table_ka_count += 1
            if table_ka_count <= 15:
                print(f"  Text: {b['text']!r} | y_range: [{b['bbox_xyxy']['y_min']:.1f}, {b['bbox_xyxy']['y_max']:.1f}] | x_center: {b['bbox_xyxy']['x_min'] + (b['bbox_xyxy']['x_max'] - b['bbox_xyxy']['x_min'])/2:.1f}")
    print(f"  Total table Kannada blocks: {table_ka_count}")

    # Reconstruct the fused blocks
    # Load preprocessed image to do the actual mapping
    # We will simulate the exact row grouping from MarketRegisterExtractor
    # Let's print out the group rows from v4 extraction method
    en_payload_fused = v4_data.get("metadata", {})
    items = v4_data.get("items", [])
    
    print("\n=== ROW RECONSTRUCTION DIAGNOSTICS ===")
    # We will load the actual fused payloads and check how they were grouped into rows
    # Let's run the actual grouping logic
    # Since we saved the v4 result, let's load it and inspect:
    print(f"Grouped rows: {v4_data['metadata']['rows_grouped']}")
    print(f"Column boundaries: {v4_data['metadata']['column_boundaries']}")

    # Check for missing row analysis
    # Ground Truth vs extracted v4
    print("\n=== MISSING ROW ANALYSIS ===")
    from generate_metrics import GROUND_TRUTH
    extracted_comms = [item["commodity"] for item in items]
    extracted_raws = [item["normalization"]["raw_name"] for item in items]
    
    missing_from_v4 = []
    for gt in GROUND_TRUTH:
        if gt["commodity"] not in extracted_comms and gt["commodity"] not in extracted_raws:
            missing_from_v4.append(gt["commodity"])
            
    print(f"Missing from v4 final JSON: {missing_from_v4}")
    
    # Check incorrect prices
    print("\n=== INCORRECT PRICE ANALYSIS ===")
    for item in items:
        comm = item["commodity"]
        price_details = item["price"]
        price = price_details.get("price")
        
        # Find expected price in ground truth
        expected = None
        for gt in GROUND_TRUTH:
            if gt["commodity"] == comm:
                expected = gt["price"]
                break
                
        if expected is not None:
            expected_val = float(expected) if expected else None
            actual_val = float(price) if price else None
            if expected_val != actual_val:
                print(f"  Commodity: {comm:20s} | Price: {actual_val} | Expected: {expected_val} | Status: {price_details['status']}")

    # Check normalization quality
    print("\n=== NORMALIZATION QUALITY ANALYSIS ===")
    for item in items:
        comm = item["commodity"]
        norm = item["normalization"]
        raw = norm["raw_name"]
        conf = norm["confidence"]
        status = norm["status"]
        
        if status == "unresolved" and conf > 0:
            print(f"  Unresolved (correct norm but low conf): raw={raw!r} -> norm={comm!r} (conf={conf:.3f})")
        elif status == "resolved" and conf < 1.0:
            print(f"  Resolved: raw={raw!r} -> norm={comm!r} (conf={conf:.3f})")


if __name__ == "__main__":
    run_diagnostics()
