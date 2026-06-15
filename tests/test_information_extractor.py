"""Tests for information_extractor.py.

Run from the project root:
    python tests/test_information_extractor.py

Covers:
    - Amount parser (European / Anglo / spaced / edge cases)
    - Regex extraction (invoice number, date, tax ID)
    - Column discovery from a synthetic block list
    - Row grouping and item parsing with real-shaped bbox data
    - Seller / Client address extraction
    - Summary extraction (Total-row scan and regex fallback)
    - End-to-end extraction from the real ocr_test_results.json
    - Unsupported payload format raises ValueError
    - Empty text_blocks raises ValueError
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from information_extractor import (  # noqa: E402
    FALLBACK_COLUMNS,
    InvoiceInformationExtractor,
    TextBlock,
    extract_invoice_from_file,
)

OCR_RESULTS_PATH = PROJECT_ROOT / "outputs" / "ocr_test_results.json"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "structured_invoice.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _block(
    text: str,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    confidence: float = 1.0,
) -> TextBlock:
    return TextBlock(
        text=text,
        confidence=confidence,
        bbox=[],
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
    )


def _make_invoice_blocks() -> list[dict[str, Any]]:
    """Return a minimal synthetic text_blocks payload matching real invoice layout."""
    # Each entry: (text, x_min, y_min, x_max, y_max)
    raw: list[tuple[str, float, float, float, float]] = [
        # Header
        ("Invoice no: INV-2024-001",  130,   73,  540,  104),
        ("04/13/2024",                800,  140,  978,  171),
        ("Date of issue:",            133,  142,  343,  170),
        # Party labels
        ("Seller:",                   130,  443,  255,  475),
        ("Client:",                   824,  443,  949,  475),
        # Seller address
        ("ACME Corp",                 139,  506,  350,  536),
        ("123 Main Street",           140,  542,  380,  570),
        ("Springfield, ST 12345",     140,  579,  470,  606),
        ("Tax Id: 123-45-6789",       138,  649,  420,  676),
        # Client address
        ("Beta Ltd",                  832,  506,  983,  536),
        ("456 Oak Avenue",            832,  542, 1100,  572),
        ("Shelbyville, ST 67890",     832,  579, 1200,  607),
        ("Tax Id: 987-65-4321",       832,  649, 1116,  676),
        # ITEMS section header
        ("ITEMS",                     129,  758,  250,  793),
        # Column headers row 1
        ("No.",                       156,  844,  205,  870),
        ("Description",               234,  843,  386,  872),
        ("Qty",                       676,  843,  728,  874),
        ("UM",                        774,  844,  822,  870),
        ("Net price",                 896,  843, 1019,  873),
        ("Net worth",                1057,  845, 1186,  869),
        ("VAT [%]",                  1227,  843, 1334,  871),
        ("Gross",                    1424,  843, 1504,  871),
        # Item 1
        ("1.",                        166,  930,  194,  958),
        ("Widget A",                  237,  931,  500,  956),
        ("3,00",                      670,  929,  729,  959),
        ("each",                      767,  931,  828,  957),
        ("100,00",                    933,  930, 1018,  959),
        ("300,00",                   1103,  930, 1187,  958),
        ("10%",                      1276,  930, 1332,  957),
        ("330,00",                   1418,  930, 1503,  958),
        # Item 2
        ("2.",                        165, 1044,  194, 1072),
        ("Gadget B",                  237, 1043,  579, 1072),
        ("5,00",                      670, 1043,  728, 1073),
        ("each",                      767, 1045,  828, 1071),
        ("50,00",                     948, 1043, 1018, 1073),
        ("250,00",                   1103, 1044, 1187, 1072),
        ("10%",                      1277, 1044, 1333, 1071),
        ("275,00",                   1418, 1044, 1502, 1072),
        # SUMMARY
        ("SUMMARY",                   131, 1713,  327, 1744),
        ("VAT [%]",                   500, 1780,  600, 1810),
        ("VAT",                       800, 1780,  900, 1810),
        ("Net worth",                1000, 1780, 1150, 1810),
        ("Gross worth",              1350, 1780, 1520, 1810),
        ("10%",                       500, 1820,  600, 1850),
        ("55,00",                     800, 1820,  900, 1850),
        ("550,00",                   1000, 1820, 1150, 1850),
        ("605,00",                   1350, 1820, 1520, 1850),
        # Total row
        ("Total",                     131, 1870,  280, 1900),
        ("$ 550,00",                  800, 1870,  950, 1900),
        ("$ 55,00",                  1000, 1870, 1150, 1900),
        ("$ 605,00",                 1350, 1870, 1520, 1900),
    ]
    blocks = []
    for text, x_min, y_min, x_max, y_max in raw:
        blocks.append({
            "text": text,
            "confidence": 1.0,
            "bbox": [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
            "bbox_xyxy": {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max},
        })
    return blocks


def _make_synthetic_payload() -> dict[str, Any]:
    blocks = _make_invoice_blocks()
    full_text = "\n".join(b["text"] for b in blocks)
    return {
        "text_blocks": blocks,
        "ocr_text_output": full_text,
        "language": "en",
        "average_confidence": 1.0,
    }


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

def test_parse_amount_european() -> None:
    """European comma-decimal format: '1 394,67' → 1394.67"""
    ex = InvoiceInformationExtractor()
    assert ex._parse_amount("1 394,67") == 1394.67, "European spaced amount"
    assert ex._parse_amount("627,00")   == 627.0,   "Simple comma-decimal"
    assert ex._parse_amount("5 640,17") == 5640.17, "Spaced comma-decimal"
    assert ex._parse_amount("207,63")   == 207.63,  "Two-decimal comma"
    print("PASS test_parse_amount_european")


def test_parse_amount_anglo() -> None:
    """Anglo dot-decimal format: '1,394.67' → 1394.67"""
    ex = InvoiceInformationExtractor()
    assert ex._parse_amount("1,394.67") == 1394.67, "Anglo comma-thousands"
    assert ex._parse_amount("5,640.17") == 5640.17, "Anglo five-thousands"
    assert ex._parse_amount("100.00")   == 100.0,   "Simple dot-decimal"
    print("PASS test_parse_amount_anglo")


def test_parse_amount_dollar_prefix() -> None:
    """Dollar-sign prefix stripped: '$ 5 640,17' → 5640.17"""
    ex = InvoiceInformationExtractor()
    assert ex._parse_amount("$ 5 640,17") == 5640.17
    assert ex._parse_amount("$100.00")     == 100.0
    print("PASS test_parse_amount_dollar_prefix")


def test_parse_amount_edge_cases() -> None:
    """Edge cases: empty, dash, integer strings."""
    ex = InvoiceInformationExtractor()
    assert ex._parse_amount("")    is None
    assert ex._parse_amount("-")   is None
    assert ex._parse_amount("abc") is None
    assert ex._parse_amount("400") == 400.0
    print("PASS test_parse_amount_edge_cases")


def test_extract_invoice_number() -> None:
    ex = InvoiceInformationExtractor()
    assert ex._extract_invoice_number("Invoice no: 51109338")    == "51109338"
    assert ex._extract_invoice_number("Invoice No: INV-2024-001") == "INV-2024-001"
    assert ex._extract_invoice_number("Invoice: ABC-001")         == "ABC-001"
    assert ex._extract_invoice_number("Date of issue: 04/13")    is None
    assert ex._extract_invoice_number("Seller: ACME Corp")        is None
    print("PASS test_extract_invoice_number")


def test_extract_invoice_date() -> None:
    ex = InvoiceInformationExtractor()
    assert ex._extract_invoice_date("Date: 04/13/2013") == "04/13/2013"
    assert ex._extract_invoice_date("Date: 01-05-2024") == "01-05-2024"
    assert ex._extract_invoice_date("No date")          is None
    print("PASS test_extract_invoice_date")


def test_column_discovery() -> None:
    """Column discovery should find all 8 canonical columns."""
    payload = _make_synthetic_payload()
    ex = InvoiceInformationExtractor()
    ocr_result = ex._select_ocr_result(payload)
    blocks = ex._normalize_blocks(ocr_result["text_blocks"])
    col_map = ex._discover_columns(blocks)

    expected_cols = {"no", "description", "qty", "unit", "net_price", "net_worth", "vat", "gross"}
    found = set(col_map.ranges.keys())
    missing = expected_cols - found
    assert not missing, f"Missing columns from discovery: {missing}"
    print(f"PASS test_column_discovery  (found: {sorted(found)})")


def test_line_items_synthetic() -> None:
    """Two line items should be extracted with correct fields."""
    payload = _make_synthetic_payload()
    ex = InvoiceInformationExtractor()
    result = ex.extract(payload)

    items = result["line_items"]
    assert len(items) == 2, f"Expected 2 items, got {len(items)}"

    item1 = items[0]
    assert item1["item_number"] == "1",        f"item_number: {item1['item_number']}"
    assert "Widget A" in (item1["description"] or ""), \
        f"description: {item1['description']}"
    assert item1["quantity"] is not None,       "qty missing"
    assert item1["unit_price"]["value"] == 100.0, \
        f"unit_price: {item1['unit_price']}"
    assert item1["net_worth"]["value"] == 300.0, \
        f"net_worth: {item1['net_worth']}"
    assert item1["vat_percent"]["value"] == 10.0, \
        f"vat_percent: {item1['vat_percent']}"
    assert item1["gross_worth"]["value"] == 330.0, \
        f"gross_worth: {item1['gross_worth']}"

    item2 = items[1]
    assert item2["item_number"] == "2"
    assert "Gadget B" in (item2["description"] or "")
    assert item2["unit_price"]["value"] == 50.0
    assert item2["gross_worth"]["value"] == 275.0
    print("PASS test_line_items_synthetic")


def test_seller_client_extraction() -> None:
    """Seller and client names and tax IDs must be correctly extracted."""
    payload = _make_synthetic_payload()
    ex = InvoiceInformationExtractor()
    result = ex.extract(payload)

    seller = result["seller"]
    client = result["client"]

    assert seller["name"] == "ACME Corp",       f"seller name: {seller['name']}"
    assert seller["tax_id"] == "123-45-6789",   f"seller tax_id: {seller['tax_id']}"
    assert client["name"] == "Beta Ltd",         f"client name: {client['name']}"
    assert client["tax_id"] == "987-65-4321",   f"client tax_id: {client['tax_id']}"
    print("PASS test_seller_client_extraction")


def test_summary_extraction_total_row() -> None:
    """Summary must extract subtotal / tax / total from the Total row."""
    payload = _make_synthetic_payload()
    ex = InvoiceInformationExtractor()
    result = ex.extract(payload)

    summary = result["summary"]
    assert summary["subtotal"]     is not None, "subtotal missing"
    assert summary["total_amount"] is not None, "total_amount missing"
    # At least one amount must parse to a positive float.
    values = [
        v["value"] for v in summary.values()
        if isinstance(v, dict) and v is not None
    ]
    assert any(isinstance(v, float) and v > 0 for v in values), \
        f"No positive float in summary: {summary}"
    print(f"PASS test_summary_extraction_total_row  (summary={summary})")


def test_unsupported_payload_raises() -> None:
    """An empty dict must raise ValueError."""
    ex = InvoiceInformationExtractor()
    try:
        ex._select_ocr_result({})
    except ValueError:
        print("PASS test_unsupported_payload_raises")
        return
    raise AssertionError("Expected ValueError for unsupported payload format")


def test_empty_text_blocks_raises() -> None:
    """A payload with no text blocks must raise ValueError."""
    ex = InvoiceInformationExtractor()
    try:
        ex.extract({"text_blocks": [], "ocr_text_output": ""})
    except ValueError:
        print("PASS test_empty_text_blocks_raises")
        return
    raise AssertionError("Expected ValueError for empty text_blocks")


def test_batch_payload_selection() -> None:
    """Batch JSON (results list) must prefer the invoice-category entry."""
    invoice_entry = {
        "category": "invoice",
        "text_blocks": _make_invoice_blocks(),
        "ocr_text_output": "Invoice no: BATCH-001",
    }
    other_entry = {
        "category": "funsd_like",
        "text_blocks": [],
        "ocr_text_output": "",
    }
    payload = {"results": [other_entry, invoice_entry]}
    ex = InvoiceInformationExtractor()
    selected = ex._select_ocr_result(payload)
    assert selected["category"] == "invoice", \
        f"Expected invoice category, got {selected['category']}"
    print("PASS test_batch_payload_selection")


def test_end_to_end_real_ocr_output() -> None:
    """Run extraction on the real ocr_test_results.json and validate output."""
    if not OCR_RESULTS_PATH.exists():
        print(f"SKIP test_end_to_end_real_ocr_output: {OCR_RESULTS_PATH} not found")
        return

    result = extract_invoice_from_file(OCR_RESULTS_PATH, output_path=OUTPUT_PATH)

    # Document type
    assert result["document_type"] == "invoice", \
        f"document_type: {result['document_type']}"

    # Must have an invoice number extracted from the real data.
    assert result["invoice_number"] is not None, \
        "invoice_number is None for real OCR output"

    # Must have a date.
    assert result["invoice_date"] is not None, \
        "invoice_date is None for real OCR output"

    # Must have at least one line item.
    items = result["line_items"]
    assert len(items) > 0, "No line items extracted from real OCR output"

    # Each item must have an item_number.
    for item in items:
        assert item["item_number"] is not None, f"item_number is None: {item}"

    # Seller name must be non-empty.
    assert result["seller"]["name"], "Seller name is empty"

    # Client name must be non-empty.
    assert result["client"]["name"], "Client name is empty"

    # Summary must have a total_amount with a positive value.
    summary = result["summary"]
    total = summary.get("total_amount")
    assert total is not None, "total_amount is None"
    assert isinstance(total["value"], float) and total["value"] > 0, \
        f"total_amount value unexpected: {total}"

    # Output file must exist.
    assert OUTPUT_PATH.exists(), f"Output file not created: {OUTPUT_PATH}"

    print(
        f"PASS test_end_to_end_real_ocr_output\n"
        f"  invoice_number : {result['invoice_number']}\n"
        f"  invoice_date   : {result['invoice_date']}\n"
        f"  seller         : {result['seller']['name']}\n"
        f"  client         : {result['client']['name']}\n"
        f"  line_items     : {len(items)} items\n"
        f"  total_amount   : {total}\n"
        f"  output written : {OUTPUT_PATH}"
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    test_parse_amount_european,
    test_parse_amount_anglo,
    test_parse_amount_dollar_prefix,
    test_parse_amount_edge_cases,
    test_extract_invoice_number,
    test_extract_invoice_date,
    test_column_discovery,
    test_line_items_synthetic,
    test_seller_client_extraction,
    test_summary_extraction_total_row,
    test_unsupported_payload_raises,
    test_empty_text_blocks_raises,
    test_batch_payload_selection,
    test_end_to_end_real_ocr_output,
]


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Information Extractor — Test Suite")
    print("=" * 60)

    passed = 0
    failed = 0
    failures: list[tuple[str, Exception]] = []

    for test_fn in TESTS:
        try:
            test_fn()
            passed += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            failures.append((test_fn.__name__, exc))
            print(f"FAIL {test_fn.__name__}: {type(exc).__name__}: {exc}")

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(TESTS)} tests")
    if failures:
        print("\nFailed tests:")
        for name, exc in failures:
            print(f"  - {name}: {exc}")
        raise SystemExit(1)
    else:
        print("All tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
