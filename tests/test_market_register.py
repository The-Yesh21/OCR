"""Tests for document classification and market register extraction.

Run from project root:
    python tests/test_market_register.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from document_classifier import classify_document
from market_register_extractor import MarketRegisterExtractor


def test_classify_document() -> None:
    # Invoices
    assert classify_document("Tax Invoice\nInvoice No: 123\nBill to: Acme") == "invoice"
    assert classify_document("Seller: Andrews\nClient: Becker\nTotal: $100") == "invoice"

    # Market registers
    assert classify_document("MYS-APMC\nVegetable market register\nDate: 18 MAR 2026") == "market_register"
    assert classify_document("ತರಕಾರಿ ಮಾರುಕಟ್ಟೆ\nಬದನೆ\nಬೆಂಡೆಕಾಯಿ") == "market_register"

    # Purchase registers
    assert classify_document("Purchase Register Book\nSupplier name\nDate") == "purchase_register"

    # Sales registers
    assert classify_document("Sales Register Book\nCustomer name\nAmount") == "sales_register"

    # Unknown
    assert classify_document("Hello World! This is some random text.") == "unknown"
    print("test_classify_document passed.")


def test_market_register_extractor() -> None:
    # Setup synthetic English and Kannada OCR payloads
    en_payload = {
        "image_path": "datasets\\test1.jpeg",
        "language": "en",
        "text_blocks": [
            # Metadata
            {"text": "MYS-APMC", "confidence": 0.99, "bbox": [[150, 1320], [390, 1320], [390, 1380], [150, 1380]]},
            {"text": "1 8 MAR 2026", "confidence": 0.99, "bbox": [[620, 1320], [960, 1320], [960, 1380], [620, 1380]]},
            
            # Row 1 (Left: price 5, Right: price 25)
            {"text": "5", "confidence": 0.95, "bbox": [[430, 740], [510, 740], [510, 780], [430, 780]]},
            {"text": "25", "confidence": 0.95, "bbox": [[840, 740], [960, 740], [960, 780], [840, 780]]},
            
            # Row 2 (Left: price 8, Right: price 16)
            {"text": "8", "confidence": 0.95, "bbox": [[420, 800], [510, 800], [510, 860], [420, 860]]},
            {"text": "16", "confidence": 0.95, "bbox": [[850, 800], [960, 800], [960, 860], [850, 860]]},
        ]
    }

    ka_payload = {
        "image_path": "datasets\\test1.jpeg",
        "language": "ka",
        "text_blocks": [
            # Metadata
            {"text": "MYS-APMC", "confidence": 0.90, "bbox": [[150, 1320], [390, 1320], [390, 1380], [150, 1380]]},
            {"text": "I g MaR 2020.", "confidence": 0.85, "bbox": [[620, 1320], [960, 1320], [960, 1380], [620, 1380]]},
            
            # Row 1 (Left: Gundu Badane, Right: Bajji Naati)
            {"text": "ಗುಂಡುಬದನೆ", "confidence": 0.92, "bbox": [[180, 740], [360, 740], [360, 780], [180, 780]]},
            {"text": "ಬಜಿ-ನಾಟಿ", "confidence": 0.91, "bbox": [[590, 740], [730, 740], [730, 780], [590, 780]]},
            
            # Row 2 (Left: Kumbalakayi, Right: Southekayi)
            {"text": "ಕಂಬಳಕಾಯು", "confidence": 0.92, "bbox": [[180, 800], [330, 800], [330, 860], [180, 860]]},
            {"text": "ಸೌ೫--4೭?", "confidence": 0.88, "bbox": [[600, 800], [770, 800], [770, 860], [600, 860]]},
        ]
    }

    extractor = MarketRegisterExtractor()
    result = extractor.extract(en_payload, ka_payload)

    # Check document metadata
    assert result["document_type"] == "market_register"
    assert result["market_name"] == "MYS-APMC"
    assert result["date"] == "18 MAR 2026"  # Reconstructed from split digit spaces in English block

    # Check extracted items
    items = result["items"]
    assert len(items) == 4

    assert items[0]["commodity"] == "ಗುಂಡುಬದನೆ"
    assert items[0]["price"] == "5"
    assert "normalization" in items[0]
    assert items[0]["normalization"]["raw_name"] == "ಗುಂಡುಬದನೆ"

    assert items[1]["commodity"] == "ಬಜ್ಜಿ-ನಾಟಿ"
    assert items[1]["price"] == "25"
    assert items[1]["normalization"]["raw_name"] == "ಬಜಿ-ನಾಟಿ"

    assert items[2]["commodity"] == "ಕುಂಬಳಕಾಯಿ"
    assert items[2]["price"] == "8"
    assert items[2]["normalization"]["raw_name"] == "ಕಂಬಳಕಾಯು"

    assert items[3]["commodity"] == "ಸೌತೆಕಾಯಿ"
    assert items[3]["price"] == "16"
    assert items[3]["normalization"]["raw_name"] == "ಸೌ೫--4೭"

    print("test_market_register_extractor passed.")


if __name__ == "__main__":
    test_classify_document()
    test_market_register_extractor()
    print("All tests passed successfully!")
