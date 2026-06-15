"""Rule-based document classification from OCR text."""

import re
import logging

logger = logging.getLogger(__name__)

# Kannada vegetable/produce keywords commonly found in market registers
KANNADA_MARKET_KEYWORDS = [
    "ಕಾಯು", "ಕೋಸು", "ಬದನೆ", "ಬೆಂಡೆ", "ಪಡವಲ", "ಸೌತೆ", "ಹಾಗಲ", 
    "ಮೆಣಸಿನ", "ಸೋರೆ", "ಮಂಜುನಾಥ", "ತರಕಾರಿ"
]

def classify_document(ocr_text: str) -> str:
    """Classify a document based on text content.

    Returns one of:
    - 'invoice'
    - 'market_register'
    - 'purchase_register'
    - 'sales_register'
    - 'unknown'
    """
    text_lower = ocr_text.lower()
    
    # Check for purchase register keywords
    if any(k in text_lower for k in ["purchase register", "purchase register book", "purchase book"]):
        return "purchase_register"
        
    # Check for sales register keywords
    if any(k in text_lower for k in ["sales register", "sales register book", "sales book", "sales day book"]):
        return "sales_register"
        
    # Check for market register keywords (APMC, vegetable names)
    if "apmc" in text_lower or any(k in ocr_text for k in KANNADA_MARKET_KEYWORDS):
        return "market_register"
        
    # Check for invoice keywords
    invoice_keywords = ["invoice", "tax invoice", "bill to", "seller:", "client:", "invoice no"]
    if any(k in text_lower for k in invoice_keywords):
        return "invoice"
        
    return "unknown"
