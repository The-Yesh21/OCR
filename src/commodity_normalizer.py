"""Commodity normalization layer using Unicode cleanup, OCR correction rules, and fuzzy matching."""

from __future__ import annotations
import re
import difflib

# Exact mappings for known OCR errors from test1.jpeg and others
EXACT_MAPPINGS = {
    "ಎಂ.೫": "ಎಂ-೫",
    "ಕೋಸು-ಚಪಾಶ": "ಕೋಸು-ಚಪಾತಿ",
    "ಸುನಾಖು": "ಸೌತೆಕಾಯಿ",
    "ಸುನಾವು": "ಸೌತೆಕಾಯಿ",
    "ಕೋಸು-ಸಾೈಂ": "ಕೋಸು-ಸಣ್ಣ",
    "ಯಳವನ": "ಮೂಲಂಗಿ",
    "ಸೀವುಬದನ": "ಸೀಮೆಬದನೆ",
    "ಸೀವಬದನೆ": "ಸೀಮೆಬದನೆ",
    "ಗುಂಡುಬದನೆ": "ಗುಂಡುಬದನೆ",
    "ಬಜಿ-ನಾಟ": "ಬಜ್ಜಿ-ನಾಟಿ",
    "ಬಜಿ-ನಾಟಿ": "ಬಜ್ಜಿ-ನಾಟಿ",
    "ಕಂಬಳಕಾಯು": "ಕುಂಬಳಕಾಯಿ",
    "ಬಜಿ-ಯಕೋನ": "ಬಜ್ಜಿ-ಲೋಕಲ್",
    "೫ೀರೆಕಯು": "ಹೀರೆಕಾಯಿ",
    "ಬೆಂಡಕಾಯು": "ಬೆಂಡೆಕಾಯಿ",
    "ಬೆಂಡೆಕಾಯು": "ಬೆಂಡೆಕಾಯಿ",
    "ಪಡವಲ": "ಪಡವಲಕಾಯಿ",
    "ಟವೋಟ-ಹುಳ": "ಟೊಮ್ಯಾಟೊ-ಹುಳಿ",
    "ಟವೋಟ--ಹುಳಿ": "ಟೊಮ್ಯಾಟೊ-ಹುಳಿ",
    "ತೊಂಡಿಕಾಯು": "ತೊಂಡೆಕಾಯಿ",
    "ತೊಂಡೆಕಾಯು": "ತೊಂಡೆಕಾಯಿ",
    "ಬುಲೆಟ್ಮಣಸು": "ಬುಲೆಟ್ ಮೆಣಸು",
    "ಬಟ್ರೋಟ್": "ಬೀಟ್ರೂಟ್",
    "ಹಾಗಲ-ಗೀನ್": "ಹಾಗಲಕಾಯಿ-ಗ್ರೀನ್",
    "ಕಾಲಿಪವರ್": "ಕಾಲಿಫ್ಲವರ್",
    "ಸೌ೫-4೭": "ಸೌತೆಕಾಯಿ",
    "ಸೌ೫--4೭": "ಸೌತೆಕಾಯಿ",
    "ಸೌ೫--4೭?": "ಸೌತೆಕಾಯಿ"
}

STANDARD_COMMODITIES = [
    "ಬೆಂಡೆಕಾಯಿ",
    "ಟೊಮ್ಯಾಟೊ",
    "ಕೋಸು-ಚಪಾತಿ",
    "ಕೋಸು-ಸಣ್ಣ",
    "ಕೋಸು-ಹೈಬ್ರಿಡ್",
    "ಕೋಸು-ಲೋಕಲ್",
    "ಸೀಮೆಬದನೆ",
    "ಗುಂಡುಬದನೆ",
    "ಬಜ್ಜಿ-ನಾಟಿ",
    "ಬಜ್ಜಿ-ಲೋಕಲ್",
    "ಕುಂಬಳಕಾಯಿ",
    "ಹೀರೆಕಾಯಿ",
    "ಪಡವಲಕಾಯಿ",
    "ತೊಂಡೆಕಾಯಿ",
    "ಬುಲೆಟ್ ಮೆಣಸು",
    "ಬೀಟ್ರೂಟ್",
    "ಹಾಗಲಕಾಯಿ-ಗ್ರೀನ್",
    "ಕಾಲಿಫ್ಲವರ್",
    "ಸೌತೆಕಾಯಿ",
    "ಎಂ-೫",
    "ಎಂ-೪",
    "ಮೂಲಂಗಿ",
    "ಬೆಳ್ಳುಳ್ಳಿ",
    "ಶುಂಠಿ",
    "ಈರುಳ್ಳಿ",
    "ಆಲೂಗಡ್ಡೆ",
    "ಕ್ಯಾರೆಟ್"
]

def clean_unicode(text: str) -> str:
    """Perform Unicode cleanup on the commodity name."""
    if not text:
        return ""
    # Strip whitespace and common noise characters
    cleaned = text.strip(". ,:-?!\t\n")
    # Normalize multiple hyphens/spaces
    cleaned = re.sub(r"-{2,}", "-", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned

def apply_ocr_rules(text: str) -> str:
    """Apply OCR correction rules for common PaddleOCR Kannada misreadings."""
    if not text:
        return ""
    
    # 1. Replace Kannada digit '೫' (5) at start with 'ಹ' (often read instead of 'ಹೀ' or 'ಹಾ')
    if text.startswith("೫"):
        text = "ಹ" + text[1:]
        
    # 2. Replace common suffix mismatches
    # e.g., 'ಕಾಯು' / 'ಕಯು' -> 'ಕಾಯಿ'
    if text.endswith("ಕಾಯು") or text.endswith("ಕಯು"):
        text = text[:-4] + "ಕಾಯಿ"
        
    # 3. Handle 'ಬಜಿ' -> 'ಬಜ್ಜಿ'
    text = text.replace("ಬಜಿ", "ಬಜ್ಜಿ")
    
    # 4. Handle 'ಟವೋಟ' -> 'ಟೊಮ್ಯಾಟೊ'
    text = text.replace("ಟವೋಟ", "ಟೊಮ್ಯಾಟೊ")
    
    return text

def is_valid_commodity(name: str) -> bool:
    """Validate commodity name candidate:
    - Must contain Kannada Unicode characters
    - Must have minimum length > 2
    - Must not be a pure English OCR artifact or isolated symbol
    """
    if not name:
        return False
        
    cleaned = clean_unicode(name)
    if len(cleaned) <= 2:
        return False
        
    # Count Kannada characters (U+0C80 to U+0CFF)
    kannada_chars = [c for c in cleaned if '\u0c80' <= c <= '\u0cff']
    if not kannada_chars:
        return False
        
    # Must contain at least 2 Kannada characters to avoid tiny noise fragments
    if len(kannada_chars) < 2:
        return False
        
    # Check if it's pure English OCR artifact or noise
    if cleaned.lower() in ("sh", "hh", "lh", "th", "apmc"):
        return False
        
    return True

def normalize_commodity(raw_name: str) -> dict[str, any]:
    """Normalize a raw commodity name.
    Returns:
        {
            "raw_name": str,
            "normalized_name": str,
            "confidence": float
        }
    """
    cleaned = clean_unicode(raw_name)
    
    # 1. Exact match check
    if cleaned in EXACT_MAPPINGS:
        return {
            "raw_name": raw_name,
            "normalized_name": EXACT_MAPPINGS[cleaned],
            "confidence": 1.0
        }
        
    # 2. Apply rules
    ruled = apply_ocr_rules(cleaned)
    if ruled in EXACT_MAPPINGS:
        return {
            "raw_name": raw_name,
            "normalized_name": EXACT_MAPPINGS[ruled],
            "confidence": 0.95
        }
    if ruled in STANDARD_COMMODITIES:
        return {
            "raw_name": raw_name,
            "normalized_name": ruled,
            "confidence": 0.95
        }
        
    # 3. Fuzzy matching
    best_match = None
    best_ratio = 0.0
    
    for std in STANDARD_COMMODITIES:
        ratio = difflib.SequenceMatcher(None, ruled, std).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = std
            
    # If fuzzy match is good enough (ratio >= 0.60)
    if best_ratio >= 0.60 and best_match:
        return {
            "raw_name": raw_name,
            "normalized_name": best_match,
            "confidence": round(best_ratio, 3)
        }
        
    # Fallback: return the cleaned/ruled name with a lower confidence
    return {
        "raw_name": raw_name,
        "normalized_name": ruled,
        "confidence": 0.50
    }
