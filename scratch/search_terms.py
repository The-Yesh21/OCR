import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

with open('outputs/test1_ocr_kannada.json', encoding='utf-8') as f:
    ka = json.load(f)
with open('outputs/test1_ocr.json', encoding='utf-8') as f:
    en = json.load(f)

print("KANNADA OCR SEARCH:")
for i, b in enumerate(ka['text_blocks']):
    text = b['text']
    for term in ["ಬುಲೆಟ್", "ಮಣಸು", "ಬಟ್ರೋಟ್", "ಹಾಗಲ", "ಕಾಲಿ", "Sh"]:
        if term in text:
            print(f"  {i}: {text} (conf={b['confidence']:.2f})")

print("\nENGLISH OCR SEARCH:")
for i, b in enumerate(en['text_blocks']):
    text = b['text']
    for term in ["bullet", "beet", "cauli", "Sh", "Green", "green", "menasu", "Menasu"]:
        if term.lower() in text.lower():
            print(f"  {i}: {text} (conf={b['confidence']:.2f})")
