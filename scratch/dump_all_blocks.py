import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

with open('outputs/test1_ocr_kannada.json', encoding='utf-8') as f:
    ka = json.load(f)
with open('outputs/test1_ocr.json', encoding='utf-8') as f:
    en = json.load(f)

print(f"Kannada blocks count: {len(ka['text_blocks'])}")
for i, b in enumerate(ka['text_blocks']):
    print(f"KA {i:2d}: {b['text']!r} (y={b['bbox_xyxy']['y_min']:.1f})")

print(f"\nEnglish blocks count: {len(en['text_blocks'])}")
for i, b in enumerate(en['text_blocks']):
    print(f"EN {i:2d}: {b['text']!r} (y={b['bbox_xyxy']['y_min']:.1f})")
