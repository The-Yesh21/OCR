import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

with open('outputs/test1_ocr.json', encoding='utf-8') as f:
    d = json.load(f)
for i, b in enumerate(d['text_blocks']):
    print(f"{i}: {b['text']} (conf={b['confidence']:.2f}, bbox_xyxy={b.get('bbox_xyxy') or b.get('bbox')})")
