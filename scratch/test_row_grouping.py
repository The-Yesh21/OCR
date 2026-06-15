import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_register_extractor import MarketRegisterExtractor

with open('outputs/test1_ocr.json', encoding='utf-8') as f:
    en_payload = json.load(f)
with open('outputs/test1_ocr_kannada.json', encoding='utf-8') as f:
    ka_payload = json.load(f)

extractor = MarketRegisterExtractor()
en_blocks = extractor._normalize_blocks(en_payload.get("text_blocks", []))
ka_blocks = extractor._normalize_blocks(ka_payload.get("text_blocks", []))
merged_blocks = extractor._merge_blocks(en_blocks, ka_blocks)

# Filter table blocks
table_blocks = [b for b in merged_blocks if 500.0 <= b.y_min < 1300.0]
rows = extractor._group_into_rows(table_blocks, tolerance=25.0)

for idx, r in enumerate(rows):
    print(f"\nRow {idx+1} (avg y_center = {sum(b.y_center for b in r)/len(r):.1f}):")
    for b in r:
        print(f"  Block: x_center={b.x_center:.1f}, x_span=[{b.x_min:.1f}, {b.x_max:.1f}], y_span=[{b.y_min:.1f}, {b.y_max:.1f}]")
        print(f"    en: {b.en_text!r}, ka: {b.ka_text!r}")
