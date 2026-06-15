import sys
from pathlib import Path
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ocr_fusion import OCRResultFusion

with open('outputs/test1_ocr.json', encoding='utf-8') as f:
    full_en = json.load(f)

# Run fusion mapping
fusion = OCRResultFusion()
mapped_en = fusion.map_coordinates(full_en.get("text_blocks", []), "original")

# Let's find blocks matching "MAR" or containing digits and see what they are mapped to
for b in mapped_en:
    if "MAR" in b["text"] or "2026" in b["text"] or "MYS" in b["text"]:
        print(f"Mapped Block: {b['text']} | bbox_xyxy={b['bbox_xyxy']}")

groups = fusion.group_overlapping_blocks(mapped_en)
print(f"\nTotal Groups: {len(groups)}")
for i, g in enumerate(groups):
    fused = fusion.fuse_price_group(g)
    if "MAR" in fused["text"] or "2026" in fused["text"] or "MYS" in fused["text"] or "APMC" in fused["text"]:
        print(f"Fused Group {i}: {fused['text']} | bbox_xyxy={fused['bbox_xyxy']}")
