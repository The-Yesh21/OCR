import sys
from pathlib import Path
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from market_register_extractor import MarketRegisterExtractor

en_payload = {
    "image_path": "datasets\\test1.jpeg",
    "language": "en",
    "text_blocks": [
        {"text": "MYS-APMC", "confidence": 0.99, "bbox": [[150, 1320], [390, 1320], [390, 1380], [150, 1380]]},
        {"text": "1 8 MAR 2026", "confidence": 0.99, "bbox": [[620, 1320], [960, 1320], [960, 1380], [620, 1380]]},
        {"text": "5", "confidence": 0.95, "bbox": [[430, 740], [510, 740], [510, 780], [430, 780]]},
        {"text": "25", "confidence": 0.95, "bbox": [[840, 740], [960, 740], [960, 780], [840, 780]]},
        {"text": "8", "confidence": 0.95, "bbox": [[420, 800], [510, 800], [510, 860], [420, 860]]},
        {"text": "16", "confidence": 0.95, "bbox": [[850, 800], [960, 800], [960, 860], [850, 860]]},
    ]
}

ka_payload = {
    "image_path": "datasets\\test1.jpeg",
    "language": "ka",
    "text_blocks": [
        {"text": "MYS-APMC", "confidence": 0.90, "bbox": [[150, 1320], [390, 1320], [390, 1380], [150, 1380]]},
        {"text": "I g MaR 2020.", "confidence": 0.85, "bbox": [[620, 1320], [960, 1320], [960, 1380], [620, 1380]]},
        {"text": "ಗುಂಡುಬದನೆ", "confidence": 0.92, "bbox": [[180, 740], [360, 740], [360, 780], [180, 780]]},
        {"text": "ಬಜಿ-ನಾಟಿ", "confidence": 0.91, "bbox": [[590, 740], [730, 740], [730, 780], [590, 780]]},
        {"text": "ಕಂಬಳಕಾಯು", "confidence": 0.92, "bbox": [[180, 800], [330, 800], [330, 860], [180, 860]]},
        {"text": "ಸೌ೫--4೭?", "confidence": 0.88, "bbox": [[600, 800], [770, 800], [770, 860], [600, 860]]},
    ]
}

extractor = MarketRegisterExtractor()
result = extractor.extract(en_payload, ka_payload)
print(json.dumps(result["items"], ensure_ascii=False, indent=2))
