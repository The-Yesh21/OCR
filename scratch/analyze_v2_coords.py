import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

with open('outputs/test1_market_register_v2.json', encoding='utf-8') as f:
    d = json.load(f)

print(f"Market name: {d['market_name']}")
print(f"Date: {d['date']}")
print("\nExtracted Items:")
for i, item in enumerate(d['items']):
    norm = item.get('normalization', {})
    print(f"Item {i+1:2d}: {item['commodity']:20s} | Price: {item['price']:5s} | Raw: {norm.get('raw_name')}")
