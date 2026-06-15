"""OCR Result Fusion Layer for Kannada market register extraction.

Integrates OCR results from full-image, region-based, and multi-scale runs,
aligning them using IoU bounding-box overlap and executing voting algorithms
for commodity names and price resolution.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def calculate_iou(box1: dict[str, float], box2: dict[str, float]) -> float:
    """Calculate Intersection-over-Union (IoU) of two axis-aligned bounding boxes."""
    x1 = max(box1["x_min"], box2["x_min"])
    y1 = max(box1["y_min"], box2["y_min"])
    x2 = min(box1["x_max"], box2["x_max"])
    y2 = min(box1["y_max"], box2["y_max"])

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1["x_max"] - box1["x_min"]) * (box1["y_max"] - box1["y_min"])
    area2 = (box2["x_max"] - box2["x_min"]) * (box2["y_max"] - box2["y_min"])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def extract_numeric(text: str | None) -> str | None:
    """Extract clean numeric value from text string, rejecting non-numeric noise."""
    if not text:
        return None
    cleaned = text.strip(". ,:-?!\t\n")
    if cleaned in ("H", "HH", "研"):
        return None
        
    # Standard character correction
    if cleaned == "I":
        cleaned = "1"
    elif cleaned == "S":
        cleaned = "5"

    match = re.search(r"\d+(?:\.\d+)?", cleaned)
    if match:
        val_str = match.group(0)
        try:
            val = float(val_str)
            if 0.0 <= val <= 500.0:
                if val.is_integer():
                    return str(int(val))
                return str(val)
        except ValueError:
            pass
    return None


class OCRResultFusion:
    """Fuses multi-run OCR results (full, region, scaled) using coordinate mapping and voting."""

    def __init__(self, iou_threshold: float = 0.3) -> None:
        self.iou_threshold = iou_threshold

    def map_coordinates(
        self,
        blocks: list[dict[str, Any]],
        source_type: str,
        region_type: str | None = None
    ) -> list[dict[str, Any]]:
        """Normalize coordinates from region-cropped or scaled images back to the original space."""
        mapped = []
        for b in blocks:
            new_b = b.copy()
            bbox = b.get("bbox", [])
            bbox_xyxy = b.get("bbox_xyxy", {})

            if not bbox or not bbox_xyxy:
                continue

            x_offset = 0.0
            y_offset = 0.0
            scale = 1.0

            if source_type == "region" and region_type:
                if region_type == "header":
                    x_offset = 0.0
                    y_offset = 0.0
                elif region_type == "left_table":
                    x_offset = 0.0
                    y_offset = 500.0
                elif region_type == "right_table":
                    x_offset = 550.0
                    y_offset = 500.0
                elif region_type == "footer":
                    x_offset = 0.0
                    y_offset = 1300.0
            elif source_type == "scaled_2x":
                scale = 2.0

            # Map the bbox_xyxy
            new_xyxy = {
                "x_min": (bbox_xyxy["x_min"] / scale) + x_offset,
                "y_min": (bbox_xyxy["y_min"] / scale) + y_offset,
                "x_max": (bbox_xyxy["x_max"] / scale) + x_offset,
                "y_max": (bbox_xyxy["y_max"] / scale) + y_offset,
            }
            new_b["bbox_xyxy"] = new_xyxy

            # Map polygon bbox points
            new_bbox = []
            for pt in bbox:
                new_bbox.append([
                    (pt[0] / scale) + x_offset,
                    (pt[1] / scale) + y_offset
                ])
            new_b["bbox"] = new_bbox
            new_b["source_run"] = f"{source_type}_{region_type}" if region_type else source_type
            mapped.append(new_b)

        return mapped

    def group_overlapping_blocks(self, all_blocks: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """Group blocks from different runs together based on bounding box overlap (IoU)."""
        groups: list[list[dict[str, Any]]] = []
        
        # Sort blocks by area descending so we group around the most prominent detections
        sorted_blocks = sorted(
            all_blocks,
            key=lambda b: (b["bbox_xyxy"]["x_max"] - b["bbox_xyxy"]["x_min"]) * (b["bbox_xyxy"]["y_max"] - b["bbox_xyxy"]["y_min"]),
            reverse=True
        )

        for b in sorted_blocks:
            matched_group = None
            for g in groups:
                # Compare IoU with any block in the group
                if any(calculate_iou(b["bbox_xyxy"], gb["bbox_xyxy"]) > self.iou_threshold for gb in g):
                    matched_group = g
                    break
            
            if matched_group is not None:
                matched_group.append(b)
            else:
                groups.append([b])

        return groups

    def fuse_commodity_group(self, group: list[dict[str, Any]]) -> dict[str, Any]:
        """Perform voting on a group of Kannada commodity blocks."""
        # Count frequency of each text string
        text_freqs = {}
        for b in group:
            txt = b["text"].strip()
            text_freqs[txt] = text_freqs.get(txt, 0) + 1

        # Find max frequency
        max_freq = max(text_freqs.values())
        candidates = [t for t, f in text_freqs.items() if f == max_freq]

        if len(candidates) == 1:
            chosen_text = candidates[0]
        else:
            # If tie, choose the candidate that has the highest confidence in the group
            chosen_text = max(group, key=lambda b: b["confidence"])["text"].strip()

        # The bounding box is inherited from the highest confidence block in the group
        best_block = max(group, key=lambda b: b["confidence"])
        
        return {
            "text": chosen_text,
            "confidence": best_block["confidence"],
            "bbox": best_block["bbox"],
            "bbox_xyxy": best_block["bbox_xyxy"],
            "metadata": {
                "fused_sources": [b.get("source_run", "unknown") for b in group],
                "vote_count": len(group)
            }
        }

    def fuse_price_group(self, group: list[dict[str, Any]]) -> dict[str, Any]:
        """Perform voting on a group of English price blocks, resolving decimal anomalies."""
        numeric_freqs = {}
        for b in group:
            num = extract_numeric(b["text"])
            if num is not None:
                numeric_freqs[num] = numeric_freqs.get(num, 0) + 1

        if numeric_freqs:
            # Find the most frequent numeric value (voting)
            max_freq = max(numeric_freqs.values())
            candidates = [n for n, f in numeric_freqs.items() if f == max_freq]
            
            if len(candidates) == 1:
                chosen_text = candidates[0]
            else:
                # If tie, select the one associated with the highest confidence block
                best_num_block = max(
                    [b for b in group if extract_numeric(b["text"]) in candidates],
                    key=lambda b: b["confidence"]
                )
                chosen_text = extract_numeric(best_num_block["text"])
        else:
            # Non-numeric text (metadata like market name or date): perform standard text voting
            text_freqs = {}
            for b in group:
                txt = b["text"].strip()
                text_freqs[txt] = text_freqs.get(txt, 0) + 1

            max_freq = max(text_freqs.values())
            candidates = [t for t, f in text_freqs.items() if f == max_freq]

            if len(candidates) == 1:
                chosen_text = candidates[0]
            else:
                chosen_text = max(group, key=lambda b: b["confidence"])["text"].strip()

        # Bounding box from the highest confidence block
        best_block = max(group, key=lambda b: b["confidence"])
        
        return {
            "text": chosen_text,
            "confidence": best_block["confidence"],
            "bbox": best_block["bbox"],
            "bbox_xyxy": best_block["bbox_xyxy"],
            "metadata": {
                "fused_sources": [b.get("source_run", "unknown") for b in group],
                "vote_count": len(group)
            }
        }

    def fuse_payloads(
        self,
        full_payload: dict[str, Any],
        region_payloads: list[tuple[dict[str, Any], str]],
        scaled_payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Fuse multiple OCR runs (original, region-based, 2x scaled) into a single result."""
        # 1. Map all coordinates back to original image space
        all_blocks = []
        
        # Add original blocks
        all_blocks.extend(self.map_coordinates(full_payload.get("text_blocks", []), "original"))
        
        # Add region blocks
        for rp, rtype in region_payloads:
            all_blocks.extend(self.map_coordinates(rp.get("text_blocks", []), "region", region_type=rtype))
            
        # Add scaled blocks
        all_blocks.extend(self.map_coordinates(scaled_payload.get("text_blocks", []), "scaled_2x"))

        # 2. Separate by language (since English is prices and Kannada is commodities)
        en_blocks = [b for b in all_blocks if b.get("source_run", "").startswith("original") or b.get("text", "").isascii() or not any('\u0c80' <= c <= '\u0cff' for c in b.get("text", ""))]
        ka_blocks = [b for b in all_blocks if any('\u0c80' <= c <= '\u0cff' for c in b.get("text", ""))]

        # 3. Group and fuse Kannada (commodity) blocks
        ka_groups = self.group_overlapping_blocks(ka_blocks)
        fused_ka = []
        for g in ka_groups:
            fused_ka.append(self.fuse_commodity_group(g))

        # 4. Group and fuse English (price) blocks
        en_groups = self.group_overlapping_blocks(en_blocks)
        fused_en = []
        for g in en_groups:
            fused_en.append(self.fuse_price_group(g))

        # Reconstruct the fused payload structure
        fused_text_blocks = fused_en + fused_ka
        full_text = "\n".join(b["text"] for b in fused_text_blocks)
        
        return {
            "image_path": full_payload.get("image_path"),
            "language": "hybrid_fused",
            "full_text": full_text,
            "text_blocks": fused_text_blocks,
            "metadata": {
                "fused_runs": ["original", "region_crops", "scaled_2x"],
                "total_blocks_before_fusion": len(all_blocks),
                "total_blocks_after_fusion": len(fused_text_blocks),
            }
        }
