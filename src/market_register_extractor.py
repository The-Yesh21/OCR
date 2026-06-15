"""Layout-aware extraction for Kannada market registers.

Extracts tabular commodities and prices from registers using a dual-column
split layout structure.
"""

from __future__ import annotations

import json
import logging
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from information_extractor import TextBlock
from commodity_normalizer import is_valid_commodity, normalize_commodity

logger = logging.getLogger(__name__)

# Regex to match dates like "18 MAR 2026" or "18-03-2026"
DATE_ALPHA_RE = re.compile(
    r"\b\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\s+\d{4}\b",
    re.IGNORECASE
)
DATE_NUM_RE = re.compile(r"\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\b")

# Regex to match APMC identifiers
APMC_RE = re.compile(r"\b[A-Z0-9\-]*APMC[A-Z0-9\-]*\b", re.IGNORECASE)


@dataclass
class MergedBlock:
    """Combines English and Kannada OCR results for a single layout region."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    en_text: str | None = None
    ka_text: str | None = None
    en_conf: float = 0.0
    ka_conf: float = 0.0

    @property
    def x_center(self) -> float:
        return (self.x_min + self.x_max) / 2.0

    @property
    def y_center(self) -> float:
        return (self.y_min + self.y_max) / 2.0

    @property
    def max_conf(self) -> float:
        return max(self.en_conf, self.ka_conf)


class MarketRegisterExtractor:
    """Extractor for Kannada APMC market registers."""

    def __init__(self, min_confidence: float = 0.0) -> None:
        self.min_confidence = min_confidence

    def extract(
        self,
        en_payload: dict[str, Any],
        ka_payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract structured register records from dual English/Kannada OCR payloads."""
        en_blocks = self._normalize_blocks(en_payload.get("text_blocks", []))
        ka_blocks = self._normalize_blocks(ka_payload.get("text_blocks", []))

        merged_blocks = self._merge_blocks(en_blocks, ka_blocks)

        # Extract metadata (Market Name and Date) from all blocks
        market_name = self._extract_market_name(merged_blocks)
        date = self._extract_date(merged_blocks)

        # Filter blocks to only keep the tabular region (500 <= y < 1300)
        table_blocks = [
            b for b in merged_blocks
            if 500.0 <= b.y_min < 1300.0
        ]

        # Group table blocks into horizontal rows
        rows = self._group_into_rows(table_blocks, tolerance=25.0)

        # Refine column boundaries dynamically based on median x positions of all blocks
        boundaries = self._calculate_column_boundaries(merged_blocks)

        # Parse commodities and prices from the rows
        items = []
        for row in rows:
            left_item = self._extract_side(row, is_left=True, boundaries=boundaries)
            if left_item:
                items.append(left_item)
            right_item = self._extract_side(row, is_left=False, boundaries=boundaries)
            if right_item:
                items.append(right_item)

        source_image = en_payload.get("image_name") or en_payload.get("image_path")
        if not source_image:
            source_image = ka_payload.get("image_name") or ka_payload.get("image_path")

        return {
            "document_type": "market_register",
            "source_image": source_image,
            "market_name": market_name,
            "date": date,
            "items": items,
            "metadata": {
                "extraction_method": "hybrid_layout_analysis_v2",
                "text_region_count_en": len(en_blocks),
                "text_region_count_ka": len(ka_blocks),
                "rows_grouped": len(rows),
                "column_boundaries": {
                    "boundary_left_comm_price": round(boundaries[0], 1),
                    "boundary_left_price_right_comm": round(boundaries[1], 1),
                    "boundary_right_comm_price": round(boundaries[2], 1),
                }
            }
        }

    def _calculate_column_boundaries(self, blocks: list[MergedBlock]) -> tuple[float, float, float]:
        """Dynamically determine the column boundaries using median horizontal positions of text blocks."""
        table_blocks = [b for b in blocks if 500.0 <= b.y_min < 1300.0]
        if not table_blocks:
            return 390.0, 550.0, 820.0

        # Initial clustering based on standard expected regions
        left_comm_xs = [b.x_center for b in table_blocks if b.x_center < 390.0]
        left_price_xs = [b.x_center for b in table_blocks if 390.0 <= b.x_center < 550.0]
        right_comm_xs = [b.x_center for b in table_blocks if 550.0 <= b.x_center < 820.0]
        right_price_xs = [b.x_center for b in table_blocks if 820.0 <= b.x_center]

        med_left_comm = statistics.median(left_comm_xs) if left_comm_xs else 250.0
        med_left_price = statistics.median(left_price_xs) if left_price_xs else 450.0
        med_right_comm = statistics.median(right_comm_xs) if right_comm_xs else 680.0
        med_right_price = statistics.median(right_price_xs) if right_price_xs else 900.0

        # Boundaries are the midpoints between column centers (simulates vertical line positions)
        boundary_1 = (med_left_comm + med_left_price) / 2.0
        boundary_2 = (med_left_price + med_right_comm) / 2.0
        boundary_3 = (med_right_comm + med_right_price) / 2.0

        logger.info(f"Dynamic column boundaries: {boundary_1:.1f}, {boundary_2:.1f}, {boundary_3:.1f}")
        return boundary_1, boundary_2, boundary_3

    def _normalize_blocks(self, raw_blocks: list[dict[str, Any]]) -> list[TextBlock]:
        """Convert raw OCR dicts into sorted TextBlock objects."""
        blocks: list[TextBlock] = []
        for raw in raw_blocks:
            text = str(raw.get("text", "")).strip()
            if not text:
                continue
            confidence = float(raw.get("confidence", 1.0))
            if confidence < self.min_confidence:
                continue
            bbox_xyxy = raw.get("bbox_xyxy") or self._bbox_to_xyxy(raw.get("bbox", []))
            blocks.append(
                TextBlock(
                    text=text,
                    confidence=confidence,
                    bbox=raw.get("bbox", []),
                    x_min=float(bbox_xyxy["x_min"]),
                    y_min=float(bbox_xyxy["y_min"]),
                    x_max=float(bbox_xyxy["x_max"]),
                    y_max=float(bbox_xyxy["y_max"]),
                )
            )
        return sorted(blocks, key=lambda b: (b.y_min, b.x_min))

    @staticmethod
    def _bbox_to_xyxy(bbox: list[list[float]]) -> dict[str, float]:
        if not bbox:
            return {"x_min": 0.0, "y_min": 0.0, "x_max": 0.0, "y_max": 0.0}
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        return {"x_min": min(xs), "y_min": min(ys), "x_max": max(xs), "y_max": max(ys)}

    def _merge_blocks(
        self,
        en_blocks: list[TextBlock],
        ka_blocks: list[TextBlock]
    ) -> list[MergedBlock]:
        """Align English and Kannada blocks based on bounding box overlap (IoU)."""
        merged: list[MergedBlock] = []
        unmatched_en = en_blocks.copy()
        unmatched_ka = ka_blocks.copy()

        # Match Kannada blocks to English blocks
        for kb in ka_blocks:
            best_iou = 0.0
            best_eb = None
            for eb in unmatched_en:
                iou = self._calculate_iou(kb, eb)
                if iou > best_iou:
                    best_iou = iou
                    best_eb = eb

            if best_iou > 0.10 and best_eb is not None:
                merged.append(
                    MergedBlock(
                        x_min=min(kb.x_min, best_eb.x_min),
                        y_min=min(kb.y_min, best_eb.y_min),
                        x_max=max(kb.x_max, best_eb.x_max),
                        y_max=max(kb.y_max, best_eb.y_max),
                        en_text=best_eb.text,
                        ka_text=kb.text,
                        en_conf=best_eb.confidence,
                        ka_conf=kb.confidence,
                    )
                )
                unmatched_en.remove(best_eb)
                unmatched_ka.remove(kb)

        # Add remaining unmatched blocks
        for eb in unmatched_en:
            merged.append(
                MergedBlock(
                    x_min=eb.x_min, y_min=eb.y_min, x_max=eb.x_max, y_max=eb.y_max,
                    en_text=eb.text, ka_text=None, en_conf=eb.confidence, ka_conf=0.0
                )
            )
        for kb in unmatched_ka:
            merged.append(
                MergedBlock(
                    x_min=kb.x_min, y_min=kb.y_min, x_max=kb.x_max, y_max=kb.y_max,
                    en_text=None, ka_text=kb.text, en_conf=0.0, ka_conf=kb.confidence
                )
            )

        return sorted(merged, key=lambda b: (b.y_min, b.x_min))

    @staticmethod
    def _calculate_iou(b1: TextBlock, b2: TextBlock) -> float:
        x_left = max(b1.x_min, b2.x_min)
        y_top = max(b1.y_min, b2.y_min)
        x_right = min(b1.x_max, b2.x_max)
        y_bottom = min(b1.y_max, b2.y_max)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = (b1.x_max - b1.x_min) * (b1.y_max - b1.y_min)
        area2 = (b2.x_max - b2.x_min) * (b2.y_max - b2.y_min)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _extract_market_name(self, blocks: list[MergedBlock]) -> str | None:
        """Find and clean the APMC market name from blocks."""
        for b in blocks:
            text = b.en_text or b.ka_text or ""
            match = APMC_RE.search(text)
            if match:
                cleaned = match.group(0).strip("- ")
                return cleaned.upper()
        return None

    def _extract_date(self, blocks: list[MergedBlock]) -> str | None:
        """Find and format the register date from blocks."""
        for b in blocks:
            text = b.en_text or ""
            normalized_text = re.sub(r"(?<=\b\d)\s+(?=\d\b)", "", text)
            alpha_match = DATE_ALPHA_RE.search(normalized_text)
            if alpha_match:
                return alpha_match.group(0)
            num_match = DATE_NUM_RE.search(normalized_text)
            if num_match:
                return num_match.group(1)

        for b in blocks:
            text = b.ka_text or ""
            normalized_text = re.sub(r"(?<=\b\d)\s+(?=\d\b)", "", text)
            alpha_match = DATE_ALPHA_RE.search(normalized_text)
            if alpha_match:
                return alpha_match.group(0)
        return None

    def _group_into_rows(
        self,
        blocks: list[MergedBlock],
        tolerance: float = 25.0
    ) -> list[list[MergedBlock]]:
        """Group blocks into rows based on vertical center coordinates."""
        rows: list[list[MergedBlock]] = []
        for b in blocks:
            y_center = b.y_center
            placed = False
            for r in rows:
                avg_y = sum(x.y_center for x in r) / len(r)
                if abs(y_center - avg_y) < tolerance:
                    r.append(b)
                    placed = True
                    break
            if not placed:
                rows.append([b])

        rows.sort(key=lambda r: sum(x.y_center for x in r) / len(r))
        for r in rows:
            r.sort(key=lambda b: b.x_min)
        return rows

    def _extract_side(
        self,
        row: list[MergedBlock],
        is_left: bool,
        boundaries: tuple[float, float, float]
    ) -> dict[str, Any] | None:
        """Extract commodity and price for either the left or right side of a row."""
        b1, b2, b3 = boundaries

        if is_left:
            comm_blocks = [b for b in row if b.x_center < b1]
            price_blocks = [b for b in row if b1 <= b.x_center < b2]
        else:
            comm_blocks = [b for b in row if b2 <= b.x_center < b3]
            price_blocks = [b for b in row if b3 <= b.x_center]

        # Ignore sides with no commodity blocks
        if not comm_blocks:
            return None

        # Clean commodity text: prefer Kannada, fallback to English
        comm_text = comm_blocks[0].ka_text or comm_blocks[0].en_text or ""
        comm_text = comm_text.strip(". ,:-?!\t\n")

        # Validate commodity
        if not is_valid_commodity(comm_text):
            logger.info(f"Commodity rejected: {comm_text!r}")
            return None

        # Normalize commodity
        norm_res = normalize_commodity(comm_text)
        normalized_commodity = norm_res["normalized_name"]

        # Validate and extract price
        def extract_numeric(text: str | None) -> str | None:
            if not text:
                return None
            cleaned = text.strip(". ,:-?!\t\n")
            # Never assign H, HH, 研 as prices
            if cleaned in ("H", "HH", "研"):
                return None
            
            # Map obvious misread single characters
            if cleaned == "I":
                cleaned = "1"
            elif cleaned == "S":
                cleaned = "5"
                
            # Search for integer or decimal values
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

        # Try extracting from the primary price block if available
        price_val = None
        primary_price_block = price_blocks[0] if price_blocks else None
        if primary_price_block:
            price_val = extract_numeric(primary_price_block.en_text) or extract_numeric(primary_price_block.ka_text)

        # If primary price block is invalid/empty/non-numeric:
        if price_val is None:
            # Search neighboring blocks in the same row/side
            candidates = []
            for b in row:
                if b == comm_blocks[0]:
                    continue
                if b == primary_price_block:
                    continue
                if is_left and b.x_center >= b2:
                    continue
                if not is_left and b.x_center < b2:
                    continue

                num_en = extract_numeric(b.en_text)
                if num_en:
                    candidates.append((num_en, b.en_conf))
                num_ka = extract_numeric(b.ka_text)
                if num_ka:
                    candidates.append((num_ka, b.ka_conf))

            if candidates:
                # Choose the highest-confidence numeric candidate
                candidates.sort(key=lambda x: x[1], reverse=True)
                price_val = candidates[0][0]

        price_text = price_val if price_val is not None else ""

        return {
            "commodity": normalized_commodity,
            "price": price_text,
            "normalization": {
                "raw_name": comm_text,
                "normalized_name": normalized_commodity,
                "confidence": norm_res["confidence"]
            }
        }
