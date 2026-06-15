"""Rule-based information extraction from OCR output.

Strategy
--------
All extraction is driven by bounding-box geometry + regular expressions.
No machine-learning models are used.

Phases
~~~~~~
1. **Header fields** – regex over the full OCR text string to pull
   invoice_number, invoice_date.

2. **Seller / Client** – locate the "Seller:" and "Client:" anchor blocks,
   derive a page midpoint from all text blocks, then collect address lines in
   the left (seller) and right (client) x-columns.  Tax IDs are extracted from
   "Tax Id:" blocks found in each column.

3. **Column discovery** – find the ITEMS header row and read the x-centers of
   the known column-header labels (No., Description, Qty, UM, Net price,
   Net worth, VAT [%], Gross).  Column boundaries are set to midpoints between
   adjacent header centers.  This makes the extractor resilient to minor layout
   shifts across invoice variants.

4. **Row grouping** – blocks matching ``r'^\\d+\\.$'`` at small x (row-number column)
   between ITEMS and SUMMARY are item anchors.  For each anchor, all blocks
   whose y_min falls in [anchor.y_min - gap, next_anchor.y_min] are collected
   and assigned to columns.  Multi-line description lines are joined.

5. **Summary** – locate the SUMMARY section label, then find the "Total"
   keyword row.  Blocks horizontally to the right of "Total" on the same y-band
   are subtotal, tax, total in order.  Falls back to regex on full text.

6. **Amount parsing** – handles European "1 394,67" and Anglo "1,394.67" styles.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled regular expressions
# ---------------------------------------------------------------------------

INVOICE_NUMBER_RE = re.compile(
    r"\bInvoice\s*(?:no\.?|number|#)?[.:#]?\s*([A-Z0-9][A-Z0-9/\-]{1,})",
    re.IGNORECASE,
)
DATE_RE = re.compile(r"\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\b")
TAX_ID_RE = re.compile(r"Tax\s*Id[.:#]?\s*([\w\-]+)", re.IGNORECASE)
ITEM_NUMBER_RE = re.compile(r"^(\d+)\.$")
VAT_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*%")
AMOUNT_SPACED_RE = re.compile(r"[\d\s]+[,.][\d]{2}")

# Labels used to discover column x-ranges from the header row.
# Each entry: (canonical_name, list of OCR text variants, case_sensitive)
COLUMN_HEADERS: list[tuple[str, list[str], bool]] = [
    ("no",          ["No.", "No"],           True),
    ("description", ["Description"],         False),
    ("qty",         ["Qty", "QTY"],          True),
    ("unit",        ["UM", "Unit"],          True),
    ("net_price",   ["Net price"],           False),
    ("net_worth",   ["Net worth"],           False),
    ("vat",         ["VAT [%]", "VAT"],      False),
    ("gross",       ["Gross", "Gross worth"],False),
]

# Fallback hardcoded x-center column ranges (pixels) derived from batch1-0001.
# Used when header discovery fails for a column.
FALLBACK_COLUMNS: dict[str, tuple[float, float]] = {
    "no":          (130,  220),
    "description": (220,  660),
    "qty":         (660,  750),
    "unit":        (750,  850),
    "net_price":   (850,  1040),
    "net_worth":   (1040, 1240),
    "vat":         (1240, 1360),
    "gross":       (1360, 1600),
}

# Row-grouping tolerance: how many pixels above an anchor's y_min to look.
ROW_Y_TOLERANCE: float = 12.0
# Vertical tolerance for deciding two blocks are on the "same row".
SAME_ROW_TOLERANCE: float = 14.0


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class TextBlock:
    """OCR text block with geometry helpers."""

    text: str
    confidence: float
    bbox: list[list[float]]
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def x_center(self) -> float:
        return (self.x_min + self.x_max) / 2.0

    @property
    def y_center(self) -> float:
        return (self.y_min + self.y_max) / 2.0

    @property
    def height(self) -> float:
        return self.y_max - self.y_min


@dataclass
class ColumnMap:
    """Discovered x-range for each invoice column."""

    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)

    def x_range(self, column: str) -> tuple[float, float]:
        return self.ranges.get(column, FALLBACK_COLUMNS.get(column, (0, float("inf"))))

    def assign(self, block: TextBlock) -> str | None:
        """Return the column name whose x-range contains block.x_center."""
        xc = block.x_center
        for col, (lo, hi) in self.ranges.items():
            if lo <= xc < hi:
                return col
        return None


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------

class InvoiceInformationExtractor:
    """Extract structured invoice fields from OCR JSON output.

    Parameters
    ----------
    min_confidence:
        Blocks below this threshold are skipped during extraction.
    """

    def __init__(self, min_confidence: float = 0.0) -> None:
        self.min_confidence = min_confidence

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract_from_file(self, input_path: str | Path) -> dict[str, Any]:
        """Load OCR JSON from disk and extract the invoice record."""
        path = Path(input_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return self.extract(payload)

    def extract(self, ocr_payload: dict[str, Any]) -> dict[str, Any]:
        """Extract a structured invoice from an OCR result dictionary.

        Accepts both:
        - A single OCR result  (has ``text_blocks`` at the top level)
        - A batch result produced by ``test_ocr_engine.py``
          (has ``results`` list; first invoice-category entry is used)
        """
        ocr_result = self._select_ocr_result(ocr_payload)
        blocks = self._normalize_blocks(ocr_result.get("text_blocks", []))
        full_text: str = (
            ocr_result.get("ocr_text_output")
            or ocr_result.get("full_text")
            or ""
        )

        if not blocks:
            raise ValueError("OCR payload does not contain usable text_blocks.")

        logger.info("Extracting from %d OCR blocks", len(blocks))

        column_map = self._discover_columns(blocks)
        items_anchor = self._find_block_by_text(blocks, {"items"})
        summary_anchor = self._find_block_by_text(blocks, {"summary"})

        invoice_number = self._extract_invoice_number(full_text)
        invoice_date   = self._extract_invoice_date(full_text)
        seller, client = self._extract_parties(blocks, items_anchor)
        line_items     = self._extract_line_items(blocks, column_map, items_anchor, summary_anchor)
        summary        = self._extract_summary(blocks, full_text, summary_anchor)

        return {
            "document_type": "invoice",
            "source_image": (
                ocr_result.get("image_name") or ocr_result.get("image_path")
            ),
            "invoice_number": invoice_number,
            "invoice_date":   invoice_date,
            "seller":         seller,
            "client":         client,
            "line_items":     line_items,
            "summary":        summary,
            "metadata": {
                "extraction_method":    "regex_and_bounding_box",
                "ocr_model_language":   ocr_result.get("ocr_model_language") or ocr_result.get("language"),
                "average_ocr_confidence": ocr_result.get("average_confidence"),
                "text_region_count":    len(blocks),
                "columns_discovered":   list(column_map.ranges.keys()),
            },
        }

    def save(
        self,
        structured_record: dict[str, Any],
        output_path: str | Path,
    ) -> None:
        """Save a structured invoice record as formatted JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(structured_record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved structured invoice JSON to %s", path)

    # ------------------------------------------------------------------
    # OCR payload selection
    # ------------------------------------------------------------------

    def _select_ocr_result(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Accept a single OCR result or a batch produced by test_ocr_engine."""
        if "text_blocks" in payload:
            return payload
        results = payload.get("results")
        if isinstance(results, list) and results:
            invoice_results = [r for r in results if r.get("category") == "invoice"]
            return invoice_results[0] if invoice_results else results[0]
        raise ValueError(
            "Unsupported OCR JSON format: no 'text_blocks' or 'results' key found."
        )

    # ------------------------------------------------------------------
    # Block normalisation
    # ------------------------------------------------------------------

    def _normalize_blocks(
        self, raw_blocks: list[dict[str, Any]]
    ) -> list[TextBlock]:
        """Convert raw OCR dicts into TextBlock objects sorted top-to-bottom."""
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

    # ------------------------------------------------------------------
    # Column discovery
    # ------------------------------------------------------------------

    def _discover_columns(self, blocks: list[TextBlock]) -> ColumnMap:
        """Build a ColumnMap by locating known header labels in the block list.

        The header row is identified as the band containing both "Description"
        and "Qty" (or their variants).  Column boundaries are midpoints between
        adjacent header x-centers.
        """
        # Collect x-centers for each canonical column name.
        # Always use case-insensitive matching for header discovery so that
        # OCR capitalisation variants ("Qty", "QTY", "qty") all match.
        header_xc: dict[str, float] = {}
        for canonical, variants, _case_sensitive in COLUMN_HEADERS:
            block = self._find_block_by_text(
                blocks,
                set(variants),           # original strings; matching is always CI
                case_insensitive=True,
            )
            if block is not None:
                header_xc[canonical] = block.x_center

        if len(header_xc) < 3:
            logger.warning(
                "Column header discovery found only %d/%d headers; "
                "falling back to hardcoded ranges.",
                len(header_xc),
                len(COLUMN_HEADERS),
            )
            return ColumnMap(ranges=dict(FALLBACK_COLUMNS))

        # Sort discovered columns by x-center.
        ordered = sorted(header_xc.items(), key=lambda kv: kv[1])

        # Build column ranges: each column spans from midpoint with previous
        # to midpoint with next.
        ranges: dict[str, tuple[float, float]] = {}
        xcs = [xc for _, xc in ordered]
        names = [name for name, _ in ordered]

        for i, (name, xc) in enumerate(ordered):
            lo = (xcs[i - 1] + xc) / 2.0 if i > 0 else 0.0
            hi = (xc + xcs[i + 1]) / 2.0 if i < len(ordered) - 1 else float("inf")
            ranges[name] = (lo, hi)

        # Ensure any undiscovered columns fall back gracefully.
        for canonical in FALLBACK_COLUMNS:
            if canonical not in ranges:
                ranges[canonical] = FALLBACK_COLUMNS[canonical]

        logger.info(
            "Discovered %d column ranges from header labels: %s",
            len(ranges),
            {k: (round(v[0], 1), round(v[1], 1)) for k, v in ranges.items()},
        )
        return ColumnMap(ranges=ranges)

    # ------------------------------------------------------------------
    # Header extraction
    # ------------------------------------------------------------------

    def _extract_invoice_number(self, full_text: str) -> str | None:
        match = INVOICE_NUMBER_RE.search(full_text)
        return match.group(1) if match else None

    def _extract_invoice_date(self, full_text: str) -> str | None:
        match = DATE_RE.search(full_text)
        return match.group(1) if match else None

    # ------------------------------------------------------------------
    # Seller / Client extraction
    # ------------------------------------------------------------------

    def _extract_parties(
        self,
        blocks: list[TextBlock],
        items_anchor: TextBlock | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Extract seller and client from labeled address columns."""
        seller_label = self._find_block_by_text(blocks, {"seller:"})
        client_label = self._find_block_by_text(blocks, {"client:"})

        if seller_label is None or client_label is None:
            logger.warning("Could not find Seller:/Client: labels.")
            return {"name": None, "address": [], "tax_id": None}, \
                   {"name": None, "address": [], "tax_id": None}

        # Dynamic midpoint between seller and client label x-centers.
        mid_x = (seller_label.x_center + client_label.x_center) / 2.0

        # y-bounds: from just below the label to just above ITEMS.
        y_top = seller_label.y_max
        y_bot = items_anchor.y_min if items_anchor else seller_label.y_max + 400.0

        # Find Tax Id blocks in the party zone.
        tax_blocks = [
            b for b in blocks
            if y_top <= b.y_min < y_bot
            and TAX_ID_RE.search(b.text)
        ]
        seller_tax = self._nearest_x(tax_blocks, seller_label.x_center)
        client_tax = self._nearest_x(tax_blocks, client_label.x_center)

        seller_y_bot = seller_tax.y_min if seller_tax else y_bot
        client_y_bot = client_tax.y_min if client_tax else y_bot

        seller_lines = self._party_address_lines(
            blocks, x_min=0, x_max=mid_x,
            y_min=y_top, y_max=seller_y_bot,
        )
        client_lines = self._party_address_lines(
            blocks, x_min=mid_x, x_max=float("inf"),
            y_min=y_top, y_max=client_y_bot,
        )

        return (
            {
                "name":    seller_lines[0] if seller_lines else None,
                "address": seller_lines[1:],
                "tax_id":  self._parse_tax_id(seller_tax.text) if seller_tax else None,
            },
            {
                "name":    client_lines[0] if client_lines else None,
                "address": client_lines[1:],
                "tax_id":  self._parse_tax_id(client_tax.text) if client_tax else None,
            },
        )

    def _party_address_lines(
        self,
        blocks: list[TextBlock],
        x_min: float, x_max: float,
        y_min: float, y_max: float,
    ) -> list[str]:
        excluded = {"seller:", "client:", "tax id", "iban"}
        lines = [
            b for b in blocks
            if x_min <= b.x_center < x_max
            and y_min <= b.y_min < y_max
            and not any(b.text.lower().startswith(ex) for ex in excluded)
        ]
        return [b.text for b in sorted(lines, key=lambda b: b.y_min)]

    def _parse_tax_id(self, text: str) -> str | None:
        match = TAX_ID_RE.search(text)
        return match.group(1) if match else None

    # ------------------------------------------------------------------
    # Line-item extraction
    # ------------------------------------------------------------------

    def _extract_line_items(
        self,
        blocks: list[TextBlock],
        column_map: ColumnMap,
        items_anchor: TextBlock | None,
        summary_anchor: TextBlock | None,
    ) -> list[dict[str, Any]]:
        """Reconstruct invoice item rows using row-number anchors."""
        if items_anchor is None:
            logger.warning("ITEMS section label not found; skipping line items.")
            return []

        y_items = items_anchor.y_min
        y_summary = summary_anchor.y_min if summary_anchor else float("inf")

        # Item-number blocks: match /^\d+\.$/ in the "no" column.
        no_lo, no_hi = column_map.x_range("no")
        anchors = sorted(
            [
                b for b in blocks
                if y_items < b.y_min < y_summary
                and no_lo <= b.x_center < no_hi
                and ITEM_NUMBER_RE.match(b.text)
            ],
            key=lambda b: b.y_min,
        )

        if not anchors:
            logger.warning("No item-number anchors found between ITEMS and SUMMARY.")
            return []

        items: list[dict[str, Any]] = []
        for idx, anchor in enumerate(anchors):
            next_y = anchors[idx + 1].y_min if idx + 1 < len(anchors) else y_summary
            row_blocks = [
                b for b in blocks
                if anchor.y_min - ROW_Y_TOLERANCE <= b.y_min < next_y - 4.0
                and y_items < b.y_min < y_summary
            ]
            items.append(self._parse_item_row(anchor, row_blocks, column_map))

        return items

    def _parse_item_row(
        self,
        anchor: TextBlock,
        row_blocks: list[TextBlock],
        column_map: ColumnMap,
    ) -> dict[str, Any]:
        """Parse one item row from its collected blocks."""
        desc_lo, desc_hi = column_map.x_range("description")
        desc_blocks = sorted(
            [b for b in row_blocks if desc_lo <= b.x_center < desc_hi],
            key=lambda b: (b.y_min, b.x_min),
        )
        description = " ".join(b.text for b in desc_blocks) or None

        return {
            "item_number":  anchor.text.rstrip("."),
            "description":  description,
            "quantity":     self._col_text(row_blocks, column_map, "qty"),
            "unit":         self._col_text(row_blocks, column_map, "unit"),
            "unit_price":   self._col_amount(row_blocks, column_map, "net_price"),
            "net_worth":    self._col_amount(row_blocks, column_map, "net_worth"),
            "vat_percent":  self._col_percent(row_blocks, column_map, "vat"),
            "gross_worth":  self._col_amount(row_blocks, column_map, "gross"),
        }

    def _col_text(
        self,
        blocks: list[TextBlock],
        column_map: ColumnMap,
        col: str,
    ) -> str | None:
        lo, hi = column_map.x_range(col)
        candidates = sorted(
            [b for b in blocks if lo <= b.x_center < hi],
            key=lambda b: b.y_min,
        )
        return candidates[0].text if candidates else None

    def _col_amount(
        self,
        blocks: list[TextBlock],
        column_map: ColumnMap,
        col: str,
    ) -> dict[str, Any] | None:
        raw = self._col_text(blocks, column_map, col)
        if raw is None:
            return None
        return {"raw": raw, "value": self._parse_amount(raw)}

    def _col_percent(
        self,
        blocks: list[TextBlock],
        column_map: ColumnMap,
        col: str,
    ) -> dict[str, Any] | None:
        raw = self._col_text(blocks, column_map, col)
        if raw is None:
            return None
        match = VAT_RE.search(raw)
        parsed = float(match.group(1).replace(",", ".")) if match else None
        return {"raw": raw, "value": parsed}

    # ------------------------------------------------------------------
    # Summary extraction
    # ------------------------------------------------------------------

    def _extract_summary(
        self,
        blocks: list[TextBlock],
        full_text: str,
        summary_anchor: TextBlock | None,
    ) -> dict[str, Any]:
        """Extract subtotal, tax, total from the invoice summary section."""
        if summary_anchor is not None:
            total_block = self._find_block_by_text(
                blocks, {"total"},
                y_min=summary_anchor.y_min,
            )
            if total_block is not None:
                # Collect all blocks on the same y-band to the right of Total.
                same_row = sorted(
                    [
                        b for b in blocks
                        if abs(b.y_center - total_block.y_center) <= SAME_ROW_TOLERANCE
                        and b.x_min > total_block.x_max
                    ],
                    key=lambda b: b.x_min,
                )
                if len(same_row) >= 3:
                    return {
                        "subtotal":     self._amount_from_block(same_row[0]),
                        "tax":          self._amount_from_block(same_row[1]),
                        "total_amount": self._amount_from_block(same_row[2]),
                    }
                if len(same_row) == 2:
                    # Some invoices show only net_worth and gross on Total row.
                    return {
                        "subtotal":     self._amount_from_block(same_row[0]),
                        "tax":          None,
                        "total_amount": self._amount_from_block(same_row[1]),
                    }

        # Fallback: scan full text for dollar/currency amounts.
        amounts = re.findall(r"\$\s*[\d\s]+[,.][\d]{2}", full_text)
        return {
            "subtotal":     self._amount_from_text(amounts[0]) if len(amounts) > 0 else None,
            "tax":          self._amount_from_text(amounts[1]) if len(amounts) > 1 else None,
            "total_amount": self._amount_from_text(amounts[2]) if len(amounts) > 2 else None,
        }

    def _amount_from_block(self, block: TextBlock) -> dict[str, Any]:
        return {"raw": block.text, "value": self._parse_amount(block.text)}

    def _amount_from_text(self, text: str) -> dict[str, Any]:
        return {"raw": text, "value": self._parse_amount(text)}

    # ------------------------------------------------------------------
    # Amount parser
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_amount(text: str) -> float | None:
        """Parse OCR amount strings into Python floats.

        Handles:
        - European: "1 394,67"  → 1394.67
        - Anglo:    "1,394.67"  → 1394.67
        - Mixed:    "$ 5 640,17" → 5640.17
        """
        # Strip currency symbols and whitespace noise, preserve digits/comma/dot/minus.
        cleaned = re.sub(r"[^\d,.\-]", "", text)
        if not cleaned:
            return None

        # Remove a leading/trailing isolated minus sign (not a number).
        if cleaned in ("-", "."):
            return None

        # Detect European vs Anglo format.
        has_comma = "," in cleaned
        has_dot   = "." in cleaned

        if has_comma and has_dot:
            # Determine which is the decimal separator by position.
            last_comma = cleaned.rfind(",")
            last_dot   = cleaned.rfind(".")
            if last_comma > last_dot:
                # Comma is decimal separator: "1.394,67" → "1394.67"
                cleaned = cleaned.replace(".", "").replace(",", ".")
            else:
                # Dot is decimal separator: "1,394.67" → "1394.67"
                cleaned = cleaned.replace(",", "")
        elif has_comma:
            # Comma only: treat as decimal separator: "394,67" → "394.67"
            cleaned = cleaned.replace(",", ".")

        try:
            return float(cleaned)
        except ValueError:
            logger.warning("Unable to parse amount: %r → %r", text, cleaned)
            return None

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _find_block_by_text(
        self,
        blocks: list[TextBlock],
        texts: set[str],
        *,
        case_insensitive: bool = True,
        y_min: float = 0.0,
    ) -> TextBlock | None:
        """Return the first block whose text matches one of the given strings."""
        compare = (lambda s: s.lower()) if case_insensitive else (lambda s: s)
        targets = {compare(t) for t in texts}
        for block in blocks:
            if block.y_min < y_min:
                continue
            if compare(block.text) in targets:
                return block
        return None

    @staticmethod
    def _nearest_x(
        blocks: list[TextBlock],
        x_center: float,
    ) -> TextBlock | None:
        if not blocks:
            return None
        return min(blocks, key=lambda b: abs(b.x_center - x_center))


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def extract_invoice_from_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    min_confidence: float = 0.0,
) -> dict[str, Any]:
    """Extract a structured invoice and optionally save it to disk.

    Parameters
    ----------
    input_path:
        Path to the OCR JSON output (single or batch format).
    output_path:
        Optional path to write the structured invoice JSON.
    min_confidence:
        OCR blocks below this confidence threshold are ignored.

    Returns
    -------
    dict
        Structured invoice record.
    """
    extractor = InvoiceInformationExtractor(min_confidence=min_confidence)
    record = extractor.extract_from_file(input_path)
    if output_path is not None:
        extractor.save(record, output_path)
    return record
