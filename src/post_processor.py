"""Post-processing and validation layer for Kannada APMC market register extraction.

Performs commodity confidence calculation using RapidFuzz, price validation,
missing value conversion, and anomaly detection.
"""

from __future__ import annotations

import re
from typing import Any
import rapidfuzz


class MarketRegisterPostProcessor:
    """Post-processor for validating and refining extracted market registers."""

    def __init__(self) -> None:
        pass

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """Process a full extracted dataset, refining all items and metadata."""
        items = data.get("items", [])
        processed_items = []

        for item in items:
            processed_items.append(self.process_item(item))

        processed_data = data.copy()
        processed_data["items"] = processed_items
        processed_data["metadata"] = data.get("metadata", {}).copy()
        processed_data["metadata"]["extraction_method"] = "hybrid_layout_analysis_v3"

        return processed_data

    def process_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """Post-process a single item containing commodity and price info."""
        # 1. Commodity Normalization & RapidFuzz Confidence
        normalization = item.get("normalization", {})
        raw_name = normalization.get("raw_name") or item.get("commodity") or ""
        normalized_name = item.get("commodity") or ""

        # Clean strings for confidence scoring
        cleaned_raw = raw_name.strip(". ,:-?!\t\n")
        cleaned_norm = normalized_name.strip(". ,:-?!\t\n")

        # Compute similarity score using RapidFuzz
        similarity = 0.0
        if cleaned_raw and cleaned_norm:
            similarity = float(rapidfuzz.fuzz.ratio(cleaned_raw, cleaned_norm) / 100.0)

        # Normalization safety rule: if confidence < 0.6, keep raw OCR value and mark unresolved
        if similarity < 0.6:
            commodity_val = raw_name
            norm_status = "unresolved"
        else:
            commodity_val = normalized_name
            norm_status = "resolved"

        updated_normalization = {
            "raw_name": raw_name,
            "normalized_name": normalized_name,
            "confidence": round(similarity, 3),
            "status": norm_status
        }

        # 2. Price Validation & Missing Value Handling & Anomaly Detection
        raw_price = item.get("price", "")
        if isinstance(raw_price, dict):
            # If price was already a dict, get the nested price
            raw_price = raw_price.get("price", "")

        price_val = None
        status = "valid"
        anomalies = []

        # Convert raw price to string and clean it
        if isinstance(raw_price, str):
            cleaned_price = raw_price.strip(". ,:-?!\t\n")
        else:
            cleaned_price = str(raw_price) if raw_price is not None else ""

        # Handle missing prices
        if cleaned_price in ("", "-", "--", "None", "null"):
            price_details = {
                "price": None,
                "status": "not_available"
            }
        else:
            # Parse numeric value
            try:
                # Find digits
                match = re.search(r"\d+(?:\.\d+)?", cleaned_price)
                if match:
                    price_val = float(match.group(0))

                    # Numeric range validation 0 <= price <= 500
                    if not (0.0 <= price_val <= 500.0):
                        status = "suspicious"
                        anomalies.append("out_of_range")

                    # OCR decimal mistake detection (e.g. 15 -> 1.5)
                    # If it has a decimal point and is very small (e.g. < 5.0)
                    if "." in match.group(0):
                        # Mysore APMC vegetable prices are typically integers.
                        # Floats like 1.5 or 1.8 are likely OCR errors for 15 or 18.
                        if price_val < 5.0:
                            status = "suspicious"
                            anomalies.append("suspected_decimal_mistake")

                    # Suspicious values (e.g. exactly 0.0 or > 150.0)
                    if price_val == 0.0:
                        status = "suspicious"
                        anomalies.append("zero_price")
                    elif price_val > 150.0:
                        status = "suspicious"
                        anomalies.append("extremely_high_price")

                    # Format clean representation (int if possible, else float)
                    if price_val.is_integer():
                        price_val = int(price_val)
                else:
                    status = "suspicious"
                    anomalies.append("non_numeric_format")
            except ValueError:
                status = "suspicious"
                anomalies.append("parsing_error")

            price_details = {
                "price": price_val,
                "status": status
            }
            if anomalies:
                price_details["anomalies"] = anomalies

        return {
            "commodity": commodity_val,
            "price": price_details,
            "normalization": updated_normalization
        }
