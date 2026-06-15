"""PaddleOCR wrapper for multilingual document OCR.

The engine returns normalized OCR blocks containing text, confidence scores,
and polygon bounding boxes. PaddleOCR is imported lazily so preprocessing and
non-OCR tests can run before the heavier OCR dependency is installed.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from preprocessing import read_image

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OCRConfig:
    """Configuration for PaddleOCR inference."""

    language: str = "en"
    use_angle_cls: bool = True
    use_gpu: bool = False
    min_confidence: float = 0.30
    sort_reading_order: bool = True


@dataclass(frozen=True)
class OCRTextBlock:
    """Single OCR text detection result."""

    text: str
    confidence: float
    bbox: list[list[float]]
    bbox_xyxy: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the text block for JSON export."""
        return asdict(self)


@dataclass(frozen=True)
class OCRResult:
    """OCR output for one document image."""

    image_path: str | None
    language: str
    text_blocks: list[OCRTextBlock]
    full_text: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize OCR results into JSON-friendly primitives."""
        return {
            "image_path": self.image_path,
            "language": self.language,
            "full_text": self.full_text,
            "text_blocks": [block.to_dict() for block in self.text_blocks],
            "metadata": self.metadata,
        }


class PaddleOCREngine:
    """High-level OCR engine backed by PaddleOCR."""

    def __init__(self, config: OCRConfig | None = None) -> None:
        self.config = config or OCRConfig()
        self._engine: Any | None = None

    @property
    def engine(self) -> Any:
        """Create and cache the PaddleOCR instance on first use."""
        if self._engine is None:
            self._engine = self._create_engine()
        return self._engine

    def _create_engine(self) -> Any:
        """Instantiate PaddleOCR with compatibility for common versions."""
        try:
            os.environ.setdefault("FLAGS_use_onednn", "0")
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise ImportError(
                "PaddleOCR is not installed. Install it with: "
                "pip install paddleocr paddlepaddle"
            ) from exc

        logger.info("Initializing PaddleOCR with language=%s", self.config.language)
        init_attempts = [
            {
                "lang": self.config.language,
                "use_doc_orientation_classify": self.config.use_angle_cls,
                "use_doc_unwarping": False,
                "use_textline_orientation": self.config.use_angle_cls,
            },
            {
                "lang": self.config.language,
                "use_angle_cls": self.config.use_angle_cls,
                "use_gpu": self.config.use_gpu,
            },
            {
                "lang": self.config.language,
                "use_textline_orientation": self.config.use_angle_cls,
            },
            {"lang": self.config.language},
        ]

        last_error: TypeError | ValueError | None = None
        for kwargs in init_attempts:
            try:
                return PaddleOCR(**kwargs)
            except (TypeError, ValueError) as exc:
                last_error = exc

        raise RuntimeError("Unable to initialize PaddleOCR") from last_error

    def recognize(
        self,
        image: str | Path | np.ndarray,
        image_path: str | None = None,
    ) -> OCRResult:
        """Run OCR on an image path or OpenCV/NumPy image array."""
        prepared_image, resolved_path = self._prepare_image(image, image_path)
        logger.info("Running OCR for %s", resolved_path or "in-memory image")

        raw_result = self._run_paddleocr(prepared_image)
        text_blocks = self._normalize_result(raw_result)
        if self.config.sort_reading_order:
            text_blocks = sorted(
                text_blocks,
                key=lambda block: (block.bbox_xyxy["y_min"], block.bbox_xyxy["x_min"]),
            )

        full_text = "\n".join(block.text for block in text_blocks)
        logger.info("OCR completed with %d text blocks", len(text_blocks))
        return OCRResult(
            image_path=resolved_path,
            language=self.config.language,
            text_blocks=text_blocks,
            full_text=full_text,
            metadata={
                "engine": "PaddleOCR",
                "image_shape": tuple(int(value) for value in prepared_image.shape),
                "min_confidence": self.config.min_confidence,
            },
        )

    def _prepare_image(
        self,
        image: str | Path | np.ndarray,
        image_path: str | None,
    ) -> tuple[np.ndarray, str | None]:
        """Load paths and normalize image arrays for PaddleOCR."""
        if isinstance(image, (str, Path)):
            path = Path(image)
            return read_image(path), str(path)

        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), image_path
        if image.ndim == 3 and image.shape[2] in {3, 4}:
            return image.copy(), image_path
        raise ValueError(f"Unsupported image shape for OCR: {image.shape}")

    def _run_paddleocr(self, image: np.ndarray) -> Any:
        """Execute PaddleOCR while handling minor API differences."""
        if hasattr(self.engine, "predict"):
            return self.engine.predict(image)

        try:
            return self.engine.ocr(image, cls=self.config.use_angle_cls)
        except TypeError:
            return self.engine.ocr(image)

    def _normalize_result(self, raw_result: Any) -> list[OCRTextBlock]:
        """Convert PaddleOCR output variants into OCRTextBlock objects."""
        blocks: list[OCRTextBlock] = []

        for item in self._iter_result_items(raw_result):
            block = self._parse_result_item(item)
            if block is None:
                continue
            if block.confidence < self.config.min_confidence:
                logger.debug(
                    "Dropping low-confidence OCR block %.3f: %s",
                    block.confidence,
                    block.text,
                )
                continue
            blocks.append(block)

        return blocks

    def _iter_result_items(self, raw_result: Any) -> Iterable[Any]:
        """Yield OCR line items from nested PaddleOCR outputs."""
        if raw_result is None:
            return

        if isinstance(raw_result, dict):
            yield from self._iter_dict_items(raw_result)
            return

        if isinstance(raw_result, list):
            if self._looks_like_ocr_line(raw_result):
                yield raw_result
                return
            for item in raw_result:
                yield from self._iter_result_items(item)

    def _iter_dict_items(self, result: dict[str, Any]) -> Iterable[dict[str, Any]]:
        """Yield line dictionaries from PaddleOCR 3.x-style outputs."""
        texts = self._first_present(result, ("rec_texts", "texts"))
        scores = self._first_present(result, ("rec_scores", "scores"))
        boxes = self._first_present(
            result,
            ("rec_polys", "rec_boxes", "dt_polys", "boxes"),
        )
        if texts is None or boxes is None:
            return

        scores = scores or [1.0] * len(texts)
        for text, score, box in zip(texts, scores, boxes, strict=False):
            yield {"text": text, "confidence": score, "bbox": box}

    @staticmethod
    def _first_present(result: dict[str, Any], keys: tuple[str, ...]) -> Any:
        """Return the first non-None dictionary value without truth-testing arrays."""
        for key in keys:
            value = result.get(key)
            if value is not None:
                return value
        return None

    def _parse_result_item(self, item: Any) -> OCRTextBlock | None:
        """Parse one PaddleOCR result item into a normalized text block."""
        if isinstance(item, dict):
            text = item.get("text")
            confidence = item.get("confidence", item.get("score", 1.0))
            bbox = item.get("bbox")
            if bbox is None:
                bbox = item.get("box")
        elif self._looks_like_ocr_line(item):
            bbox = item[0]
            text_info = item[1]
            text = text_info[0]
            confidence = text_info[1]
        else:
            return None

        if not text or bbox is None:
            logger.warning("Skipping malformed OCR item without text or bbox: %s", item)
            return None

        try:
            normalized_bbox = self._normalize_bbox(bbox)
        except (TypeError, ValueError) as exc:
            logger.warning("Skipping OCR item with invalid bbox %s: %s", bbox, exc)
            return None

        bbox_xyxy = self._bbox_to_xyxy(normalized_bbox)
        return OCRTextBlock(
            text=str(text).strip(),
            confidence=float(confidence),
            bbox=normalized_bbox,
            bbox_xyxy=bbox_xyxy,
        )

    @staticmethod
    def _looks_like_ocr_line(item: Any) -> bool:
        """Detect PaddleOCR 2.x line format: [box, (text, confidence)]."""
        if not isinstance(item, list) or len(item) < 2:
            return False
        text_info = item[1]
        return isinstance(text_info, (tuple, list)) and len(text_info) >= 2

    @staticmethod
    def _normalize_bbox(bbox: Any) -> list[list[float]]:
        """Normalize a polygon or rectangle into four [x, y] points."""
        array = np.asarray(bbox, dtype=float)
        if array.shape == (4,):
            x_min, y_min, x_max, y_max = array.tolist()
            return [
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max],
            ]
        if array.ndim == 2 and array.shape[1] == 2:
            return [[float(x), float(y)] for x, y in array.tolist()]
        raise ValueError(f"Unsupported OCR bounding box format: {bbox}")

    @staticmethod
    def _bbox_to_xyxy(bbox: list[list[float]]) -> dict[str, float]:
        """Compute axis-aligned box limits from polygon points."""
        points = np.asarray(bbox, dtype=float)
        return {
            "x_min": float(points[:, 0].min()),
            "y_min": float(points[:, 1].min()),
            "x_max": float(points[:, 0].max()),
            "y_max": float(points[:, 1].max()),
        }


def run_ocr(
    image: str | Path | np.ndarray,
    config: OCRConfig | None = None,
    image_path: str | None = None,
) -> OCRResult:
    """Convenience function for one-off OCR inference."""
    return PaddleOCREngine(config=config).recognize(image, image_path=image_path)
