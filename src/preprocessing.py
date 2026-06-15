"""Image preprocessing utilities for OCR-ready document images.

This module prepares scanned or photographed document images before OCR.
The pipeline is intentionally conservative: it improves contrast, removes
small noise, binarizes the page, and deskews it without changing document
layout or attempting classification/table extraction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration for OCR image preprocessing."""

    denoise_kernel_size: int = 3
    adaptive_threshold_block_size: int = 31
    adaptive_threshold_c: int = 15
    deskew_max_angle: float = 15.0
    min_foreground_pixels_for_deskew: int = 100

    def validate(self) -> None:
        """Validate OpenCV-sensitive configuration values."""
        if self.denoise_kernel_size < 1 or self.denoise_kernel_size % 2 == 0:
            raise ValueError("denoise_kernel_size must be a positive odd integer")
        if (
            self.adaptive_threshold_block_size < 3
            or self.adaptive_threshold_block_size % 2 == 0
        ):
            raise ValueError(
                "adaptive_threshold_block_size must be an odd integer >= 3"
            )


@dataclass(frozen=True)
class PreprocessingResult:
    """Container returned by the preprocessing pipeline."""

    original_path: str | None
    image: np.ndarray
    grayscale: np.ndarray
    thresholded: np.ndarray
    skew_angle: float
    metadata: dict[str, Any]


def read_image(image_path: str | Path) -> np.ndarray:
    """Read an image from disk as a BGR OpenCV array.

    ``cv2.imread`` can fail with some Unicode paths on Windows. Reading bytes
    through NumPy keeps this module reliable for local Kannada document paths.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image_bytes = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to decode image: {path}")

    logger.info("Loaded image %s with shape %s", path, image.shape)
    return image


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR or BGRA image to grayscale."""
    if image.ndim == 2:
        return image.copy()
    if image.ndim != 3:
        raise ValueError(f"Unsupported image dimensions: {image.shape}")
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported channel count: {image.shape[2]}")


def remove_noise(grayscale: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Remove small speckle noise while preserving text edges."""
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")

    denoised = cv2.medianBlur(grayscale, kernel_size)
    logger.debug("Applied median blur with kernel size %d", kernel_size)
    return denoised


def apply_adaptive_threshold(
    grayscale: np.ndarray,
    block_size: int = 31,
    c_value: int = 15,
) -> np.ndarray:
    """Binarize the image using local adaptive thresholding."""
    if block_size < 3 or block_size % 2 == 0:
        raise ValueError("block_size must be an odd integer >= 3")

    thresholded = cv2.adaptiveThreshold(
        grayscale,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=block_size,
        C=c_value,
    )
    logger.debug(
        "Applied adaptive thresholding with block_size=%d, c_value=%d",
        block_size,
        c_value,
    )
    return thresholded


def estimate_skew_angle(
    binary_image: np.ndarray,
    min_foreground_pixels: int = 100,
    max_angle: float = 15.0,
) -> float:
    """Estimate document skew angle in degrees from foreground pixels."""
    if binary_image.ndim != 2:
        raise ValueError("binary_image must be a single-channel image")

    foreground = np.column_stack(np.where(binary_image < 255))[:, ::-1]
    if foreground.shape[0] < min_foreground_pixels:
        logger.warning(
            "Skipping deskew estimation; only %d foreground pixels found",
            foreground.shape[0],
        )
        return 0.0

    angle = cv2.minAreaRect(foreground)[-1]
    if angle > 45:
        angle -= 90
    elif angle < -45:
        angle += 90

    # Different OpenCV builds may report upright pages near 0 or 90 degrees.
    # Normalizing keeps deskew conservative and avoids rotating valid pages.
    skew_angle = float(angle)
    if abs(skew_angle) > max_angle:
        logger.warning(
            "Ignoring implausible skew angle %.2f outside +/- %.2f degrees",
            skew_angle,
            max_angle,
        )
        return 0.0

    logger.info("Estimated skew angle: %.2f degrees", skew_angle)
    return skew_angle


def rotate_image(image: np.ndarray, angle: float, border_value: int = 255) -> np.ndarray:
    """Rotate an image around its center without cropping."""
    if abs(angle) < 0.01:
        return image.copy()

    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(rotation_matrix[0, 0])
    sin = abs(rotation_matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    rotation_matrix[0, 2] += (new_width / 2.0) - center[0]
    rotation_matrix[1, 2] += (new_height / 2.0) - center[1]

    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (new_width, new_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    logger.debug("Rotated image by %.2f degrees", angle)
    return rotated


def deskew_image(
    image: np.ndarray,
    thresholded: np.ndarray,
    max_angle: float = 15.0,
    min_foreground_pixels: int = 100,
) -> tuple[np.ndarray, float]:
    """Deskew an image using the estimated angle from its thresholded version."""
    skew_angle = estimate_skew_angle(
        thresholded,
        min_foreground_pixels=min_foreground_pixels,
        max_angle=max_angle,
    )
    corrected = rotate_image(image, -skew_angle)
    return corrected, skew_angle


def preprocess_image(
    image: np.ndarray,
    config: PreprocessingConfig | None = None,
    original_path: str | None = None,
) -> PreprocessingResult:
    """Run the full Phase 1 preprocessing pipeline on an image array."""
    config = config or PreprocessingConfig()
    config.validate()

    logger.info("Starting image preprocessing")
    grayscale = convert_to_grayscale(image)
    denoised = remove_noise(grayscale, config.denoise_kernel_size)
    thresholded = apply_adaptive_threshold(
        denoised,
        block_size=config.adaptive_threshold_block_size,
        c_value=config.adaptive_threshold_c,
    )
    deskewed, skew_angle = deskew_image(
        thresholded,
        thresholded,
        max_angle=config.deskew_max_angle,
        min_foreground_pixels=config.min_foreground_pixels_for_deskew,
    )

    result = PreprocessingResult(
        original_path=original_path,
        image=deskewed,
        grayscale=grayscale,
        thresholded=thresholded,
        skew_angle=skew_angle,
        metadata={
            "original_shape": tuple(int(value) for value in image.shape),
            "processed_shape": tuple(int(value) for value in deskewed.shape),
            "preprocessing_steps": [
                "grayscale_conversion",
                "median_noise_removal",
                "adaptive_thresholding",
                "deskew_correction",
            ],
        },
    )
    logger.info("Completed image preprocessing")
    return result


def preprocess_image_file(
    image_path: str | Path,
    config: PreprocessingConfig | None = None,
) -> PreprocessingResult:
    """Read and preprocess an image from disk."""
    path = Path(image_path)
    image = read_image(path)
    return preprocess_image(image, config=config, original_path=str(path))


def save_image(image: np.ndarray, output_path: str | Path) -> None:
    """Save an OpenCV image array to disk, creating parent directories if needed."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    extension = path.suffix or ".png"
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise ValueError(f"Unable to encode image for output path: {path}")

    encoded.tofile(path)
    logger.info("Saved image to %s", path)
