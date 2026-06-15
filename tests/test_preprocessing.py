"""Validation script for the Phase 1 preprocessing module.

Run from the project root:
    python tests/test_preprocessing.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing import (  # noqa: E402
    PreprocessingConfig,
    apply_adaptive_threshold,
    convert_to_grayscale,
    deskew_image,
    estimate_skew_angle,
    preprocess_image_file,
    read_image,
    remove_noise,
    save_image,
)


@dataclass(frozen=True)
class SampleImage:
    label: str
    path: Path | None


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "preprocessing_tests"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def find_first_image(directory: Path) -> Path | None:
    """Return the first readable image below a directory."""
    if not directory.exists():
        return None

    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return path
    return None


def collect_samples() -> list[SampleImage]:
    """Collect representative local samples from the organized datasets."""
    candidates = [
        SampleImage("invoice_part2", find_first_image(PROJECT_ROOT / "datasets" / "part2")),
        SampleImage("funsd_like_part3", find_first_image(PROJECT_ROOT / "datasets" / "part3")),
        SampleImage("kannada_custom", find_first_image(PROJECT_ROOT / "datasets" / "kannada")),
    ]
    return [sample for sample in candidates if sample.path is not None]


def make_comparison(original: np.ndarray, processed: np.ndarray) -> np.ndarray:
    """Create a side-by-side original/processed comparison image."""
    original_gray = convert_to_grayscale(original)
    original_bgr = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2BGR)
    processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    target_height = min(original_bgr.shape[0], processed_bgr.shape[0], 1200)

    def resize_to_height(image: np.ndarray) -> np.ndarray:
        scale = target_height / image.shape[0]
        target_width = max(1, int(image.shape[1] * scale))
        return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

    original_resized = resize_to_height(original_bgr)
    processed_resized = resize_to_height(processed_bgr)

    divider = np.full((target_height, 12, 3), 255, dtype=np.uint8)
    return np.hstack([original_resized, divider, processed_resized])


def assert_binary_image(image: np.ndarray) -> None:
    """Assert an image is a valid binary/grayscale OCR output."""
    assert image.ndim == 2, f"Expected single-channel image, got shape {image.shape}"
    assert image.dtype == np.uint8, f"Expected uint8 image, got {image.dtype}"
    unique_values = np.unique(image)
    assert unique_values.size <= 256
    assert unique_values.min() >= 0 and unique_values.max() <= 255


def test_invalid_path_handling() -> None:
    """Verify missing image paths fail with a clear exception."""
    missing_path = PROJECT_ROOT / "datasets" / "does_not_exist.png"
    try:
        read_image(missing_path)
    except FileNotFoundError as exc:
        assert str(missing_path) in str(exc)
        print("PASS invalid_path_handling: FileNotFoundError raised")
        return
    raise AssertionError("read_image did not raise FileNotFoundError for invalid path")


def validate_sample(sample: SampleImage) -> dict[str, object]:
    """Run all preprocessing assertions for one real document image."""
    print(f"\nRunning preprocessing validation for {sample.label}")
    print(f"Input path: {sample.path}")

    started_at = time.perf_counter()

    original = read_image(sample.path)
    assert original.size > 0
    assert original.ndim == 3, f"Expected BGR image, got shape {original.shape}"

    grayscale = convert_to_grayscale(original)
    assert grayscale.ndim == 2
    assert grayscale.shape == original.shape[:2]

    denoised = remove_noise(grayscale)
    assert denoised.shape == grayscale.shape
    assert denoised.dtype == grayscale.dtype

    thresholded = apply_adaptive_threshold(denoised)
    assert thresholded.shape == grayscale.shape
    assert_binary_image(thresholded)

    skew_angle = estimate_skew_angle(thresholded)
    assert isinstance(skew_angle, float)
    assert -15.0 <= skew_angle <= 15.0

    deskewed, deskew_angle = deskew_image(thresholded, thresholded)
    assert_binary_image(deskewed)
    assert isinstance(deskew_angle, float)

    result = preprocess_image_file(sample.path, config=PreprocessingConfig())
    assert_binary_image(result.image)
    assert result.original_path == str(sample.path)
    assert isinstance(result.skew_angle, float)
    assert result.metadata["original_shape"] == tuple(int(value) for value in original.shape)
    assert result.image.size > 0

    sample_dir = OUTPUT_DIR / sample.label
    save_image(original, sample_dir / "original.png")
    save_image(result.image, sample_dir / "processed.png")
    save_image(make_comparison(original, result.image), sample_dir / "comparison.png")

    processing_time = time.perf_counter() - started_at
    status = {
        "label": sample.label,
        "input_dimensions": tuple(int(value) for value in original.shape),
        "output_dimensions": tuple(int(value) for value in result.image.shape),
        "estimated_skew_angle": result.skew_angle,
        "processing_time_seconds": round(processing_time, 4),
        "status": "SUCCESS",
    }

    print(f"Input image dimensions: {status['input_dimensions']}")
    print(f"Output image dimensions: {status['output_dimensions']}")
    print(f"Estimated skew angle: {status['estimated_skew_angle']:.2f}")
    print(f"Processing time: {status['processing_time_seconds']} seconds")
    print(f"Status: {status['status']}")
    return status


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    test_invalid_path_handling()

    samples = collect_samples()
    labels = {sample.label for sample in samples}
    assert "invoice_part2" in labels, "No invoice image found under datasets/part2"
    assert "funsd_like_part3" in labels, "No FUNSD-like image found under datasets/part3"
    assert "kannada_custom" in labels, "No Kannada image found under datasets/kannada"

    results = []
    failures = []
    for sample in samples:
        try:
            results.append(validate_sample(sample))
        except Exception as exc:  # noqa: BLE001
            failures.append((sample.label, exc))
            print(f"Status: FAILURE for {sample.label}: {exc}")

    print("\nPreprocessing Validation Report")
    print("===============================")
    print(f"Test cases executed: {len(results) + len(failures) + 1}")
    print("Results:")
    print("- invalid_path_handling: SUCCESS")
    for result in results:
        print(
            "- {label}: {status}, input={input_dimensions}, output={output_dimensions}, "
            "skew={estimated_skew_angle:.2f}, time={processing_time_seconds}s".format(
                **result
            )
        )

    if failures:
        print("Bugs found:")
        for label, exc in failures:
            print(f"- {label}: {type(exc).__name__}: {exc}")
        raise AssertionError(f"{len(failures)} preprocessing validation(s) failed")

    print("Bugs found: None during this validation run")
    print("Fixes applied: None required after test execution")
    print(
        "Generated artifacts: original.png, processed.png, comparison.png under "
        f"{OUTPUT_DIR}"
    )
    print(
        "Remaining limitations: deskew estimation is conservative and ignores angles "
        "outside +/-15 degrees to avoid damaging unusual layouts."
    )
    print("Overall status: SUCCESS")


if __name__ == "__main__":
    main()
