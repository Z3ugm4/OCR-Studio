"""
Vision Studio — Video frame extraction, OCR, and dialogue tools
===============================================================
A polished PyQt6 GUI for video-to-image extraction, batch OCR, and transcript
cleanup.

Features
--------
* Extract video frames by count, interval, all frames, or custom fps.
* Drag & drop or pick multiple images (PNG/JPG/JPEG/BMP/WebP/TIFF).
* Per-image preview, full result list, and confidence scores.
* Async worker thread keeps the UI responsive (cancel supported).
* PaddleOCR, EasyOCR, and MinerU2.5-Pro document-parsing backends.
* OCR dialogue cleanup and frame-by-frame transcript consolidation.
* Guided Overall Run: video frames, OCR, and cleaned text in one workflow.
* Language selector (en, ch, chinese_cht, fr, de, japan, korean, ...).
* Export results to TXT (readable) or JSON (structured).
* Modern dark theme, custom QSS, no external QSS files needed.

Usage
-----
    pip install paddleocr paddlepaddle PyQt6 Pillow numpy
    python paddle_ocr_studio.py

Tested with: paddleocr >= 2.7, paddlepaddle >= 2.5, PyQt6 >= 6.5.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from ocr_dialogue_cleaner import CleanerConfig, clean_file, summary as cleaner_summary
from video_ocr_dialogue_extractor import (
    extract_transcript,
    format_transcript,
    parse_aliases,
)

from PyQt6.QtCore import (
    Qt,
    QThread,
    pyqtSignal,
    QSettings,
    QSize,
    QMimeData,
    QRectF,
    QUrl,
)
from PyQt6.QtGui import (
    QAction,
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QIcon,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QProgressDialog,
    QFrame,
    QButtonGroup,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpacerItem,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QStyle,
    QTabWidget,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

# OpenCV powers the Video → Images tab. Keep the OCR tab usable and provide a
# friendly in-app error if OpenCV is not installed.
try:
    import cv2  # type: ignore
    CV2_OK = True
except Exception as _exc:  # pragma: no cover - import-time guard
    cv2 = None  # type: ignore
    CV2_OK = False
    _CV2_IMPORT_ERROR = _exc

# ----------------------------------------------------------------------------
# Lazy PaddleOCR import — show a friendlier error if it's missing.
# ----------------------------------------------------------------------------
try:
    from paddleocr import PaddleOCR  # type: ignore
    PADDLE_OK = True
except Exception as _exc:  # pragma: no cover - import-time guard
    PaddleOCR = None  # type: ignore
    PADDLE_OK = False
    _PADDLE_IMPORT_ERROR = _exc

# Lazy EasyOCR import — optional backend.
try:
    import easyocr  # type: ignore
    EASY_OK = True
except Exception as _exc:  # pragma: no cover - import-time guard
    easyocr = None  # type: ignore
    EASY_OK = False
    _EASY_IMPORT_ERROR = _exc


# ============================================================================
# Domain types
# ============================================================================

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

# PaddleOCR language codes (most common subset).
LANGUAGES_PADDLE = [
    ("English",            "en"),
    ("Chinese (Simplified)", "ch"),
    ("Chinese (Traditional)", "chinese_cht"),
    ("French",             "fr"),
    ("German",             "german"),
    ("Japanese",           "japan"),
    ("Korean",             "korean"),
    ("Russian",            "ru"),
    ("Arabic",             "ar"),
    ("Hindi",              "hi"),
    ("Spanish",            "es"),
    ("Portuguese",         "pt"),
    ("Italian",            "it"),
    ("Vietnamese",         "vi"),
    ("Thai",               "th"),
    ("Greek",              "el"),
]

# EasyOCR uses different short codes than PaddleOCR.
# Full list: https://www.jaided.ai/easyocr/ (80+ languages supported)
LANGUAGES_EASY = [
    ("English",              "en"),
    ("Chinese (Simplified)", "ch_sim"),
    ("Chinese (Traditional)", "ch_tra"),
    ("French",               "fr"),
    ("German",               "de"),
    ("Japanese",             "ja"),
    ("Korean",               "ko"),
    ("Russian",              "ru"),
    ("Arabic",               "ar"),
    ("Hindi",                "hi"),
    ("Spanish",              "es"),
    ("Portuguese",           "pt"),
    ("Italian",              "it"),
    ("Vietnamese",           "vi"),
    ("Thai",                 "th"),
    ("Greek",                "el"),
]

BACKENDS = [
    ("PaddleOCR", "paddle"),
    ("EasyOCR",   "easy"),
    ("MinerU2.5-Pro", "mineru"),
]

BACKEND_LABELS = dict((key, name) for name, key in BACKENDS)
MINERU_MODEL_ID = "opendatalab/MinerU2.5-Pro-2604-1.2B"
LANGUAGES_MINERU = [("Automatic (multilingual)", "auto")]


@dataclass
class OCRLine:
    """A single detected text line."""
    text: str
    confidence: float
    box: list[list[float]] = field(default_factory=list)  # 4-point polygon
    kind: str = "text"


@dataclass
class OCRResult:
    """OCR result for one image."""
    path: str
    name: str
    lines: list[OCRLine] = field(default_factory=list)
    elapsed_sec: float = 0.0
    error: Optional[str] = None

    @property
    def is_ok(self) -> bool:
        return self.error is None

    @property
    def full_text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    def to_dict(self) -> dict:
        d = {
            "path": self.path,
            "name": self.name,
            "elapsed_sec": round(self.elapsed_sec, 3),
            "line_count": len(self.lines),
            "lines": [
                {
                    "text": ln.text,
                    "confidence": round(ln.confidence, 4),
                    "box": ln.box,
                    "type": ln.kind,
                }
                for ln in self.lines
            ],
            "error": self.error,
        }
        return d


# ============================================================================
# Worker — runs PaddleOCR off the UI thread
# ============================================================================

class OCRWorker(QThread):
    """Process a queue of images, emitting progress / result / done signals."""

    progress = pyqtSignal(int, int, str)           # index, total, current_name
    download_progress = pyqtSignal(object, object, str, str)
    file_done = pyqtSignal(int, object)            # index, OCRResult
    finished_ok = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(
        self,
        paths: list[str],
        lang: str,
        use_angle_cls: bool = True,
    ) -> None:
        super().__init__()
        self.paths = paths
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    # ---- PaddleOCR version compatibility ---------------------------------
    # v3 dropped `show_log` / `enable_mkldnn`, renamed `use_angle_cls` to
    # `use_textline_orientation`, and replaced `ocr()` with `predict()` whose
    # return shape is structured (`rec_texts` / `rec_scores` / `rec_boxes`).
    # We detect v3 by checking for the `predict` method on the instance, which
    # is more reliable than reading `__version__` (some builds don't expose it).

    @staticmethod
    def _is_v3_instance(ocr: "PaddleOCR") -> bool:
        return hasattr(ocr, "predict")

    def _build_ocr(self) -> "PaddleOCR":
        # Try v3 first (modern). If the installed paddleocr is v2, the v2 kwargs
        # below will work; we decide which API to call at runtime based on the
        # instance's methods.
        try:
            return PaddleOCR(
                lang=self.lang,
                use_textline_orientation=self.use_angle_cls,
            )
        except (TypeError, ValueError):
            # v2 fallback
            return PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=self.lang,
                show_log=False,
                enable_mkldnn=True,
            )

    @staticmethod
    def _parse_v2(raw) -> list[OCRLine]:
        """v2 result shape: [ [[box, (text, score)], ...], ... ]."""
        if raw and isinstance(raw[0], list):
            page = raw[0]
        else:
            page = raw or []
        lines: list[OCRLine] = []
        for det in page:
            if not det or len(det) < 2:
                continue
            box, (text, score) = det[0], det[1]
            lines.append(
                OCRLine(
                    text=str(text),
                    confidence=float(score),
                    box=[[float(x), float(y)] for x, y in box],
                )
            )
        return lines

    @staticmethod
    def _to_dict(item) -> Optional[dict]:
        """Best-effort conversion of a v3 result item into a plain dict."""
        if isinstance(item, dict):
            return item
        # 1. `.json` may be a method or a property depending on build
        j = getattr(item, "json", None)
        if j is not None:
            try:
                if callable(j):
                    j = j()
                if isinstance(j, dict):
                    return j
            except Exception:
                pass
        # 2. fall back to object's __dict__
        d = getattr(item, "__dict__", None)
        if isinstance(d, dict):
            return d
        return None

    @staticmethod
    def _find_rec_keys(payload: dict, depth: int = 0) -> Optional[dict]:
        """Walk a (possibly nested) dict looking for `rec_texts`.

        v3 result shapes vary across builds — `rec_texts` may sit at the top
        level, under a `res` key, inside a list of page dicts, etc.
        """
        if depth > 6 or not isinstance(payload, dict):
            return None
        if "rec_texts" in payload:
            return payload
        for v in payload.values():
            if isinstance(v, dict):
                hit = OCRWorker._find_rec_keys(v, depth + 1)
                if hit is not None:
                    return hit
            elif isinstance(v, list) and v:
                # dive into the first list element that looks like a dict
                for el in v:
                    if isinstance(el, dict):
                        hit = OCRWorker._find_rec_keys(el, depth + 1)
                        if hit is not None:
                            return hit
        return None

    @classmethod
    def _parse_v3(cls, raw, debug_label: str = "") -> list[OCRLine]:
        """Robust v3 parser: handles .json method/attr, nested `res`, page lists."""
        lines: list[OCRLine] = []
        for idx, item in enumerate(raw or []):
            data = cls._to_dict(item)
            if data is None:
                continue

            inner = cls._find_rec_keys(data) or {}

            if not inner and idx == 0:
                # Dump once so we can see what shape we got
                import json as _json
                try:
                    snippet = _json.dumps(data, default=str)[:600]
                except Exception:
                    snippet = repr(data)[:600]
                print(
                    f"[ocr] v3 result has no rec_texts ({debug_label}); "
                    f"first-item keys={list(data.keys())[:10]}; "
                    f"snippet={snippet}",
                    file=sys.stderr,
                )

            # NOTE: do not use `or []` here — rec_scores / rec_boxes can be
            # numpy arrays, and `numpy_array or []` raises
            # "truth value of an array is ambiguous".
            texts = inner.get("rec_texts")
            if texts is None:
                texts = []
            scores = inner.get("rec_scores")
            if scores is None:
                scores = []
            boxes = inner.get("rec_boxes")
            if boxes is None:
                boxes = inner.get("rec_polys") or []
            if boxes is None:
                boxes = []

            for j, text in enumerate(texts):
                score = float(scores[j]) if j < len(scores) else 0.0
                box: list[list[float]] = []
                if j < len(boxes):
                    raw_box = boxes[j]
                    # Normalize box shape to [[x, y], [x, y], ...]
                    if isinstance(raw_box, (list, tuple)) and raw_box:
                        if isinstance(raw_box[0], (list, tuple)):
                            box = [[float(x), float(y)] for x, y in raw_box]
                        else:
                            # Flat [x1, y1, x2, y2, ...] -> 4-point polygon
                            flat = [float(v) for v in raw_box]
                            box = [
                                [flat[0], flat[1]],
                                [flat[2], flat[3]],
                                [flat[4], flat[5]],
                                [flat[6], flat[7]],
                            ] if len(flat) >= 8 else []
                lines.append(
                    OCRLine(text=str(text), confidence=score, box=box)
                )
        return lines

    def run(self) -> None:  # noqa: D401
        try:
            self.download_progress.emit(0, 0, "loading", "PaddleOCR")
            with track_tqdm_model_downloads(
                self.download_progress,
                "PaddleOCR",
                lambda: self._cancel,
            ):
                ocr = self._build_ocr()
        except ModelDownloadCancelled:
            self.finished_ok.emit()
            return
        except Exception as exc:
            self.failed.emit(f"Failed to init PaddleOCR: {exc}\n{traceback.format_exc()}")
            return

        is_v3 = self._is_v3_instance(ocr)
        print(f"[ocr] detected {'v3' if is_v3 else 'v2'} PaddleOCR API",
              file=sys.stderr)

        total = len(self.paths)
        for i, path in enumerate(self.paths):
            if self._cancel:
                break
            self.progress.emit(i, total, os.path.basename(path))
            t0 = time.perf_counter()
            err: Optional[str] = None
            lines: list[OCRLine] = []
            try:
                if is_v3:
                    raw = list(ocr.predict(path))
                    lines = self._parse_v3(raw, debug_label=os.path.basename(path))
                else:
                    raw = ocr.ocr(path, cls=self.use_angle_cls)
                    lines = self._parse_v2(raw)
            except Exception as exc:
                err = f"{exc}"
                traceback.print_exc()

            elapsed = time.perf_counter() - t0
            self.file_done.emit(
                i,
                OCRResult(
                    path=path,
                    name=os.path.basename(path),
                    lines=lines,
                    elapsed_sec=elapsed,
                    error=err,
                ),
            )

        if not self._cancel:
            self.progress.emit(total, total, "")
        self.finished_ok.emit()


# ============================================================================
# EasyOCR worker — same interface as OCRWorker (progress/file_done/finished_ok)
# ============================================================================

class EasyOCRWorker(QThread):
    """Batch OCR via EasyOCR (JaidedAI).

    EasyOCR's `Reader.readtext` returns [(box, text, confidence), ...] where
    `box` is already a 4-point polygon in [[x, y], ...] form, so mapping into
    our OCRLine dataclass is straightforward.
    """

    progress = pyqtSignal(int, int, str)
    download_progress = pyqtSignal(object, object, str, str)
    file_done = pyqtSignal(int, object)
    finished_ok = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(
        self,
        paths: list[str],
        lang: str,
        gpu: bool = False,
        paragraph: bool = False,
    ) -> None:
        super().__init__()
        self.paths = paths
        self.lang = lang
        self.gpu = gpu
        self.paragraph = paragraph
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:  # noqa: D401
        if not EASY_OK:
            self.failed.emit(
                "EasyOCR is not installed. Install it with:\n\n"
                "    pip install easyocr\n\n"
                f"Underlying error:\n{_EASY_IMPORT_ERROR}"
            )
            return

        try:
            # First-time use will download models to ~/.EasyOCR/model/
            self.download_progress.emit(0, 0, "loading", "EasyOCR")
            from easyocr import utils as easyocr_utils

            original_progress_factory = easyocr_utils.printProgressBar

            def progress_factory(*_args, **_kwargs):
                def report(count, block_size, total_size):
                    if self._cancel:
                        raise ModelDownloadCancelled()
                    if total_size and total_size > 0:
                        current = min(count * block_size, total_size)
                        self.download_progress.emit(
                            current,
                            total_size,
                            "bytes",
                            "EasyOCR",
                        )

                return report

            easyocr_utils.printProgressBar = progress_factory
            try:
                reader = easyocr.Reader([self.lang], gpu=self.gpu, verbose=True)
            finally:
                easyocr_utils.printProgressBar = original_progress_factory
        except ModelDownloadCancelled:
            self.finished_ok.emit()
            return
        except Exception as exc:
            self.failed.emit(
                f"Failed to init EasyOCR: {exc}\n{traceback.format_exc()}"
            )
            return

        total = len(self.paths)
        for i, path in enumerate(self.paths):
            if self._cancel:
                break
            self.progress.emit(i, total, os.path.basename(path))
            t0 = time.perf_counter()
            err: Optional[str] = None
            lines: list[OCRLine] = []
            try:
                # detail=1 gives [(box, text, conf), ...]
                raw = reader.readtext(
                    path,
                    detail=1,
                    paragraph=self.paragraph,
                )
                for det in raw or []:
                    if not det or len(det) < 3:
                        continue
                    box, text, conf = det[0], det[1], det[2]
                    norm_box: list[list[float]] = []
                    if isinstance(box, (list, tuple)) and box:
                        try:
                            norm_box = [
                                [float(x), float(y)] for x, y in box
                            ]
                        except Exception:
                            norm_box = []
                    lines.append(
                        OCRLine(
                            text=str(text),
                            confidence=float(conf),
                            box=norm_box,
                        )
                    )
            except Exception as exc:
                err = f"{exc}"
                traceback.print_exc()

            elapsed = time.perf_counter() - t0
            self.file_done.emit(
                i,
                OCRResult(
                    path=path,
                    name=os.path.basename(path),
                    lines=lines,
                    elapsed_sec=elapsed,
                    error=err,
                ),
            )

        if not self._cancel:
            self.progress.emit(total, total, "")
        self.finished_ok.emit()


# ============================================================================
# MinerU2.5-Pro worker — layout-aware text, table, and equation extraction
# ============================================================================

class ModelDownloadCancelled(Exception):
    """Raised inside the Hub progress callback when the user cancels."""


class _NullProgressWriter:
    """Discard terminal tqdm output while the GUI shows the same progress."""

    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        pass


class track_tqdm_model_downloads:
    """Temporarily bridge tqdm download bars into a Qt progress signal."""

    def __init__(self, signal, model_label: str, is_cancelled) -> None:
        self.signal = signal
        self.model_label = model_label
        self.is_cancelled = is_cancelled
        self._tqdm_class = None
        self._original_update = None
        self._original_refresh = None

    def __enter__(self):
        try:
            from tqdm.std import tqdm as tqdm_class
        except Exception:
            return self

        self._tqdm_class = tqdm_class
        self._original_update = tqdm_class.update
        self._original_refresh = tqdm_class.refresh
        bridge = self

        def report(bar) -> None:
            if bridge.is_cancelled():
                raise ModelDownloadCancelled()
            total = float(getattr(bar, "total", 0) or 0)
            current = float(getattr(bar, "n", 0) or 0)
            if total <= 0:
                return
            unit = str(getattr(bar, "unit", "") or "")
            stage = "bytes" if unit.upper().endswith("B") else "files"
            bridge.signal.emit(
                min(current, total),
                total,
                stage,
                bridge.model_label,
            )

        def tracked_update(bar, amount=1):
            result = bridge._original_update(bar, amount)
            report(bar)
            return result

        def tracked_refresh(bar, *args, **kwargs):
            result = bridge._original_refresh(bar, *args, **kwargs)
            if hasattr(bar, "n"):
                report(bar)
            return result

        tqdm_class.update = tracked_update
        tqdm_class.refresh = tracked_refresh
        return self

    def __exit__(self, exc_type, exc_value, traceback_obj) -> None:
        if self._tqdm_class is not None:
            self._tqdm_class.update = self._original_update
            self._tqdm_class.refresh = self._original_refresh


class MinerUWorker(QThread):
    """Batch document parsing with MinerU2.5-Pro via Transformers."""

    progress = pyqtSignal(int, int, str)
    download_progress = pyqtSignal(object, object, str, str)
    file_done = pyqtSignal(int, object)
    finished_ok = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, paths: list[str]) -> None:
        super().__init__()
        self.paths = paths
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    @staticmethod
    def _format_content(kind: str, content: str) -> str:
        """Keep non-text document blocks readable in TXT/Markdown-style output."""
        content = content.strip()
        if kind == "title":
            return f"# {content}"
        if kind in {"equation", "equation_block"}:
            return f"$$\n{content}\n$$"
        if kind in {"code", "algorithm"}:
            return f"```\n{content}\n```"
        return content

    @staticmethod
    def _bbox_to_polygon(
        bbox,
        image_width: int,
        image_height: int,
    ) -> list[list[float]]:
        """Convert MinerU's normalized [x1, y1, x2, y2] box to pixels."""
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return []
        try:
            x1, y1, x2, y2 = (float(value) for value in bbox)
        except (TypeError, ValueError):
            return []
        return [
            [x1 * image_width, y1 * image_height],
            [x2 * image_width, y1 * image_height],
            [x2 * image_width, y2 * image_height],
            [x1 * image_width, y2 * image_height],
        ]

    @classmethod
    def _parse_blocks(
        cls,
        blocks,
        image_width: int,
        image_height: int,
    ) -> list[OCRLine]:
        lines: list[OCRLine] = []
        for block in blocks or []:
            if not isinstance(block, dict):
                continue
            content = block.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            kind = str(block.get("type") or "text")
            text = cls._format_content(kind, content)
            box = cls._bbox_to_polygon(
                block.get("bbox"),
                image_width,
                image_height,
            )
            lines.append(
                OCRLine(
                    text=text,
                    confidence=0.0,  # MinerU does not return per-block confidence.
                    box=box,
                    kind=kind,
                )
            )
        return lines

    def _download_model_snapshot(self) -> str:
        """Return a complete local model snapshot, reporting missing downloads."""
        from huggingface_hub import snapshot_download

        # Avoid any network request or download UI when this fixed model is
        # already complete in the local Hugging Face cache.
        try:
            cached_path = snapshot_download(
                MINERU_MODEL_ID,
                local_files_only=True,
            )
            required_files = (
                "config.json",
                "model.safetensors",
                "preprocessor_config.json",
                "tokenizer.json",
            )
            if all((Path(cached_path) / name).is_file() for name in required_files):
                return str(cached_path)
        except Exception:
            pass

        if self._cancel:
            raise ModelDownloadCancelled()

        self.download_progress.emit(0, 0, "checking", "MinerU2.5-Pro")

        # snapshot_download accepts a tqdm-compatible class. Current Hub
        # releases create an aggregate byte bar; older releases expose only the
        # outer file-count bar, which remains a useful fallback.
        from tqdm.auto import tqdm as base_tqdm

        worker = self

        class DownloadTqdm(base_tqdm):
            preferred_bytes = None
            fallback_files = None

            def __init__(self, *args, **kwargs):
                # Newer huggingface_hub versions pass their own grouping name,
                # which base tqdm does not accept.
                kwargs.pop("name", None)
                kwargs["file"] = _NullProgressWriter()
                kwargs["disable"] = False
                super().__init__(*args, **kwargs)

                description = str(getattr(self, "desc", "") or "").lower()
                unit = str(getattr(self, "unit", "") or "")
                if unit.upper() == "B":
                    if "reconstruct" in description:
                        type(self).preferred_bytes = self
                    elif type(self).preferred_bytes is None:
                        type(self).preferred_bytes = self
                elif description.startswith(("fetching", "downloading")):
                    type(self).fallback_files = self
                self._report()

            def _report(self) -> None:
                if worker._cancel:
                    raise ModelDownloadCancelled()

                preferred = type(self).preferred_bytes
                fallback = type(self).fallback_files
                if preferred is not None:
                    if self is not preferred:
                        return
                    unit = "bytes"
                else:
                    if self is not fallback:
                        return
                    unit = "files"

                total = float(getattr(self, "total", 0) or 0)
                current = float(getattr(self, "n", 0) or 0)
                if total > 0:
                    worker.download_progress.emit(
                        current,
                        total,
                        unit,
                        "MinerU2.5-Pro",
                    )

            def update(self, amount=1):
                result = super().update(amount)
                self._report()
                return result

            def refresh(self, *args, **kwargs):
                result = super().refresh(*args, **kwargs)
                # During __init__, the preferred/fallback attributes have not
                # necessarily been assigned yet.
                if hasattr(self, "n"):
                    self._report()
                return result

        snapshot_path = snapshot_download(
            MINERU_MODEL_ID,
            tqdm_class=DownloadTqdm,
        )
        if self._cancel:
            raise ModelDownloadCancelled()
        self.download_progress.emit(1, 1, "complete", "MinerU2.5-Pro")
        return str(snapshot_path)

    def run(self) -> None:  # noqa: D401
        try:
            from mineru_vl_utils import MinerUClient
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

            snapshot_path = self._download_model_snapshot()
            self.download_progress.emit(0, 0, "loading", "MinerU2.5-Pro")
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                snapshot_path,
                dtype="auto",
                device_map="auto",
                local_files_only=True,
            )
            processor = AutoProcessor.from_pretrained(
                snapshot_path,
                use_fast=True,
                local_files_only=True,
            )
            client = MinerUClient(
                backend="transformers",
                model=model,
                processor=processor,
                image_analysis=False,
            )
        except ModelDownloadCancelled:
            self.finished_ok.emit()
            return
        except Exception as exc:
            self.failed.emit(
                f"Failed to init MinerU2.5-Pro: {exc}\n{traceback.format_exc()}"
            )
            return

        total = len(self.paths)
        for i, path in enumerate(self.paths):
            if self._cancel:
                break
            self.progress.emit(i, total, os.path.basename(path))
            t0 = time.perf_counter()
            err: Optional[str] = None
            lines: list[OCRLine] = []
            try:
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    width, height = image.size
                    blocks = client.two_step_extract(image)
                lines = self._parse_blocks(blocks, width, height)
            except Exception as exc:
                err = f"{exc}"
                traceback.print_exc()

            elapsed = time.perf_counter() - t0
            self.file_done.emit(
                i,
                OCRResult(
                    path=path,
                    name=os.path.basename(path),
                    lines=lines,
                    elapsed_sec=elapsed,
                    error=err,
                ),
            )

        if not self._cancel:
            self.progress.emit(total, total, "")
        self.finished_ok.emit()


# ============================================================================
# Video → Images — native PyQt tab and worker
# ============================================================================

VIDEO_FILTER = (
    "Video files (*.mp4 *.mov *.mkv *.avi *.webm *.flv *.m4v *.wmv "
    "*.mpg *.mpeg *.ts *.3gp);;All files (*.*)"
)


class VideoExtractWorker(QThread):
    """Extract selected video frames without blocking the GUI."""

    progress = pyqtSignal(int, int, str)
    outputs_ready = pyqtSignal(list)
    completed = pyqtSignal(bool, str)  # cancelled, message
    failed = pyqtSignal(str)

    def __init__(
        self,
        video_path: str,
        output_dir: str,
        mode: str,
        count: int,
        custom_fps: float,
        image_format: str,
        quality: int,
    ) -> None:
        super().__init__()
        self.video_path = video_path
        self.output_dir = output_dir
        self.mode = mode
        self.count = count
        self.custom_fps = custom_fps
        self.image_format = image_format
        self.quality = quality
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    @staticmethod
    def _target_indices(
        total: int,
        source_fps: float,
        mode: str,
        count: int,
        custom_fps: float,
    ):
        if mode == "all":
            return range(total)
        if mode == "persec":
            return range(0, total, max(int(round(source_fps)), 1))
        if mode == "fps4":
            return range(0, total, max(int(round(source_fps / 4.0)), 1))
        if mode == "fpscustom":
            return range(0, total, max(int(round(source_fps / custom_fps)), 1))
        if count <= 1 or total == 1:
            return [0]

        step = max(total - 1, 1) / (count - 1)
        raw = [min(total - 1, int(round(i * step))) for i in range(count)]
        return list(dict.fromkeys(raw))

    def run(self) -> None:  # noqa: D401
        cap = None
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise RuntimeError(
                    "Failed to open the video. It may be corrupt or use an "
                    "unsupported codec."
                )

            source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if total_frames <= 0:
                raise RuntimeError("The video has no readable frames.")

            indices = self._target_indices(
                total_frames,
                source_fps,
                self.mode,
                self.count,
                self.custom_fps,
            )
            total_to_save = len(indices)
            duration = total_frames / source_fps if source_fps > 0 else 0.0
            self.progress.emit(
                0,
                total_to_save,
                f"{width}×{height} • {source_fps:.2f} fps • "
                f"{duration:.2f}s • extracting {total_to_save} frame(s)…",
            )

            extension = {
                "jpg": ".jpg",
                "png": ".png",
                "webp": ".webp",
            }[self.image_format]
            if self.image_format == "jpg":
                encoding = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
            elif self.image_format == "webp":
                encoding = [int(cv2.IMWRITE_WEBP_QUALITY), self.quality]
            else:
                encoding = []

            stem = Path(self.video_path).stem
            saved = 0
            saved_paths: list[str] = []
            for position, frame_index in enumerate(indices, start=1):
                if self._cancel:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                destination = os.path.join(
                    self.output_dir,
                    f"{stem}_{frame_index:06d}{extension}",
                )
                if cv2.imwrite(destination, frame, encoding):
                    saved += 1
                    saved_paths.append(destination)
                self.progress.emit(
                    position,
                    total_to_save,
                    f"Saving {saved}/{total_to_save}",
                )

            self.outputs_ready.emit(saved_paths)
            if self._cancel:
                message = (
                    f"Cancelled — saved {saved} image(s) to {self.output_dir}"
                )
            else:
                message = f"Done — saved {saved} image(s) to {self.output_dir}"
            self.completed.emit(self._cancel, message)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            if cap is not None:
                cap.release()


class VideoToImagesWidget(QWidget):
    """PyQt version of the Video → Images utility."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.worker: Optional[VideoExtractWorker] = None
        self._mode = "count"
        self._build_ui()
        self._refresh_mode_controls()
        self._refresh_estimate()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(10)

        title = QLabel("Video → Images")
        title.setObjectName("pageTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(title)

        subtitle = QLabel(
            "Extract frames from any video file. Pick a mode, then hit Extract."
        )
        subtitle.setObjectName("muted")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(subtitle)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        form = QWidget()
        self.form_widget = form
        form_layout = QVBoxLayout(form)
        form_layout.setContentsMargins(4, 4, 4, 4)
        form_layout.setSpacing(10)
        scroll.setWidget(form)
        root.addWidget(scroll, 1)

        source = self._card("Source")
        self.source_card = source
        source_layout = source.layout()

        video_row = QHBoxLayout()
        video_row.addWidget(QLabel("Video"), 0)
        self.video_path = QLineEdit()
        self.video_path.setPlaceholderText("Choose a video file…")
        self.video_path.editingFinished.connect(self._inspect_video)
        video_row.addWidget(self.video_path, 1)
        self.btn_video = QPushButton("Browse…")
        self.btn_video.clicked.connect(self._pick_video)
        video_row.addWidget(self.btn_video)
        source_layout.addLayout(video_row)

        self.video_info = QLabel("Pick a video to begin.")
        self.video_info.setObjectName("muted")
        self.video_info.setWordWrap(True)
        source_layout.addWidget(self.video_info)

        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Output"), 0)
        self.output_dir = QLineEdit()
        self.output_dir.setPlaceholderText("Choose an output folder…")
        output_row.addWidget(self.output_dir, 1)
        self.btn_output = QPushButton("Browse…")
        self.btn_output.clicked.connect(self._pick_output)
        output_row.addWidget(self.btn_output)
        source_layout.addLayout(output_row)
        form_layout.addWidget(source)

        mode_card = self._card("Extraction Mode")
        self.mode_card = mode_card
        mode_layout = mode_card.layout()
        self.mode_group = QButtonGroup(self)
        self.mode_buttons: dict[str, QRadioButton] = {}

        self.count_input = QSpinBox()
        self.count_input.setRange(1, 10_000_000)
        self.count_input.setValue(12)
        self.count_input.valueChanged.connect(self._refresh_estimate)
        count_row = self._mode_row(
            "count",
            "Total count",
            "images, evenly spaced across the video",
            self.count_input,
        )
        mode_layout.addLayout(count_row)
        mode_layout.addLayout(
            self._mode_row("persec", "1 image / second", "one frame every second")
        )
        mode_layout.addLayout(
            self._mode_row("all", "All frames", "save every readable frame")
        )
        mode_layout.addLayout(
            self._mode_row("fps4", "4 images / second", "four frames every second")
        )

        self.custom_fps_input = QDoubleSpinBox()
        self.custom_fps_input.setRange(0.01, 1000.0)
        self.custom_fps_input.setDecimals(2)
        self.custom_fps_input.setValue(2.0)
        self.custom_fps_input.valueChanged.connect(self._refresh_estimate)
        mode_layout.addLayout(
            self._mode_row(
                "fpscustom",
                "Custom fps",
                "frames per second",
                self.custom_fps_input,
            )
        )

        self.estimate = QLabel("")
        self.estimate.setObjectName("estimate")
        self.estimate.setWordWrap(True)
        mode_layout.addWidget(self.estimate)
        form_layout.addWidget(mode_card)

        format_card = self._card("Output Format")
        self.format_card = format_card
        format_layout = format_card.layout()
        format_row = QHBoxLayout()
        format_row.addWidget(QLabel("Format"))
        self.format_combo = QComboBox()
        self.format_combo.addItem("JPG", "jpg")
        self.format_combo.addItem("PNG", "png")
        self.format_combo.addItem("WebP", "webp")
        self.format_combo.currentIndexChanged.connect(self._on_format_changed)
        format_row.addWidget(self.format_combo)
        format_row.addSpacing(12)

        self.quality_label = QLabel("JPG quality 95")
        format_row.addWidget(self.quality_label)
        self.quality_slider = QSlider(Qt.Orientation.Horizontal)
        self.quality_slider.setRange(1, 100)
        self.quality_slider.setValue(95)
        self.quality_slider.valueChanged.connect(self._on_quality_changed)
        format_row.addWidget(self.quality_slider, 1)
        format_layout.addLayout(format_row)
        form_layout.addWidget(format_card)

        progress_card = self._card("Progress")
        progress_layout = progress_card.layout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("%v / %m  (%p%)")
        progress_layout.addWidget(self.progress)
        self.status = QLabel("Idle")
        self.status.setObjectName("muted")
        self.status.setWordWrap(True)
        progress_layout.addWidget(self.status)
        form_layout.addWidget(progress_card)
        form_layout.addStretch(1)

        actions = QHBoxLayout()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("danger")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel)
        actions.addWidget(self.cancel_btn)

        self.start_btn = QPushButton("Extract Images")
        self.start_btn.setObjectName("primary")
        self.start_btn.setMinimumHeight(44)
        self.start_btn.clicked.connect(self.start)
        actions.addWidget(self.start_btn, 1)
        root.addLayout(actions)

    def _card(self, title: str) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)
        heading = QLabel(title)
        heading.setObjectName("cardTitle")
        layout.addWidget(heading)
        return card

    def _mode_row(
        self,
        key: str,
        title: str,
        description: str,
        control: Optional[QWidget] = None,
    ) -> QHBoxLayout:
        row = QHBoxLayout()
        button = QRadioButton(title)
        button.setChecked(key == "count")
        button.toggled.connect(
            lambda checked, mode=key: self._select_mode(mode) if checked else None
        )
        self.mode_group.addButton(button)
        self.mode_buttons[key] = button
        row.addWidget(button)
        if control is not None:
            row.addWidget(control)
        detail = QLabel(description)
        detail.setObjectName("muted")
        row.addWidget(detail)
        row.addStretch(1)
        return row

    def _select_mode(self, mode: str) -> None:
        self._mode = mode
        self._refresh_mode_controls()
        self._refresh_estimate()

    def _refresh_mode_controls(self) -> None:
        self.count_input.setEnabled(self._mode == "count")
        self.custom_fps_input.setEnabled(self._mode == "fpscustom")

    def _pick_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select video",
            self.video_path.text(),
            VIDEO_FILTER,
        )
        if not path:
            return
        self.video_path.setText(path)
        if not self.output_dir.text().strip():
            source = Path(path)
            self.output_dir.setText(str(source.parent / f"{source.stem}_frames"))
        self._inspect_video()

    def _pick_output(self) -> None:
        start = self.output_dir.text().strip() or str(Path.home())
        path = QFileDialog.getExistingDirectory(
            self,
            "Select output folder",
            start,
        )
        if path:
            self.output_dir.setText(path)

    def _metadata(self):
        path = self.video_path.text().strip()
        if not CV2_OK or not path or not Path(path).is_file():
            return None
        cap = cv2.VideoCapture(path)
        try:
            if not cap.isOpened():
                return None
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if fps <= 0 or total <= 0:
                return None
            return width, height, fps, total, total / fps
        finally:
            cap.release()

    def _inspect_video(self) -> None:
        metadata = self._metadata()
        if metadata is None:
            self.video_info.setText("Could not read video metadata.")
        else:
            width, height, fps, total, duration = metadata
            self.video_info.setText(
                f"{width}×{height} • {fps:.2f} fps • "
                f"{duration:.2f}s • {total} frames"
            )
        self._refresh_estimate()

    def _refresh_estimate(self, *_args) -> None:
        metadata = self._metadata()
        if metadata is None:
            self.estimate.setText("")
            return
        _width, _height, fps, total, duration = metadata
        if self._mode == "all":
            text = f"Will save all {total} frames."
        elif self._mode == "persec":
            text = (
                f"Will save about {math.ceil(duration)} image(s) "
                f"(1/sec over {duration:.2f}s)."
            )
        elif self._mode == "fps4":
            text = (
                f"Will save about {math.ceil(duration * 4)} image(s) "
                f"(4/sec over {duration:.2f}s)."
            )
        elif self._mode == "fpscustom":
            target = self.custom_fps_input.value()
            effective = min(target, fps)
            text = (
                f"Will save about {math.ceil(duration * effective)} image(s) "
                f"({target:g}/sec requested)."
            )
        else:
            text = (
                f"Will save {min(self.count_input.value(), total)} image(s), "
                f"evenly spaced across {duration:.2f}s."
            )
        self.estimate.setText(text)

    def _on_format_changed(self, *_args) -> None:
        image_format = self.format_combo.currentData()
        self.quality_slider.setEnabled(image_format != "png")
        self._on_quality_changed(self.quality_slider.value())

    def _on_quality_changed(self, value: int) -> None:
        image_format = self.format_combo.currentData()
        if image_format == "png":
            self.quality_label.setText("PNG is lossless")
        else:
            self.quality_label.setText(
                f"{str(image_format).upper()} quality {value}"
            )

    def _set_running(self, running: bool) -> None:
        for card in (self.source_card, self.mode_card, self.format_card):
            card.setEnabled(not running)
        self.start_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)

    def start(self) -> None:
        if self.worker and self.worker.isRunning():
            return
        if not CV2_OK:
            QMessageBox.critical(
                self,
                "OpenCV not available",
                "Video → Images needs OpenCV.\n\n"
                "Install it with:\n\n    pip install opencv-python\n\n"
                f"Underlying error:\n{_CV2_IMPORT_ERROR}",
            )
            return

        video_path = self.video_path.text().strip()
        output_dir = self.output_dir.text().strip()
        if not video_path or not Path(video_path).is_file():
            QMessageBox.warning(
                self,
                "Missing video",
                "Please choose a valid video file.",
            )
            return
        if not output_dir:
            QMessageBox.warning(
                self,
                "Missing output",
                "Please choose an output folder.",
            )
            return
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot create output folder", str(exc))
            return

        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.status.setText("Starting…")
        self.worker = VideoExtractWorker(
            video_path=video_path,
            output_dir=output_dir,
            mode=self._mode,
            count=self.count_input.value(),
            custom_fps=self.custom_fps_input.value(),
            image_format=str(self.format_combo.currentData()),
            quality=self.quality_slider.value(),
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.completed.connect(self._on_completed)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()
        self._set_running(True)

    def cancel(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.cancel_btn.setEnabled(False)
            self.status.setText("Cancelling…")

    def _on_progress(self, current: int, total: int, message: str) -> None:
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(current)
        self.status.setText(message)

    def _on_completed(self, cancelled: bool, message: str) -> None:
        self._set_running(False)
        self.status.setText(message)
        if not cancelled:
            self.progress.setValue(self.progress.maximum())

    def _on_failed(self, message: str) -> None:
        self._set_running(False)
        self.status.setText(f"Error: {message}")
        QMessageBox.critical(self, "Video extraction failed", message)

    def is_running(self) -> bool:
        return bool(self.worker and self.worker.isRunning())

    def cancel_and_wait(self, timeout_ms: int = 3000) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(timeout_ms)


# ============================================================================
# Dialogue tools — native PyQt tab and worker
# ============================================================================

class DialogueToolsWorker(QThread):
    """Run dialogue cleaning or frame-transcript extraction off the UI thread."""

    completed = pyqtSignal(str, str, str)
    failed = pyqtSignal(str)

    def __init__(
        self,
        *,
        mode: str,
        input_path: str,
        output_path: str,
        save_report: bool,
        profile: str,
        include_narration: bool,
        show_frames: bool,
        minimum_confidence: str,
        aliases: str,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.save_report = save_report
        self.profile = profile
        self.include_narration = include_narration
        self.show_frames = show_frames
        self.minimum_confidence = minimum_confidence
        self.aliases = aliases

    def run(self) -> None:
        try:
            report_path = (
                self.output_path.with_name(
                    f"{self.output_path.stem}_report.json"
                )
                if self.save_report
                else None
            )
            if self.mode == "clean":
                report = clean_file(
                    self.input_path,
                    self.output_path,
                    report_path,
                    CleanerConfig.for_profile(self.profile),
                )
                details = cleaner_summary(report)
            else:
                source = self.input_path.read_text(encoding="utf-8-sig")
                alias_values = [
                    item.strip()
                    for item in self.aliases.replace("\n", ";").split(";")
                    if item.strip()
                ]
                entries, report = extract_transcript(
                    source,
                    speaker_aliases=parse_aliases(alias_values),
                    include_narration=self.include_narration,
                )
                confidence_rank = {"low": 0, "medium": 1, "high": 2}
                minimum_rank = confidence_rank[self.minimum_confidence]
                printable_entries = [
                    entry
                    for entry in entries
                    if confidence_rank[entry.confidence] >= minimum_rank
                ]
                report.entries_excluded_by_confidence = (
                    len(entries) - len(printable_entries)
                )
                rendered = format_transcript(
                    printable_entries,
                    show_frames=self.show_frames,
                )
                self.output_path.parent.mkdir(parents=True, exist_ok=True)
                self.output_path.write_text(rendered, encoding="utf-8")
                if report_path:
                    report_path.parent.mkdir(parents=True, exist_ok=True)
                    report_path.write_text(
                        json.dumps(
                            asdict(report),
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                details = (
                    f"Input: {report.input_frames} OCR frames\n"
                    f"Output: {len(printable_entries)} transcript entries\n"
                    f"Named dialogue: {report.named_dialogue_entries}\n"
                    f"Narration/screen text: {report.narration_entries}\n"
                    f"Low-confidence entries: {report.low_confidence_entries}\n"
                    "Excluded by confidence filter: "
                    f"{report.entries_excluded_by_confidence}"
                )

            preview = self.output_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
            report_note = str(report_path) if report_path else ""
            self.completed.emit(details, preview, report_note)
        except Exception as exc:
            self.failed.emit(str(exc))


class DialogueToolsWidget(QWidget):
    """PyQt interface for the OCR dialogue cleaner and video extractor."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.worker: Optional[DialogueToolsWorker] = None
        self._build_ui()
        self._on_mode_changed()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(10)

        title = QLabel("Dialogue Tools")
        title.setObjectName("pageTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(title)

        self.subtitle = QLabel()
        self.subtitle.setObjectName("muted")
        self.subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle.setWordWrap(True)
        root.addWidget(self.subtitle)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        form = QWidget()
        self.form_widget = form
        form_layout = QVBoxLayout(form)
        form_layout.setContentsMargins(4, 4, 4, 4)
        form_layout.setSpacing(10)
        scroll.setWidget(form)
        root.addWidget(scroll, 1)

        source_card = self._card("Source and Output")
        source_layout = source_card.layout()

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Tool"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Clean OCR dialogue", "clean")
        self.mode_combo.addItem("Extract video transcript", "extract")
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.mode_combo, 1)
        source_layout.addLayout(mode_row)

        input_row = QHBoxLayout()
        input_row.addWidget(QLabel("OCR text"))
        self.input_path = QLineEdit()
        self.input_path.setPlaceholderText("Choose a combined OCR .txt file…")
        self.input_path.editingFinished.connect(self._suggest_output)
        input_row.addWidget(self.input_path, 1)
        input_button = QPushButton("Browse…")
        input_button.clicked.connect(self._pick_input)
        input_row.addWidget(input_button)
        source_layout.addLayout(input_row)

        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Output"))
        self.output_path = QLineEdit()
        self.output_path.setPlaceholderText("Choose where to save the result…")
        output_row.addWidget(self.output_path, 1)
        output_button = QPushButton("Browse…")
        output_button.clicked.connect(self._pick_output)
        output_row.addWidget(output_button)
        source_layout.addLayout(output_row)

        self.save_report = QCheckBox(
            "Save a JSON review report next to the output"
        )
        self.save_report.setChecked(True)
        source_layout.addWidget(self.save_report)
        form_layout.addWidget(source_card)

        self.cleaner_card = self._card("Cleaning Settings")
        cleaner_layout = self.cleaner_card.layout()
        profile_row = QHBoxLayout()
        profile_row.addWidget(QLabel("Cleaning level"))
        self.profile_combo = QComboBox()
        self.profile_combo.addItem("Conservative", "conservative")
        self.profile_combo.addItem("Balanced", "balanced")
        self.profile_combo.addItem("Aggressive", "aggressive")
        self.profile_combo.setCurrentIndex(1)
        profile_row.addWidget(self.profile_combo)
        profile_row.addStretch(1)
        cleaner_layout.addLayout(profile_row)
        cleaner_hint = QLabel(
            "Balanced is recommended. Aggressive catches more corrupted repeats "
            "but should be checked using the JSON report."
        )
        cleaner_hint.setObjectName("muted")
        cleaner_hint.setWordWrap(True)
        cleaner_layout.addWidget(cleaner_hint)
        form_layout.addWidget(self.cleaner_card)

        self.extractor_card = self._card("Transcript Settings")
        extractor_layout = self.extractor_card.layout()
        self.include_narration = QCheckBox(
            "Include narration and on-screen prose"
        )
        self.include_narration.setChecked(True)
        extractor_layout.addWidget(self.include_narration)
        self.show_frames = QCheckBox("Include source frame ranges")
        self.show_frames.setChecked(True)
        extractor_layout.addWidget(self.show_frames)

        confidence_row = QHBoxLayout()
        confidence_row.addWidget(QLabel("Minimum confidence"))
        self.confidence_combo = QComboBox()
        self.confidence_combo.addItem("Low — include [CHECK] entries", "low")
        self.confidence_combo.addItem("Medium", "medium")
        self.confidence_combo.addItem("High", "high")
        self.confidence_combo.setCurrentIndex(1)
        confidence_row.addWidget(self.confidence_combo)
        confidence_row.addStretch(1)
        extractor_layout.addLayout(confidence_row)

        alias_row = QHBoxLayout()
        alias_row.addWidget(QLabel("Speaker aliases"))
        self.aliases = QLineEdit()
        self.aliases.setPlaceholderText(
            "OCR Spelling=Preferred Name; another alias=Display Name"
        )
        alias_row.addWidget(self.aliases, 1)
        extractor_layout.addLayout(alias_row)
        alias_hint = QLabel(
            "Separate multiple aliases with semicolons. Aliases are optional."
        )
        alias_hint.setObjectName("muted")
        extractor_layout.addWidget(alias_hint)
        form_layout.addWidget(self.extractor_card)

        result_card = self._card("Result Preview")
        result_layout = result_card.layout()
        self.preview = QTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setMinimumHeight(220)
        self.preview.setPlaceholderText(
            "The cleaned dialogue or transcript will appear here…"
        )
        result_layout.addWidget(self.preview)
        form_layout.addWidget(result_card, 1)
        form_layout.addStretch(1)

        progress_row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        progress_row.addWidget(self.progress, 1)
        self.start_button = QPushButton("Clean File")
        self.start_button.setObjectName("primary")
        self.start_button.setMinimumHeight(44)
        self.start_button.setMinimumWidth(170)
        self.start_button.clicked.connect(self.start)
        progress_row.addWidget(self.start_button)
        root.addLayout(progress_row)

        self.status = QLabel("Choose an OCR text file to begin.")
        self.status.setObjectName("muted")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

    def _card(self, title: str) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)
        heading = QLabel(title)
        heading.setObjectName("cardTitle")
        layout.addWidget(heading)
        return card

    def _on_mode_changed(self, *_args) -> None:
        clean_mode = self.mode_combo.currentData() == "clean"
        self.cleaner_card.setVisible(clean_mode)
        self.extractor_card.setVisible(not clean_mode)
        self.start_button.setText(
            "Clean File" if clean_mode else "Extract Transcript"
        )
        self.subtitle.setText(
            "Remove OCR repetition, join wrapped dialogue, and flag cut-offs."
            if clean_mode
            else (
                "Consolidate progressive frame-by-frame OCR into a readable "
                "dialogue transcript."
            )
        )
        self._suggest_output(force=True)

    def _pick_input(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose OCR text",
            self.input_path.text(),
            "Text files (*.txt);;All files (*.*)",
        )
        if path:
            self.input_path.setText(path)
            self._suggest_output(force=True)

    def _pick_output(self) -> None:
        suggested = self.output_path.text().strip()
        if not suggested:
            source = Path(self.input_path.text().strip())
            suggested = str(source.parent if source.parent else Path.home())
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save dialogue output",
            suggested,
            "Text files (*.txt)",
        )
        if path:
            self.output_path.setText(path)

    def _suggest_output(self, *_args, force: bool = False) -> None:
        source_text = self.input_path.text().strip()
        if not source_text:
            return
        if self.output_path.text().strip() and not force:
            return
        source = Path(source_text)
        suffix = (
            "cleaned"
            if self.mode_combo.currentData() == "clean"
            else "transcript"
        )
        self.output_path.setText(
            str(source.with_name(f"{source.stem}_{suffix}.txt"))
        )

    def _set_running(self, running: bool) -> None:
        self.form_widget.setEnabled(not running)
        self.start_button.setEnabled(not running)
        if running:
            self.progress.setRange(0, 0)
            self.status.setText("Processing…")
        else:
            self.progress.setRange(0, 1)
            self.progress.setValue(1)

    def start(self) -> None:
        if self.worker and self.worker.isRunning():
            return
        input_path = self.input_path.text().strip()
        output_path = self.output_path.text().strip()
        if not input_path or not Path(input_path).is_file():
            QMessageBox.warning(
                self,
                "Missing input",
                "Please choose an existing OCR text file.",
            )
            return
        if not output_path:
            QMessageBox.warning(
                self,
                "Missing output",
                "Please choose where to save the result.",
            )
            return
        if Path(input_path).resolve() == Path(output_path).resolve():
            QMessageBox.warning(
                self,
                "Choose another output",
                "The output file must be different from the input file.",
            )
            return

        self.preview.clear()
        self.worker = DialogueToolsWorker(
            mode=str(self.mode_combo.currentData()),
            input_path=input_path,
            output_path=output_path,
            save_report=self.save_report.isChecked(),
            profile=str(self.profile_combo.currentData()),
            include_narration=self.include_narration.isChecked(),
            show_frames=self.show_frames.isChecked(),
            minimum_confidence=str(self.confidence_combo.currentData()),
            aliases=self.aliases.text(),
        )
        self.worker.completed.connect(self._on_completed)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_thread_finished)
        self.worker.start()
        self._set_running(True)

    def _on_completed(
        self,
        details: str,
        preview: str,
        report_path: str,
    ) -> None:
        self.preview.setPlainText(preview)
        destination = self.output_path.text().strip()
        report_note = f"\nReport: {report_path}" if report_path else ""
        self.status.setText(
            f"{details.replace(chr(10), '  •  ')}  •  Saved to {destination}"
        )
        QMessageBox.information(
            self,
            "Dialogue processing finished",
            f"{details}\n\nSaved to:\n{destination}{report_note}",
        )

    def _on_failed(self, message: str) -> None:
        self.status.setText(f"Error: {message}")
        QMessageBox.critical(self, "Dialogue processing failed", message)

    def _on_thread_finished(self) -> None:
        self._set_running(False)
        if self.worker:
            self.worker.deleteLater()
        self.worker = None

    def is_running(self) -> bool:
        return bool(self.worker and self.worker.isRunning())

    def wait_for_finish(self, timeout_ms: int = 30_000) -> bool:
        if not self.worker or not self.worker.isRunning():
            return True
        self.worker.requestInterruption()
        return self.worker.wait(timeout_ms)


# ============================================================================
# Overall Run — video → frames → OCR → cleaner
# ============================================================================

class OverallRunWidget(QWidget):
    """Chain the three tools into one guided, cancellable workflow."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.extract_worker: Optional[VideoExtractWorker] = None
        self.ocr_worker: Optional[QThread] = None
        self.clean_worker: Optional[DialogueToolsWorker] = None
        self.frame_paths: list[str] = []
        self.ocr_results: dict[int, OCRResult] = {}
        self._running = False
        self._cancelled = False
        self._build_ui()
        self._refresh_extraction_controls()
        self._on_backend_changed()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(10)

        title = QLabel("Overall Run")
        title.setObjectName("pageTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(title)

        subtitle = QLabel(
            "Run the complete workflow: video → temp_img_ocr → OCR → "
            "cleaned text."
        )
        subtitle.setObjectName("muted")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        form = QWidget()
        self.form_widget = form
        form_layout = QVBoxLayout(form)
        form_layout.setContentsMargins(4, 4, 4, 4)
        form_layout.setSpacing(10)
        scroll.setWidget(form)
        root.addWidget(scroll, 1)

        source_card = self._card("Source and Destination")
        source_layout = source_card.layout()

        video_row = QHBoxLayout()
        video_row.addWidget(QLabel("Video"))
        self.video_path = QLineEdit()
        self.video_path.setPlaceholderText("Choose the source video…")
        video_row.addWidget(self.video_path, 1)
        video_button = QPushButton("Browse…")
        video_button.clicked.connect(self._pick_video)
        video_row.addWidget(video_button)
        source_layout.addLayout(video_row)

        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Output folder"))
        self.output_dir = QLineEdit()
        self.output_dir.setPlaceholderText(
            "Temporary frames and final text will be saved here…"
        )
        output_row.addWidget(self.output_dir, 1)
        output_button = QPushButton("Browse…")
        output_button.clicked.connect(self._pick_output)
        output_row.addWidget(output_button)
        source_layout.addLayout(output_row)

        temp_note = QLabel(
            "Extracted images go into an automatically created "
            "temp_img_ocr subfolder."
        )
        temp_note.setObjectName("muted")
        source_layout.addWidget(temp_note)
        form_layout.addWidget(source_card)

        extract_card = self._card("Step 1 — Extract Images")
        extract_layout = extract_card.layout()

        extract_mode_row = QHBoxLayout()
        extract_mode_row.addWidget(QLabel("Mode"))
        self.extract_mode = QComboBox()
        self.extract_mode.addItem("Evenly spaced total count", "count")
        self.extract_mode.addItem("1 image / second", "persec")
        self.extract_mode.addItem("All frames", "all")
        self.extract_mode.addItem("4 images / second", "fps4")
        self.extract_mode.addItem("Custom fps", "fpscustom")
        self.extract_mode.currentIndexChanged.connect(
            self._refresh_extraction_controls
        )
        extract_mode_row.addWidget(self.extract_mode, 1)
        self.extract_count = QSpinBox()
        self.extract_count.setRange(1, 10_000_000)
        self.extract_count.setValue(60)
        extract_mode_row.addWidget(self.extract_count)
        self.custom_fps = QDoubleSpinBox()
        self.custom_fps.setRange(0.01, 1000.0)
        self.custom_fps.setDecimals(2)
        self.custom_fps.setValue(2.0)
        extract_mode_row.addWidget(self.custom_fps)
        extract_layout.addLayout(extract_mode_row)

        format_row = QHBoxLayout()
        format_row.addWidget(QLabel("Image format"))
        self.image_format = QComboBox()
        self.image_format.addItem("JPG", "jpg")
        self.image_format.addItem("PNG", "png")
        self.image_format.addItem("WebP", "webp")
        self.image_format.currentIndexChanged.connect(
            self._refresh_extraction_controls
        )
        format_row.addWidget(self.image_format)
        format_row.addWidget(QLabel("Quality"))
        self.image_quality = QSlider(Qt.Orientation.Horizontal)
        self.image_quality.setRange(1, 100)
        self.image_quality.setValue(95)
        format_row.addWidget(self.image_quality, 1)
        self.quality_value = QLabel("95")
        self.image_quality.valueChanged.connect(
            lambda value: self.quality_value.setText(str(value))
        )
        format_row.addWidget(self.quality_value)
        extract_layout.addLayout(format_row)
        form_layout.addWidget(extract_card)

        ocr_card = self._card("Step 2 — OCR")
        ocr_layout = ocr_card.layout()
        ocr_row = QHBoxLayout()
        ocr_row.addWidget(QLabel("Backend"))
        self.backend = QComboBox()
        for name, key in BACKENDS:
            self.backend.addItem(name, key)
        self.backend.currentIndexChanged.connect(self._on_backend_changed)
        ocr_row.addWidget(self.backend)
        ocr_row.addWidget(QLabel("Language"))
        self.language = QComboBox()
        ocr_row.addWidget(self.language, 1)
        ocr_layout.addLayout(ocr_row)

        ocr_options = QHBoxLayout()
        self.detect_rotation = QCheckBox("Detect rotation (Paddle)")
        self.detect_rotation.setChecked(True)
        ocr_options.addWidget(self.detect_rotation)
        self.easy_gpu = QCheckBox("Use GPU (EasyOCR)")
        ocr_options.addWidget(self.easy_gpu)
        ocr_options.addStretch(1)
        ocr_layout.addLayout(ocr_options)
        form_layout.addWidget(ocr_card)

        cleaner_card = self._card("Step 3 — Clean OCR Text")
        cleaner_layout = cleaner_card.layout()
        cleaner_row = QHBoxLayout()
        cleaner_row.addWidget(QLabel("Cleaning level"))
        self.cleaner_profile = QComboBox()
        self.cleaner_profile.addItem("Conservative", "conservative")
        self.cleaner_profile.addItem("Balanced", "balanced")
        self.cleaner_profile.addItem("Aggressive", "aggressive")
        self.cleaner_profile.setCurrentIndex(1)
        cleaner_row.addWidget(self.cleaner_profile)
        cleaner_row.addStretch(1)
        cleaner_layout.addLayout(cleaner_row)
        cleaner_note = QLabel(
            "The raw OCR file is kept. The cleaner removes repeated readings, "
            "joins wrapped dialogue, and saves a JSON review report."
        )
        cleaner_note.setObjectName("muted")
        cleaner_note.setWordWrap(True)
        cleaner_layout.addWidget(cleaner_note)
        form_layout.addWidget(cleaner_card)

        warning = QLabel(
            "Important: immediately before the run starts, you will be asked "
            "to verify every setting. OCR model downloads may be large."
        )
        warning.setObjectName("estimate")
        warning.setWordWrap(True)
        form_layout.addWidget(warning)

        progress_card = self._card("Run Progress")
        progress_layout = progress_card.layout()
        self.stage_label = QLabel("Ready")
        self.stage_label.setObjectName("muted")
        progress_layout.addWidget(self.stage_label)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("%v / %m  (%p%)")
        progress_layout.addWidget(self.progress)

        self.model_progress_label = QLabel("")
        self.model_progress_label.setObjectName("muted")
        self.model_progress_label.setVisible(False)
        progress_layout.addWidget(self.model_progress_label)
        self.model_progress = QProgressBar()
        self.model_progress.setRange(0, 1000)
        self.model_progress.setValue(0)
        self.model_progress.setVisible(False)
        progress_layout.addWidget(self.model_progress)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(150)
        self.log.setPlaceholderText("Run details will appear here…")
        progress_layout.addWidget(self.log)
        form_layout.addWidget(progress_card, 1)
        form_layout.addStretch(1)

        actions = QHBoxLayout()
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("danger")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel)
        actions.addWidget(self.cancel_button)
        self.start_button = QPushButton("Check Settings and Run")
        self.start_button.setObjectName("primary")
        self.start_button.setMinimumHeight(44)
        self.start_button.clicked.connect(self.start)
        actions.addWidget(self.start_button, 1)
        root.addLayout(actions)

    def _card(self, title: str) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)
        heading = QLabel(title)
        heading.setObjectName("cardTitle")
        layout.addWidget(heading)
        return card

    def _pick_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select source video",
            self.video_path.text(),
            VIDEO_FILTER,
        )
        if not path:
            return
        self.video_path.setText(path)
        if not self.output_dir.text().strip():
            source = Path(path)
            self.output_dir.setText(
                str(source.parent / f"{source.stem}_ocr_output")
            )

    def _pick_output(self) -> None:
        start = self.output_dir.text().strip() or str(Path.home())
        path = QFileDialog.getExistingDirectory(
            self,
            "Select output folder",
            start,
        )
        if path:
            self.output_dir.setText(path)

    def _refresh_extraction_controls(self, *_args) -> None:
        mode = self.extract_mode.currentData()
        self.extract_count.setVisible(mode == "count")
        self.custom_fps.setVisible(mode == "fpscustom")
        self.image_quality.setEnabled(self.image_format.currentData() != "png")
        self.quality_value.setEnabled(self.image_format.currentData() != "png")

    def _on_backend_changed(self, *_args) -> None:
        backend = self.backend.currentData()
        if backend == "paddle":
            languages = LANGUAGES_PADDLE
        elif backend == "easy":
            languages = LANGUAGES_EASY
        else:
            languages = LANGUAGES_MINERU
        self.language.clear()
        for name, code in languages:
            self.language.addItem(name, code)
        self.language.setEnabled(backend != "mineru")
        self.detect_rotation.setEnabled(backend == "paddle")
        self.easy_gpu.setEnabled(backend == "easy")

    def _check_dependencies(self) -> bool:
        if not CV2_OK:
            QMessageBox.critical(
                self,
                "OpenCV not available",
                "Overall Run needs OpenCV.\n\n"
                "Install it with:\n\n    pip install opencv-python",
            )
            return False
        backend = self.backend.currentData()
        if backend == "paddle" and not PADDLE_OK:
            QMessageBox.critical(
                self,
                "PaddleOCR not available",
                "Install PaddleOCR and PaddlePaddle before using this backend.",
            )
            return False
        if backend == "easy" and not EASY_OK:
            QMessageBox.critical(
                self,
                "EasyOCR not available",
                "Install it with:\n\n    pip install easyocr",
            )
            return False
        if backend == "mineru":
            required = ("mineru_vl_utils", "transformers", "huggingface_hub")
            missing = [
                name
                for name in required
                if importlib.util.find_spec(name) is None
            ]
            if missing:
                QMessageBox.critical(
                    self,
                    "MinerU2.5-Pro not available",
                    'Install it with:\n\n    pip install -U '
                    '"mineru-vl-utils[transformers]"\n\n'
                    f"Missing: {', '.join(missing)}",
                )
                return False
        return True

    def _extraction_description(self) -> str:
        mode = self.extract_mode.currentData()
        if mode == "count":
            return f"{self.extract_count.value()} evenly spaced images"
        if mode == "fpscustom":
            return f"{self.custom_fps.value():g} images/second"
        labels = {
            "persec": "1 image/second",
            "all": "all video frames",
            "fps4": "4 images/second",
        }
        return labels[str(mode)]

    def _confirm_settings(self, temp_dir: Path) -> bool:
        backend = BACKEND_LABELS.get(
            str(self.backend.currentData()),
            self.backend.currentText(),
        )
        image_format = str(self.image_format.currentData()).upper()
        quality_note = (
            ""
            if image_format == "PNG"
            else f", quality {self.image_quality.value()}"
        )
        summary = (
            f"Video: {self.video_path.text().strip()}\n"
            f"Output folder: {self.output_dir.text().strip()}\n"
            f"Temporary images: {temp_dir}\n\n"
            f"Extraction: {self._extraction_description()}\n"
            f"Image format: {image_format}{quality_note}\n"
            f"OCR backend: {backend}\n"
            f"OCR language: {self.language.currentText()}\n"
            f"Detect rotation: "
            f"{'Yes' if self.detect_rotation.isChecked() else 'No'}\n"
            f"EasyOCR GPU: {'Yes' if self.easy_gpu.isChecked() else 'No'}\n"
            f"Cleaner: {self.cleaner_profile.currentText()}\n\n"
            "Please check these settings carefully before starting. The first "
            "OCR run may download model files, and matching temporary frame "
            "files may be overwritten."
        )
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Check Settings Before Overall Run")
        box.setText("Are all Overall Run settings correct?")
        box.setInformativeText(summary)
        box.setStandardButtons(
            QMessageBox.StandardButton.Ok
            | QMessageBox.StandardButton.Cancel
        )
        box.button(QMessageBox.StandardButton.Ok).setText("Run Now")
        return box.exec() == QMessageBox.StandardButton.Ok

    def _set_running(self, running: bool) -> None:
        self._running = running
        self.form_widget.setEnabled(not running)
        self.start_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        if not running:
            self.model_progress.setVisible(False)
            self.model_progress_label.setVisible(False)

    def _append_log(self, text: str) -> None:
        self.log.append(text)

    def start(self) -> None:
        if self._running:
            return
        video = self.video_path.text().strip()
        output = self.output_dir.text().strip()
        if not video or not Path(video).is_file():
            QMessageBox.warning(
                self,
                "Missing video",
                "Please choose an existing video file.",
            )
            return
        if not output:
            QMessageBox.warning(
                self,
                "Missing output folder",
                "Please choose an output folder.",
            )
            return
        if not self._check_dependencies():
            return

        output_dir = Path(output)
        temp_dir = output_dir / "temp_img_ocr"
        if not self._confirm_settings(temp_dir):
            return
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            temp_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QMessageBox.critical(self, "Cannot create output folder", str(exc))
            return

        self._cancelled = False
        self.frame_paths = []
        self.ocr_results = {}
        self.log.clear()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.stage_label.setText("Step 1/3 — Extracting images…")
        self._append_log(f"Temporary images: {temp_dir}")
        self._set_running(True)

        self.extract_worker = VideoExtractWorker(
            video_path=video,
            output_dir=str(temp_dir),
            mode=str(self.extract_mode.currentData()),
            count=self.extract_count.value(),
            custom_fps=self.custom_fps.value(),
            image_format=str(self.image_format.currentData()),
            quality=self.image_quality.value(),
        )
        self.extract_worker.progress.connect(self._on_extract_progress)
        self.extract_worker.outputs_ready.connect(self._on_frames_ready)
        self.extract_worker.completed.connect(self._on_extract_completed)
        self.extract_worker.failed.connect(self._on_stage_failed)
        self.extract_worker.start()

    def _on_extract_progress(
        self,
        current: int,
        total: int,
        message: str,
    ) -> None:
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(current)
        self.stage_label.setText(f"Step 1/3 — {message}")

    def _on_frames_ready(self, paths: list[str]) -> None:
        self.frame_paths = list(paths)

    def _on_extract_completed(self, cancelled: bool, message: str) -> None:
        self._append_log(message)
        if cancelled or self._cancelled:
            self._finish_cancelled()
            return
        if not self.frame_paths:
            self._on_stage_failed("No video frames were saved.")
            return
        self._start_ocr()

    def _start_ocr(self) -> None:
        backend = self.backend.currentData()
        language = str(self.language.currentData())
        if backend == "paddle":
            worker: QThread = OCRWorker(
                self.frame_paths,
                lang=language,
                use_angle_cls=self.detect_rotation.isChecked(),
            )
        elif backend == "easy":
            worker = EasyOCRWorker(
                self.frame_paths,
                lang=language,
                gpu=self.easy_gpu.isChecked(),
            )
        else:
            worker = MinerUWorker(self.frame_paths)
        self.ocr_worker = worker
        worker.progress.connect(self._on_ocr_progress)
        worker.file_done.connect(self._on_ocr_file_done)
        worker.finished_ok.connect(self._on_ocr_finished)
        worker.failed.connect(self._on_stage_failed)
        worker.download_progress.connect(self._on_model_progress)
        self.stage_label.setText(
            f"Step 2/3 — Starting {BACKEND_LABELS[str(backend)]}…"
        )
        self.progress.setRange(0, len(self.frame_paths))
        self.progress.setValue(0)
        self._append_log(
            f"OCR input: {len(self.frame_paths)} extracted image(s)"
        )
        worker.start()

    def _on_ocr_progress(
        self,
        current: int,
        total: int,
        name: str,
    ) -> None:
        self.model_progress.setVisible(False)
        self.model_progress_label.setVisible(False)
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(current)
        detail = f": {name}" if name else ""
        self.stage_label.setText(
            f"Step 2/3 — OCR {min(current + 1, total)}/{total}{detail}"
        )

    def _on_ocr_file_done(self, index: int, result: OCRResult) -> None:
        self.ocr_results[index] = result

    @staticmethod
    def _human_size(value: float) -> str:
        size = float(value)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if abs(size) < 1024.0 or unit == "TB":
                return (
                    f"{size:.0f} {unit}"
                    if unit == "B"
                    else f"{size:.1f} {unit}"
                )
            size /= 1024.0
        return f"{size:.1f} TB"

    def _on_model_progress(
        self,
        current,
        total,
        stage: str,
        model_label: str,
    ) -> None:
        self.model_progress.setVisible(True)
        self.model_progress_label.setVisible(True)
        if stage in {"checking", "loading"}:
            self.model_progress.setRange(0, 0)
            verb = "Checking" if stage == "checking" else "Loading"
            label = f"{verb} {model_label} model…"
        else:
            total_value = float(total or 0)
            current_value = (
                min(float(current or 0), total_value)
                if total_value
                else 0.0
            )
            self.model_progress.setRange(0, 1000)
            fraction = (
                max(0.0, min(1.0, current_value / total_value))
                if total_value
                else 0.0
            )
            self.model_progress.setValue(round(fraction * 1000))
            if stage == "bytes":
                detail = (
                    f"{self._human_size(current_value)} / "
                    f"{self._human_size(total_value)}"
                )
            else:
                detail = f"{int(current_value)} / {int(total_value)} files"
            label = f"Downloading {model_label} — {detail}"
        self.model_progress_label.setText(label)
        self.stage_label.setText(f"Step 2/3 — {label}")

    def _on_ocr_finished(self) -> None:
        self.model_progress.setVisible(False)
        self.model_progress_label.setVisible(False)
        if self._cancelled:
            self._finish_cancelled()
            return
        usable = [
            result
            for _, result in sorted(self.ocr_results.items())
            if result.is_ok and result.full_text.strip()
        ]
        if not usable:
            self._on_stage_failed(
                "OCR finished, but no readable text was detected."
            )
            return

        output_dir = Path(self.output_dir.text().strip())
        stem = Path(self.video_path.text().strip()).stem
        raw_path = output_dir / f"{stem}_ocr_raw.txt"
        clean_path = output_dir / f"{stem}_cleaned.txt"
        try:
            raw_text = "\n\n".join(
                result.full_text.strip() for result in usable
            ).strip()
            raw_path.write_text(raw_text + "\n", encoding="utf-8")
        except OSError as exc:
            self._on_stage_failed(f"Could not save raw OCR text: {exc}")
            return

        failed_count = sum(
            not result.is_ok for result in self.ocr_results.values()
        )
        self._append_log(
            f"Raw OCR: {raw_path}\n"
            f"Readable frames: {len(usable)}; failed frames: {failed_count}"
        )
        self.stage_label.setText("Step 3/3 — Cleaning OCR text…")
        self.progress.setRange(0, 0)
        self.clean_worker = DialogueToolsWorker(
            mode="clean",
            input_path=str(raw_path),
            output_path=str(clean_path),
            save_report=True,
            profile=str(self.cleaner_profile.currentData()),
            include_narration=True,
            show_frames=True,
            minimum_confidence="medium",
            aliases="",
        )
        self.clean_worker.completed.connect(self._on_clean_completed)
        self.clean_worker.failed.connect(self._on_stage_failed)
        self.clean_worker.start()

    def _on_clean_completed(
        self,
        details: str,
        preview: str,
        report_path: str,
    ) -> None:
        if self._cancelled:
            self._finish_cancelled()
            return
        clean_path = (
            Path(self.output_dir.text().strip())
            / f"{Path(self.video_path.text().strip()).stem}_cleaned.txt"
        )
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.stage_label.setText("Complete — cleaned text is ready.")
        self._append_log(
            f"{details}\nCleaned text: {clean_path}\nReport: {report_path}"
        )
        if preview:
            sample = preview[:4000]
            self._append_log(f"\nCleaned preview:\n{sample}")
        self._set_running(False)
        QMessageBox.information(
            self,
            "Overall Run complete",
            "All three steps finished successfully.\n\n"
            f"Cleaned text:\n{clean_path}\n\n"
            f"Review report:\n{report_path}\n\n"
            f"Temporary images:\n"
            f"{Path(self.output_dir.text().strip()) / 'temp_img_ocr'}",
        )

    def _on_stage_failed(self, message: str) -> None:
        if not self._running:
            return
        self.stage_label.setText(f"Failed — {message}")
        self._append_log(f"ERROR: {message}")
        self._set_running(False)
        QMessageBox.critical(self, "Overall Run failed", message)

    def cancel(self) -> None:
        if not self._running:
            return
        self._cancelled = True
        self.cancel_button.setEnabled(False)
        self.stage_label.setText("Cancelling safely…")
        if self.extract_worker and self.extract_worker.isRunning():
            self.extract_worker.cancel()
        if self.ocr_worker and self.ocr_worker.isRunning():
            self.ocr_worker.cancel()
        if self.clean_worker and self.clean_worker.isRunning():
            self.clean_worker.requestInterruption()

    def _finish_cancelled(self) -> None:
        if not self._running:
            return
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.stage_label.setText("Cancelled.")
        self._append_log("Overall Run cancelled.")
        self._set_running(False)

    def is_running(self) -> bool:
        return self._running

    def cancel_and_wait(self, timeout_ms: int = 30_000) -> bool:
        self.cancel()
        workers = (
            self.extract_worker,
            self.ocr_worker,
            self.clean_worker,
        )
        for worker in workers:
            if worker and worker.isRunning() and not worker.wait(timeout_ms):
                return False
        return True


# ============================================================================
# Custom widgets
# ============================================================================

class DropListWidget(QListWidget):
    """QListWidget with image-friendly drag & drop."""

    files_dropped = pyqtSignal(list)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setIconSize(QSize(64, 64))
        self.setUniformItemSizes(True)
        self.setMovement(QListWidget.Movement.Static)
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        if event.mimeData().hasUrls():
            paths: list[str] = []
            for url in event.mimeData().urls():
                local = url.toLocalFile()
                if local:
                    paths.append(local)
            self.files_dropped.emit(paths)
            event.acceptProposedAction()


class ImagePreviewLabel(QLabel):
    """Auto-scaling image preview that keeps aspect ratio."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(220)
        self.setStyleSheet(
            "background:#1e2230;border:1px solid #2a2f42;border-radius:10px;"
        )
        self._pixmap: Optional[QPixmap] = None

    def set_pixmap(self, pix: Optional[QPixmap]) -> None:
        self._pixmap = pix
        self._refresh()

    def clear_preview(self) -> None:
        self._pixmap = None
        self.setText("Drop or add images to begin")
        self.setPixmap(QPixmap())

    def _refresh(self) -> None:
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event):  # noqa: D401
        super().resizeEvent(event)
        self._refresh()


# ============================================================================
# Main window
# ============================================================================

class MainWindow(QMainWindow):
    APP_TITLE = "Vision Studio"

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(self.APP_TITLE)
        self.resize(1280, 820)

        self.results: dict[int, OCRResult] = {}     # row -> result
        self.paths: list[str] = []
        self.worker: Optional[QThread] = None
        self._job_cancelled = False

        # Cache of base thumbnail pixmaps so badge updates don't re-decode images
        self._icon_cache: dict[str, QPixmap] = {}

        # Auto-export state (persisted via QSettings)
        self._settings = QSettings("PaddleOCR Studio", "PaddleOCR Studio")
        self._auto_export_folder: str = str(
            self._settings.value("auto_export_folder", "", type=str) or ""
        )

        self._build_style()
        self._build_ui()
        self._build_menu()
        self._wire()

        # Restore auto-export checkbox + folder label
        self.chk_auto_export.setChecked(
            bool(self._settings.value("auto_export_enabled", False, type=bool))
        )
        self._refresh_folder_label()

        self._set_running(False)
        self._on_main_tab_changed(self.main_tabs.currentIndex())

    # ----- styling ---------------------------------------------------------
    def _build_style(self) -> None:
        # Native dark-ish palette (works on every platform without external QSS).
        QApplication.setStyle("Fusion")
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background-color: #15182a;
                color: #e6e8ef;
                font-family: "Inter", "Segoe UI", "SF Pro Text", sans-serif;
                font-size: 13px;
            }
            QToolBar {
                background: #1b1f33;
                border: none;
                padding: 6px;
                spacing: 6px;
            }
            QToolBar QLabel { color: #aab1c5; padding: 0 6px; }

            QPushButton {
                background-color: #2a3252;
                color: #f3f5fb;
                border: 1px solid #3a4470;
                border-radius: 8px;
                padding: 8px 14px;
                font-weight: 500;
            }
            QPushButton:hover    { background-color: #34406b; }
            QPushButton:pressed  { background-color: #232a44; }
            QPushButton:disabled { background-color: #1d2238; color: #6b7191; border-color: #2a2f42; }
            QPushButton#primary {
                background-color: #4f7cff; border-color: #6a92ff;
            }
            QPushButton#primary:hover    { background-color: #6891ff; }
            QPushButton#primary:disabled { background-color: #2c3a66; color: #8a93b5; border-color: #34406b; }
            QPushButton#danger {
                background-color: #c0394a; border-color: #e05566;
            }
            QPushButton#danger:hover { background-color: #d64a5b; }

            QLabel#pageTitle {
                color: #ffffff;
                font-size: 24px;
                font-weight: 700;
                padding-top: 2px;
            }
            QLabel#muted { color: #8f98b3; }
            QLabel#estimate { color: #c9d4e0; padding-top: 4px; }
            QLabel#cardTitle {
                color: #ffffff;
                font-size: 14px;
                font-weight: 600;
            }
            QFrame#card {
                background: #1b1f33;
                border: 1px solid #2a2f42;
                border-radius: 12px;
            }
            QFrame#card QLabel, QFrame#card QRadioButton {
                background: transparent;
                border: none;
            }
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollArea > QWidget > QWidget { background: transparent; }

            QListWidget {
                background: #1a1e30;
                border: 1px solid #2a2f42;
                border-radius: 10px;
                padding: 6px;
                outline: 0;
            }
            QListWidget::item {
                padding: 8px 10px;
                border-radius: 6px;
                margin: 2px 2px;
            }
            QListWidget::item:selected {
                background-color: #34406b;
                color: #ffffff;
            }
            QListWidget::item:hover:!selected { background-color: #232a44; }

            QTextEdit, QPlainTextEdit {
                background: #1a1e30;
                border: 1px solid #2a2f42;
                border-radius: 10px;
                padding: 8px;
                selection-background-color: #4f7cff;
                selection-color: #ffffff;
                font-family: "JetBrains Mono", "Consolas", "Menlo", monospace;
                font-size: 12px;
            }
            QTabWidget::pane {
                border: 1px solid #2a2f42;
                border-radius: 10px;
                top: -1px;
                background: #1a1e30;
            }
            QTabBar::tab {
                background: #1a1e30;
                color: #aab1c5;
                padding: 8px 16px;
                border: 1px solid #2a2f42;
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #232a44;
                color: #ffffff;
            }
            QTabBar::tab:hover:!selected { background: #1f2438; }

            QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {
                background: #2a3252;
                color: #f3f5fb;
                border: 1px solid #3a4470;
                border-radius: 8px;
                padding: 6px 10px;
            }
            QComboBox { min-width: 140px; }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border-color: #6a92ff;
            }
            QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {
                background: #34406b;
            }
            QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
                background: #1d2238;
                color: #6b7191;
                border-color: #2a2f42;
            }
            QComboBox::drop-down { border: none; width: 22px; }
            QComboBox QAbstractItemView {
                background: #1f2438;
                color: #e6e8ef;
                border: 1px solid #3a4470;
                selection-background-color: #4f7cff;
            }

            QCheckBox { color: #cfd3e1; padding: 0 8px; }
            QCheckBox::indicator {
                width: 16px; height: 16px;
                border-radius: 4px;
                border: 1px solid #3a4470;
                background: #1a1e30;
            }
            QCheckBox::indicator:checked {
                background: #4f7cff; border-color: #6a92ff;
            }
            QRadioButton { color: #cfd3e1; spacing: 8px; }
            QRadioButton::indicator {
                width: 16px; height: 16px;
                border-radius: 9px;
                border: 1px solid #3a4470;
                background: #1a1e30;
            }
            QRadioButton::indicator:checked {
                background: #4f7cff;
                border: 3px solid #8aa9ff;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #1a1e30;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #4f7cff;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                width: 16px;
                margin: -5px 0;
                background: #8aa9ff;
                border: 1px solid #b6c8ff;
                border-radius: 8px;
            }

            QProgressBar {
                background: #1a1e30;
                border: 1px solid #2a2f42;
                border-radius: 8px;
                text-align: center;
                color: #e6e8ef;
                height: 18px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4f7cff, stop:1 #8aa9ff);
                border-radius: 8px;
            }

            QStatusBar {
                background: #1b1f33;
                color: #aab1c5;
                border-top: 1px solid #2a2f42;
            }

            QSplitter::handle { background: #1b1f33; }
            QSplitter::handle:horizontal { width: 6px; }
            QSplitter::handle:vertical   { height: 6px; }
            """
        )

    # ----- ui --------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)

        shell_layout = QVBoxLayout(central)
        shell_layout.setContentsMargins(8, 8, 8, 8)
        shell_layout.setSpacing(0)

        self.main_tabs = QTabWidget()
        self.main_tabs.setObjectName("mainTabs")
        shell_layout.addWidget(self.main_tabs)

        self.video_tab = VideoToImagesWidget()
        self.ocr_page = QWidget()
        self.dialogue_tab = DialogueToolsWidget()
        self.overall_tab = OverallRunWidget()
        self.main_tabs.addTab(self.video_tab, "Video → Images")
        self.main_tabs.addTab(self.ocr_page, "OCR")
        self.main_tabs.addTab(self.dialogue_tab, "Dialogue Tools")
        self.main_tabs.addTab(self.overall_tab, "Overall Run")

        root = QVBoxLayout(self.ocr_page)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # Toolbar ----------------------------------------------------------------
        toolbar = QToolBar("Main")
        self.ocr_toolbar = toolbar
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(18, 18))
        self.addToolBar(toolbar)

        self.act_add = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton),
                               "Add Images", self)
        self.act_add.setShortcut(QKeySequence("Ctrl+O"))
        toolbar.addAction(self.act_add)

        self.act_clear = QAction(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogResetButton),
                                 "Clear", self)
        toolbar.addAction(self.act_clear)
        toolbar.addSeparator()

        toolbar.addWidget(QLabel("Backend:"))
        self.cmb_backend = QComboBox()
        for name, key in BACKENDS:
            self.cmb_backend.addItem(name, key)
        toolbar.addWidget(self.cmb_backend)

        toolbar.addWidget(QLabel("Language:"))
        self.cmb_lang = QComboBox()
        self._populate_lang_list()
        toolbar.addWidget(self.cmb_lang)

        self.chk_angle = QCheckBox("Detect rotation (Paddle)")
        self.chk_angle.setChecked(True)
        toolbar.addWidget(self.chk_angle)

        self.chk_gpu = QCheckBox("GPU (Easy)")
        self.chk_gpu.setChecked(False)
        self.chk_gpu.setToolTip("Use CUDA for EasyOCR (needs a working torch+CUDA install).")
        toolbar.addWidget(self.chk_gpu)

        self.cmb_backend.currentIndexChanged.connect(self._on_backend_changed)

        toolbar.addSeparator()
        self.chk_auto_export = QCheckBox("Auto-export on completion")
        self.chk_auto_export.setToolTip(
            "When checked, every finished OCR run is automatically saved as\n"
            "TXT + JSON into the chosen folder."
        )
        toolbar.addWidget(self.chk_auto_export)

        self.btn_pick_folder = QPushButton("Folder…")
        self.btn_pick_folder.setToolTip("Choose the folder used for auto-export.")
        toolbar.addWidget(self.btn_pick_folder)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        self.lbl_summary = QLabel("0 images")
        toolbar.addWidget(self.lbl_summary)

        # Body ------------------------------------------------------------------
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)

        # Left: file list --------------------------------------------------------
        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(0, 0, 0, 0)
        left_l.setSpacing(8)

        left_header = QHBoxLayout()
        left_header.addWidget(QLabel("Images"))
        left_header.addStretch(1)

        self.btn_remove = QPushButton("Remove")
        self.btn_remove.setEnabled(False)
        left_header.addWidget(self.btn_remove)

        left_l.addLayout(left_header)

        self.list = DropListWidget()
        left_l.addWidget(self.list, 1)

        hint = QLabel("Tip: drag & drop image files here, or use Ctrl+O")
        hint.setStyleSheet("color:#6b7191;font-size:11px;")
        left_l.addWidget(hint)

        splitter.addWidget(left)
        splitter.setStretchFactor(0, 5)

        # Right: preview + tabs --------------------------------------------------
        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(0, 0, 0, 0)
        right_l.setSpacing(8)

        self.preview = ImagePreviewLabel()
        self.preview.clear_preview()
        right_l.addWidget(self.preview, 3)

        self.tabs = QTabWidget()
        self.txt_text = QTextEdit()
        self.txt_text.setReadOnly(True)
        self.txt_text.setPlaceholderText("Recognized text will appear here…")
        self.txt_json = QTextEdit()
        self.txt_json.setReadOnly(True)
        self.txt_json.setPlaceholderText("Structured JSON will appear here…")

        self.tabs.addTab(self.txt_text, "Text")
        self.tabs.addTab(self.txt_json, "JSON")
        right_l.addWidget(self.tabs, 4)

        splitter.addWidget(right)
        splitter.setStretchFactor(1, 7)
        splitter.setSizes([420, 820])

        # Model download/loading progress shared by all OCR backends.
        self.model_progress_panel = QWidget()
        model_progress_layout = QHBoxLayout(self.model_progress_panel)
        model_progress_layout.setContentsMargins(0, 0, 0, 0)
        model_progress_layout.setSpacing(8)

        self.lbl_model_progress = QLabel("Downloading model…")
        self.lbl_model_progress.setMinimumWidth(300)
        model_progress_layout.addWidget(self.lbl_model_progress)

        self.model_progress = QProgressBar()
        self.model_progress.setRange(0, 1000)
        self.model_progress.setValue(0)
        self.model_progress.setFormat("%p%")
        model_progress_layout.addWidget(self.model_progress, 1)
        self.model_progress_panel.setVisible(False)
        root.addWidget(self.model_progress_panel)

        # Bottom: OCR progress + actions ----------------------------------------
        bottom = QHBoxLayout()
        bottom.setSpacing(8)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("%v / %m  (%p%)")
        bottom.addWidget(self.progress, 1)

        self.btn_start = QPushButton("Start OCR")
        self.btn_start.setObjectName("primary")
        self.btn_start.setMinimumWidth(120)
        bottom.addWidget(self.btn_start)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setObjectName("danger")
        self.btn_cancel.setEnabled(False)
        bottom.addWidget(self.btn_cancel)

        self.btn_export_txt = QPushButton("Export TXT")
        self.btn_export_txt.setEnabled(False)
        bottom.addWidget(self.btn_export_txt)

        self.btn_export_json = QPushButton("Export JSON")
        self.btn_export_json.setEnabled(False)
        bottom.addWidget(self.btn_export_json)

        self.btn_export_all = QPushButton("Export All…")
        self.btn_export_all.setToolTip(
            "Pick a folder — saves a combined TXT and a JSON of every image at once."
        )
        self.btn_export_all.setEnabled(False)
        bottom.addWidget(self.btn_export_all)

        root.addLayout(bottom)

        # Status bar -------------------------------------------------------------
        self.setStatusBar(QStatusBar())
        self.status_lbl = QLabel("")
        self.statusBar().addPermanentWidget(self.status_lbl, 1)
        self.main_tabs.currentChanged.connect(self._on_main_tab_changed)
        self.main_tabs.setCurrentIndex(0)
        self.ocr_toolbar.setVisible(False)

    def _build_menu(self) -> None:
        m = self.menuBar()
        file_menu = m.addMenu("&File")

        file_menu.addAction(self.act_add)
        act_export_txt = QAction("Export as TXT…", self)
        act_export_txt.setShortcut(QKeySequence("Ctrl+E"))
        act_export_txt.triggered.connect(self.export_txt)
        file_menu.addAction(act_export_txt)

        act_export_json = QAction("Export as JSON…", self)
        act_export_json.setShortcut(QKeySequence("Ctrl+Shift+E"))
        act_export_json.triggered.connect(self.export_json)
        file_menu.addAction(act_export_json)

        file_menu.addSeparator()
        self.act_export_all = QAction("Export All (TXT + JSON)…", self)
        self.act_export_all.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self.act_export_all.triggered.connect(self.export_all)
        file_menu.addAction(self.act_export_all)

        self.act_pick_folder = QAction("Choose Auto-Export Folder…", self)
        self.act_pick_folder.triggered.connect(self.pick_auto_export_folder)
        file_menu.addAction(self.act_pick_folder)

        file_menu.addSeparator()
        act_quit = QAction("Quit", self)
        act_quit.setShortcut(QKeySequence("Ctrl+Q"))
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        help_menu = m.addMenu("&Help")
        act_about = QAction("About", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    def _wire(self) -> None:
        self.act_add.triggered.connect(self.add_files_dialog)
        self.act_clear.triggered.connect(self.clear_files)
        self.btn_remove.clicked.connect(self.remove_selected)
        self.list.files_dropped.connect(self.add_paths)
        self.list.itemSelectionChanged.connect(self._on_select)
        self.btn_start.clicked.connect(self.start_ocr)
        self.btn_cancel.clicked.connect(self.cancel_ocr)
        self.btn_export_txt.clicked.connect(self.export_txt)
        self.btn_export_json.clicked.connect(self.export_json)
        self.btn_export_all.clicked.connect(self.export_all)
        self.btn_pick_folder.clicked.connect(self.pick_auto_export_folder)
        self.chk_auto_export.toggled.connect(self._on_auto_export_toggled)

    # ----- helpers ----------------------------------------------------------
    def _on_main_tab_changed(self, index: int) -> None:
        ocr_active = index == 1
        self.ocr_toolbar.setVisible(ocr_active)
        ocr_running = bool(self.worker and self.worker.isRunning())
        self.act_add.setEnabled(ocr_active and not ocr_running)
        if not ocr_running:
            messages = {
                0: "Video → Images ready.",
                1: "OCR ready.",
                2: "Dialogue Tools ready.",
                3: "Overall Run ready.",
            }
            self._update_status(messages.get(index, "Ready."))

    def _set_running(self, running: bool) -> None:
        self.btn_start.setEnabled(not running and bool(self.paths))
        self.btn_cancel.setEnabled(running)
        self.btn_export_txt.setEnabled(not running and bool(self.results))
        self.btn_export_json.setEnabled(not running and bool(self.results))
        self.btn_export_all.setEnabled(not running and bool(self.results))
        self.act_add.setEnabled(
            not running and self.main_tabs.currentIndex() == 1
        )
        self.btn_remove.setEnabled(not running and bool(self.list.selectedItems()))
        self.list.setAcceptDrops(not running)
        self.cmb_backend.setEnabled(not running)
        self._refresh_backend_controls(running)

    def _update_status(self, text: str) -> None:
        self.status_lbl.setText(text)

    def _update_summary(self) -> None:
        n = len(self.paths)
        ok = sum(1 for r in self.results.values() if r.is_ok and len(r.lines) > 0)
        empty = sum(
            1 for r in self.results.values() if r.is_ok and len(r.lines) == 0
        )
        fail = sum(1 for r in self.results.values() if not r.is_ok)
        done = ok + empty + fail
        if n == 0:
            self.lbl_summary.setText("0 images")
        elif done == 0:
            self.lbl_summary.setText(f"{n} image{'s' if n != 1 else ''}")
        else:
            parts = [f"{ok} with text"]
            if empty:
                parts.append(f"{empty} empty")
            if fail:
                parts.append(f"{fail} failed")
            self.lbl_summary.setText(
                f"{n} image{'s' if n != 1 else ''}  •  " + ", ".join(parts)
            )

    def _show_about(self) -> None:
        QMessageBox.information(
            self,
            "About",
            "<h3>Vision Studio</h3>"
            "<p>A unified PyQt desktop app for extracting video frames and "
            "running batch OCR and dialogue cleanup.</p>"
            "<p>The first tab converts videos to JPG, PNG, or WebP images. "
            "The second tab provides PaddleOCR, EasyOCR, and MinerU2.5-Pro. "
            "The third tab cleans OCR dialogue and extracts readable transcripts "
            "from frame-by-frame OCR output. The fourth tab runs video extraction, "
            "OCR, and dialogue cleaning as one guided workflow.</p>"
            "<p style='color:#aab1c5;font-size:11px;'>"
            "Video decoding: OpenCV<br>"
            "PaddleOCR: paddleocr + paddlepaddle<br>"
            "EasyOCR: <a href='https://github.com/JaidedAI/EasyOCR'>github.com/JaidedAI/EasyOCR</a><br>"
            "MinerU2.5-Pro: <a href='https://huggingface.co/opendatalab/MinerU2.5-Pro-2604-1.2B'>"
            "huggingface.co/opendatalab/MinerU2.5-Pro-2604-1.2B</a>"
            "</p>",
        )

    # ----- file management --------------------------------------------------
    def add_files_dialog(self) -> None:
        filt = "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff)"
        files, _ = QFileDialog.getOpenFileNames(
            self, "Add images", "", filt
        )
        if files:
            self.add_paths(files)

    def add_paths(self, paths: list[str]) -> None:
        # 1) Filter the incoming list to supported, non-duplicate images.
        candidates: list[str] = []
        skipped_dup = 0
        skipped_bad = 0
        for p in paths:
            if not p:
                continue
            ext = os.path.splitext(p)[1].lower()
            if ext not in SUPPORTED_EXTS:
                skipped_bad += 1
                continue
            if p in self.paths:
                skipped_dup += 1
                continue
            candidates.append(p)

        if not candidates:
            msg_parts = []
            if skipped_dup:
                msg_parts.append(f"{skipped_dup} already in list")
            if skipped_bad:
                msg_parts.append(f"{skipped_bad} unsupported format")
            if msg_parts:
                self._update_status("No new images added (" + ", ".join(msg_parts) + ").")
            return

        # 2) Show a progress dialog while we add + thumbnail each image.
        #    Thumbnail decoding is the slow part for big batches (large JPEGs etc.),
        #    so we want visible feedback and the ability to bail out.
        progress = QProgressDialog(
            f"Adding {len(candidates)} image(s)…",
            "Cancel",
            0, len(candidates),
            self,
        )
        progress.setWindowTitle("Adding Images")
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setMinimumDuration(0)             # show immediately
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setValue(0)
        # Match the dark theme so it doesn't look out of place.
        progress.setStyleSheet(
            """
            QProgressDialog {
                background-color: #1b1f33;
                color: #e6e8ef;
                min-width: 380px;
            }
            QProgressDialog QLabel {
                color: #e6e8ef;
                font-size: 12px;
            }
            QProgressDialog QProgressBar {
                background: #1a1e30;
                border: 1px solid #2a2f42;
                border-radius: 8px;
                text-align: center;
                color: #e6e8ef;
                height: 18px;
            }
            QProgressDialog QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4f7cff, stop:1 #8aa9ff);
                border-radius: 8px;
            }
            QProgressDialog QPushButton {
                background-color: #2a3252;
                color: #f3f5fb;
                border: 1px solid #3a4470;
                border-radius: 8px;
                padding: 6px 14px;
            }
            QProgressDialog QPushButton:hover { background-color: #34406b; }
            """
        )

        added = 0
        cancelled = False
        for i, p in enumerate(candidates):
            if progress.wasCanceled():
                cancelled = True
                break
            self.paths.append(p)
            item = QListWidgetItem(self._make_icon(p), os.path.basename(p))
            item.setToolTip(p)
            item.setData(Qt.ItemDataRole.UserRole, p)
            self.list.addItem(item)
            added += 1

            done = i + 1
            total = len(candidates)
            pct = int(done * 100 / total)
            progress.setValue(done)
            progress.setLabelText(
                f"Adding image {done} of {total}  ({pct}%)\n{os.path.basename(p)}"
            )
            # Keep the dialog (and the rest of the UI) responsive while we churn.
            QApplication.processEvents()

        progress.close()

        if added:
            self._update_summary()
            self._set_running(self.worker is not None and self.worker.isRunning())
            self.btn_start.setEnabled(self.worker is None or not self.worker.isRunning())
            if cancelled:
                self._update_status(
                    f"Stopped: added {added} of {len(candidates)} image(s)."
                )
            else:
                self._update_status(f"Added {added} image(s).")
        elif cancelled:
            self._update_status("Add cancelled — no images added.")

    def clear_files(self) -> None:
        if self.worker and self.worker.isRunning():
            return
        self.paths.clear()
        self.results.clear()
        self._icon_cache.clear()
        self.list.clear()
        self.txt_text.clear()
        self.txt_json.clear()
        self.preview.clear_preview()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self._update_summary()
        self._set_running(False)
        self._update_status("Ready.")

    def remove_selected(self) -> None:
        if self.worker and self.worker.isRunning():
            return
        rows = sorted({i.row() for i in self.list.selectedIndexes()}, reverse=True)
        if not rows:
            return
        for r in rows:
            item = self.list.takeItem(r)
            if item is None:
                continue
            try:
                removed_path = self.paths.pop(r)
                self._icon_cache.pop(removed_path, None)
            except IndexError:
                pass
            self.results.pop(r, None)
        # re-pack results: keep keys stable by clearing and re-indexing later
        self._repack_results()
        self._on_select()
        self._update_summary()
        self._set_running(False)
        self.btn_start.setEnabled(bool(self.paths))

    def _repack_results(self) -> None:
        """After deletion, remap remaining results to current row indexes."""
        new_results: dict[int, OCRResult] = {}
        for new_row in range(self.list.count()):
            path = self.list.item(new_row).data(Qt.ItemDataRole.UserRole)
            for r in self.results.values():
                if r.path == path:
                    new_results[new_row] = r
                    break
        self.results = new_results

    def _make_icon(self, path: str, badge: Optional[str] = None) -> QIcon:
        """Build a thumbnail QIcon, optionally with a status badge.

        badge:
            None  - no badge
            "x"   - empty result (red circle with white X)   → no text found
            "!"   - failed OCR  (orange circle with white !)
            "ok"  - successful OCR (green check)
        """
        base = self._icon_cache.get(path)
        if base is None:
            try:
                with Image.open(path) as im:
                    im.thumbnail((64, 64))
                    if im.mode != "RGB":
                        im = im.convert("RGB")
                    data = im.tobytes("raw", "RGB")
                    qimg = QImage(
                        data, im.width, im.height, im.width * 3,
                        QImage.Format.Format_RGB888,
                    ).copy()
                    base = QPixmap.fromImage(qimg)
            except Exception:
                base = self.style().standardIcon(
                    QStyle.StandardPixmap.SP_FileIcon
                ).pixmap(64, 64)
            self._icon_cache[path] = base

        if not badge:
            return QIcon(base)

        canvas = base.copy()
        painter = QPainter(canvas)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            size = 18
            rect = QRectF(1.5, 1.5, size, size)

            if badge == "x":
                bg, fg, glyph = QColor("#c0394a"), QColor("#ffffff"), "x"
            elif badge == "!":
                bg, fg, glyph = QColor("#e67e22"), QColor("#ffffff"), "!"
            elif badge == "ok":
                bg, fg, glyph = QColor("#27ae60"), QColor("#ffffff"), "v"
            else:
                painter.end()
                return QIcon(canvas)

            # Filled circle background
            painter.setBrush(bg)
            painter.setPen(QPen(QColor("#ffffff"), 1))
            painter.drawEllipse(rect)

            # Glyph
            font = painter.font()
            font.setBold(True)
            font.setPointSize(11)
            painter.setFont(font)
            painter.setPen(fg)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, glyph)
        finally:
            painter.end()
        return QIcon(canvas)

    # ----- preview ----------------------------------------------------------
    def _on_select(self) -> None:
        self._set_running(self.worker is not None and self.worker.isRunning())
        items = self.list.selectedItems()
        if not items:
            self.preview.clear_preview()
            self.txt_text.clear()
            self.txt_json.clear()
            return
        row = self.list.row(items[0])
        path = items[0].data(Qt.ItemDataRole.UserRole)
        self._show_image(path)
        res = self.results.get(row)
        if res is None:
            self.txt_text.setPlainText("(not processed yet)")
            self.txt_json.setPlainText("(not processed yet)")
        elif not res.is_ok:
            self.txt_text.setPlainText(f"Error: {res.error}")
            self.txt_json.setPlainText(json.dumps(res.to_dict(), indent=2, ensure_ascii=False))
        else:
            self.txt_text.setPlainText(res.full_text or "(no text detected)")
            self.txt_json.setPlainText(json.dumps(res.to_dict(), indent=2, ensure_ascii=False))

    def _show_image(self, path: str) -> None:
        pix = QPixmap(path)
        if pix.isNull():
            self.preview.setText(f"Cannot load: {path}")
            return
        self.preview.set_pixmap(pix)

    # ----- OCR --------------------------------------------------------------
    def _check_paddle(self) -> bool:
        if PADDLE_OK:
            return True
        QMessageBox.critical(
            self,
            "PaddleOCR not available",
            "Could not import paddleocr. Install it with:\n\n"
            "    pip install paddleocr paddlepaddle\n\n"
            f"Underlying error:\n{_PADDLE_IMPORT_ERROR}",
        )
        return False

    def _check_easy(self) -> bool:
        if EASY_OK:
            return True
        QMessageBox.critical(
            self,
            "EasyOCR not available",
            "Could not import easyocr. Install it with:\n\n"
            "    pip install easyocr\n\n"
            f"Underlying error:\n{_EASY_IMPORT_ERROR}",
        )
        return False

    def _check_mineru(self) -> bool:
        try:
            import importlib.util

            missing = [
                package
                for package in ("mineru_vl_utils", "transformers")
                if importlib.util.find_spec(package) is None
            ]
        except Exception as exc:
            missing = [str(exc)]
        if not missing:
            return True
        QMessageBox.critical(
            self,
            "MinerU2.5-Pro not available",
            "MinerU2.5-Pro needs its optional local inference packages.\n\n"
            'Install them with:\n\n    pip install -U "mineru-vl-utils[transformers]"\n\n'
            f"Missing: {', '.join(missing)}",
        )
        return False

    def _populate_lang_list(self) -> None:
        """Fill the language combo with codes for the currently selected backend."""
        backend = self.cmb_backend.currentData() if hasattr(self, "cmb_backend") else "paddle"
        if backend == "paddle":
            langs = LANGUAGES_PADDLE
        elif backend == "easy":
            langs = LANGUAGES_EASY
        else:
            langs = LANGUAGES_MINERU
        self.cmb_lang.blockSignals(True)
        self.cmb_lang.clear()
        for name, code in langs:
            self.cmb_lang.addItem(name, code)
        self.cmb_lang.blockSignals(False)

    def _refresh_backend_controls(self, running: bool = False) -> None:
        backend = self.cmb_backend.currentData()
        self.cmb_lang.setEnabled(not running and backend != "mineru")
        self.chk_angle.setEnabled(not running and backend == "paddle")
        self.chk_gpu.setEnabled(not running and backend == "easy")

    def _on_backend_changed(self, _idx: int) -> None:
        self._populate_lang_list()
        backend = self.cmb_backend.currentData()
        self._refresh_backend_controls()
        label = BACKEND_LABELS.get(backend, str(backend))
        if backend == "mineru":
            self._update_status(
                "Backend: MinerU2.5-Pro. Multilingual layout model; first run "
                "downloads the model."
            )
        else:
            self._update_status(f"Backend: {label}. Models may download on first run.")

    def start_ocr(self) -> None:
        if self.worker and self.worker.isRunning():
            return
        if not self.paths:
            return
        backend = self.cmb_backend.currentData()
        if backend == "paddle":
            if not self._check_paddle():
                return
        elif backend == "easy":
            if not self._check_easy():
                return
        elif not self._check_mineru():
            return

        # Reset progress
        self._job_cancelled = False
        self.model_progress_panel.setVisible(False)
        self.results.clear()
        self.txt_text.clear()
        self.txt_json.clear()
        self.progress.setRange(0, len(self.paths))
        self.progress.setValue(0)

        lang = self.cmb_lang.currentData()

        if backend == "paddle":
            self.worker = OCRWorker(
                self.paths,
                lang=lang,
                use_angle_cls=self.chk_angle.isChecked(),
            )
        elif backend == "easy":
            self.worker = EasyOCRWorker(
                self.paths,
                lang=lang,
                gpu=self.chk_gpu.isChecked(),
            )
        else:
            self.worker = MinerUWorker(self.paths)

        self.worker.progress.connect(self._on_progress)
        self.worker.file_done.connect(self._on_file_done)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.download_progress.connect(self._on_model_download_progress)
        self.worker.start()

        self._set_running(True)
        backend_label = BACKEND_LABELS.get(backend, str(backend))
        suffix = "" if backend == "mineru" else f" ({lang})"
        self._update_status(f"Running {backend_label}{suffix}…")

    def cancel_ocr(self) -> None:
        if self.worker and self.worker.isRunning():
            self._job_cancelled = True
            self.worker.cancel()
            self._update_status("Cancelling…")

    @staticmethod
    def _human_size(value: float) -> str:
        size = float(value)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if abs(size) < 1024.0 or unit == "TB":
                return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"

    def _on_model_download_progress(
        self,
        current,
        total,
        stage: str,
        model_label: str,
    ) -> None:
        self.model_progress_panel.setVisible(True)

        if stage in {"checking", "loading"}:
            self.model_progress.setRange(0, 0)
            self.model_progress.setFormat("")
            label = (
                f"Checking {model_label} model files…"
                if stage == "checking"
                else f"Loading {model_label} model into memory…"
            )
            self.lbl_model_progress.setText(label)
            self._update_status(label)
            return

        total_value = float(total or 0)
        current_value = min(float(current or 0), total_value) if total_value else 0.0
        self.model_progress.setRange(0, 1000)
        if total_value > 0:
            fraction = max(0.0, min(1.0, current_value / total_value))
            self.model_progress.setValue(round(fraction * 1000))
        else:
            self.model_progress.setValue(0)
        self.model_progress.setFormat("%p%")

        if stage == "bytes":
            detail = (
                f"{self._human_size(current_value)} / "
                f"{self._human_size(total_value)}"
            )
            label = f"Downloading {model_label} model — {detail}"
        elif stage == "files":
            label = (
                f"Downloading {model_label} model files — "
                f"{int(current_value)} / {int(total_value)}"
            )
        else:
            label = f"{model_label} model download complete."
        self.lbl_model_progress.setText(label)
        self._update_status(label)

    def _on_progress(self, index: int, total: int, current: str) -> None:
        self.model_progress_panel.setVisible(False)
        self.progress.setRange(0, total)
        self.progress.setValue(index)
        if current:
            self._update_status(f"Processing {index + 1}/{total}: {current}")

    def _on_file_done(self, index: int, result: OCRResult) -> None:
        self.results[index] = result
        # Refresh the list icon & tooltip for status
        item = self.list.item(index)
        if item is not None:
            if result.is_ok:
                tip = (
                    f"{result.path}\n"
                    f"{len(result.lines)} line(s) • {result.elapsed_sec:.2f}s"
                )
            else:
                tip = f"{result.path}\nError: {result.error}"
            item.setToolTip(tip)

            # Pick a badge + text color based on outcome
            if not result.is_ok:
                badge = "!"
                item.setForeground(QColor("#ff7a8a"))
                tip += "\n(OCR failed)"
            elif len(result.lines) == 0:
                badge = "x"
                item.setForeground(QColor("#ffb380"))
                tip += "\n(no text detected)"
            else:
                badge = "ok"
                item.setForeground(QColor("#e6e8ef"))
            item.setToolTip(tip)

            # Rebuild icon with the badge (cached base pixmap, fast)
            item.setIcon(self._make_icon(result.path, badge=badge))
        self._update_summary()
        # Auto-select first finished
        if self.list.currentRow() == -1 and self.list.count() > 0:
            self.list.setCurrentRow(0)

    def _on_finished(self) -> None:
        self.model_progress_panel.setVisible(False)
        ok = sum(1 for r in self.results.values() if r.is_ok)
        fail = sum(1 for r in self.results.values() if not r.is_ok)
        if self._job_cancelled:
            self._update_status("Cancelled.")
        else:
            self._update_status(f"Done. {ok} ok, {fail} failed.")
        self._set_running(False)
        self.btn_start.setEnabled(bool(self.paths))
        # Auto-render first result into preview/tabs
        if self.results:
            first = sorted(self.results.keys())[0]
            self.list.setCurrentRow(first)
        # Save to disk automatically if the toggle is on
        if not self._job_cancelled:
            self._auto_export_if_enabled()

    def _on_failed(self, msg: str) -> None:
        self.model_progress_panel.setVisible(False)
        self._update_status("Worker failed.")
        self._set_running(False)
        QMessageBox.critical(self, "OCR failed", msg)

    # ----- export -----------------------------------------------------------
    def export_txt(self) -> None:
        if not self.results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export TXT", "ocr_results.txt", "Text files (*.txt)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# Vision Studio — OCR results ({len(self.results)} files)\n")
                f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                for i in sorted(self.results):
                    r = self.results[i]
                    f.write("=" * 78 + "\n")
                    f.write(f"[{i + 1}/{len(self.results)}] {r.name}\n")
                    f.write(f"Path:    {r.path}\n")
                    f.write(f"Elapsed: {r.elapsed_sec:.3f}s\n")
                    if not r.is_ok:
                        f.write(f"Error:   {r.error}\n")
                        f.write("\n")
                        continue
                    f.write(f"Lines:   {len(r.lines)}\n")
                    f.write("-" * 78 + "\n")
                    for ln in r.lines:
                        f.write(f"{ln.text}\n")
                    f.write("\n")
            self._update_status(f"Exported TXT → {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    def export_json(self) -> None:
        if not self.results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export JSON", "ocr_results.json", "JSON files (*.json)"
        )
        if not path:
            return
        try:
            payload = {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "app": "Vision Studio",
                "count": len(self.results),
                "results": [self.results[i].to_dict() for i in sorted(self.results)],
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            self._update_status(f"Exported JSON → {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    # ----- bulk export (TXT + JSON in one folder) -------------------------
    def _ts(self) -> str:
        """Filesystem-safe timestamp for filenames."""
        return time.strftime("%Y%m%d_%H%M%S")

    def _filter_results_for_export(self) -> tuple[dict[int, "OCRResult"], int]:
        """Drop successful-but-empty results from the export set.

        Errors are kept (so the user still sees them); only results that
        completed cleanly but found zero text are skipped.
        """
        kept: dict[int, OCRResult] = {}
        skipped = 0
        for i, r in self.results.items():
            if r.is_ok and len(r.lines) == 0:
                skipped += 1
                continue
            kept[i] = r
        return kept, skipped

    def _write_combined_txt(self, path: str) -> int:
        """Compact, human-readable dump of every image's text in one file.
        Returns the number of empty-result images that were skipped.
        """
        kept, skipped = self._filter_results_for_export()
        total = len(kept)
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Vision Studio — combined OCR results\n")
            f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(
                f"# Images with text: {total}"
                + (f"  ({skipped} empty skipped)" if skipped else "")
                + "\n\n"
            )
            for n, i in enumerate(sorted(kept), start=1):
                r = kept[i]
                f.write(f"--- [{n}/{total}] {r.name} ---\n")
                if not r.is_ok:
                    f.write(f"[error] {r.error}\n\n")
                    continue
                for ln in r.lines:
                    f.write(f"{ln.text}\n")
                f.write("\n")
        return skipped

    def _write_combined_json(self, path: str) -> int:
        """Structured dump of every image's lines + boxes + scores.
        Returns the number of empty-result images that were skipped.
        """
        kept, skipped = self._filter_results_for_export()
        payload = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "app": "Vision Studio",
            "count": len(kept),
            "skipped_empty": skipped,
            "results": [kept[i].to_dict() for i in sorted(kept)],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return skipped

    def export_all(self, folder: Optional[str] = None) -> None:
        """Pick a folder, then save both combined TXT + JSON in one shot.
        Empty (no-text) images are skipped by default."""
        if not self.results:
            return
        if not folder:
            start = self._auto_export_folder or str(Path.home())
            folder = QFileDialog.getExistingDirectory(
                self, "Choose folder for combined TXT + JSON", start
            )
            if not folder:
                return
        try:
            folder_path = Path(folder)
            folder_path.mkdir(parents=True, exist_ok=True)
            ts = self._ts()
            txt_path = folder_path / f"ocr_results_{ts}.txt"
            json_path = folder_path / f"ocr_results_{ts}.json"
            skipped_txt = self._write_combined_txt(str(txt_path))
            skipped_json = self._write_combined_json(str(json_path))
            skipped = max(skipped_txt, skipped_json)
            suffix = f" (skipped {skipped} empty image{'s' if skipped != 1 else ''})" if skipped else ""
            self._update_status(
                f"Exported all \u2192 {txt_path.name} + {json_path.name} in {folder_path}{suffix}"
            )
            # remember last folder for next time
            self._auto_export_folder = str(folder_path)
            self._settings.setValue("auto_export_folder", self._auto_export_folder)
            self._refresh_folder_label()
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    # ----- auto-export on completion --------------------------------------
    def _refresh_folder_label(self) -> None:
        if self._auto_export_folder:
            short = self._auto_export_folder
            if len(short) > 60:
                short = "\u2026" + short[-57:]
            self.btn_pick_folder.setText(f"Folder: {short}")
            self.btn_pick_folder.setToolTip(self._auto_export_folder)
        else:
            self.btn_pick_folder.setText("Folder…")
            self.btn_pick_folder.setToolTip("Choose the folder used for auto-export.")

    def pick_auto_export_folder(self) -> None:
        start = self._auto_export_folder or str(Path.home())
        folder = QFileDialog.getExistingDirectory(
            self, "Choose auto-export folder", start
        )
        if not folder:
            return
        self._auto_export_folder = folder
        self._settings.setValue("auto_export_folder", folder)
        self._refresh_folder_label()
        self._update_status(f"Auto-export folder set \u2192 {folder}")

    def _on_auto_export_toggled(self, checked: bool) -> None:
        self._settings.setValue("auto_export_enabled", checked)
        if checked and not self._auto_export_folder:
            # prompt right away so the user doesn't forget
            self.pick_auto_export_folder()

    def _auto_export_if_enabled(self) -> None:
        if not self.chk_auto_export.isChecked():
            return
        if not self.results:
            return
        if not self._auto_export_folder:
            self.pick_auto_export_folder()
            if not self._auto_export_folder:
                return
        # silent save (no dialog), still surfaces errors
        try:
            self.export_all(folder=self._auto_export_folder)
        except Exception as exc:
            QMessageBox.critical(self, "Auto-export failed", str(exc))

    # ----- shutdown ---------------------------------------------------------
    def closeEvent(self, event):  # noqa: D401
        ocr_running = bool(self.worker and self.worker.isRunning())
        video_running = self.video_tab.is_running()
        dialogue_running = self.dialogue_tab.is_running()
        overall_running = self.overall_tab.is_running()
        if ocr_running or video_running or dialogue_running or overall_running:
            ans = QMessageBox.question(
                self,
                "Work in progress",
                "A video, OCR, dialogue, or Overall Run job is still running. "
                "Cancel active work and quit?",
            )
            if ans != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            if ocr_running:
                self.worker.cancel()
                self.worker.wait(3000)
            if video_running:
                self.video_tab.cancel_and_wait(3000)
            if dialogue_running and not self.dialogue_tab.wait_for_finish():
                QMessageBox.warning(
                    self,
                    "Still processing",
                    "Dialogue processing is still finishing. Please try closing "
                    "again in a moment.",
                )
                event.ignore()
                return
            if overall_running and not self.overall_tab.cancel_and_wait():
                QMessageBox.warning(
                    self,
                    "Still processing",
                    "Overall Run is still finishing safely. Please try closing "
                    "again in a moment.",
                )
                event.ignore()
                return
        event.accept()


# ============================================================================
# Entry point
# ============================================================================

def main() -> int:
    if not CV2_OK:
        print(f"[warn] opencv import failed: {_CV2_IMPORT_ERROR}",
              file=sys.stderr)
    if not PADDLE_OK:
        print(f"[warn] paddleocr import failed: {_PADDLE_IMPORT_ERROR}",
              file=sys.stderr)
    if not EASY_OK:
        print(f"[warn] easyocr import failed: {_EASY_IMPORT_ERROR}",
              file=sys.stderr)

    app = QApplication(sys.argv)
    app.setApplicationName("Vision Studio")
    app.setOrganizationName("Vision Studio")

    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
