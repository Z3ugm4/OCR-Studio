# Vision Studio

A unified, dark-themed PyQt desktop app with four top-level tabs:

- **Video → Images** extracts frames from video files.
- **OCR** processes image batches with PaddleOCR, EasyOCR, or MinerU2.5-Pro.
- **Dialogue Tools** cleans OCR dialogue and turns frame-by-frame OCR dumps into
  readable transcripts.
- **Overall Run** chains video extraction, OCR, and dialogue cleaning into one
  guided workflow.

![preview](docs/preview.png)

---

## Features

### Video → Images

- Five extraction modes: total count, 1 image/second, all frames,
  4 images/second, and custom fps.
- Live resolution, fps, duration, and frame-count metadata.
- Output estimates before extraction.
- JPG, PNG, and WebP output with quality controls.
- Background extraction with progress and cancellation.
- Stable output names such as `clip_000012.jpg`.

### OCR

- Drag & drop or pick multiple images (`png / jpg / jpeg / bmp / webp / tif / tiff`).
- Threaded PaddleOCR worker — UI stays responsive, **Cancel** supported.
- Three selectable OCR backends:
  - **PaddleOCR** for fast general-purpose OCR.
  - **EasyOCR** as a lightweight optional alternative.
  - **MinerU2.5-Pro** for multilingual, layout-aware document parsing,
    including text, titles, tables, equations, and code blocks.
- Model download/loading bar for PaddleOCR, EasyOCR, and MinerU, with percentage,
  downloaded size, and cancellation whenever the backend exposes byte totals.
- 16 language models out of the box (English, Chinese, Japanese, Korean,
  French, German, Arabic, Hindi, …).
- Optional rotation / angle classifier.
- Per-file preview, full text view, and JSON view with bounding boxes +
  confidence scores.
- Export all results to **TXT** or **JSON** in one click.
- **Export All…** — pick a folder, get both `ocr_results_<timestamp>.txt` and
  `ocr_results_<timestamp>.json` written at the same time.
- **Auto-export on completion** — flip the checkbox, pick a folder once, and
  every finished OCR run is saved automatically. The choice is remembered.
- Modern Fusion dark theme via pure QSS — no external style files needed.

### Dialogue Tools

- Clean regular OCR text by removing exact and fuzzy repetition.
- Keep more complete readings and join wrapped or hyphenated dialogue.
- Normalize dialogue written as `SPEAKER: text` or a speaker on its own line.
- Flag possibly incomplete text without inventing missing words.
- Consolidate progressive typewriter-style readings from frame-by-frame OCR.
- Filter transcript entries by confidence and optionally include narration,
  source frame ranges, and custom speaker aliases.
- Preview the finished result and optionally save a detailed JSON review report.

### Overall Run

- Runs the full sequence automatically:
  `video → temp_img_ocr → OCR → cleaned text`.
- Collects the extraction, backend, language, and cleaner settings in one place.
- Shows a mandatory settings warning and complete summary immediately before
  execution.
- Displays extraction progress, per-image OCR progress, and OCR model-download
  progress.
- Supports safe cancellation between and during processing stages.
- Saves temporary frames to `temp_img_ocr` inside the selected output folder.
- Saves `<video>_ocr_raw.txt`, `<video>_cleaned.txt`, and a JSON cleaning report
  in the selected output folder.

---

## Install

```bash
# 1. Create a fresh env (recommended)
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 2. Install the shared UI, image, and video dependencies
pip install PyQt6 Pillow numpy opencv-python

# 3. Install PaddlePaddle (CPU build is enough for most cases)
#    Pick the wheel for your platform from:
#    https://www.paddlepaddle.org.cn/install/quick
pip install paddlepaddle

# 4. Install PaddleOCR
pip install paddleocr
```

> On first run PaddleOCR will auto-download the detection / recognition
> models for the language you pick. They live in
> `~/.paddleocr/`, so this only happens once.

### Optional backends

EasyOCR:

```bash
pip install easyocr
```

MinerU2.5-Pro (local Transformers backend):

```bash
pip install -U "mineru-vl-utils[transformers]"
```

Select **MinerU2.5-Pro** in the Backend menu. Language detection is automatic.
The first run downloads
[`opendatalab/MinerU2.5-Pro-2604-1.2B`](https://huggingface.co/opendatalab/MinerU2.5-Pro-2604-1.2B)
from Hugging Face. The app uses the simple Transformers backend, which runs on
the device selected automatically by PyTorch/Accelerate; a CUDA GPU is strongly
recommended for practical speed.

## Run

```bash
python paddle_ocr_studio.py
```

---

## OCR output formats

### TXT

Plain, copy-pasteable text. Section per file, with metadata:

```
==============================================================================
[1/2] receipt.png
Path:    /tmp/receipt.png
Elapsed: 1.234s
Lines:   12
------------------------------------------------------------------------------
COFFEE SHOP
2x Latte     8.50
1x Croissant 3.20
...
```

### JSON

Full structured dump with bounding boxes and confidence:

```json
{
  "generated_at": "2026-07-03T12:34:56",
  "app": "Vision Studio",
  "count": 2,
  "results": [
    {
      "path": "/tmp/receipt.png",
      "name": "receipt.png",
      "elapsed_sec": 1.234,
      "line_count": 12,
      "lines": [
        {
          "text": "COFFEE SHOP",
          "confidence": 0.9876,
          "box": [[12.0, 30.0], [220.0, 30.0], [220.0, 60.0], [12.0, 60.0]],
          "type": "text"
        }
      ],
      "error": null
    }
  ]
}
```

---

## Project layout

```
paddle_ocr_studio.py     # the entire application (single file)
ocr_dialogue_cleaner.py  # dialogue cleanup engine
video_ocr_dialogue_extractor.py  # frame transcript engine
requirements.txt
README.md
```

The PyQt tabbed UI and its workers live in `paddle_ocr_studio.py`. The dialogue
algorithms remain in focused standard-library modules so they can also be used
from the command line.

---

## License

MIT — do whatever you want, no warranty.
