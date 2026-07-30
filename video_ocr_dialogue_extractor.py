#!/usr/bin/env python3
"""Extract a readable transcript from frame-by-frame PaddleOCR Studio output.

This program is designed for OCR dumps where the same typewriter-style game
dialogue appears in many consecutive screenshots, gradually becoming complete.
It keeps the best stable reading from each temporal sequence.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path


FRAME_HEADER = re.compile(
    r"^--- \[(?P<number>\d+)/(?P<total>\d+)\] (?P<filename>.+?) ---\s*$"
)
WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:['’\-][A-Za-zÀ-ÖØ-öø-ÿ]+)*")
END_PUNCTUATION = re.compile(r'[.!?…]["”’)\]]*$')

DEFAULT_SPEAKER_ALIASES = {
    "Mei": "Mei",
    "RaidenMei": "Raiden Mei",
    "Raiden Mei": "Raiden Mei",
    "Carole": "Carole",
    "Theresa": "Theresa",
    "Welt": "Welt",
    "Lewis": "Lewis",
    "Joyce": "Joyce",
    "Mysterious Whisper": "Mysterious Whisper",
    "Unidentified Comms": "Unidentified Comms",
}

UI_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^(?:field comms|raise rating|objective|skip|confirm|return)$",
        r"^(?:audio|files|photos|album|database|collection)(?:\s+\d+/\d+)?[.:]?$",
        r"^(?:dialogue event|sound recording \d+|a post-honkai odyssey tips)$",
        r"^(?:light|heavy|shield|rating raised|final sale)$",
        r"^(?:enter your name|this is how you will be addressed).*$",
    )
]
UI_PATTERNS.append(re.compile(r"^[A-Z0-9 ._'’\-]{2,}$"))


@dataclass
class Frame:
    number: int
    total: int
    filename: str
    lines: list[str]


@dataclass
class Reading:
    frame: int
    speaker: str | None
    text: str


@dataclass
class TranscriptEntry:
    speaker: str | None
    text: str
    first_frame: int
    last_frame: int
    observations: int
    stable_observations: int
    confidence: str
    source_file: str


@dataclass
class ExtractionReport:
    input_frames: int = 0
    frames_with_readings: int = 0
    transcript_entries: int = 0
    named_dialogue_entries: int = 0
    narration_entries: int = 0
    duplicate_entries_removed: int = 0
    partial_entries_removed: int = 0
    low_confidence_entries: int = 0
    entries_excluded_by_confidence: int = 0
    speakers: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def normalize_line(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "").replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r" +([,.;:!?])", r"\1", text)
    return text


def comparison_key(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n\"'.,;:!?-–—")


def parse_frames(text: str) -> list[Frame]:
    frames: list[Frame] = []
    current: Frame | None = None
    for raw in text.splitlines():
        match = FRAME_HEADER.match(raw)
        if match:
            if current is not None:
                frames.append(current)
            current = Frame(
                number=int(match.group("number")),
                total=int(match.group("total")),
                filename=match.group("filename"),
                lines=[],
            )
            continue
        if current is not None:
            line = normalize_line(raw)
            if line:
                current.lines.append(line)
    if current is not None:
        frames.append(current)
    return frames


def join_wrapped(lines: list[str]) -> str:
    result = ""
    for line in lines:
        if not result:
            result = line
        elif result.endswith("-") and line[:1].islower():
            result = result[:-1] + line
        else:
            result += " " + line
    return re.sub(r"\s+", " ", result).strip()


def is_ui_line(line: str) -> bool:
    return any(pattern.match(line) for pattern in UI_PATTERNS)


def useful_prose(line: str) -> bool:
    words = WORD.findall(line)
    lowercase = sum(ch.islower() for ch in line)
    return len(words) >= 2 and lowercase >= 2 and not is_ui_line(line)


def discover_global_noise(frames: list[Frame], speaker_aliases: dict[str, str]) -> set[str]:
    counts = Counter(
        comparison_key(line)
        for frame in frames
        for line in frame.lines
        if line not in speaker_aliases and comparison_key(line)
    )
    # Dialogue is normally held for fewer than 30 sampled frames. Text repeated
    # much more often is overwhelmingly a HUD label, location sign, or watermark.
    return {key for key, count in counts.items() if count >= 45}


def frame_reading(
    frame: Frame,
    speaker_aliases: dict[str, str],
    global_noise: set[str],
    include_narration: bool,
) -> Reading | None:
    speaker_index: int | None = None
    speaker: str | None = None
    for index, line in enumerate(frame.lines):
        if line in speaker_aliases:
            speaker_index = index
            speaker = speaker_aliases[line]

    if speaker_index is not None:
        content = [
            line
            for line in frame.lines[speaker_index + 1 :]
            if comparison_key(line) not in global_noise and not is_ui_line(line)
        ]
        text = join_wrapped(content)
        if text and WORD.search(text):
            return Reading(frame.number, speaker, text)
        return None

    if not include_narration:
        return None

    candidates = [
        line
        for line in frame.lines
        if useful_prose(line) and comparison_key(line) not in global_noise
    ]
    if not candidates:
        return None
    text = join_wrapped(candidates)
    if len(WORD.findall(text)) < 3:
        return None
    return Reading(frame.number, None, text)


def progression_similarity(left: str, right: str) -> float:
    a, b = comparison_key(left), comparison_key(right)
    if not a or not b:
        return 0.0
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) >= 3 and longer.startswith(shorter):
        return 1.0
    ratio = SequenceMatcher(None, a, b, autojunk=False).ratio()
    opening = min(14, len(shorter))
    if opening >= 5 and a[:opening] != b[:opening]:
        ratio -= 0.18
    return ratio


def _reading_quality(text: str, frequency: int) -> float:
    suspicious = len(re.findall(r"[^A-Za-zÀ-ÖØ-öø-ÿ0-9\s.,!?…'’\"“”():;\-—/&]", text))
    punctuation_bonus = 12 if END_PUNCTUATION.search(text) else 0
    return frequency * 20 + min(len(text), 220) * 0.18 + punctuation_bonus - suspicious * 4


def choose_best_reading(
    readings: list[Reading], frames_by_number: dict[int, Frame]
) -> TranscriptEntry:
    variants = Counter(comparison_key(item.text) for item in readings)
    best = max(
        readings,
        key=lambda item: _reading_quality(
            item.text, variants[comparison_key(item.text)]
        ),
    )
    stable = variants[comparison_key(best.text)]
    observations = len(readings)
    if stable >= 3 and END_PUNCTUATION.search(best.text):
        confidence = "high"
    elif stable >= 2 or END_PUNCTUATION.search(best.text):
        confidence = "medium"
    else:
        confidence = "low"
    frame = frames_by_number[best.frame]
    return TranscriptEntry(
        speaker=best.speaker,
        text=best.text,
        first_frame=readings[0].frame,
        last_frame=readings[-1].frame,
        observations=observations,
        stable_observations=stable,
        confidence=confidence,
        source_file=frame.filename,
    )


def consolidate_readings(
    readings: list[Reading], frames: list[Frame], report: ExtractionReport
) -> list[TranscriptEntry]:
    frames_by_number = {frame.number: frame for frame in frames}
    groups: list[list[Reading]] = []
    current: list[Reading] = []

    for reading in readings:
        if not current:
            current = [reading]
            continue
        previous = current[-1]
        same_speaker = previous.speaker == reading.speaker
        close_in_time = reading.frame - previous.frame <= 5
        similarity = progression_similarity(previous.text, reading.text)
        if same_speaker and close_in_time and similarity >= 0.66:
            current.append(reading)
        else:
            groups.append(current)
            current = [reading]
    if current:
        groups.append(current)

    entries = [choose_best_reading(group, frames_by_number) for group in groups]
    deduped: list[TranscriptEntry] = []
    for entry in entries:
        if deduped:
            previous = deduped[-1]
            similarity = progression_similarity(previous.text, entry.text)
            nearby = entry.first_frame - previous.last_frame <= 12
            compatible_speaker = (
                entry.speaker == previous.speaker
                or entry.speaker is None
                or previous.speaker is None
            )
            if nearby and compatible_speaker and similarity >= 0.92:
                report.duplicate_entries_removed += 1
                # Prefer a named reading, then the more stable/complete reading.
                old_score = (
                    (1 if previous.speaker else 0),
                    previous.stable_observations,
                    len(previous.text),
                )
                new_score = (
                    (1 if entry.speaker else 0),
                    entry.stable_observations,
                    len(entry.text),
                )
                if new_score > old_score:
                    deduped[-1] = entry
                continue
        deduped.append(entry)
    def looks_superseded(partial: TranscriptEntry, complete: TranscriptEntry) -> bool:
        if (
            partial.confidence != "low"
            or partial.speaker != complete.speaker
            or complete.first_frame - partial.last_frame > 3
            or len(complete.text) <= len(partial.text)
        ):
            return False
        old, new = comparison_key(partial.text), comparison_key(complete.text)
        if len(old) <= 12:
            return True
        common = 0
        for left_char, right_char in zip(old, new):
            if left_char != right_char:
                break
            common += 1
        if common >= 8:
            return True
        old_words, new_words = WORD.findall(old), WORD.findall(new)
        if len(old_words) >= 2 and new_words[:2] == old_words[:2]:
            return True
        return False

    polished: list[TranscriptEntry] = []
    for index, entry in enumerate(deduped):
        if index + 1 < len(deduped) and looks_superseded(entry, deduped[index + 1]):
            report.partial_entries_removed += 1
            continue
        polished.append(entry)
    return polished


def extract_transcript(
    text: str,
    speaker_aliases: dict[str, str] | None = None,
    include_narration: bool = True,
) -> tuple[list[TranscriptEntry], ExtractionReport]:
    aliases = dict(DEFAULT_SPEAKER_ALIASES)
    if speaker_aliases:
        aliases.update(speaker_aliases)
    frames = parse_frames(text)
    report = ExtractionReport(input_frames=len(frames))
    noise = discover_global_noise(frames, aliases)
    readings = [
        reading
        for frame in frames
        if (
            reading := frame_reading(
                frame, aliases, noise, include_narration=include_narration
            )
        )
    ]
    report.frames_with_readings = len(readings)
    entries = consolidate_readings(readings, frames, report)
    report.transcript_entries = len(entries)
    report.named_dialogue_entries = sum(entry.speaker is not None for entry in entries)
    report.narration_entries = sum(entry.speaker is None for entry in entries)
    report.low_confidence_entries = sum(entry.confidence == "low" for entry in entries)
    report.speakers = dict(
        Counter(entry.speaker for entry in entries if entry.speaker is not None)
    )
    if report.low_confidence_entries:
        report.warnings.append(
            "Low-confidence entries should be compared with their referenced source frame."
        )
    report.warnings.append(
        "Dialogue choices can remain under the previous on-screen speaker label; "
        "frame references are retained so these can be reviewed."
    )
    return entries, report


def format_transcript(entries: list[TranscriptEntry], show_frames: bool = True) -> str:
    lines: list[str] = []
    for entry in entries:
        prefix = f"[frames {entry.first_frame}–{entry.last_frame}] " if show_frames else ""
        confidence = " [CHECK]" if entry.confidence == "low" else ""
        if entry.speaker:
            lines.append(f"{prefix}{entry.speaker}: {entry.text}{confidence}")
        else:
            lines.append(f"{prefix}[Narration/Screen]: {entry.text}{confidence}")
    return "\n\n".join(lines).strip() + ("\n" if lines else "")


def parse_aliases(values: list[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for value in values:
        if "=" in value:
            raw, canonical = value.split("=", 1)
        else:
            raw = canonical = value
        raw, canonical = raw.strip(), canonical.strip()
        if not raw or not canonical:
            raise ValueError(f"Invalid speaker alias: {value!r}")
        aliases[raw] = canonical
    return aliases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Consolidate progressive, repeated dialogue from frame-by-frame "
            "PaddleOCR Studio text."
        )
    )
    parser.add_argument("input", type=Path, help="combined PaddleOCR Studio .txt file")
    parser.add_argument("-o", "--output", type=Path, help="output transcript .txt")
    parser.add_argument("--report", type=Path, help="optional extraction report .json")
    parser.add_argument(
        "--speaker",
        action="append",
        default=[],
        metavar="OCR_NAME=DISPLAY_NAME",
        help="add a speaker or OCR name alias; may be repeated",
    )
    parser.add_argument(
        "--dialogue-only",
        action="store_true",
        help="exclude automatically detected narration/screen prose",
    )
    parser.add_argument(
        "--no-frame-references",
        action="store_true",
        help="omit source frame ranges from the readable transcript",
    )
    parser.add_argument(
        "--minimum-confidence",
        choices=("low", "medium", "high"),
        default="medium",
        help=(
            "lowest confidence included in the transcript (default: medium); "
            "use low for a complete audit containing [CHECK] entries"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input.is_file():
        print(f"Error: input file does not exist: {args.input}", file=sys.stderr)
        return 2
    output = args.output or args.input.with_name(f"{args.input.stem}_transcript.txt")
    try:
        aliases = parse_aliases(args.speaker)
        source = args.input.read_text(encoding="utf-8-sig")
        entries, report = extract_transcript(
            source,
            speaker_aliases=aliases,
            include_narration=not args.dialogue_only,
        )
        confidence_rank = {"low": 0, "medium": 1, "high": 2}
        minimum_rank = confidence_rank[args.minimum_confidence]
        printable_entries = [
            entry
            for entry in entries
            if confidence_rank[entry.confidence] >= minimum_rank
        ]
        report.entries_excluded_by_confidence = len(entries) - len(printable_entries)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            format_transcript(
                printable_entries, show_frames=not args.no_frame_references
            ),
            encoding="utf-8",
        )
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(asdict(report), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Read {report.input_frames} OCR frames")
    print(f"Created {report.transcript_entries} transcript entries")
    print(f"Named dialogue: {report.named_dialogue_entries}")
    print(f"Narration/screen text: {report.narration_entries}")
    print(f"Low-confidence entries: {report.low_confidence_entries}")
    if report.entries_excluded_by_confidence:
        print(
            "Excluded by confidence filter: "
            f"{report.entries_excluded_by_confidence}"
        )
    print(f"Saved transcript to: {output}")
    if args.report:
        print(f"Saved report to: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
