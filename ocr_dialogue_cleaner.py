#!/usr/bin/env python3
"""Clean repeated and fragmented dialogue from OCR-produced text files.

The cleaner is intentionally dependency-free.  Run it with no arguments for a
small desktop interface, or pass an input file to use it from a terminal.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable


SPEAKER_WITH_TEXT = re.compile(
    r"^\s*(?:[-–—]\s*)?"
    r"(?P<speaker>[A-ZÀ-ÖØ-Þ][A-ZÀ-ÖØ-Þ0-9 ._'’\-]{0,38}?)"
    r"\s*(?P<separator>:|—|–|-)\s*(?P<text>.*)$"
)
SPEAKER_ONLY = re.compile(
    r"^\s*(?:[-–—]\s*)?"
    r"(?P<speaker>[A-ZÀ-ÖØ-Þ][A-ZÀ-ÖØ-Þ0-9 .'’\-]{0,30})"
    r"\s*:?\s*$"
)
TIMESTAMP = re.compile(
    r"^\s*(?:\[\s*)?(?:\d{1,2}:)?\d{1,2}:\d{2}(?:[.,]\d{1,3})?(?:\s*\])?\s*$"
)
PAGE_NUMBER = re.compile(
    r"^\s*(?:page\s+)?(?:\d{1,4}|[ivxlcdm]{1,8})(?:\s+of\s+\d{1,4})?\s*$",
    re.IGNORECASE,
)
WORD = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)
END_PUNCTUATION = re.compile(r'[.!?]["”’)\]]*$')
SOFT_END = re.compile(r'[,;:—–-]["”’)\]]*$')


@dataclass
class CleanerConfig:
    """Tunable settings for the cleaning pipeline."""

    profile: str = "balanced"
    fuzzy_threshold: float = 0.91
    recent_window: int = 18
    block_min_lines: int = 2
    min_fuzzy_chars: int = 16
    remove_page_numbers: bool = True
    keep_blank_lines: bool = True
    speaker_only_labels: bool = True

    @classmethod
    def for_profile(cls, profile: str) -> "CleanerConfig":
        profiles = {
            "conservative": cls(
                profile="conservative",
                fuzzy_threshold=0.97,
                recent_window=8,
                block_min_lines=3,
                min_fuzzy_chars=24,
            ),
            "balanced": cls(profile="balanced"),
            "aggressive": cls(
                profile="aggressive",
                fuzzy_threshold=0.84,
                recent_window=35,
                block_min_lines=2,
                min_fuzzy_chars=10,
            ),
        }
        try:
            return profiles[profile]
        except KeyError as exc:
            raise ValueError(f"Unknown profile: {profile}") from exc


@dataclass
class Change:
    action: str
    line: int | None
    text: str
    reason: str


@dataclass
class CleanReport:
    input_lines: int = 0
    output_lines: int = 0
    exact_duplicates_removed: int = 0
    repeated_block_lines_removed: int = 0
    fuzzy_duplicates_removed: int = 0
    fragments_replaced_by_longer_text: int = 0
    page_numbers_removed: int = 0
    wrapped_lines_joined: int = 0
    incomplete_lines: list[dict[str, object]] = field(default_factory=list)
    changes: list[Change] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        return result


@dataclass
class SourceLine:
    text: str
    number: int
    blank: bool = False


def normalize_line(text: str) -> str:
    """Normalize OCR whitespace and compatibility characters without losing accents."""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "")  # soft hyphen
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" +([,.;:!?])", r"\1", text)
    return text.strip()


def comparison_key(text: str) -> str:
    """Create a forgiving key for duplicate comparison."""

    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"^[\-–—•]+\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n\"'.,;:!?-–—")


def word_count(text: str) -> int:
    return len(WORD.findall(text))


def parse_speaker(text: str, allow_speaker_only: bool = True) -> tuple[str, str] | None:
    """Return ``(speaker, dialogue)`` when a line looks like a speaker cue."""

    match = SPEAKER_WITH_TEXT.match(text)
    if match:
        speaker = re.sub(r"\s+", " ", match.group("speaker")).strip(" .")
        # Avoid treating ordinary capitalized prose ending in a dash as a label.
        if word_count(speaker) <= 5:
            return speaker, match.group("text").strip()

    if allow_speaker_only:
        match = SPEAKER_ONLY.match(text)
        if match:
            speaker = re.sub(r"\s+", " ", match.group("speaker")).strip(" .")
            if 1 <= word_count(speaker) <= 4 and any(ch.isalpha() for ch in speaker):
                return speaker, ""
    return None


def _prepare_lines(text: str, config: CleanerConfig, report: CleanReport) -> list[SourceLine]:
    raw_lines = text.splitlines()
    report.input_lines = len(raw_lines)
    prepared: list[SourceLine] = []

    for number, raw in enumerate(raw_lines, start=1):
        line = normalize_line(raw)
        if not line:
            if config.keep_blank_lines and prepared and not prepared[-1].blank:
                prepared.append(SourceLine("", number, blank=True))
            continue
        if TIMESTAMP.match(line):
            report.changes.append(Change("remove", number, line, "standalone timestamp"))
            continue
        if config.remove_page_numbers and PAGE_NUMBER.match(line):
            report.page_numbers_removed += 1
            report.changes.append(Change("remove", number, line, "standalone page number"))
            continue
        prepared.append(SourceLine(line, number))

    while prepared and prepared[-1].blank:
        prepared.pop()
    return prepared


def _exact_and_block_dedupe(
    lines: list[SourceLine], config: CleanerConfig, report: CleanReport
) -> list[SourceLine]:
    """Remove immediate duplicates and repeated multi-line OCR/page overlaps."""

    result: list[SourceLine] = []
    index = 0
    while index < len(lines):
        current = lines[index]
        if current.blank:
            if result and not result[-1].blank:
                result.append(current)
            index += 1
            continue

        key = comparison_key(current.text)
        previous_nonblank = next((x for x in reversed(result) if not x.blank), None)
        if previous_nonblank and key and key == comparison_key(previous_nonblank.text):
            report.exact_duplicates_removed += 1
            report.changes.append(
                Change("remove", current.number, current.text, "exact repeated line")
            )
            index += 1
            continue

        # Detect a repeated block, usually caused by overlap between OCR pages.
        best_length = 0
        compact_result = [x for x in result if not x.blank]
        candidate_positions = [
            pos
            for pos, old in enumerate(compact_result[-config.recent_window :])
            if key and comparison_key(old.text) == key
        ]
        for relative_pos in candidate_positions:
            old_pos = max(0, len(compact_result) - config.recent_window) + relative_pos
            length = 0
            scan = index
            while (
                old_pos + length < len(compact_result)
                and scan < len(lines)
                and length < config.recent_window
            ):
                if lines[scan].blank:
                    scan += 1
                    continue
                if comparison_key(compact_result[old_pos + length].text) != comparison_key(
                    lines[scan].text
                ):
                    break
                length += 1
                scan += 1
            best_length = max(best_length, length)

        if best_length >= config.block_min_lines:
            removed = 0
            while index < len(lines) and removed < best_length:
                if not lines[index].blank:
                    report.changes.append(
                        Change(
                            "remove",
                            lines[index].number,
                            lines[index].text,
                            "line from repeated OCR block",
                        )
                    )
                    removed += 1
                index += 1
            report.repeated_block_lines_removed += removed
            continue

        result.append(current)
        index += 1

    while result and result[-1].blank:
        result.pop()
    return result


def _fuzzy_dedupe(
    lines: list[SourceLine], config: CleanerConfig, report: CleanReport
) -> list[SourceLine]:
    """Collapse near-duplicate OCR variants, keeping the more complete version."""

    result: list[SourceLine] = []
    for current in lines:
        if current.blank:
            if result and not result[-1].blank:
                result.append(current)
            continue

        current_key = comparison_key(current.text)
        if len(current_key) < config.min_fuzzy_chars or word_count(current_key) < 3:
            result.append(current)
            continue

        recent_indices = [
            i
            for i in range(max(0, len(result) - config.recent_window), len(result))
            if not result[i].blank
        ]
        duplicate_index: int | None = None
        best_ratio = 0.0
        for candidate_index in reversed(recent_indices):
            candidate_key = comparison_key(result[candidate_index].text)
            if not candidate_key:
                continue
            ratio = SequenceMatcher(None, candidate_key, current_key, autojunk=False).ratio()
            same_opening = (
                candidate_key[: min(12, len(candidate_key))]
                == current_key[: min(12, len(current_key))]
            )
            prefix_variant = (
                min(len(candidate_key), len(current_key)) >= config.min_fuzzy_chars
                and (
                    candidate_key.startswith(current_key)
                    or current_key.startswith(candidate_key)
                )
            )
            if (ratio >= config.fuzzy_threshold and same_opening) or prefix_variant:
                if ratio > best_ratio:
                    best_ratio = ratio
                    duplicate_index = candidate_index

        if duplicate_index is None:
            result.append(current)
            continue

        old = result[duplicate_index]
        if len(current_key) > len(comparison_key(old.text)) + 2:
            result[duplicate_index] = current
            report.fragments_replaced_by_longer_text += 1
            report.changes.append(
                Change(
                    "replace",
                    old.number,
                    old.text,
                    f"short OCR fragment replaced by line {current.number}: {current.text}",
                )
            )
        else:
            report.fuzzy_duplicates_removed += 1
            report.changes.append(
                Change(
                    "remove",
                    current.number,
                    current.text,
                    f"near-duplicate of line {old.number} ({best_ratio:.0%} similar)",
                )
            )
    return result


def _join_text(left: str, right: str) -> str:
    """Join OCR line-wrap fragments, including words split by a hyphen."""

    if not left:
        return right
    if not right:
        return left
    if left.endswith("-") and right[:1].islower():
        return left[:-1] + right
    return left.rstrip() + " " + right.lstrip()


def _should_join_plain(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if parse_speaker(right):
        return False
    if left.endswith(("-", "‐")):
        return True
    if SOFT_END.search(left):
        return True
    if not END_PUNCTUATION.search(left):
        return True
    first_word = WORD.search(right)
    return bool(first_word and first_word.group(0)[:1].islower())


def _merge_fragments(
    lines: list[SourceLine], config: CleanerConfig, report: CleanReport
) -> list[SourceLine]:
    """Join wrapped dialogue and prose while keeping speaker turns separate."""

    output: list[SourceLine] = []
    current: SourceLine | None = None
    current_speaker: str | None = None

    def flush() -> None:
        nonlocal current, current_speaker
        if current is not None and current.text:
            output.append(current)
        current = None
        current_speaker = None

    for item in lines:
        if item.blank:
            flush()
            if config.keep_blank_lines and output and not output[-1].blank:
                output.append(item)
            continue

        speaker_data = parse_speaker(item.text, config.speaker_only_labels)
        if speaker_data:
            flush()
            speaker, dialogue = speaker_data
            current_speaker = speaker
            formatted = f"{speaker}:"
            if dialogue:
                formatted += f" {dialogue}"
            current = SourceLine(formatted, item.number)
            continue

        if current is None:
            current = SourceLine(item.text, item.number)
            continue

        if current_speaker is not None or _should_join_plain(current.text, item.text):
            current.text = _join_text(current.text, item.text)
            report.wrapped_lines_joined += 1
            report.changes.append(
                Change("join", item.number, item.text, f"joined to line {current.number}")
            )
        else:
            flush()
            current = SourceLine(item.text, item.number)

    flush()
    while output and output[-1].blank:
        output.pop()
    return output


def _find_incomplete(lines: Iterable[SourceLine], report: CleanReport) -> None:
    """Flag possible cut-offs for human review; never manufacture missing words."""

    for output_number, item in enumerate((x for x in lines if not x.blank), start=1):
        text = item.text
        dialogue = parse_speaker(text)
        content = dialogue[1] if dialogue else text
        reason = ""
        if content.endswith(("-", "‐")):
            reason = "ends with a split word or dash"
        elif content.endswith(("...", "…")):
            reason = "ends with an ellipsis"
        elif content and not END_PUNCTUATION.search(content) and word_count(content) >= 4:
            reason = "has no closing punctuation"
        if reason:
            report.incomplete_lines.append(
                {
                    "output_line": output_number,
                    "source_line": item.number,
                    "text": text,
                    "reason": reason,
                }
            )


def clean_text(
    text: str, config: CleanerConfig | None = None
) -> tuple[str, CleanReport]:
    """Clean OCR text and return ``(cleaned_text, report)``."""

    config = config or CleanerConfig.for_profile("balanced")
    report = CleanReport()
    lines = _prepare_lines(text, config, report)
    lines = _exact_and_block_dedupe(lines, config, report)
    lines = _fuzzy_dedupe(lines, config, report)
    lines = _merge_fragments(lines, config, report)
    _find_incomplete(lines, report)
    report.output_lines = sum(not line.blank for line in lines)
    cleaned = "\n".join("" if line.blank else line.text for line in lines).strip()
    if cleaned:
        cleaned += "\n"
    return cleaned, report


def read_text_file(path: Path, encoding: str) -> str:
    if encoding != "auto":
        return path.read_text(encoding=encoding)
    for candidate in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return path.read_text(encoding=candidate)
        except (UnicodeError, UnicodeDecodeError):
            pass
    return path.read_text(encoding="utf-8", errors="replace")


def clean_file(
    input_path: Path,
    output_path: Path,
    report_path: Path | None,
    config: CleanerConfig,
    encoding: str = "auto",
) -> CleanReport:
    text = read_text_file(input_path, encoding)
    cleaned, report = clean_text(text, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(cleaned, encoding="utf-8")
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return report


def summary(report: CleanReport) -> str:
    removed = (
        report.exact_duplicates_removed
        + report.repeated_block_lines_removed
        + report.fuzzy_duplicates_removed
    )
    return (
        f"Input: {report.input_lines} lines\n"
        f"Output: {report.output_lines} text lines\n"
        f"Duplicates removed: {removed}\n"
        f"Longer versions recovered: {report.fragments_replaced_by_longer_text}\n"
        f"Wrapped lines joined: {report.wrapped_lines_joined}\n"
        f"Possible incomplete lines: {len(report.incomplete_lines)}"
    )


def launch_gui() -> None:
    """Launch a basic Tkinter interface; imports are delayed for headless systems."""

    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ImportError:
        raise SystemExit(
            "Tkinter is not installed. Pass an input filename to use command-line mode."
        )

    root = tk.Tk()
    root.title("OCR Dialogue Cleaner")
    root.geometry("720x380")
    root.minsize(620, 340)

    input_var = tk.StringVar()
    output_var = tk.StringVar()
    report_var = tk.BooleanVar(value=True)
    profile_var = tk.StringVar(value="balanced")
    status_var = tk.StringVar(value="Choose an OCR text file to begin.")

    frame = ttk.Frame(root, padding=20)
    frame.pack(fill="both", expand=True)
    frame.columnconfigure(1, weight=1)

    def choose_input() -> None:
        selected = filedialog.askopenfilename(
            title="Choose OCR text",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if selected:
            input_var.set(selected)
            source = Path(selected)
            output_var.set(str(source.with_name(f"{source.stem}_cleaned.txt")))

    def choose_output() -> None:
        selected = filedialog.asksaveasfilename(
            title="Save cleaned text",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
        )
        if selected:
            output_var.set(selected)

    def run_cleaner() -> None:
        source = Path(input_var.get().strip())
        destination = Path(output_var.get().strip())
        if not source.is_file():
            messagebox.showerror("Missing input", "Please choose an existing text file.")
            return
        if not str(destination):
            messagebox.showerror("Missing output", "Please choose where to save the result.")
            return
        try:
            report_path = (
                destination.with_name(f"{destination.stem}_report.json")
                if report_var.get()
                else None
            )
            report = clean_file(
                source,
                destination,
                report_path,
                CleanerConfig.for_profile(profile_var.get()),
            )
        except Exception as exc:  # GUI boundary: show a useful message instead of crashing
            messagebox.showerror("Could not clean file", str(exc))
            return
        status_var.set(summary(report).replace("\n", "  •  "))
        report_note = f"\n\nReview log: {report_path}" if report_path else ""
        messagebox.showinfo(
            "Cleaning finished",
            f"{summary(report)}\n\nSaved to:\n{destination}{report_note}",
        )

    ttk.Label(frame, text="OCR Dialogue Cleaner", font=("", 17, "bold")).grid(
        row=0, column=0, columnspan=3, sticky="w", pady=(0, 8)
    )
    ttk.Label(
        frame,
        text=(
            "Removes repeated OCR passages, keeps the longest version, joins broken "
            "dialogue lines, and flags possible cut-offs."
        ),
        wraplength=650,
    ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 20))

    ttk.Label(frame, text="OCR file").grid(row=2, column=0, sticky="w", padx=(0, 10))
    ttk.Entry(frame, textvariable=input_var).grid(row=2, column=1, sticky="ew")
    ttk.Button(frame, text="Browse…", command=choose_input).grid(
        row=2, column=2, padx=(10, 0)
    )

    ttk.Label(frame, text="Cleaned file").grid(
        row=3, column=0, sticky="w", padx=(0, 10), pady=12
    )
    ttk.Entry(frame, textvariable=output_var).grid(row=3, column=1, sticky="ew", pady=12)
    ttk.Button(frame, text="Browse…", command=choose_output).grid(
        row=3, column=2, padx=(10, 0), pady=12
    )

    ttk.Label(frame, text="Cleaning level").grid(
        row=4, column=0, sticky="w", padx=(0, 10)
    )
    ttk.Combobox(
        frame,
        textvariable=profile_var,
        values=("conservative", "balanced", "aggressive"),
        state="readonly",
        width=18,
    ).grid(row=4, column=1, sticky="w")

    ttk.Checkbutton(
        frame,
        text="Save a JSON review log next to the cleaned file",
        variable=report_var,
    ).grid(row=5, column=1, sticky="w", pady=(12, 20))

    ttk.Button(frame, text="Clean file", command=run_cleaner).grid(
        row=6, column=1, sticky="w"
    )
    ttk.Label(frame, textvariable=status_var, wraplength=650).grid(
        row=7, column=0, columnspan=3, sticky="w", pady=(22, 0)
    )
    root.mainloop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Remove repeated OCR dialogue, recover longer variants, join wrapped "
            "lines, and report possible incomplete text."
        )
    )
    parser.add_argument("input", nargs="?", type=Path, help="OCR .txt file")
    parser.add_argument("-o", "--output", type=Path, help="cleaned UTF-8 .txt file")
    parser.add_argument(
        "--report",
        type=Path,
        help="optional JSON file containing every change and review warning",
    )
    parser.add_argument(
        "--profile",
        choices=("conservative", "balanced", "aggressive"),
        default="balanced",
        help="how readily near-duplicates are removed (default: balanced)",
    )
    parser.add_argument(
        "--encoding",
        default="auto",
        help="input encoding; default tries UTF-8, UTF-16, then Windows-1252",
    )
    parser.add_argument(
        "--no-page-numbers",
        action="store_true",
        help="keep standalone page numbers instead of removing them",
    )
    parser.add_argument(
        "--no-blank-lines",
        action="store_true",
        help="do not preserve paragraph breaks",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="open the desktop interface",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.gui or args.input is None:
        launch_gui()
        return 0

    if not args.input.is_file():
        print(f"Error: input file does not exist: {args.input}", file=sys.stderr)
        return 2

    output = args.output or args.input.with_name(f"{args.input.stem}_cleaned.txt")
    config = CleanerConfig.for_profile(args.profile)
    config.remove_page_numbers = not args.no_page_numbers
    config.keep_blank_lines = not args.no_blank_lines
    try:
        report = clean_file(args.input, output, args.report, config, args.encoding)
    except (OSError, UnicodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(summary(report))
    print(f"Saved cleaned text to: {output}")
    if args.report:
        print(f"Saved review log to: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
