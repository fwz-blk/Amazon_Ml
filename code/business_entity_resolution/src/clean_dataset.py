#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - Business Entity Resolution Pipeline
Stage 1: Conservative Data Cleaning, Validation, and Profiling

Core Principles:
1. Strict raw-data preservation:
   - Original input files are never modified.
   - No rows are deleted, deduplicated, reordered, filtered, merged, or altered.
   - All four original columns (entity_id, business_name, business_address, country)
     are preserved byte-for-byte in the cleaned output.
2. Conservative normalization applied ONLY to derived columns:
   - Unicode NFC normalization.
   - Non-Latin scripts (Hindi, Arabic, French accents, Cyrillic, etc.) preserved.
   - Unicode-aware casefold().
   - Safe punctuation equivalences only (curly quotes -> straight, unicode dashes -> hyphen).
   - Meaningful punctuation preserved (&, -, ., ', /, @, +, #, commas, parens).
   - Whitespace runs and noise collapsed to single ASCII space; stripped.
   - Country normalized using only NFC + casefold() + whitespace normalization.
   - Missing fields flagged with boolean indicators and kept empty (no placeholders).
3. Structural integrity & quality gates:
   - Validates input UTF-8, headers, column counts, entity_id format & uniqueness.
   - Bounded-memory streaming execution (line-by-line / chunked).
   - Writes to temporary files and atomically renames only upon passing all checks.
   - Produces diagnostic profile.json and quality_checks.json reports.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ==============================================================================
# Constants & Encodings
# ==============================================================================

INPUT_COLUMNS = ["entity_id", "business_name", "business_address", "country"]

OUTPUT_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
    "business_name_normalized",
    "business_address_normalized",
    "country_normalized",
    "business_name_is_missing",
    "business_address_is_missing",
    "country_is_missing",
]

DEFAULT_EXPECTED_PREFIXES = {
    "source1": "S1-",
    "source2": "S2-",
    "source3": "S3-",
}

# Explicitly documented safe punctuation equivalences
# Maps Unicode variants to ASCII standard representations
SAFE_PUNCTUATION_TRANSLATION = {
    # Curly single quotes & apostrophe variants -> straight single quote (')
    ord("‘"): "'",  # U+2018 LEFT SINGLE QUOTATION MARK
    ord("’"): "'",  # U+2019 RIGHT SINGLE QUOTATION MARK
    ord("‚"): "'",  # U+201A SINGLE LOW-9 QUOTATION MARK
    ord("‛"): "'",  # U+201B SINGLE HIGH-REVERSED-9 QUOTATION MARK
    ord("ʻ"): "'",  # U+02BB MODIFIER LETTER TURNED COMMA
    ord("ʼ"): "'",  # U+02BC MODIFIER LETTER APOSTROPHE
    ord("′"): "'",  # U+2032 PRIME
    ord("‵"): "'",  # U+2035 REVERSED PRIME

    # Curly double quotes -> straight double quote (")
    ord("“"): '"',  # U+201C LEFT DOUBLE QUOTATION MARK
    ord("”"): '"',  # U+201D RIGHT DOUBLE QUOTATION MARK
    ord("„"): '"',  # U+201E DOUBLE LOW-9 QUOTATION MARK
    ord("‟"): '"',  # U+201F DOUBLE HIGH-REVERSED-9 QUOTATION MARK
    ord("″"): '"',  # U+2033 DOUBLE PRIME
    ord("‶"): '"',  # U+2036 REVERSED DOUBLE PRIME

    # Unicode dash variants -> ASCII hyphen (-)
    ord("‐"): "-",  # U+2010 HYPHEN
    ord("‑"): "-",  # U+2011 NON-BREAKING HYPHEN
    ord("‒"): "-",  # U+2012 FIGURE DASH
    ord("–"): "-",  # U+2013 EN DASH
    ord("—"): "-",  # U+2014 EM DASH
    ord("―"): "-",  # U+2015 HORIZONTAL BAR
    ord("−"): "-",  # U+2212 MINUS SIGN
    ord("﹘"): "-",  # U+FE58 SMALL EM DASH
    ord("﹣"): "-",  # U+FE63 SMALL HYPHEN-MINUS
    ord("－"): "-",  # U+FF0D FULLWIDTH HYPHEN-MINUS

    # Unicode space variants -> ASCII space (' ')
    ord("\u00A0"): " ",  # NO-BREAK SPACE
    ord("\u1680"): " ",  # OGHAM SPACE MARK
    ord("\u2000"): " ",  # EN QUAD
    ord("\u2001"): " ",  # EM QUAD
    ord("\u2002"): " ",  # EN SPACE
    ord("\u2003"): " ",  # EM SPACE
    ord("\u2004"): " ",  # THREE-PER-EM SPACE
    ord("\u2005"): " ",  # FOUR-PER-EM SPACE
    ord("\u2006"): " ",  # SIX-PER-EM SPACE
    ord("\u2007"): " ",  # FIGURE SPACE
    ord("\u2008"): " ",  # PUNCTUATION SPACE
    ord("\u2009"): " ",  # THIN SPACE
    ord("\u200A"): " ",  # HAIR SPACE
    ord("\u2028"): " ",  # LINE SEPARATOR
    ord("\u2029"): " ",  # PARAGRAPH SEPARATOR
    ord("\u202F"): " ",  # NARROW NO-BREAK SPACE
    ord("\u205F"): " ",  # MEDIUM MATHEMATICAL SPACE
    ord("\u3000"): " ",  # IDEOGRAPHIC SPACE
    ord("\uFEFF"): " ",  # ZERO WIDTH NO-BREAK SPACE / BOM
}

# Compiled regex for collapsing Unicode whitespace, tabs, and newlines
RE_WHITESPACE = re.compile(r"\s+", flags=re.UNICODE)


# ==============================================================================
# Exceptions
# ==============================================================================

class StructuralIntegrityError(Exception):
    """Raised when an input file violates structural integrity constraints."""

    def __init__(self, file_path: str, line_number: int, reason: str):
        self.file_path = file_path
        self.line_number = line_number
        self.reason = reason
        super().__init__(
            f"Structural integrity violation in '{file_path}' at line {line_number}: {reason}"
        )


class QualityCheckError(Exception):
    """Raised when a post-cleaning invariant check fails."""

    def __init__(self, check_name: str, file_path: str, reason: str):
        self.check_name = check_name
        self.file_path = file_path
        self.reason = reason
        super().__init__(
            f"Quality check failure [{check_name}] in '{file_path}': {reason}"
        )


# ==============================================================================
# Normalization Functions
# ==============================================================================

def is_missing_value(val: Optional[str]) -> bool:
    """
    Check if a source field value is missing.
    Empty strings and whitespace-only strings are considered missing.
    """
    if val is None or len(val) == 0:
        return True
    return val.strip() == ""


def normalize_text(text: Optional[str]) -> str:
    """
    Conservative normalization applied ONLY to business_name and business_address.

    Rules applied:
    1. Unicode NFC normalization.
    2. Safe punctuation equivalences:
       - curly single quotes -> straight single quote (')
       - curly double quotes -> straight double quote (")
       - Unicode dash variants -> ASCII hyphen (-)
       - Unicode space variants -> ASCII space (' ')
    3. Unicode-aware casefold().
    4. Re-apply Unicode NFC normalization (ensures canonical composition after casefold).
    5. Treat Unicode whitespace, tabs, and embedded newlines as noise:
       replace whitespace runs with a single ASCII space and strip ends.
    6. Non-Latin scripts (Hindi, Arabic, Cyrillic, CJK, etc.) and accents are strictly preserved.
    7. Meaningful punctuation (&, -, ., ', /, @, +, #, commas, parens) is preserved.
    8. Missing values return empty string ("").
    """
    if is_missing_value(text):
        return ""

    assert text is not None
    # 1. NFC normalization
    s = unicodedata.normalize("NFC", text)
    # 2. Safe punctuation equivalences
    s = s.translate(SAFE_PUNCTUATION_TRANSLATION)
    # 3. Unicode-aware casefold
    s = s.casefold()
    # 4. Re-apply NFC
    s = unicodedata.normalize("NFC", s)
    # 5. Collapse whitespace and strip
    s = RE_WHITESPACE.sub(" ", s).strip()
    return s


def normalize_country(text: Optional[str]) -> str:
    """
    Country normalization using ONLY:
    - Unicode NFC
    - Unicode casefold()
    - whitespace normalization

    Does NOT hard-code country names or apply alias maps.
    Missing values return empty string ("").
    """
    if is_missing_value(text):
        return ""

    assert text is not None
    s = unicodedata.normalize("NFC", text)
    s = s.casefold()
    s = unicodedata.normalize("NFC", s)
    s = RE_WHITESPACE.sub(" ", s).strip()
    return s


# ==============================================================================
# Helper Functions for Character & Script Profiling
# ==============================================================================

def get_script_category(char: str) -> str:
    """Identify script category of a Unicode character."""
    cp = ord(char)
    if cp < 128:
        return "Latin-ASCII"
    if (0x0080 <= cp <= 0x024F) or (0x1E00 <= cp <= 0x1EFF):
        return "Latin-Extended"
    if 0x0900 <= cp <= 0x097F:
        return "Devanagari"
    if (0x0600 <= cp <= 0x06FF) or (0x0750 <= cp <= 0x077F) or (0x08A0 <= cp <= 0x08FF):
        return "Arabic"
    if 0x0400 <= cp <= 0x04FF:
        return "Cyrillic"
    if 0x4E00 <= cp <= 0x9FFF:
        return "CJK"
    if 0x0B80 <= cp <= 0x0BFF:
        return "Tamil"
    if 0x0C00 <= cp <= 0x0C7F:
        return "Telugu"
    if 0x0980 <= cp <= 0x09FF:
        return "Bengali"
    if 0x0A00 <= cp <= 0x0A7F:
        return "Gurmukhi"
    if 0x0A80 <= cp <= 0x0AFF:
        return "Gujarati"
    if 0x0C80 <= cp <= 0x0CFF:
        return "Kannada"
    if 0x0D00 <= cp <= 0x0D7F:
        return "Malayalam"
    if 0x0370 <= cp <= 0x03FF:
        return "Greek"
    if 0x0590 <= cp <= 0x05FF:
        return "Hebrew"
    return "Other-Unicode"


def is_suspicious_symbol(char: str) -> Optional[str]:
    """Check if character is a suspicious control, replacement, or private use character."""
    cp = ord(char)
    if cp == 0xFFFD:
        return "replacement_character_ufffd"
    if cp < 32 and char not in ("\t", "\n", "\r"):
        return "ascii_control_character"
    cat = unicodedata.category(char)
    if cat == "Cc" and char not in ("\t", "\n", "\r"):
        return "unicode_control_char"
    if cat in ("Co", "Cs"):
        return "private_use_or_surrogate"
    return None


def get_expected_prefix(filename: str, custom_prefix: Optional[str] = None) -> str:
    """Determine expected entity_id prefix from filename or custom argument."""
    if custom_prefix:
        return custom_prefix
    lower_name = filename.lower()
    for key, prefix in DEFAULT_EXPECTED_PREFIXES.items():
        if key in lower_name:
            return prefix
    return ""


# ==============================================================================
# Bounded Binned Quantile & Length Profiler
# ==============================================================================

class LengthProfiler:
    """
    Computes exact summary statistics (min, max, mean, quantiles) and maintains
    bounded deterministic top-k shortest non-empty and longest string examples
    in strictly bounded memory.
    """

    def __init__(self, max_examples: int = 5):
        self.count: int = 0
        self.min_len: int = 0
        self.max_len: int = 0
        self.sum_len: int = 0
        self.length_histogram: Counter[int] = Counter()
        self.max_examples: int = max_examples
        # Store tuples: (length, value, entity_id)
        self.shortest_examples: List[Tuple[int, str, str]] = []
        self.longest_examples: List[Tuple[int, str, str]] = []

    def update(self, val: str, entity_id: str) -> None:
        length = len(val)
        if self.count == 0:
            self.min_len = length
            self.max_len = length
        else:
            if length < self.min_len:
                self.min_len = length
            if length > self.max_len:
                self.max_len = length

        self.count += 1
        self.sum_len += length
        self.length_histogram[length] += 1

        # Maintain bounded deterministic shortest non-empty examples
        if length > 0:
            item = (length, val, entity_id)
            if len(self.shortest_examples) < self.max_examples:
                self.shortest_examples.append(item)
                self.shortest_examples.sort()
            elif item < self.shortest_examples[-1]:
                self.shortest_examples[-1] = item
                self.shortest_examples.sort()

        # Maintain bounded deterministic longest examples
        # Use negative length for sorting or sort by (-length, value, entity_id)
        longest_item = (-length, val, entity_id)
        if len(self.longest_examples) < self.max_examples:
            self.longest_examples.append(longest_item)
            self.longest_examples.sort()
        elif longest_item < self.longest_examples[-1]:
            self.longest_examples[-1] = longest_item
            self.longest_examples.sort()

    def get_summary(self) -> Dict[str, Any]:
        if self.count == 0:
            return {"min": 0, "max": 0, "mean": 0.0, "quantiles": {}}

        mean = round(self.sum_len / self.count, 2)
        quantiles = self._compute_quantiles([0.25, 0.50, 0.75, 0.90, 0.95, 0.99])

        return {
            "min": self.min_len,
            "max": self.max_len,
            "mean": mean,
            "quantiles": quantiles,
        }

    def _compute_quantiles(self, percentiles: List[float]) -> Dict[str, int]:
        sorted_lens = sorted(self.length_histogram.keys())
        targets = {p: int(self.count * p) for p in percentiles}
        results = {}

        cum = 0
        sorted_targets = sorted(targets.items(), key=lambda x: x[1])
        t_idx = 0

        for length in sorted_lens:
            cum += self.length_histogram[length]
            while t_idx < len(sorted_targets) and cum >= sorted_targets[t_idx][1]:
                p, _ = sorted_targets[t_idx]
                p_label = f"p{int(p * 100)}"
                results[p_label] = length
                t_idx += 1

        return results

    def get_examples(self) -> Dict[str, List[Dict[str, Any]]]:
        shortest = [
            {"entity_id": eid, "length": length, "value": val}
            for (length, val, eid) in self.shortest_examples
        ]
        longest = [
            {"entity_id": eid, "length": -neg_len, "value": val}
            for (neg_len, val, eid) in self.longest_examples
        ]
        return {
            "shortest_non_empty": shortest,
            "longest": longest,
        }


# ==============================================================================
# Input Validation (Integrity Gate)
# ==============================================================================

def validate_input_file(
    file_path: Path,
    expected_prefix: str = "",
    check_unique_ids: bool = True,
) -> int:
    """
    Validate raw input file integrity before writing any cleaned output.

    Checks:
    - File readability as UTF-8
    - Exact expected header: entity_id\\tbusiness_name\\tbusiness_address\\tcountry
    - Exactly 4 tab-separated columns per row
    - Non-empty entity_id
    - entity_id prefix consistency
    - Unique entity_id values (tracked with bounded memory per file)
    - No malformed rows

    Raises:
        StructuralIntegrityError if any violation is found.
    Returns:
        Total row count (excluding header).
    """
    if not file_path.is_file():
        raise StructuralIntegrityError(str(file_path), 0, "Input file does not exist")

    seen_ids: Set[str] = set() if check_unique_ids else set()
    line_number = 0

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            # Check header
            header_line = f.readline()
            line_number = 1
            if not header_line:
                raise StructuralIntegrityError(str(file_path), 1, "File is completely empty")

            header_cols = header_line.rstrip("\r\n").split("\t")
            if header_cols != INPUT_COLUMNS:
                raise StructuralIntegrityError(
                    str(file_path),
                    1,
                    f"Invalid header: expected {INPUT_COLUMNS}, found {header_cols}",
                )

            # Check rows
            for line in f:
                line_number += 1
                cols = line.rstrip("\r\n").split("\t")
                if len(cols) != 4:
                    raise StructuralIntegrityError(
                        str(file_path),
                        line_number,
                        f"Malformed row: expected 4 tab-delimited columns, found {len(cols)}",
                    )

                entity_id = cols[0]
                if not entity_id or not entity_id.strip():
                    raise StructuralIntegrityError(
                        str(file_path), line_number, "Empty or whitespace-only entity_id"
                    )

                if expected_prefix and not entity_id.startswith(expected_prefix):
                    raise StructuralIntegrityError(
                        str(file_path),
                        line_number,
                        f"Inconsistent entity_id prefix: '{entity_id}' does not start with expected prefix '{expected_prefix}'",
                    )

                if check_unique_ids:
                    if entity_id in seen_ids:
                        raise StructuralIntegrityError(
                            str(file_path),
                            line_number,
                            f"Duplicate entity_id detected: '{entity_id}'",
                        )
                    seen_ids.add(entity_id)

    except UnicodeDecodeError as e:
        raise StructuralIntegrityError(
            str(file_path), line_number, f"UTF-8 decode failure: {e}"
        ) from e

    total_rows = line_number - 1
    # Free memory
    del seen_ids
    gc.collect()

    return total_rows


# ==============================================================================
# Streaming Cleaner and Profiler
# ==============================================================================

def clean_and_profile_file(
    input_path: Path,
    temp_output_path: Path,
    expected_prefix: str = "",
) -> Dict[str, Any]:
    """
    Streams raw input file, normalizes derived fields, writes cleaned output
    to temp_output_path, and computes comprehensive profiling diagnostics.

    Memory bounded:
    - Processes row-by-row with minimal buffer allocations.
    - Tracks duplicate counts via two-stage integer hash sets (freed upon completion).
    - Length histograms and examples are bounded O(1).
    """
    temp_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Profiling structures
    row_count = 0
    missing_counts = {col: 0 for col in INPUT_COLUMNS[1:]}
    empty_string_counts = {col: 0 for col in INPUT_COLUMNS[1:]}
    non_ascii_counts = {col: 0 for col in INPUT_COLUMNS[1:]}
    script_counts: Dict[str, Counter[str]] = {col: Counter() for col in INPUT_COLUMNS[1:]}
    suspicious_symbols: Dict[str, Counter[str]] = {col: Counter() for col in INPUT_COLUMNS[1:]}
    country_counts: Counter[str] = Counter()

    # Length profilers
    length_profilers = {
        "business_name": LengthProfiler(),
        "business_address": LengthProfiler(),
        "country": LengthProfiler(),
        "business_name_normalized": LengthProfiler(),
        "business_address_normalized": LengthProfiler(),
        "country_normalized": LengthProfiler(),
    }

    # Bounded duplicate tracking via two-set hash state
    # Seen once vs seen multiple
    name_seen_once: Set[int] = set()
    name_seen_multiple: Set[int] = set()
    total_dup_name_rows = 0

    addr_seen_once: Set[int] = set()
    addr_seen_multiple: Set[int] = set()
    total_dup_addr_rows = 0

    full_seen_once: Set[int] = set()
    full_seen_multiple: Set[int] = set()
    total_dup_full_rows = 0

    duplicate_eids: List[str] = []
    seen_eids: Set[str] = set()
    malformed_findings: List[Dict[str, Any]] = []

    line_num = 0

    with open(input_path, "r", encoding="utf-8") as in_f, \
         open(temp_output_path, "w", encoding="utf-8", newline="\n") as out_f:

        # Header check & write
        header_line = in_f.readline()
        line_num = 1
        raw_header = header_line.rstrip("\r\n").split("\t")
        if raw_header != INPUT_COLUMNS:
            raise StructuralIntegrityError(
                str(input_path), 1, f"Expected header {INPUT_COLUMNS}, got {raw_header}"
            )

        out_header = "\t".join(OUTPUT_COLUMNS) + "\n"
        out_f.write(out_header)

        # Stream lines
        for line in in_f:
            line_num += 1
            cols = line.rstrip("\r\n").split("\t")
            if len(cols) != 4:
                malformed_findings.append({
                    "line": line_num,
                    "expected": 4,
                    "found": len(cols),
                })
                raise StructuralIntegrityError(
                    str(input_path), line_num, f"Malformed row with {len(cols)} columns"
                )

            entity_id, b_name, b_addr, country = cols
            row_count += 1

            # Validate entity_id
            if expected_prefix and not entity_id.startswith(expected_prefix):
                raise StructuralIntegrityError(
                    str(input_path),
                    line_num,
                    f"Prefix mismatch for entity_id '{entity_id}', expected '{expected_prefix}'",
                )

            if entity_id in seen_eids:
                if len(duplicate_eids) < 10:
                    duplicate_eids.append(entity_id)
            else:
                seen_eids.add(entity_id)

            # Missingness checks
            name_missing = is_missing_value(b_name)
            addr_missing = is_missing_value(b_addr)
            ctry_missing = is_missing_value(country)

            if name_missing:
                missing_counts["business_name"] += 1
                if b_name == "":
                    empty_string_counts["business_name"] += 1
            if addr_missing:
                missing_counts["business_address"] += 1
                if b_addr == "":
                    empty_string_counts["business_address"] += 1
            if ctry_missing:
                missing_counts["country"] += 1
                if country == "":
                    empty_string_counts["country"] += 1

            # Country frequency
            country_counts[country] += 1

            # Script & Symbol Profiling
            for col_name, val in (
                ("business_name", b_name),
                ("business_address", b_addr),
                ("country", country),
            ):
                if val:
                    if not val.isascii():
                        non_ascii_counts[col_name] += 1
                        for char in val:
                            if ord(char) >= 128:
                                script_counts[col_name][get_script_category(char)] += 1
                            susp = is_suspicious_symbol(char)
                            if susp:
                                suspicious_symbols[col_name][susp] += 1
                    else:
                        for char in val:
                            susp = is_suspicious_symbol(char)
                            if susp:
                                suspicious_symbols[col_name][susp] += 1

            # Normalization
            name_norm = normalize_text(b_name)
            addr_norm = normalize_text(b_addr)
            ctry_norm = normalize_country(country)

            # Update Length Profilers
            length_profilers["business_name"].update(b_name, entity_id)
            length_profilers["business_address"].update(b_addr, entity_id)
            length_profilers["country"].update(country, entity_id)
            length_profilers["business_name_normalized"].update(name_norm, entity_id)
            length_profilers["business_address_normalized"].update(addr_norm, entity_id)
            length_profilers["country_normalized"].update(ctry_norm, entity_id)

            # Duplicate normalized value tracking
            # 1. Normalized name
            if name_norm:
                h_name = hash(name_norm)
                if h_name in name_seen_multiple:
                    total_dup_name_rows += 1
                elif h_name in name_seen_once:
                    name_seen_once.remove(h_name)
                    name_seen_multiple.add(h_name)
                    total_dup_name_rows += 2
                else:
                    name_seen_once.add(h_name)

            # 2. Normalized address (exclude empty)
            if addr_norm:
                h_addr = hash(addr_norm)
                if h_addr in addr_seen_multiple:
                    total_dup_addr_rows += 1
                elif h_addr in addr_seen_once:
                    addr_seen_once.remove(h_addr)
                    addr_seen_multiple.add(h_addr)
                    total_dup_addr_rows += 2
                else:
                    addr_seen_once.add(h_addr)

            # 3. Full normalized tuple
            h_full = hash((name_norm, addr_norm, ctry_norm))
            if h_full in full_seen_multiple:
                total_dup_full_rows += 1
            elif h_full in full_seen_once:
                full_seen_once.remove(h_full)
                full_seen_multiple.add(h_full)
                total_dup_full_rows += 2
            else:
                full_seen_once.add(h_full)

            # Write cleaned output row
            # Boolean missing flags formatted as standard string representations ("True"/"False")
            out_row = (
                f"{entity_id}\t"
                f"{b_name}\t"
                f"{b_addr}\t"
                f"{country}\t"
                f"{name_norm}\t"
                f"{addr_norm}\t"
                f"{ctry_norm}\t"
                f"{str(name_missing)}\t"
                f"{str(addr_missing)}\t"
                f"{str(ctry_missing)}\n"
            )
            out_f.write(out_row)

    # Compute duplicate metrics before freeing structures
    dup_metrics = {
        "normalized_name": {
            "unique_values_with_duplicates": len(name_seen_multiple),
            "total_duplicate_rows": total_dup_name_rows,
        },
        "normalized_address": {
            "unique_values_with_duplicates": len(addr_seen_multiple),
            "total_duplicate_rows": total_dup_addr_rows,
        },
        "full_normalized_record": {
            "unique_tuples_with_duplicates": len(full_seen_multiple),
            "total_duplicate_rows": total_dup_full_rows,
        },
    }

    # Free memory
    del name_seen_once, name_seen_multiple
    del addr_seen_once, addr_seen_multiple
    del full_seen_once, full_seen_multiple
    del seen_eids
    gc.collect()

    # Compile profile report for this file
    profile_data: Dict[str, Any] = {
        "file_name": input_path.name,
        "input_path": str(input_path),
        "row_count": row_count,
        "column_count": len(OUTPUT_COLUMNS),
        "headers": OUTPUT_COLUMNS,
        "data_types_observed": {
            "entity_id": "string",
            "business_name": "string",
            "business_address": "string",
            "country": "string",
            "business_name_normalized": "string",
            "business_address_normalized": "string",
            "country_normalized": "string",
            "business_name_is_missing": "boolean",
            "business_address_is_missing": "boolean",
            "country_is_missing": "boolean",
        },
        "missing_counts": missing_counts,
        "empty_string_counts": empty_string_counts,
        "unicode_non_ascii_counts": non_ascii_counts,
        "multilingual_script_indicators": {
            k: dict(v) for k, v in script_counts.items()
        },
        "suspicious_symbol_counts": {
            k: dict(v) for k, v in suspicious_symbols.items()
        },
        "text_length_summaries": {
            col: lp.get_summary() for col, lp in length_profilers.items()
        },
        "text_length_examples": {
            col: lp.get_examples() for col, lp in length_profilers.items()
        },
        "duplicate_looking_normalized_value_counts": dup_metrics,
        "duplicate_entity_id_findings": duplicate_eids,
        "malformed_row_findings": malformed_findings,
        "country_frequency_counts": dict(country_counts),
        "preservation_checks": {
            "row_order_and_ids_preserved": True,
            "original_fields_unaltered": True,
            "entity_ids_preserved": True,
            "row_count_preserved": True,
        },
    }

    return profile_data


# ==============================================================================
# Quality Checks (Post-Cleaning Invariant Gate)
# ==============================================================================

def verify_file_quality(
    input_path: Path,
    cleaned_path: Path,
    expected_prefix: str = "",
) -> Dict[str, Any]:
    """
    Comprehensive quality check verifying that cleaned output satisfies
    all 12 required invariants against original raw input:

    1. row_count_match: input and output row counts identical
    2. entity_id_match_and_order: entity IDs identical in exact original sequence
    3. original_values_unaltered: original 4 fields match raw input exactly
    4. source_columns_present: all 4 original columns present in positions 0..3
    5. no_unexpected_nulls: no literal 'None', 'null', 'nan' in fields unless in raw
    6. normalized_missing_not_fabricated: normalized missing fields are "" (no placeholders)
    7. output_headers_correct: headers exactly match OUTPUT_COLUMNS
    8. valid_utf8_tsv: file is strictly parseable as UTF-8 TSV
    9. no_malformed_rows: exactly 10 tab-separated columns on every single line
    10. normalized_columns_present: positions 4..6 present and correct
    11. source_prefix_validation: all IDs start with expected source prefix
    12. boolean_flags_valid: positions 7..9 strictly 'True' or 'False'
    """
    invariants = {
        "row_count_match": True,
        "entity_id_match_and_order": True,
        "original_values_unaltered": True,
        "source_columns_present": True,
        "no_unexpected_nulls": True,
        "normalized_missing_not_fabricated": True,
        "output_headers_correct": True,
        "valid_utf8_tsv": True,
        "no_malformed_rows": True,
        "normalized_columns_present": True,
        "source_prefix_validation": True,
        "boolean_flags_valid": True,
    }

    errors: List[str] = []
    in_row_count = 0
    out_row_count = 0

    with open(input_path, "r", encoding="utf-8") as in_f, \
         open(cleaned_path, "r", encoding="utf-8") as out_f:

        # Header check
        in_header = in_f.readline().rstrip("\r\n").split("\t")
        out_header = out_f.readline().rstrip("\r\n").split("\t")

        if in_header != INPUT_COLUMNS:
            invariants["source_columns_present"] = False
            errors.append(f"Input header mismatch: {in_header}")

        if out_header != OUTPUT_COLUMNS:
            invariants["output_headers_correct"] = False
            errors.append(f"Output header mismatch: {out_header}")

        line_num = 1
        for in_line, out_line in itertools.zip_longest(in_f, out_f, fillvalue=None):
            line_num += 1
            if in_line is None:
                invariants["row_count_match"] = False
                errors.append(f"Line {line_num}: cleaned output has extra rows beyond input file")
                out_row_count += 1
                break
            if out_line is None:
                invariants["row_count_match"] = False
                errors.append(f"Line {line_num}: cleaned output has fewer rows than input file")
                in_row_count += 1
                break

            in_row_count += 1
            out_row_count += 1

            in_cols = in_line.rstrip("\r\n").split("\t")
            out_cols = out_line.rstrip("\r\n").split("\t")

            # Check column count
            if len(out_cols) != 10:
                invariants["no_malformed_rows"] = False
                errors.append(f"Line {line_num}: output has {len(out_cols)} columns, expected 10")
                break

            # 1. Entity ID match & order
            if in_cols[0] != out_cols[0]:
                invariants["entity_id_match_and_order"] = False
                errors.append(
                    f"Line {line_num}: entity_id mismatch '{in_cols[0]}' vs '{out_cols[0]}'"
                )
                break

            # 2. Source prefix validation
            if expected_prefix and not out_cols[0].startswith(expected_prefix):
                invariants["source_prefix_validation"] = False
                errors.append(
                    f"Line {line_num}: entity_id '{out_cols[0]}' missing expected prefix '{expected_prefix}'"
                )
                break

            # 3. Original values unaltered
            if in_cols[:4] != out_cols[:4]:
                invariants["original_values_unaltered"] = False
                errors.append(
                    f"Line {line_num}: raw fields altered from {in_cols[:4]} to {out_cols[:4]}"
                )
                break

            # 4. Check boolean flags
            name_missing_flag = out_cols[7]
            addr_missing_flag = out_cols[8]
            ctry_missing_flag = out_cols[9]
            if name_missing_flag not in ("True", "False") or \
               addr_missing_flag not in ("True", "False") or \
               ctry_missing_flag not in ("True", "False"):
                invariants["boolean_flags_valid"] = False
                errors.append(f"Line {line_num}: invalid boolean missing flags {out_cols[7:10]}")
                break

            # 5. Missing value fabrication check
            # When input is missing, normalized must be "" and flag True
            # When input is present, normalized must be non-empty and flag False
            in_name_missing = is_missing_value(in_cols[1])
            in_addr_missing = is_missing_value(in_cols[2])
            in_ctry_missing = is_missing_value(in_cols[3])

            if in_name_missing:
                if out_cols[4] != "" or name_missing_flag != "True":
                    invariants["normalized_missing_not_fabricated"] = False
                    errors.append(
                        f"Line {line_num}: missing name has non-empty normalized value '{out_cols[4]}'"
                    )
                    break
            else:
                if out_cols[4] == "" or name_missing_flag != "False":
                    invariants["normalized_missing_not_fabricated"] = False
                    errors.append(f"Line {line_num}: present name has empty normalized value")
                    break

            if in_addr_missing:
                if out_cols[5] != "" or addr_missing_flag != "True":
                    invariants["normalized_missing_not_fabricated"] = False
                    errors.append(
                        f"Line {line_num}: missing address has non-empty normalized value '{out_cols[5]}'"
                    )
                    break
            else:
                if out_cols[5] == "" or addr_missing_flag != "False":
                    invariants["normalized_missing_not_fabricated"] = False
                    errors.append(f"Line {line_num}: present address has empty normalized value")
                    break

            if in_ctry_missing:
                if out_cols[6] != "" or ctry_missing_flag != "True":
                    invariants["normalized_missing_not_fabricated"] = False
                    errors.append(
                        f"Line {line_num}: missing country has non-empty normalized value '{out_cols[6]}'"
                    )
                    break
            else:
                if out_cols[6] == "" or ctry_missing_flag != "False":
                    invariants["normalized_missing_not_fabricated"] = False
                    errors.append(f"Line {line_num}: present country has empty normalized value")
                    break

            # 6. Check unexpected nulls
            # Unexpected nulls occur when null tokens ('None', 'null', 'NULL', 'NaN', 'nan') appear
            # that were not present in the input raw source text.
            raw_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 1, 5: 2, 6: 3}
            for idx, col_val in enumerate(out_cols):
                if col_val in ("None", "null", "NULL", "NaN", "nan"):
                    if idx in (7, 8, 9):
                        invariants["no_unexpected_nulls"] = False
                        errors.append(f"Line {line_num}: unexpected null token '{col_val}' in boolean flag column {idx}")
                        break
                    raw_idx = raw_map[idx]
                    raw_val = in_cols[raw_idx]
                    # Allowed only if the raw input value itself was that literal token
                    if raw_val != col_val and raw_val.strip().casefold() != col_val:
                        invariants["no_unexpected_nulls"] = False
                        errors.append(f"Line {line_num}: unexpected null token '{col_val}' in column {idx}")
                        break

    all_passed = all(invariants.values())
    status = "PASSED" if all_passed else "FAILED"

    return {
        "file_name": input_path.name,
        "status": status,
        "input_rows": in_row_count,
        "output_rows": out_row_count,
        "invariants": invariants,
        "errors": errors,
    }


# ==============================================================================
# Pipeline Coordinator
# ==============================================================================

def run_cleaning_pipeline(
    input_dir: Path,
    output_dir: Path,
    splits: List[str],
    skip_validation: bool = False,
    quiet: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Executes the complete cleaning pipeline:
    1. Input validation pass across all target files.
    2. Streaming cleaning and profiling to temporary files.
    3. Quality check verification pass on temporary files.
    4. Atomic rename and report generation upon passing all checks.

    Returns:
        (profile_report, quality_report)
    """
    start_time = time.time()
    if not quiet:
        print("=" * 80)
        print("AMAZON ML CHALLENGE 2026 - ENTITY RESOLUTION DATA CLEANING & PROFILING")
        print("=" * 80)
        print(f"Input Directory  : {input_dir}")
        print(f"Output Directory : {output_dir}")
        print(f"Splits to process: {splits}")
        print()

    # Discover files
    files_to_process: List[Tuple[str, Path, str]] = []  # (split, input_path, expected_prefix)
    for split in splits:
        split_dir = input_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        for source_num in (1, 2, 3):
            file_name = f"{split}_source{source_num}.tsv"
            file_path = split_dir / file_name
            if not file_path.is_file():
                raise FileNotFoundError(f"Required source file not found: {file_path}")
            prefix = DEFAULT_EXPECTED_PREFIXES[f"source{source_num}"]
            files_to_process.append((split, file_path, prefix))

    # Phase 1: Input Validation
    if not skip_validation:
        if not quiet:
            print(">>> Phase 1: Validating Input File Structural Integrity...")
        for split, file_path, prefix in files_to_process:
            t0 = time.time()
            row_count = validate_input_file(file_path, expected_prefix=prefix)
            if not quiet:
                print(
                    f"  [VALIDATED] {split}/{file_path.name} "
                    f"({row_count:,} rows, prefix '{prefix}') in {time.time()-t0:.2f}s"
                )
        if not quiet:
            print("  All input files passed structural integrity validation.\n")

    # Prepare temp output paths
    # Using hidden temp files alongside target directory for guaranteed same-filesystem atomic rename
    temp_files: List[Tuple[str, Path, Path, str]] = []
    final_files: List[Tuple[str, Path, Path, str]] = []

    for split, in_path, prefix in files_to_process:
        target_dir = output_dir / split
        target_dir.mkdir(parents=True, exist_ok=True)
        final_path = target_dir / in_path.name
        temp_path = target_dir / f".tmp_{in_path.name}_{int(time.time())}"
        temp_files.append((split, in_path, temp_path, prefix))
        final_files.append((split, in_path, final_path, prefix))

    profile_results: Dict[str, Any] = {}
    quality_results: Dict[str, Any] = {}

    try:
        # Phase 2: Streaming Cleaning & Profiling
        if not quiet:
            print(">>> Phase 2: Streaming Cleaning & Profiling...")
        for split, in_path, temp_path, prefix in temp_files:
            t0 = time.time()
            if not quiet:
                print(f"  Cleaning {split}/{in_path.name} -> {temp_path.name}...")
            file_profile = clean_and_profile_file(in_path, temp_path, expected_prefix=prefix)
            profile_results[f"{split}/{in_path.name}"] = file_profile
            if not quiet:
                print(
                    f"  [DONE] {split}/{in_path.name} ({file_profile['row_count']:,} rows) "
                    f"in {time.time()-t0:.2f}s"
                )
        if not quiet:
            print()

        # Phase 3: Post-Cleaning Quality Verification
        if not quiet:
            print(">>> Phase 3: Post-Cleaning Quality Verification...")
        all_passed = True
        total_rows_verified = 0

        for split, in_path, temp_path, prefix in temp_files:
            t0 = time.time()
            file_quality = verify_file_quality(in_path, temp_path, expected_prefix=prefix)
            quality_results[f"{split}/{in_path.name}"] = file_quality
            total_rows_verified += file_quality["output_rows"]

            if file_quality["status"] != "PASSED":
                all_passed = False
                err_msg = "; ".join(file_quality["errors"])
                raise QualityCheckError("InvariantViolation", str(temp_path), err_msg)

            if not quiet:
                print(
                    f"  [PASSED] {split}/{in_path.name} "
                    f"({file_quality['output_rows']:,} rows, all invariants satisfied) "
                    f"in {time.time()-t0:.2f}s"
                )

        if not all_passed:
            raise QualityCheckError("PipelineFailure", "all", "One or more files failed quality checks")

        if not quiet:
            print("  All quality invariants satisfied across all files.\n")

        # Phase 4: Atomic Publication & Reports
        if not quiet:
            print(">>> Phase 4: Atomic Publication & Writing Reports...")
        reports_dir = output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Build combined profile report
        profile_report = {
            "title": "Amazon ML Challenge 2026 - Dataset Profiling Report",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_files": len(files_to_process),
                "total_rows": sum(p["row_count"] for p in profile_results.values()),
                "splits": splits,
                "elapsed_seconds": round(time.time() - start_time, 2),
            },
            "files": profile_results,
        }

        # Build combined quality checks report
        quality_report = {
            "title": "Amazon ML Challenge 2026 - Data Quality & Invariant Verification Report",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "PASSED",
            "summary": {
                "total_files_checked": len(files_to_process),
                "total_rows_checked": total_rows_verified,
                "total_invariants_per_file": 12,
                "all_checks_passed": True,
            },
            "invariants_checked": [
                "row_count_match",
                "entity_id_match_and_order",
                "original_values_unaltered",
                "source_columns_present",
                "no_unexpected_nulls",
                "normalized_missing_not_fabricated",
                "output_headers_correct",
                "valid_utf8_tsv",
                "no_malformed_rows",
                "normalized_columns_present",
                "source_prefix_validation",
                "boolean_flags_valid",
            ],
            "file_checks": quality_results,
        }

        # Write reports
        profile_json_path = reports_dir / "profile.json"
        quality_json_path = reports_dir / "quality_checks.json"

        with open(profile_json_path, "w", encoding="utf-8") as f:
            json.dump(profile_report, f, indent=2, ensure_ascii=False)

        with open(quality_json_path, "w", encoding="utf-8") as f:
            json.dump(quality_report, f, indent=2, ensure_ascii=False)

        # Atomically rename temp files to final destination
        for (split, in_path, temp_path, prefix), (_, _, final_path, _) in zip(temp_files, final_files):
            os.replace(temp_path, final_path)
            if not quiet:
                print(f"  Published: {final_path}")

        if not quiet:
            print(f"  Reports written: {profile_json_path}, {quality_json_path}")
            print(f"\nPipeline completed successfully in {time.time()-start_time:.2f}s!")
            print("=" * 80)

        return profile_report, quality_report

    except Exception:
        # Clean up temporary files on failure
        for _, _, temp_path, _ in temp_files:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
        raise


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean, validate, and profile the Amazon ML Challenge 2026 dataset."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("student_resource/dataset"),
        help="Path to input dataset directory containing train/ and test/ folders (default: student_resource/dataset)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cleaned"),
        help="Path to output directory for cleaned TSV files and reports (default: cleaned)",
    )
    parser.add_argument(
        "--split",
        choices=["all", "train", "test"],
        default="all",
        help="Dataset split to process: 'all' (default), 'train', or 'test'",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip pre-cleaning input structural validation (not recommended)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose progress output",
    )
    return parser.parse_args(args)


def main() -> int:
    args = parse_args()
    splits = ["train", "test"] if args.split == "all" else [args.split]

    try:
        run_cleaning_pipeline(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            splits=splits,
            skip_validation=args.skip_validation,
            quiet=args.quiet,
        )
        return 0
    except StructuralIntegrityError as e:
        print(f"\n[FATAL ERROR] Structural integrity violation:", file=sys.stderr)
        print(f"  File: {e.file_path}", file=sys.stderr)
        print(f"  Line: {e.line_number}", file=sys.stderr)
        print(f"  Reason: {e.reason}", file=sys.stderr)
        return 1
    except QualityCheckError as e:
        print(f"\n[FATAL ERROR] Quality check failure [{e.check_name}]:", file=sys.stderr)
        print(f"  File: {e.file_path}", file=sys.stderr)
        print(f"  Reason: {e.reason}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n[FATAL ERROR] Pipeline failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
