#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - Business Entity Resolution Pipeline
Stage 1: Conservative Data Cleaning, Validation, and Profiling

Core Principles:
1. Strict raw-data preservation:
   - Raw input files under student_resource/dataset are never modified.
   - No rows are deleted, deduplicated, reordered, filtered, merged, or altered.
   - Original parsed field values are retained in the first four output columns.
   - Output files are deterministic UTF-8 TSV serializations.
2. Conservative normalization applied ONLY to derived columns:
   - Unicode NFC normalization.
   - Non-Latin scripts (Hindi, Arabic, French accents, Cyrillic, etc.) preserved.
   - Unicode-aware casefold().
   - Safe punctuation equivalences only (curly quotes -> straight, unicode dashes -> hyphen).
   - Primes (U+2032/U+2033) and language-specific modifier letters remain untouched.
   - Meaningful punctuation preserved (&, -, ., ', /, @, +, #, commas, parens).
   - Whitespace runs and noise (including BOM/zero-width space) collapsed to single ASCII space; stripped.
   - Country normalized using only NFC + casefold() + whitespace normalization.
   - Missing fields flagged with boolean indicators and kept empty (no placeholders).
3. Structural integrity & quality gates:
   - Mandatory validation of UTF-8, headers, column counts, entity_id format & uniqueness.
   - Bounded in-memory streaming transformation with disk-backed SQLite aggregation
     for exact uniqueness, length distributions, and collision-resistant SHA-256 duplicate profiling.
   - Staged generation directory with rollback-safe multi-directory transaction publication.
   - Recomputes all normalized values during quality verification to guarantee 100% derivation accuracy.
   - Produces diagnostic profile.json, quality_checks.json, and manifest.json reports.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import sqlite3
import sys
import time
import unicodedata
from collections import Counter
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

# Narrowed, explicitly documented safe punctuation equivalences
# Primes (U+2032/U+2033) and modifier letters (U+02BB/U+02BC) are explicitly NOT mapped.
SAFE_PUNCTUATION_TRANSLATION = {
    # Curly single quotes -> straight single quote (')
    ord("‘"): "'",  # U+2018 LEFT SINGLE QUOTATION MARK
    ord("’"): "'",  # U+2019 RIGHT SINGLE QUOTATION MARK
    ord("‚"): "'",  # U+201A SINGLE LOW-9 QUOTATION MARK
    ord("‛"): "'",  # U+201B SINGLE HIGH-REVERSED-9 QUOTATION MARK

    # Curly double quotes -> straight double quote (")
    ord("“"): '"',  # U+201C LEFT DOUBLE QUOTATION MARK
    ord("”"): '"',  # U+201D RIGHT DOUBLE QUOTATION MARK
    ord("„"): '"',  # U+201E DOUBLE LOW-9 QUOTATION MARK
    ord("‟"): '"',  # U+201F DOUBLE HIGH-REVERSED-9 QUOTATION MARK

    # Documented Unicode dash variants -> ASCII hyphen (-)
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

# Regex matching missing values: empty or purely Unicode whitespace / BOM / zero-width space
RE_MISSING_WHITESPACE = re.compile(r"^[\s\ufeff\u200b]*$", flags=re.UNICODE)


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


class PublicationTransactionError(Exception):
    """Raised when publication transaction fails and triggers automatic rollback."""
    pass


# ==============================================================================
# Normalization Functions
# ==============================================================================

def is_missing_value(val: Optional[str]) -> bool:
    """
    Check if a source field value is missing.
    Empty strings, whitespace-only strings, BOM (\ufeff), and zero-width spaces
    are considered missing. Real business names such as 'NAN', 'Null', or 'None'
    are preserved and NOT considered missing.
    """
    if val is None or len(val) == 0:
        return True
    return bool(RE_MISSING_WHITESPACE.match(val))


def normalize_text(text: Optional[str]) -> str:
    """
    Conservative normalization applied ONLY to business_name and business_address.

    Rules applied:
    1. Check missingness: empty, whitespace-only, or BOM-only returns "".
    2. Unicode NFC normalization.
    3. Safe punctuation equivalences:
       - curly single quotes -> straight single quote (')
       - curly double quotes -> straight double quote (")
       - Unicode dash variants -> ASCII hyphen (-)
       - Unicode space variants & BOM -> ASCII space (' ')
    4. Unicode-aware casefold().
    5. Re-apply Unicode NFC normalization (ensures canonical composition after casefold).
    6. Treat Unicode whitespace, tabs, and embedded newlines as noise:
       replace whitespace runs with a single ASCII space and strip ends.
    7. Non-Latin scripts (Hindi, Arabic, Cyrillic, CJK, etc.) and accents are strictly preserved.
    8. Meaningful punctuation (&, -, ., ', /, @, +, #, commas, parens) is preserved.
    9. Primes (U+2032/U+2033) and language-specific modifier letters remain untouched.
    """
    if is_missing_value(text):
        return ""

    assert text is not None
    # 1. NFC normalization
    s = unicodedata.normalize("NFC", text)
    # 2. Safe punctuation translation
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
    - Missingness check
    - Unicode NFC
    - Unicode casefold()
    - Whitespace normalization

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
# Helper Functions for Character & Script Profiling and Chunked Hashing
# ==============================================================================

def sha256_file(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """
    Compute SHA-256 hex digest of a file in streaming 1MB chunks (strictly bounded memory).
    Never loads multi-gigabyte files into RAM.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


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
# Disk-Backed Profiler (Bounded-Memory, SQLite Length Counts, SHA-256)
# ==============================================================================

class DiskBackedProfiler:
    """
    Disk-backed profiler providing exact entity_id uniqueness validation,
    exact collision-resistant SHA-256 duplicate metrics, disk-backed length aggregation,
    and country frequency counts with strictly bounded in-memory footprint.

    Ephemeral SQLite tables with capped in-memory page cache (4MB) ensure that memory
    usage is strictly bounded regardless of dataset scale.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(db_path))
        self.cur = self.con.cursor()
        self.cur.execute("PRAGMA synchronous = OFF")
        self.cur.execute("PRAGMA journal_mode = OFF")
        self.cur.execute("PRAGMA cache_size = -4000")  # Capped at ~4MB RAM

        self.cur.execute("CREATE TABLE entity_ids (eid TEXT PRIMARY KEY)")
        self.cur.execute("CREATE TABLE name_digests (digest TEXT PRIMARY KEY, cnt INTEGER)")
        self.cur.execute("CREATE TABLE addr_digests (digest TEXT PRIMARY KEY, cnt INTEGER)")
        self.cur.execute("CREATE TABLE record_digests (digest TEXT PRIMARY KEY, cnt INTEGER)")
        self.cur.execute("CREATE TABLE countries (country TEXT PRIMARY KEY, cnt INTEGER)")
        self.cur.execute(
            "CREATE TABLE length_counts (col TEXT, length INTEGER, cnt INTEGER, PRIMARY KEY (col, length))"
        )

        self.batch_size = 50000
        self.eid_batch: List[Tuple[str]] = []
        self.name_batch: List[Tuple[str]] = []
        self.addr_batch: List[Tuple[str]] = []
        self.record_batch: List[Tuple[str]] = []
        self.country_batch: List[Tuple[str]] = []
        self.length_batch: List[Tuple[str, int]] = []

    def record_row(
        self,
        entity_id: str,
        name_norm: str,
        addr_norm: str,
        ctry_norm: str,
        raw_country: str,
    ) -> None:
        self.eid_batch.append((entity_id,))
        if raw_country:
            self.country_batch.append((raw_country,))

        if name_norm:
            d_name = hashlib.sha256(name_norm.encode("utf-8")).hexdigest()
            self.name_batch.append((d_name,))

        if addr_norm:
            d_addr = hashlib.sha256(addr_norm.encode("utf-8")).hexdigest()
            self.addr_batch.append((d_addr,))

        d_record = hashlib.sha256(f"{name_norm}\t{addr_norm}\t{ctry_norm}".encode("utf-8")).hexdigest()
        self.record_batch.append((d_record,))

        if len(self.eid_batch) >= self.batch_size:
            self.flush()

    def record_length(self, col: str, length: int) -> None:
        self.length_batch.append((col, length))

    def flush(self) -> None:
        if self.eid_batch:
            try:
                self.cur.executemany("INSERT INTO entity_ids VALUES (?)", self.eid_batch)
            except sqlite3.IntegrityError as e:
                for (eid,) in self.eid_batch:
                    self.cur.execute("SELECT eid FROM entity_ids WHERE eid = ?", (eid,))
                    if self.cur.fetchone():
                        raise StructuralIntegrityError(
                            str(self.db_path), 0, f"Duplicate entity_id detected: '{eid}'"
                        ) from e
                    self.cur.execute("INSERT INTO entity_ids VALUES (?)", (eid,))
            self.eid_batch.clear()

        if self.name_batch:
            self.cur.executemany(
                "INSERT INTO name_digests VALUES (?, 1) ON CONFLICT(digest) DO UPDATE SET cnt = cnt + 1",
                self.name_batch,
            )
            self.name_batch.clear()

        if self.addr_batch:
            self.cur.executemany(
                "INSERT INTO addr_digests VALUES (?, 1) ON CONFLICT(digest) DO UPDATE SET cnt = cnt + 1",
                self.addr_batch,
            )
            self.addr_batch.clear()

        if self.record_batch:
            self.cur.executemany(
                "INSERT INTO record_digests VALUES (?, 1) ON CONFLICT(digest) DO UPDATE SET cnt = cnt + 1",
                self.record_batch,
            )
            self.record_batch.clear()

        if self.country_batch:
            self.cur.executemany(
                "INSERT INTO countries VALUES (?, 1) ON CONFLICT(country) DO UPDATE SET cnt = cnt + 1",
                self.country_batch,
            )
            self.country_batch.clear()

        if self.length_batch:
            self.cur.executemany(
                "INSERT INTO length_counts VALUES (?, ?, 1) ON CONFLICT(col, length) DO UPDATE SET cnt = cnt + 1",
                self.length_batch,
            )
            self.length_batch.clear()

        self.con.commit()

    def compute_length_quantiles(
        self, col: str, total_count: int, percentiles: List[float]
    ) -> Dict[str, int]:
        """Compute exact quantiles directly from disk-backed SQLite length frequencies."""
        self.flush()
        if total_count == 0:
            return {}

        self.cur.execute(
            "SELECT length, cnt FROM length_counts WHERE col = ? ORDER BY length", (col,)
        )
        rows = self.cur.fetchall()
        targets = {p: max(1, math.ceil(total_count * p)) for p in percentiles}
        sorted_targets = sorted(targets.items(), key=lambda x: x[1])
        results = {}

        cum = 0
        t_idx = 0
        for length, cnt in rows:
            cum += cnt
            while t_idx < len(sorted_targets) and cum >= sorted_targets[t_idx][1]:
                p, _ = sorted_targets[t_idx]
                results[f"p{int(p * 100)}"] = length
                t_idx += 1

        return results

    def get_duplicate_metrics(self) -> Dict[str, Any]:
        self.flush()

        self.cur.execute("SELECT count(*), coalesce(sum(cnt), 0) FROM name_digests WHERE cnt > 1")
        name_unique, name_total = self.cur.fetchone()

        self.cur.execute("SELECT count(*), coalesce(sum(cnt), 0) FROM addr_digests WHERE cnt > 1")
        addr_unique, addr_total = self.cur.fetchone()

        self.cur.execute("SELECT count(*), coalesce(sum(cnt), 0) FROM record_digests WHERE cnt > 1")
        rec_unique, rec_total = self.cur.fetchone()

        return {
            "digest_algorithm": "sha256",
            "storage_backend": "disk_backed_sqlite",
            "normalized_name": {
                "unique_values_with_duplicates": name_unique,
                "total_duplicate_rows": name_total,
            },
            "normalized_address": {
                "unique_values_with_duplicates": addr_unique,
                "total_duplicate_rows": addr_total,
            },
            "full_normalized_record": {
                "unique_tuples_with_duplicates": rec_unique,
                "total_duplicate_rows": rec_total,
            },
        }

    def get_country_counts(self) -> Dict[str, int]:
        self.flush()
        self.cur.execute("SELECT country, cnt FROM countries")
        return dict(self.cur.fetchall())

    def close(self) -> None:
        try:
            self.con.close()
        except Exception:
            pass
        if self.db_path.exists():
            try:
                self.db_path.unlink()
            except OSError:
                pass


# ==============================================================================
# Bounded Length Profiler (No In-Memory Histogram)
# ==============================================================================

class LengthProfiler:
    """
    Maintains bounded deterministic summary statistics (min, max, count, sum)
    and bounded top-k shortest non-empty and longest string examples in strictly
    bounded O(1) in-memory storage.

    Exact length distributions are stored on disk in DiskBackedProfiler.
    """

    def __init__(self, max_examples: int = 5):
        self.count: int = 0
        self.min_len: int = 0
        self.max_len: int = 0
        self.sum_len: int = 0
        self.max_examples: int = max_examples
        self.shortest_examples: List[Tuple[int, str, str]] = []
        self.longest_examples: List[Tuple[int, str, str]] = []

    def update(self, val: str, entity_id: str) -> int:
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

        if length > 0:
            item = (length, val, entity_id)
            if len(self.shortest_examples) < self.max_examples:
                self.shortest_examples.append(item)
                self.shortest_examples.sort()
            elif item < self.shortest_examples[-1]:
                self.shortest_examples[-1] = item
                self.shortest_examples.sort()

        longest_item = (-length, val, entity_id)
        if len(self.longest_examples) < self.max_examples:
            self.longest_examples.append(longest_item)
            self.longest_examples.sort()
        elif longest_item < self.longest_examples[-1]:
            self.longest_examples[-1] = longest_item
            self.longest_examples.sort()

        return length

    def get_summary(self, quantiles: Dict[str, int]) -> Dict[str, Any]:
        if self.count == 0:
            return {"min": 0, "max": 0, "mean": 0.0, "quantiles": {}}

        mean = round(self.sum_len / self.count, 2)
        return {
            "min": self.min_len,
            "max": self.max_len,
            "mean": mean,
            "quantiles": quantiles,
        }

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
# Mandatory Input Validation Gate
# ==============================================================================

def validate_input_file(
    file_path: Path,
    expected_prefix: str = "",
    temp_dir: Optional[Path] = None,
) -> int:
    """
    Validate raw input file integrity before writing any cleaned output.
    Uses a disk-backed SQLite index to check entity_id uniqueness with strictly
    bounded memory.

    Non-negotiable checks:
    - UTF-8 readability
    - Exact expected header: entity_id\\tbusiness_name\\tbusiness_address\\tcountry
    - Exactly 4 tab-separated columns per row
    - Non-empty entity_id
    - entity_id prefix consistency
    - Unique entity_id values (disk-backed SQLite table)
    - No malformed rows

    Raises:
        StructuralIntegrityError if any violation is found.
    Returns:
        Total row count (excluding header).
    """
    if not file_path.is_file():
        raise StructuralIntegrityError(str(file_path), 0, "Input file does not exist")

    scratch_dir = temp_dir or file_path.parent
    scratch_dir.mkdir(parents=True, exist_ok=True)
    db_file = scratch_dir / f".val_{file_path.name}_{int(time.time()*1000)}_{os.getpid()}.db"
    con = sqlite3.connect(str(db_file))
    cur = con.cursor()
    cur.execute("PRAGMA synchronous = OFF")
    cur.execute("PRAGMA journal_mode = OFF")
    cur.execute("PRAGMA cache_size = -4000")
    cur.execute("CREATE TABLE eids (eid TEXT PRIMARY KEY)")

    batch: List[Tuple[str]] = []
    line_number = 0

    try:
        with open(file_path, "r", encoding="utf-8") as f:
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

                batch.append((entity_id,))
                if len(batch) >= 50000:
                    try:
                        cur.executemany("INSERT INTO eids VALUES (?)", batch)
                        con.commit()
                        batch.clear()
                    except sqlite3.IntegrityError:
                        for (eid,) in batch:
                            cur.execute("SELECT eid FROM eids WHERE eid = ?", (eid,))
                            if cur.fetchone():
                                raise StructuralIntegrityError(
                                    str(file_path),
                                    line_number,
                                    f"Duplicate entity_id detected: '{eid}'",
                                )
                            cur.execute("INSERT INTO eids VALUES (?)", (eid,))
                        con.commit()
                        batch.clear()

            if batch:
                try:
                    cur.executemany("INSERT INTO eids VALUES (?)", batch)
                    con.commit()
                    batch.clear()
                except sqlite3.IntegrityError:
                    for (eid,) in batch:
                        cur.execute("SELECT eid FROM eids WHERE eid = ?", (eid,))
                        if cur.fetchone():
                            raise StructuralIntegrityError(
                                str(file_path),
                                line_number,
                                f"Duplicate entity_id detected: '{eid}'",
                            )
                        cur.execute("INSERT INTO eids VALUES (?)", (eid,))
                    con.commit()
                    batch.clear()

    except UnicodeDecodeError as e:
        raise StructuralIntegrityError(
            str(file_path), line_number, f"UTF-8 decode failure: {e}"
        ) from e
    finally:
        con.close()
        if db_file.exists():
            try:
                db_file.unlink()
            except OSError:
                pass

    total_rows = line_number - 1
    return total_rows


# ==============================================================================
# Streaming Cleaner and Profiler
# ==============================================================================

def clean_and_profile_file(
    input_path: Path,
    staged_output_path: Path,
    expected_prefix: str = "",
    temp_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Streams raw input file, normalizes derived fields, writes cleaned output
    to staged_output_path, and computes comprehensive profiling diagnostics
    via DiskBackedProfiler.

    Memory bounded:
    - Processes row-by-row with streaming I/O.
    - Aggregates duplicate metrics, entity IDs, and text length frequencies via disk-backed SQLite.
    - Memory footprint is strictly bounded O(1).
    """
    staged_output_path.parent.mkdir(parents=True, exist_ok=True)
    scratch_dir = temp_dir or staged_output_path.parent
    db_path = scratch_dir / f".prof_{input_path.name}_{int(time.time()*1000)}_{os.getpid()}.db"
    profiler = DiskBackedProfiler(db_path)

    row_count = 0
    missing_counts = {col: 0 for col in INPUT_COLUMNS[1:]}
    empty_string_counts = {col: 0 for col in INPUT_COLUMNS[1:]}
    non_ascii_counts = {col: 0 for col in INPUT_COLUMNS[1:]}
    script_counts: Dict[str, Counter[str]] = {col: Counter() for col in INPUT_COLUMNS[1:]}
    suspicious_symbols: Dict[str, Counter[str]] = {col: Counter() for col in INPUT_COLUMNS[1:]}

    length_profilers = {
        "business_name": LengthProfiler(),
        "business_address": LengthProfiler(),
        "country": LengthProfiler(),
        "business_name_normalized": LengthProfiler(),
        "business_address_normalized": LengthProfiler(),
        "country_normalized": LengthProfiler(),
    }

    line_num = 0

    try:
        with open(input_path, "r", encoding="utf-8") as in_f, \
             open(staged_output_path, "w", encoding="utf-8", newline="\n") as out_f:

            header_line = in_f.readline()
            line_num = 1
            raw_header = header_line.rstrip("\r\n").split("\t")
            if raw_header != INPUT_COLUMNS:
                raise StructuralIntegrityError(
                    str(input_path), 1, f"Expected header {INPUT_COLUMNS}, got {raw_header}"
                )

            out_header = "\t".join(OUTPUT_COLUMNS) + "\n"
            out_f.write(out_header)

            for line in in_f:
                line_num += 1
                cols = line.rstrip("\r\n").split("\t")
                if len(cols) != 4:
                    raise StructuralIntegrityError(
                        str(input_path), line_num, f"Malformed row with {len(cols)} columns"
                    )

                entity_id, b_name, b_addr, country = cols
                row_count += 1

                if expected_prefix and not entity_id.startswith(expected_prefix):
                    raise StructuralIntegrityError(
                        str(input_path),
                        line_num,
                        f"Prefix mismatch for entity_id '{entity_id}', expected '{expected_prefix}'",
                    )

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

                name_norm = normalize_text(b_name)
                addr_norm = normalize_text(b_addr)
                ctry_norm = normalize_country(country)

                len_b_name = length_profilers["business_name"].update(b_name, entity_id)
                len_b_addr = length_profilers["business_address"].update(b_addr, entity_id)
                len_country = length_profilers["country"].update(country, entity_id)
                len_name_norm = length_profilers["business_name_normalized"].update(name_norm, entity_id)
                len_addr_norm = length_profilers["business_address_normalized"].update(addr_norm, entity_id)
                len_ctry_norm = length_profilers["country_normalized"].update(ctry_norm, entity_id)

                profiler.record_length("business_name", len_b_name)
                profiler.record_length("business_address", len_b_addr)
                profiler.record_length("country", len_country)
                profiler.record_length("business_name_normalized", len_name_norm)
                profiler.record_length("business_address_normalized", len_addr_norm)
                profiler.record_length("country_normalized", len_ctry_norm)

                profiler.record_row(entity_id, name_norm, addr_norm, ctry_norm, country)

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

            out_f.flush()
            os.fsync(out_f.fileno())

        # Compute quantiles from disk-backed SQLite length counts
        percentiles = [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        text_length_summaries = {}
        for col, lp in length_profilers.items():
            quantiles = profiler.compute_length_quantiles(col, lp.count, percentiles)
            text_length_summaries[col] = lp.get_summary(quantiles)

        dup_metrics = profiler.get_duplicate_metrics()
        country_counts = profiler.get_country_counts()

    finally:
        profiler.close()

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
        "text_length_summaries": text_length_summaries,
        "text_length_examples": {
            col: lp.get_examples() for col, lp in length_profilers.items()
        },
        "duplicate_looking_normalized_value_counts": dup_metrics,
        "duplicate_entity_id_findings": {
            "status": "none_detected_during_mandatory_validation"
        },
        "malformed_row_findings": {
            "status": "none_detected_during_mandatory_validation"
        },
        "country_frequency_counts": country_counts,
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
    10. normalized_columns_present: normalized columns exist and contain the exact expected
        recomputed derived values (recomputes normalize_text and normalize_country)
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

            # 6. Recompute and verify exact normalized values
            expected_name_norm = normalize_text(in_cols[1])
            expected_addr_norm = normalize_text(in_cols[2])
            expected_ctry_norm = normalize_country(in_cols[3])

            if out_cols[4] != expected_name_norm:
                invariants["normalized_columns_present"] = False
                errors.append(
                    f"Line {line_num}: business_name_normalized mismatch: expected '{expected_name_norm}', found '{out_cols[4]}'"
                )
                break

            if out_cols[5] != expected_addr_norm:
                invariants["normalized_columns_present"] = False
                errors.append(
                    f"Line {line_num}: business_address_normalized mismatch: expected '{expected_addr_norm}', found '{out_cols[5]}'"
                )
                break

            if out_cols[6] != expected_ctry_norm:
                invariants["normalized_columns_present"] = False
                errors.append(
                    f"Line {line_num}: country_normalized mismatch: expected '{expected_ctry_norm}', found '{out_cols[6]}'"
                )
                break

            # 7. Check unexpected nulls
            raw_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 1, 5: 2, 6: 3}
            for idx, col_val in enumerate(out_cols):
                if col_val in ("None", "null", "NULL", "NaN", "nan"):
                    if idx in (7, 8, 9):
                        invariants["no_unexpected_nulls"] = False
                        errors.append(f"Line {line_num}: unexpected null token '{col_val}' in boolean flag column {idx}")
                        break
                    raw_idx = raw_map[idx]
                    raw_val = in_cols[raw_idx]
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
# Pipeline Coordinator with Generation Staging & Rollback-Safe Transaction
# ==============================================================================

def run_cleaning_pipeline(
    input_dir: Path,
    output_dir: Path,
    splits: List[str],
    quiet: bool = False,
    _inject_failure_point: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Executes the complete cleaning pipeline:
    1. Mandatory structural integrity validation pass across all target files.
    2. Streaming cleaning and profiling to staged generation directory.
    3. Post-cleaning quality verification pass recomputing all derived values.
    4. Rollback-safe multi-directory publication transaction.

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

    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover target files
    files_to_process: List[Tuple[str, Path, str]] = []
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

    # Phase 1: Mandatory Structural Integrity Validation
    if not quiet:
        print(">>> Phase 1: Validating Input File Structural Integrity (Mandatory)...")
    for split, file_path, prefix in files_to_process:
        t0 = time.time()
        row_count = validate_input_file(file_path, expected_prefix=prefix, temp_dir=output_dir)
        if not quiet:
            print(
                f"  [VALIDATED] {split}/{file_path.name} "
                f"({row_count:,} rows, prefix '{prefix}') in {time.time()-t0:.2f}s"
            )
    if not quiet:
        print("  All input files passed structural integrity validation.\n")

    # Phase 2 & 3: Generation Staging Directory
    staging_dir = output_dir / f".generation_staging_{int(time.time()*1000)}_{os.getpid()}"
    staging_dir.mkdir(parents=True, exist_ok=True)

    staged_files: List[Tuple[str, Path, Path, str]] = []
    for split, in_path, prefix in files_to_process:
        split_stage_dir = staging_dir / split
        split_stage_dir.mkdir(parents=True, exist_ok=True)
        staged_path = split_stage_dir / in_path.name
        staged_files.append((split, in_path, staged_path, prefix))

    profile_results: Dict[str, Any] = {}
    quality_results: Dict[str, Any] = {}

    try:
        # Phase 2: Streaming Cleaning & Disk-Backed Profiling into Staging
        if not quiet:
            print(">>> Phase 2: Streaming Cleaning & Profiling into Staging Directory...")
        for split, in_path, staged_path, prefix in staged_files:
            t0 = time.time()
            if not quiet:
                print(f"  Cleaning {split}/{in_path.name} -> {staged_path.relative_to(output_dir)}...")
            file_profile = clean_and_profile_file(
                in_path, staged_path, expected_prefix=prefix, temp_dir=staging_dir
            )
            profile_results[f"{split}/{in_path.name}"] = file_profile
            if not quiet:
                print(
                    f"  [DONE] {split}/{in_path.name} ({file_profile['row_count']:,} rows) "
                    f"in {time.time()-t0:.2f}s"
                )
        if not quiet:
            print()

        # Phase 3: Post-Cleaning Quality Verification on Staged Files
        if not quiet:
            print(">>> Phase 3: Post-Cleaning Quality Verification (Recomputing Normalized Values)...")
        all_passed = True
        total_rows_verified = 0

        for split, in_path, staged_path, prefix in staged_files:
            t0 = time.time()
            file_quality = verify_file_quality(in_path, staged_path, expected_prefix=prefix)
            quality_results[f"{split}/{in_path.name}"] = file_quality
            total_rows_verified += file_quality["output_rows"]

            if file_quality["status"] != "PASSED":
                all_passed = False
                err_msg = "; ".join(file_quality["errors"])
                raise QualityCheckError("InvariantViolation", str(staged_path), err_msg)

            if not quiet:
                print(
                    f"  [PASSED] {split}/{in_path.name} "
                    f"({file_quality['output_rows']:,} rows, all 12 invariants satisfied) "
                    f"in {time.time()-t0:.2f}s"
                )

        if not all_passed:
            raise QualityCheckError("PipelineFailure", "all", "One or more files failed quality checks")

        if not quiet:
            print("  All quality invariants satisfied across all staged files.\n")

        # Phase 4: Stage Reports & Manifest
        staged_reports_dir = staging_dir / "reports"
        staged_reports_dir.mkdir(parents=True, exist_ok=True)

        profile_report = {
            "title": "Amazon ML Challenge 2026 - Dataset Profiling Report",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_files": len(files_to_process),
                "total_rows": sum(p["row_count"] for p in profile_results.values()),
                "splits": splits,
                "elapsed_seconds": round(time.time() - start_time, 2),
                "memory_model": "streaming_row_transformation_with_disk_backed_sqlite_aggregation",
            },
            "files": profile_results,
        }

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

        # Chunked SHA-256 calculation for manifest (never loads multi-GB file into RAM)
        manifest = {
            "generation_id": staging_dir.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "splits": splits,
            "files": {
                f"{split}/{in_p.name}": {
                    "rows": profile_results[f"{split}/{in_p.name}"]["row_count"],
                    "sha256": sha256_file(staged_p),
                }
                for split, in_p, staged_p, _ in staged_files
            },
        }

        staged_prof_path = staged_reports_dir / "profile.json"
        staged_qual_path = staged_reports_dir / "quality_checks.json"
        staged_man_path = staged_reports_dir / "manifest.json"

        with open(staged_prof_path, "w", encoding="utf-8") as f:
            json.dump(profile_report, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        with open(staged_qual_path, "w", encoding="utf-8") as f:
            json.dump(quality_report, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        with open(staged_man_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        # Phase 5: Rollback-Safe Multi-Directory Publication Transaction
        if not quiet:
            print(">>> Phase 4: Rollback-Safe Multi-Directory Publication Transaction...")

        backup_dir = output_dir / f".backup_{int(time.time()*1000)}_{os.getpid()}"
        published_items = ["train", "test", "reports"]
        backed_up: List[Tuple[Path, Path]] = []
        moved_to_dest: List[Tuple[Path, Path]] = []

        try:
            # Step 1: Back up existing destination directories
            for item in published_items:
                target_path = output_dir / item
                if target_path.exists():
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    dest_backup = backup_dir / item
                    shutil.move(str(target_path), str(dest_backup))
                    backed_up.append((dest_backup, target_path))

                    if _inject_failure_point == "after_one_backup" and len(backed_up) == 1:
                        raise PublicationTransactionError("Injected failure after one destination directory backup")

            # Step 2: Publish staged directories to destination
            for item in published_items:
                src_path = staging_dir / item
                if src_path.exists():
                    target_path = output_dir / item
                    shutil.move(str(src_path), str(target_path))
                    moved_to_dest.append((src_path, target_path))

                    if _inject_failure_point == "after_one_publish" and len(moved_to_dest) == 1:
                        raise PublicationTransactionError("Injected failure after one staged directory published")
                    if _inject_failure_point == "after_two_publish" and len(moved_to_dest) == 2:
                        raise PublicationTransactionError("Injected failure after two staged directories published")

            # Step 3: Transaction committed successfully -> clean up backup
            if backup_dir.exists():
                shutil.rmtree(backup_dir, ignore_errors=True)

        except Exception as pub_err:
            # Transaction Rollback
            if not quiet:
                print(f"  [ROLLBACK] Publication error: {pub_err}. Restoring previous state...", file=sys.stderr)
            for _, target_path in moved_to_dest:
                if target_path.exists():
                    shutil.rmtree(target_path, ignore_errors=True)

            for dest_backup, target_path in backed_up:
                if dest_backup.exists():
                    shutil.move(str(dest_backup), str(target_path))

            if backup_dir.exists():
                shutil.rmtree(backup_dir, ignore_errors=True)

            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)

            raise pub_err

        # Clean up staging directory
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)

        if not quiet:
            print(f"  Published: {output_dir / 'train'}")
            print(f"  Published: {output_dir / 'test'}")
            print(f"  Published: {output_dir / 'reports' / 'profile.json'}")
            print(f"  Published: {output_dir / 'reports' / 'quality_checks.json'}")
            print(f"  Published: {output_dir / 'reports' / 'manifest.json'}")
            print(f"\nPipeline completed successfully in {time.time()-start_time:.2f}s!")
            print("=" * 80)

        return profile_report, quality_report

    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
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
    except PublicationTransactionError as e:
        print(f"\n[FATAL ERROR] Publication transaction failure (rolled back): {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n[FATAL ERROR] Pipeline failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
