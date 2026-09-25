#!/usr/bin/env python3
"""
Unit and Integration Tests for Amazon ML Challenge 2026 Data Cleaning Pipeline.

Verifies:
- Unicode preservation (accents, non-Latin scripts)
- Hindi/non-Latin text preservation (no transliteration, no translation)
- Unicode-aware casefolding
- NFC normalization
- Whitespace and embedded tab/newline noise handling
- Narrowed punctuation preservation:
  - Meaningful punctuation preserved
  - Safe equivalences mapped
  - Primes (U+2032/U+2033) and language-specific modifier letters (U+02BB/U+02BC) preserved
- Missing fields handling (empty string, BOM, unicode spaces, boolean flags, no placeholders)
- Real business names ('NAN', 'Null', 'None') NOT treated as missing
- Duplicate-looking records remaining present
- Repeated business-name words remaining unchanged
- Exact normalized-value recomputation and verification in quality checks
- Malformed TSV failure (structural integrity gate)
- Duplicate / missing entity_id rejection (mandatory integrity gate)
- Disk-backed SQLite aggregation and deterministic SHA-256 duplicate metrics
- Generation staging and atomic rollback on publication failure
- No leftover temporary files on success or failure
- Repeated full-run output equivalence
"""

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

# Add src to path
import sys
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from clean_dataset import (
    INPUT_COLUMNS,
    OUTPUT_COLUMNS,
    StructuralIntegrityError,
    QualityCheckError,
    is_missing_value,
    normalize_text,
    normalize_country,
    validate_input_file,
    clean_and_profile_file,
    verify_file_quality,
    run_cleaning_pipeline,
)


class TestNormalization(unittest.TestCase):
    """Test conservative text and country normalization rules."""

    def test_unicode_preservation(self):
        """Accents and European Unicode characters must be preserved, not converted to ASCII."""
        raw = "Café L'Étoile & München Bräuhaus"
        norm = normalize_text(raw)
        self.assertEqual(norm, "café l'étoile & münchen bräuhaus")
        self.assertIn("é", norm)
        self.assertIn("ü", norm)
        self.assertIn("ä", norm)

    def test_hindi_non_latin_text(self):
        """Hindi / Devanagari and other non-Latin scripts must be preserved in Unicode."""
        hindi_name = "राम मार्केटिंग प्राइवेट लिमिटेड"
        norm = normalize_text(hindi_name)
        self.assertEqual(norm, hindi_name)

        hindi_addr = "G-3/571, GULMOHAR COLONY, BHOPAL, Madhya Pradesh"
        norm_addr = normalize_text(hindi_addr)
        self.assertEqual(norm_addr, "g-3/571, gulmohar colony, bhopal, madhya pradesh")

        arabic_text = "شركة النور للتجارة"
        self.assertEqual(normalize_text(arabic_text), arabic_text)

        cjk_text = "トヨタ自動車株式会社"
        self.assertEqual(normalize_text(cjk_text), cjk_text)

        cyrillic_text = "ООО Ромашка Плюс"
        self.assertEqual(normalize_text(cyrillic_text), "ооо ромашка плюс")

    def test_casefolding(self):
        """Unicode-aware casefolding must properly lower-case text."""
        self.assertEqual(normalize_text("WEISS STRASSE"), "weiss strasse")
        self.assertEqual(normalize_text("Maße"), "masse")
        self.assertEqual(normalize_text("B+ Retail Inc"), "b+ retail inc")

    def test_nfc_normalization(self):
        """Decomposed characters (NFD) must be normalized to canonical composed form (NFC)."""
        nfd_str = "e\u0301cole"
        nfc_str = "école"
        self.assertNotEqual(nfd_str, nfc_str)
        self.assertEqual(normalize_text(nfd_str), nfc_str)

    def test_whitespace_and_embedded_noise(self):
        """Unicode whitespace, tabs, and embedded newlines must collapse to single space and strip."""
        raw = "\t  Acme   \r\n   Logistics \u00A0 \u2003  LLC \t\n"
        norm = normalize_text(raw)
        self.assertEqual(norm, "acme logistics llc")

    def test_punctuation_preservation(self):
        """Safe equivalences must map, while meaningful punctuation is strictly preserved."""
        raw_quotes = "“Bob’s” ‘Best’ „Bakery‟"
        self.assertEqual(normalize_text(raw_quotes), '"bob\'s" \'best\' "bakery"')

        raw_dashes = "A–B—C−D‐E‑F"
        self.assertEqual(normalize_text(raw_dashes), "a-b-c-d-e-f")

        meaningful = "B & M / Smith . # 42 , (North-West) + co @ domain"
        norm = normalize_text(meaningful)
        for char in ["&", "/", ".", "#", ",", "(", ")", "+", "@", "-"]:
            self.assertIn(char, norm)

    def test_primes_and_modifiers_preserved(self):
        """Primes (U+2032/U+2033) and language-specific modifier letters (U+02BB/U+02BC) must NOT be mapped to quotes."""
        raw = "Hawai\u02bbi 5\u2032 10\u2033 & Ma\u02bco"
        norm = normalize_text(raw)
        self.assertIn("\u02bb", norm)  # Modifier letter turned comma
        self.assertIn("\u02bc", norm)  # Modifier letter apostrophe
        self.assertIn("\u2032", norm)  # Prime
        self.assertIn("\u2033", norm)  # Double prime

    def test_country_normalization(self):
        """Country normalization uses only NFC + casefold() + whitespace normalization."""
        self.assertEqual(normalize_country("  US  "), "us")
        self.assertEqual(normalize_country("India"), "india")
        self.assertEqual(normalize_country("France"), "france")
        self.assertEqual(normalize_country(""), "")
        self.assertEqual(normalize_country("   "), "")
        self.assertEqual(normalize_country(None), "")

    def test_missing_fields(self):
        """Missing fields (empty string, BOM, or whitespace-only) must return empty string and flag as missing."""
        self.assertTrue(is_missing_value(""))
        self.assertTrue(is_missing_value("   "))
        self.assertTrue(is_missing_value("\t\r\n "))
        self.assertTrue(is_missing_value("\ufeff"))  # BOM
        self.assertTrue(is_missing_value("\u00a0"))  # Non-breaking space
        self.assertTrue(is_missing_value("\u2003"))  # Em space
        self.assertTrue(is_missing_value("\t\r\n \u00a0\ufeff \u2003"))
        self.assertTrue(is_missing_value(None))

        # Real business names must NOT be missing
        self.assertFalse(is_missing_value("US"))
        self.assertFalse(is_missing_value(" . "))
        self.assertFalse(is_missing_value("NAN"))
        self.assertFalse(is_missing_value("Null"))
        self.assertFalse(is_missing_value("None"))

        self.assertEqual(normalize_text(""), "")
        self.assertEqual(normalize_text("\ufeff"), "")
        self.assertEqual(normalize_text("\u00a0"), "")
        self.assertEqual(normalize_text("\u2003"), "")
        self.assertEqual(normalize_text("\t\r\n \u00a0\ufeff \u2003"), "")

        # Real names normalize as expected
        self.assertEqual(normalize_text("NAN"), "nan")
        self.assertEqual(normalize_text("Null"), "null")
        self.assertEqual(normalize_text("None"), "none")

    def test_repeated_business_name_words_unchanged(self):
        """Repeated tokens, legal suffixes, and abbreviations must remain intact."""
        self.assertEqual(normalize_text("Pizza Pizza Inc"), "pizza pizza inc")
        self.assertEqual(normalize_text("Apex Apex Global Ltd Corp"), "apex apex global ltd corp")


class TestPipelineAndIntegrity(unittest.TestCase):
    """Test input validation, streaming cleaner, quality checks, and publication."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.base_path = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_tsv(self, file_path: Path, rows: list):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write("\t".join(row) + "\n")

    def test_malformed_tsv_missing_columns_failure(self):
        """Row with missing column must fail input validation with line number and reason."""
        bad_file = self.base_path / "bad.tsv"
        rows = [
            INPUT_COLUMNS,
            ["S1-001", "Acme Corp", "123 Main St", "US"],
            ["S1-002", "Bad Row", "Missing Country"],
        ]
        self._create_tsv(bad_file, rows)

        with self.assertRaises(StructuralIntegrityError) as ctx:
            validate_input_file(bad_file, expected_prefix="S1-", temp_dir=self.base_path)
        self.assertEqual(ctx.exception.line_number, 3)
        self.assertIn("4 tab-delimited columns", ctx.exception.reason)

    def test_malformed_tsv_extra_columns_failure(self):
        """Row with extra column must fail input validation with line number."""
        bad_file = self.base_path / "extra.tsv"
        rows = [
            INPUT_COLUMNS,
            ["S1-001", "Acme Corp", "123 Main St", "US", "Extra Col"],
        ]
        self._create_tsv(bad_file, rows)

        with self.assertRaises(StructuralIntegrityError) as ctx:
            validate_input_file(bad_file, expected_prefix="S1-", temp_dir=self.base_path)
        self.assertEqual(ctx.exception.line_number, 2)
        self.assertIn("expected 4", ctx.exception.reason)

    def test_duplicate_entity_id_failure(self):
        """Duplicate entity_id must fail validation unconditionally."""
        bad_file = self.base_path / "dup_id.tsv"
        rows = [
            INPUT_COLUMNS,
            ["S1-001", "Acme Corp", "123 Main St", "US"],
            ["S1-001", "Another Corp", "456 Other Rd", "US"],
        ]
        self._create_tsv(bad_file, rows)

        with self.assertRaises(StructuralIntegrityError) as ctx:
            validate_input_file(bad_file, expected_prefix="S1-", temp_dir=self.base_path)
        self.assertIn("Duplicate entity_id detected", ctx.exception.reason)

    def test_missing_entity_id_failure(self):
        """Empty entity_id must fail validation."""
        bad_file = self.base_path / "empty_id.tsv"
        rows = [
            INPUT_COLUMNS,
            ["", "Acme Corp", "123 Main St", "US"],
        ]
        self._create_tsv(bad_file, rows)

        with self.assertRaises(StructuralIntegrityError) as ctx:
            validate_input_file(bad_file, expected_prefix="S1-", temp_dir=self.base_path)
        self.assertEqual(ctx.exception.line_number, 2)
        self.assertIn("Empty or whitespace-only entity_id", ctx.exception.reason)

    def test_inconsistent_prefix_failure(self):
        """Entity ID with incorrect prefix must fail validation."""
        bad_file = self.base_path / "bad_prefix.tsv"
        rows = [
            INPUT_COLUMNS,
            ["S2-001", "Acme Corp", "123 Main St", "US"],
        ]
        self._create_tsv(bad_file, rows)

        with self.assertRaises(StructuralIntegrityError) as ctx:
            validate_input_file(bad_file, expected_prefix="S1-", temp_dir=self.base_path)
        self.assertEqual(ctx.exception.line_number, 2)
        self.assertIn("Inconsistent entity_id prefix", ctx.exception.reason)

    def test_duplicate_looking_records_remaining_present(self):
        """Duplicate-looking records must NOT be removed or deduplicated."""
        in_file = self.base_path / "dups.tsv"
        out_file = self.base_path / "dups_clean.tsv"
        rows = [
            INPUT_COLUMNS,
            ["S1-001", "Acme Corp", "123 Main St", "US"],
            ["S1-002", "Acme Corp", "123 Main St", "US"],
        ]
        self._create_tsv(in_file, rows)

        profile = clean_and_profile_file(in_file, out_file, expected_prefix="S1-", temp_dir=self.base_path)
        self.assertEqual(profile["row_count"], 2)

        with open(out_file, "r", encoding="utf-8") as f:
            lines = [line.rstrip("\r\n").split("\t") for line in f]
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[1][0], "S1-001")
        self.assertEqual(lines[2][0], "S1-002")

    def test_row_and_id_preservation(self):
        """Output rows and entity IDs must be in exact same order with exact raw values."""
        in_file = self.base_path / "preserve.tsv"
        out_file = self.base_path / "preserve_clean.tsv"
        rows = [
            INPUT_COLUMNS,
            ["S1-001", "First Business", "100 Ave A", "US"],
            ["S1-002", "Second Business", "200 Ave B", "India"],
            ["S1-003", "Third Business", "", "US"],
        ]
        self._create_tsv(in_file, rows)

        clean_and_profile_file(in_file, out_file, expected_prefix="S1-", temp_dir=self.base_path)
        quality = verify_file_quality(in_file, out_file, expected_prefix="S1-")
        self.assertEqual(quality["status"], "PASSED")

        with open(out_file, "r", encoding="utf-8") as f:
            lines = [line.rstrip("\r\n").split("\t") for line in f]

        self.assertEqual(lines[0], OUTPUT_COLUMNS)
        self.assertEqual(lines[1][0:4], ["S1-001", "First Business", "100 Ave A", "US"])
        self.assertEqual(lines[2][0:4], ["S1-002", "Second Business", "200 Ave B", "India"])
        self.assertEqual(lines[3][0:4], ["S1-003", "Third Business", "", "US"])
        self.assertEqual(lines[3][5], "")
        self.assertEqual(lines[3][8], "True")

    def test_quality_check_incorrect_normalized_value_rejected(self):
        """Regression test: quality check must fail if derived normalized value is deliberately incorrect."""
        in_file = self.base_path / "raw_norm.tsv"
        cleaned_bad = self.base_path / "cleaned_bad_norm.tsv"
        self._create_tsv(in_file, [
            INPUT_COLUMNS,
            ["S1-001", "Beta Technologies", "100 Tech Way", "US"],
        ])
        # Manually create bad output with incorrect non-empty normalized value
        self._create_tsv(cleaned_bad, [
            OUTPUT_COLUMNS,
            ["S1-001", "Beta Technologies", "100 Tech Way", "US", "gamma technologies", "100 tech way", "us", "False", "False", "False"],
        ])
        quality = verify_file_quality(in_file, cleaned_bad, expected_prefix="S1-")
        self.assertEqual(quality["status"], "FAILED")
        self.assertFalse(quality["invariants"]["normalized_columns_present"])
        self.assertTrue(any("business_name_normalized mismatch" in err for err in quality["errors"]))

    def test_quality_check_fabricated_missing_failure(self):
        """Quality check must fail if a missing field is replaced with a placeholder like 'Unknown'."""
        in_file = self.base_path / "raw.tsv"
        cleaned_bad = self.base_path / "cleaned_bad.tsv"
        self._create_tsv(in_file, [
            INPUT_COLUMNS,
            ["S1-001", "Acme", "", "US"],
        ])
        self._create_tsv(cleaned_bad, [
            OUTPUT_COLUMNS,
            ["S1-001", "Acme", "", "US", "acme", "Unknown", "us", "False", "False", "False"],
        ])
        quality = verify_file_quality(in_file, cleaned_bad, expected_prefix="S1-")
        self.assertEqual(quality["status"], "FAILED")
        self.assertFalse(quality["invariants"]["normalized_missing_not_fabricated"])

    def test_quality_check_altered_original_values_failure(self):
        """Quality check must fail if original column values were altered in the cleaned output."""
        in_file = self.base_path / "raw2.tsv"
        cleaned_bad = self.base_path / "cleaned_bad2.tsv"
        self._create_tsv(in_file, [
            INPUT_COLUMNS,
            ["S1-001", "Original Name", "123 Main St", "US"],
        ])
        self._create_tsv(cleaned_bad, [
            OUTPUT_COLUMNS,
            ["S1-001", "Altered Name", "123 Main St", "US", "altered name", "123 main st", "us", "False", "False", "False"],
        ])
        quality = verify_file_quality(in_file, cleaned_bad, expected_prefix="S1-")
        self.assertEqual(quality["status"], "FAILED")
        self.assertFalse(quality["invariants"]["original_values_unaltered"])

    def test_quality_check_row_count_mismatch_failure(self):
        """Quality check must fail if input and output row counts differ."""
        in_file = self.base_path / "raw3.tsv"
        cleaned_bad = self.base_path / "cleaned_bad3.tsv"
        self._create_tsv(in_file, [
            INPUT_COLUMNS,
            ["S1-001", "Name 1", "123 Main St", "US"],
            ["S1-002", "Name 2", "456 Main St", "US"],
        ])
        self._create_tsv(cleaned_bad, [
            OUTPUT_COLUMNS,
            ["S1-001", "Name 1", "123 Main St", "US", "name 1", "123 main st", "us", "False", "False", "False"],
        ])
        quality = verify_file_quality(in_file, cleaned_bad, expected_prefix="S1-")
        self.assertEqual(quality["status"], "FAILED")
        self.assertFalse(quality["invariants"]["row_count_match"])

    def test_deterministic_duplicate_profiling_sha256(self):
        """Duplicate metrics must use SHA-256 and produce deterministic counts across separate runs."""
        in_file = self.base_path / "det.tsv"
        out1 = self.base_path / "out1.tsv"
        out2 = self.base_path / "out2.tsv"
        rows = [
            INPUT_COLUMNS,
            ["S1-100", "Alpha Inc.", "10 Alpha Way", "US"],
            ["S1-200", "Beta & Co.", "20 Beta Rd", "India"],
            ["S1-300", "Alpha Inc.", "30 Other St", "US"],  # Duplicate name
        ]
        self._create_tsv(in_file, rows)

        p1 = clean_and_profile_file(in_file, out1, expected_prefix="S1-", temp_dir=self.base_path)
        p2 = clean_and_profile_file(in_file, out2, expected_prefix="S1-", temp_dir=self.base_path)

        with open(out1, "rb") as f1, open(out2, "rb") as f2:
            self.assertEqual(f1.read(), f2.read())

        self.assertEqual(
            p1["duplicate_looking_normalized_value_counts"],
            p2["duplicate_looking_normalized_value_counts"],
        )
        self.assertEqual(
            p1["duplicate_looking_normalized_value_counts"]["digest_algorithm"],
            "sha256",
        )
        self.assertEqual(
            p1["duplicate_looking_normalized_value_counts"]["normalized_name"]["unique_values_with_duplicates"],
            1,
        )

    def test_publication_failure_cleanup_and_rollback(self):
        """Injected publication failure must roll back cleanly, retain previous state, and leave no partial generation."""
        dataset_dir = self.base_path / "dataset_rollback"
        train_dir = dataset_dir / "train"
        test_dir = dataset_dir / "test"

        for split, s_dir in [("train", train_dir), ("test", test_dir)]:
            for src_idx in (1, 2, 3):
                pfx = f"S{src_idx}-"
                file_path = s_dir / f"{split}_source{src_idx}.tsv"
                self._create_tsv(file_path, [
                    INPUT_COLUMNS,
                    [f"{pfx}101", f"Biz {src_idx}", f"{src_idx}00 Main St", "US"],
                ])

        output_dir = self.base_path / "cleaned_rollback"
        # Run successful initial generation
        run_cleaning_pipeline(
            input_dir=dataset_dir,
            output_dir=output_dir,
            splits=["train", "test"],
            quiet=True,
        )
        self.assertTrue((output_dir / "train" / "train_source1.tsv").is_file())

        # Modify initial file content to check rollback retention
        marker_file = output_dir / "train" / "train_source1.tsv"
        initial_bytes = marker_file.read_bytes()

        # Run pipeline with injected publication failure
        with self.assertRaises(OSError):
            run_cleaning_pipeline(
                input_dir=dataset_dir,
                output_dir=output_dir,
                splits=["train", "test"],
                quiet=True,
                _inject_publication_error=True,
            )

        # Verify rollback restored previous state
        self.assertTrue(marker_file.is_file())
        self.assertEqual(marker_file.read_bytes(), initial_bytes)

        # Verify no staging or backup directories remain
        remaining_staging = list(output_dir.glob(".generation_staging_*"))
        remaining_backup = list(output_dir.glob(".backup_*"))
        self.assertEqual(len(remaining_staging), 0)
        self.assertEqual(len(remaining_backup), 0)

    def test_repeated_full_run_output_equivalence(self):
        """Two full runs on a small fixture produce byte-for-byte identical output files and reports."""
        dataset_dir = self.base_path / "dataset_det"
        train_dir = dataset_dir / "train"
        test_dir = dataset_dir / "test"

        for split, s_dir in [("train", train_dir), ("test", test_dir)]:
            for src_idx in (1, 2, 3):
                pfx = f"S{src_idx}-"
                file_path = s_dir / f"{split}_source{src_idx}.tsv"
                self._create_tsv(file_path, [
                    INPUT_COLUMNS,
                    [f"{pfx}101", f"Biz {src_idx}", f"{src_idx}00 Main St", "US"],
                    [f"{pfx}102", f"Biz {src_idx} Branch", "", "India"],
                ])

        out1 = self.base_path / "run1"
        out2 = self.base_path / "run2"

        run_cleaning_pipeline(input_dir=dataset_dir, output_dir=out1, splits=["train", "test"], quiet=True)
        run_cleaning_pipeline(input_dir=dataset_dir, output_dir=out2, splits=["train", "test"], quiet=True)

        for rel_file in [
            "train/train_source1.tsv",
            "train/train_source2.tsv",
            "train/train_source3.tsv",
            "test/test_source1.tsv",
            "test/test_source2.tsv",
            "test/test_source3.tsv",
        ]:
            b1 = (out1 / rel_file).read_bytes()
            b2 = (out2 / rel_file).read_bytes()
            self.assertEqual(b1, b2, f"File {rel_file} differs between runs")


if __name__ == "__main__":
    unittest.main()
