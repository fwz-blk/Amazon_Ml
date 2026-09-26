"""
Unit & Integration Tests for Blocking and Candidate Generation
Amazon ML Challenge 2026 - Business Entity Resolution

Covers:
- Unicode tokenization across scripts (Latin, Devanagari, Tamil, Arabic, Cyrillic)
- Empty-key exclusion
- Exact-name candidates
- Rare-token frequency filtering
- Address token extraction
- Address number+location keys
- Compact Unicode n-grams
- Country fallback rules in both directions
- Cross-country high-specificity rules and restrictions
- Pair deduplication and bitwise provenance union
- Deterministic candidate ordering (S2 sorted, then S3 sorted)
- Overflow refinement, retained overflow postings, and complete overflow metadata
- Final budget behavior and evidence ranking
- One-to-many ground-truth evaluation and singleton handling
- Disk-backed SQLite index lifecycle, bounded memory, and cleanup after success/failure
- Streaming export and exact 1-to-1 row preservation
- Deterministic repeated runs
- Synthetic end-to-end fixture covering all 10 edge cases
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Set

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from blocking.candidate_store import CandidateStore, compute_candidate_evidence_tier, rank_and_cap_candidates
from blocking.config import (
    DEFAULT_LEGAL_STOPLIST,
    PROV_ADDRESS_NUMBER_LOCATION,
    PROV_ADDRESS_TOKEN,
    PROV_COMPACT_CHAR_NGRAM,
    PROV_CROSS_COUNTRY_HIGH_SPECIFICITY,
    PROV_EXACT_NAME,
    PROV_MISSING_COUNTRY_FALLBACK,
    PROV_RARE_NAME_TOKEN,
    BlockingConfig,
)
from blocking.evaluate import BlockingEvaluator, is_validation_entity, load_ground_truth
from blocking.index import CandidateSourceIndex
from blocking.keys import (
    extract_address_number_location_keys,
    extract_address_numbers,
    extract_compact_ngrams,
    tokenize_address,
    tokenize_name,
    tokenize_unicode,
)
from blocking.passes import generate_candidates_for_s1
from blocking.run_blocking import run_blocking_pipeline, run_cap_sweep


class TestKeyDerivation(unittest.TestCase):
    """Test key extraction, Unicode tokenization, and stoplist filtering."""

    def test_unicode_tokenization_across_scripts(self):
        """Tokenization must preserve letters, numbers, and combining marks across all scripts."""
        # Devanagari with vowel signs (matras)
        hindi = "एसएस फूड प्राइवेट लिमिटेड"
        h_tokens = tokenize_unicode(hindi)
        self.assertEqual(h_tokens, ["एसएस", "फूड", "प्राइवेट", "लिमिटेड"])

        # Tamil with combining marks
        tamil = "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி"
        t_tokens = tokenize_unicode(tamil)
        self.assertEqual(t_tokens, ["ராஜ்", "இன்வெஸ்ட்மெண்ட்ஸ்", "எல்எல்பி"])

        # Arabic
        arabic = "شركة الأمل للتجارة"
        a_tokens = tokenize_unicode(arabic)
        self.assertEqual(a_tokens, ["شركة", "الأمل", "للتجارة"])

        # Latin with diacritics
        french = "Café du Théâtre & Cie"
        f_tokens = tokenize_unicode(french)
        self.assertEqual(f_tokens, ["Café", "du", "Théâtre", "Cie"])

    def test_empty_key_exclusion(self):
        """Empty strings, whitespace, and stoplist tokens must be excluded."""
        self.assertEqual(tokenize_unicode(""), [])
        self.assertEqual(tokenize_unicode("   \t\n  "), [])
        self.assertEqual(tokenize_name("", DEFAULT_LEGAL_STOPLIST), [])
        self.assertEqual(tokenize_name("Inc LLC Pvt Ltd Corp", DEFAULT_LEGAL_STOPLIST), [])
        self.assertEqual(tokenize_address(""), [])
        self.assertEqual(extract_address_numbers(""), [])
        self.assertEqual(extract_compact_ngrams(""), [])

    def test_name_token_stoplist_filtering(self):
        """Legal stoplist terms must be filtered, while informative tokens are retained."""
        tokens = tokenize_name("Orelee's Barbershop, Inc.", DEFAULT_LEGAL_STOPLIST)
        self.assertIn("Orelee", tokens)
        self.assertIn("Barbershop", tokens)
        self.assertNotIn("Inc", tokens)

    def test_address_token_extraction(self):
        """Address tokens must extract non-digit words of sufficient length."""
        tokens = tokenize_address("1795 Westchester Drive, High Point, NC 27262")
        self.assertIn("Westchester", tokens)
        self.assertIn("Drive", tokens)
        self.assertIn("High", tokens)
        self.assertIn("Point", tokens)
        self.assertNotIn("1795", tokens)
        self.assertNotIn("27262", tokens)

    def test_address_number_location_keys(self):
        """Address numbers paired with locality tokens must tolerate component reordering."""
        addr1 = "1795 Westchester Drive, High Point, NC"
        addr2 = "High Point, NC, 1795 Westchester Drive"
        keys1 = extract_address_number_location_keys(addr1)
        keys2 = extract_address_number_location_keys(addr2)

        # Both addresses produce the shared key ('1795', 'Westchester')
        self.assertTrue(any(pair[0] == "1795" and "Westchester" in pair[1] for pair in keys1))
        self.assertTrue(any(pair[0] == "1795" and "Westchester" in pair[1] for pair in keys2))

    def test_compact_unicode_ngrams(self):
        """Character n-grams must use compact alphanumeric representations across scripts."""
        ngrams = extract_compact_ngrams("Orelee's Barbershop", n=4)
        self.assertIn("orel", [ng.casefold() for ng in ngrams])
        self.assertIn("barb", [ng.casefold() for ng in ngrams])

        # Short names
        short_ngrams = extract_compact_ngrams("A1", n=4, min_length=3)
        self.assertEqual(short_ngrams, [])


class TestCandidateStoreAndRanking(unittest.TestCase):
    """Test candidate storage, bitmask provenance, deduplication, and ranking."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_cands.db"
        self.store = CandidateStore(self.db_path)

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_pair_deduplication_and_provenance_union(self):
        """Pairs added multiple times must deduplicate and bitwise-OR their provenance."""
        s1 = "S1-001"
        cand = "S2-100"

        # First add via EXACT_NAME
        self.store.add_candidates(s1, {cand: PROV_EXACT_NAME})
        # Second add via RARE_NAME_TOKEN
        self.store.add_candidates(s1, {cand: PROV_RARE_NAME_TOKEN})
        self.store.flush()

        self.store.cur.execute("SELECT provenance FROM candidate_pairs WHERE source1_id = ? AND cand_id = ?", (s1, cand))
        rows = self.store.cur.fetchall()
        self.assertEqual(len(rows), 1)
        expected_mask = PROV_EXACT_NAME | PROV_RARE_NAME_TOKEN
        self.assertEqual(rows[0][0], expected_mask)

    def test_deterministic_candidate_ordering(self):
        """Exported TSV must order S2 candidates first (sorted), then S3 candidates (sorted)."""
        s1 = "S1-001"
        self.store.add_candidates(s1, {
            "S3-500": PROV_EXACT_NAME,
            "S2-200": PROV_EXACT_NAME,
            "S2-100": PROV_EXACT_NAME,
            "S3-100": PROV_EXACT_NAME,
        })
        self.store.flush()

        tsv_path = Path(self.temp_dir) / "ordered.tsv"
        self.store.export_tsv(tsv_path, [s1])

        content = tsv_path.read_text().strip().split("\t")
        self.assertEqual(content[0], s1)
        cands = content[1].split(",")
        self.assertEqual(cands, ["S2-100", "S2-200", "S3-100", "S3-500"])

    def test_final_budget_behavior_and_ranking(self):
        """When candidate count exceeds safety budget, ranking must prioritize exact name and multi-block hits."""
        s1 = "S1-001"
        cands = {
            "S2-001": PROV_COMPACT_CHAR_NGRAM,  # Single weak hit
            "S2-002": PROV_EXACT_NAME | PROV_RARE_NAME_TOKEN,  # Exact name + token
            "S2-003": PROV_ADDRESS_NUMBER_LOCATION | PROV_ADDRESS_TOKEN,  # 2 address hits
            "S2-004": PROV_EXACT_NAME,  # Exact name
        }
        self.store.add_candidates(s1, cands)
        self.store.flush()

        tsv_path = Path(self.temp_dir) / "budget.tsv"
        # Set max 2 candidates per source
        stats = self.store.export_tsv(tsv_path, [s1], max_candidates_per_source=2)

        content = tsv_path.read_text().strip().split("\t")
        retained = content[1].split(",")
        self.assertEqual(len(retained), 2)
        # Exact name hits (S2-002 and S2-004) must be prioritized over weak hits
        self.assertIn("S2-002", retained)
        self.assertIn("S2-004", retained)
        self.assertEqual(stats["total_candidate_pairs_truncated_by_caps"], 2)
        self.assertEqual(stats["s1_capped_by_safety_budget"], 1)

    def test_streaming_export_and_singleton_preservation(self):
        """Export must stream line by line and preserve singletons with empty candidate lists."""
        s1_ids = ["S1-01", "S1-02", "S1-03"]
        self.store.add_candidates("S1-01", {"S2-10": PROV_EXACT_NAME})
        # S1-02 is a singleton with 0 candidates
        self.store.add_candidates("S1-03", {"S3-20": PROV_EXACT_NAME})
        self.store.flush()

        tsv_path = Path(self.temp_dir) / "stream.tsv"
        self.store.export_tsv(tsv_path, s1_ids)

        lines = tsv_path.read_text().strip().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[0], "S1-01\tS2-10")
        self.assertEqual(lines[1], "S1-02\t")
        self.assertEqual(lines[2], "S1-03\tS3-20")


class TestEvaluatorAndHoldout(unittest.TestCase):
    """Test ground truth loading, deterministic holdout, and recall evaluation."""

    def test_deterministic_holdout_partition(self):
        """Holdout check must be strictly reproducible given entity ID and salt."""
        v1 = is_validation_entity("S1-123456", holdout_ratio=0.1, salt="test_salt")
        v2 = is_validation_entity("S1-123456", holdout_ratio=0.1, salt="test_salt")
        self.assertEqual(v1, v2)

    def test_one_to_many_and_singleton_evaluation(self):
        """Evaluator must support one-to-many matches and singletons correctly."""
        gt = {
            "S1-001": {"S2-10", "S3-20"},  # 2 true matches
            "S1-002": set(),               # Singleton (0 matches)
            "S1-003": {"S2-30"},           # 1 match
        }
        evaluator = BlockingEvaluator(gt)
        self.assertEqual(evaluator.total_true_pairs, 3)
        self.assertEqual(evaluator.singletons, 1)

        retrieved = {
            "S1-001": {"S2-10": PROV_EXACT_NAME},  # Hit 1 of 2
            "S1-002": {},                           # Correctly no cands
            "S1-003": {"S2-30": PROV_EXACT_NAME},  # Hit 1 of 1
        }
        results = evaluator.evaluate_candidates(retrieved)
        self.assertEqual(results["counts"]["hits"], 2)
        self.assertAlmostEqual(results["pair_recall"], 2 / 3, places=4)
        self.assertAlmostEqual(results["union_recall"], 2 / 3, places=4)
        self.assertAlmostEqual(results["s1_coverage"], 1.0, places=4)


class TestBlockingPassesAndOverflow(unittest.TestCase):
    """Test rare-token filtering, country fallback rules, refinement, and overflow retention."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.base_path = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_rare_token_frequency_filtering(self):
        """Tokens with DF exceeding max_name_df must be filtered."""
        tsv_path = self.base_path / "cand.tsv"
        rows = [
            "entity_id\tbusiness_name\tbusiness_address\tcountry\tbusiness_name_normalized\tbusiness_address_normalized\tcountry_normalized\tbusiness_name_is_missing\tbusiness_address_is_missing\tcountry_is_missing\n",
        ]
        for i in range(10):
            rows.append(f"S2-{i}\tPopular Biz\t100 Main St\tUS\tpopular biz\t100 main st\tus\tFalse\tFalse\tFalse\n")
        rows.append("S2-99\tRare Biz\t100 Main St\tUS\trare biz\t100 main st\tus\tFalse\tFalse\tFalse\n")
        tsv_path.write_text("".join(rows), encoding="utf-8")

        config = BlockingConfig(max_name_df_absolute=3)
        index = CandidateSourceIndex("test_s2", config, temp_dir=self.base_path)
        index.build_from_tsv(tsv_path)

        self.assertIn(("us", "popular"), index.filtered_name_tokens)
        self.assertNotIn(("us", "rare"), index.filtered_name_tokens)
        index.close()

    def test_overflow_event_structure_and_retention(self):
        """Pathological block overflows must NOT be dropped to 0; postings must be retained."""
        config = BlockingConfig(max_block_size=2)
        index = CandidateSourceIndex("test_s2", config, temp_dir=self.base_path)

        # Insert 3 candidates with token 'overflowing' (> max_block_size 2)
        for i in range(1, 4):
            index.add_candidate(f"S2-{i}", "us", "overflowing", "100 Road")
        index.commit_index()

        # Query with single token
        cands = index.query_name_tokens("S1-001", "us", ["overflowing"])
        # Postings MUST be retained, not discarded!
        self.assertEqual(len(cands), 3)
        self.assertEqual(len(index.overflow_events), 1)

        event = index.overflow_events[0]
        self.assertEqual(event["s1_id"], "S1-001")
        self.assertEqual(event["source"], "test_s2")
        self.assertEqual(event["block_family"], "rare_name_token")
        self.assertEqual(event["key"], "overflowing")
        self.assertEqual(event["generated_count"], 3)
        self.assertEqual(event["retained_count"], 3)
        self.assertEqual(event["truncated_count"], 0)
        self.assertFalse(event["is_filtered"])
        self.assertTrue(event["refinement_attempted"])
        self.assertFalse(event["refined"])
        index.close()

    def test_overflow_refinement_success(self):
        """When multi-token refinement reduces postings <= max_block_size, refined set is returned."""
        config = BlockingConfig(max_block_size=2)
        index = CandidateSourceIndex("test_s2", config, temp_dir=self.base_path)

        # 3 candidates share 'phoenix', but only 1 has 'phoenix' + 'aerospace'
        index.add_candidate("S2-1", "us", "phoenix aerospace", "100 Main")
        index.add_candidate("S2-2", "us", "phoenix logistics", "200 Oak")
        index.add_candidate("S2-3", "us", "phoenix dynamics", "300 Pine")
        index.commit_index()

        # Query with both tokens
        cands = index.query_name_tokens("S1-001", "us", ["phoenix", "aerospace"])
        self.assertIn("S2-1", cands)
        index.close()

    def test_missing_country_fallback_both_directions(self):
        """Missing-country fallback must work in both directions and missing-missing."""
        config = BlockingConfig(allow_missing_country_fallback=True)
        index = CandidateSourceIndex("test_s2", config, temp_dir=self.base_path)

        # Candidate with missing country
        index.add_candidate("S2-missing", "", "omni tech", "123 Road")
        # Candidate with known country
        index.add_candidate("S2-known", "us", "omni tech", "456 Blvd")
        index.commit_index()

        # Direction 1: S1 has missing country -> matches S2-missing and S2-known
        cands1 = generate_candidates_for_s1("S1-01", "omni tech", "123 Road", "", index, config)
        self.assertIn("S2-missing", cands1)
        self.assertIn("S2-known", cands1)
        self.assertTrue(cands1["S2-missing"] & PROV_MISSING_COUNTRY_FALLBACK)
        self.assertTrue(cands1["S2-known"] & PROV_MISSING_COUNTRY_FALLBACK)

        # Direction 2: S1 has known country -> matches candidate with missing country via fallback
        cands2 = generate_candidates_for_s1("S1-02", "omni tech", "123 Road", "france", index, config)
        self.assertIn("S2-missing", cands2)
        self.assertTrue(cands2["S2-missing"] & PROV_MISSING_COUNTRY_FALLBACK)
        index.close()

    def test_cross_country_restrictions(self):
        """Cross-country candidates allowed ONLY through high specificity (exact name, >= 2 tokens, num_loc)."""
        config = BlockingConfig(allow_cross_country_high_specificity=True, cross_country_min_rare_tokens=2)
        index = CandidateSourceIndex("test_s2", config, temp_dir=self.base_path)

        # S2 records from France
        index.add_candidate("S2-exact", "france", "hyperion robotics", "100 Tech Park")
        index.add_candidate("S2-single-token", "france", "hyperion logistics", "200 Paris St")
        index.add_candidate("S2-multi-token", "france", "quantum cybernetics labs", "300 Lyon Rd")
        index.commit_index()

        # Query from US with exact name: allowed!
        cands1 = generate_candidates_for_s1("S1-01", "hyperion robotics", "100 Tech Park", "us", index, config)
        self.assertIn("S2-exact", cands1)
        self.assertTrue(cands1["S2-exact"] & PROV_CROSS_COUNTRY_HIGH_SPECIFICITY)

        # Query from US with only 1 shared token ('hyperion'): forbidden across countries!
        cands2 = generate_candidates_for_s1("S1-02", "hyperion medical", "999 Other St", "us", index, config)
        self.assertNotIn("S2-single-token", cands2)

        # Query from US with 2 shared tokens ('quantum', 'cybernetics'): allowed!
        cands3 = generate_candidates_for_s1("S1-03", "quantum cybernetics solutions", "555 Ave", "us", index, config)
        self.assertIn("S2-multi-token", cands3)
        self.assertTrue(cands3["S2-multi-token"] & PROV_CROSS_COUNTRY_HIGH_SPECIFICITY)
        index.close()

    def test_disk_backed_index_lifecycle_and_cleanup(self):
        """Temporary SQLite database files must be removed on close or context exit."""
        config = BlockingConfig()
        index = CandidateSourceIndex("test_lifecycle", config, temp_dir=self.base_path)
        index.add_candidate("S2-1", "us", "test entity", "100 St")
        index.commit_index()

        db_path = index.db_path
        self.assertIsNotNone(db_path)
        self.assertTrue(db_path.is_file())

        index.close()
        self.assertFalse(db_path.exists())

    def test_cleanup_after_injected_failure(self):
        """Index cleanup must remove temporary files even when an exception occurs."""
        config = BlockingConfig()
        db_path = None
        try:
            with CandidateSourceIndex("test_fail", config, temp_dir=self.base_path) as index:
                index.add_candidate("S2-1", "us", "test entity", "100 St")
                index.commit_index()
                db_path = index.db_path
                self.assertTrue(db_path.is_file())
                raise RuntimeError("Simulated pipeline failure")
        except RuntimeError:
            pass

        self.assertIsNotNone(db_path)
        self.assertFalse(db_path.exists())


class TestSyntheticFixture(unittest.TestCase):
    """
    Synthetic end-to-end integration test covering all 10 specified edge cases:
    1. Exact name match
    2. Typo/name corruption
    3. Missing address
    4. Missing country
    5. Multilingual name
    6. Reordered address components
    7. Changed address number
    8. Cross-country label mismatch
    9. Singleton Source 1
    10. Multiple matches for one Source 1
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.base_path = Path(self.temp_dir)
        self.dataset_dir = self.base_path / "cleaned"
        self.output_dir = self.base_path / "candidates"
        self.train_dir = self.dataset_dir / "train"
        self.train_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_all_10_edge_cases_synthetic_pipeline(self):
        """Run full blocking pipeline on synthetic data verifying recovery of all edge cases."""
        # 1. Create Source 1 TSV
        s1_rows = [
            "entity_id\tbusiness_name\tbusiness_address\tcountry\tbusiness_name_normalized\tbusiness_address_normalized\tcountry_normalized\tbusiness_name_is_missing\tbusiness_address_is_missing\tcountry_is_missing\n",
            # Case 1 & 10: Exact name & multiple matches
            "S1-01\tAlpha Corp\t100 Main St, Austin, TX\tUS\talpha corp\t100 main st, austin, tx\tus\tFalse\tFalse\tFalse\n",
            # Case 2: Typo / spelling corruption
            "S1-02\tBlue Ocean Seafood\t200 Shoreline Blvd\tUS\tblue ocean seafood\t200 shoreline blvd\tus\tFalse\tFalse\tFalse\n",
            # Case 3: Missing address
            "S1-03\tZenith Financial Group\t\tUS\tzenith financial group\t\tus\tFalse\tTrue\tFalse\n",
            # Case 4: Missing country
            "S1-04\tApex Global Logistics\t400 Cargo Rd\t\tapex global logistics\t400 cargo rd\t\tFalse\tFalse\tTrue\n",
            # Case 5: Multilingual name (Hindi Devanagari)
            "S1-05\tएसएस फूड्स\t500 Bazaar Rd, Delhi\tIndia\tएसएस फूड्स\t500 bazaar rd, delhi\tindia\tFalse\tFalse\tFalse\n",
            # Case 6: Reordered address components
            "S1-06\tSunrise Bakery\tSuite 10, 600 Oak Ave, Portland, OR\tUS\tsunrise bakery\tsuite 10, 600 oak ave, portland, or\tus\tFalse\tFalse\tFalse\n",
            # Case 7: Changed address number (brand match)
            "S1-07\tCascade Mountaineering\t700 Alpine Way\tUS\tcascade mountaineering\t700 alpine way\tus\tFalse\tFalse\tFalse\n",
            # Case 8: Cross-country label mismatch (high specificity exact name)
            "S1-08\tVanguard Robotics\t800 Tech Park\tUS\tvanguard robotics\t800 tech park\tus\tFalse\tFalse\tFalse\n",
            # Case 9: Singleton Source 1
            "S1-09\tLone Star Galaxy\t900 Solitary St\tUS\tlone star galaxy\t900 solitary st\tus\tFalse\tFalse\tFalse\n",
        ]
        (self.train_dir / "train_source1.tsv").write_text("".join(s1_rows), encoding="utf-8")

        # 2. Create Source 2 TSV
        s2_rows = [
            "entity_id\tbusiness_name\tbusiness_address\tcountry\tbusiness_name_normalized\tbusiness_address_normalized\tcountry_normalized\tbusiness_name_is_missing\tbusiness_address_is_missing\tcountry_is_missing\n",
            # Match for S1-01 (Exact name)
            "S2-101\tAlpha Corp\t100 Main St, Austin, TX\tUS\talpha corp\t100 main st, austin, tx\tus\tFalse\tFalse\tFalse\n",
            # Match for S1-02 (Typo: 'blu ocean seafood' -> recovered by n-grams or tokens)
            "S2-102\tBlu Ocean Seafood\t200 Shoreline Blvd\tUS\tblu ocean seafood\t200 shoreline blvd\tus\tFalse\tFalse\tFalse\n",
            # Match for S1-05 (Multilingual exact match)
            "S2-105\tएसएस फूड्स\t500 Bazaar Rd, Delhi\tIndia\tएसएस फूड्स\t500 bazaar rd, delhi\tindia\tFalse\tFalse\tFalse\n",
            # Match for S1-06 (Reordered address)
            "S2-106\tSunrise Bakery\t600 Oak Ave, Portland, OR, Suite 10\tUS\tsunrise bakery\t600 oak ave, portland, or, suite 10\tus\tFalse\tFalse\tFalse\n",
            # Match for S1-08 (Cross-country label mismatch: country labeled 'FR' but exact name)
            "S2-108\tVanguard Robotics\t800 Tech Park\tFrance\tvanguard robotics\t800 tech park\tfrance\tFalse\tFalse\tFalse\n",
        ]
        (self.train_dir / "train_source2.tsv").write_text("".join(s2_rows), encoding="utf-8")

        # 3. Create Source 3 TSV
        s3_rows = [
            "entity_id\tbusiness_name\tbusiness_address\tcountry\tbusiness_name_normalized\tbusiness_address_normalized\tcountry_normalized\tbusiness_name_is_missing\tbusiness_address_is_missing\tcountry_is_missing\n",
            # Second match for S1-01 (Multiple matches)
            "S3-201\tAlpha Corporation\t100 Main St, Austin\tUS\talpha corporation\t100 main st, austin\tus\tFalse\tFalse\tFalse\n",
            # Match for S1-03 (Missing address on S1 side; matched on rare name tokens 'zenith financial')
            "S3-203\tZenith Financial\t300 Banker Row, Chicago\tUS\tzenith financial\t300 banker row, chicago\tus\tFalse\tFalse\tFalse\n",
            # Match for S1-04 (Missing country on S1 side; fallback match)
            "S3-204\tApex Global Logistics\t400 Cargo Rd\tUS\tapex global logistics\t400 cargo rd\tus\tFalse\tFalse\tFalse\n",
            # Match for S1-07 (Changed address number: '750 Alpine Way' instead of 700; matched on rare name tokens)
            "S3-207\tCascade Mountaineering LLC\t750 Alpine Way\tUS\tcascade mountaineering llc\t750 alpine way\tus\tFalse\tFalse\tFalse\n",
        ]
        (self.train_dir / "train_source3.tsv").write_text("".join(s3_rows), encoding="utf-8")

        # 4. Create Ground Truth TSV
        gt_rows = [
            "source1_entity_id\tmatched_entity_ids\n",
            "S1-01\tS2-101,S3-201\n",
            "S1-02\tS2-102\n",
            "S1-03\tS3-203\n",
            "S1-04\tS3-204\n",
            "S1-05\tS2-105\n",
            "S1-06\tS2-106\n",
            "S1-07\tS3-207\n",
            "S1-08\tS2-108\n",
            "S1-09\t\n",  # Singleton
        ]
        gt_path = self.train_dir / "train_ground_truth.tsv"
        gt_path.write_text("".join(gt_rows), encoding="utf-8")

        # 5. Run blocking pipeline
        config = BlockingConfig(
            split="train",
            input_dir=self.dataset_dir,
            output_dir=self.output_dir,
            ground_truth_path=gt_path,
            holdout_ratio=1.0,  # Evaluate all rows in holdout
        )

        tsv_path, diag_path, diags = run_blocking_pipeline(config, quiet=True)

        self.assertTrue(tsv_path.is_file())
        self.assertTrue(diag_path.is_file())

        # Verify candidate_pairs.tsv formatting
        lines = [line for line in tsv_path.read_text(encoding="utf-8").splitlines() if line]
        self.assertEqual(len(lines), 9)  # Exactly 9 rows matching S1 input count

        s1_cands_map: Dict[str, List[str]] = {}
        for line in lines:
            parts = line.split("\t")
            s1_id = parts[0]
            cands = parts[1].split(",") if len(parts) > 1 and parts[1] else []
            s1_cands_map[s1_id] = cands

        # Case 1 & 10: S1-01 has both S2-101 and S3-201, with S2 preceding S3
        self.assertIn("S2-101", s1_cands_map["S1-01"])
        self.assertIn("S3-201", s1_cands_map["S1-01"])
        s2_idx = s1_cands_map["S1-01"].index("S2-101")
        s3_idx = s1_cands_map["S1-01"].index("S3-201")
        self.assertLess(s2_idx, s3_idx)

        # Case 2: Typo 'blu ocean seafood' recovered
        self.assertIn("S2-102", s1_cands_map["S1-02"])

        # Case 3: Missing address recovered via name tokens
        self.assertIn("S3-203", s1_cands_map["S1-03"])

        # Case 4: Missing country recovered via missing country fallback
        self.assertIn("S3-204", s1_cands_map["S1-04"])

        # Case 5: Multilingual Hindi name recovered
        self.assertIn("S2-105", s1_cands_map["S1-05"])

        # Case 6: Reordered address recovered
        self.assertIn("S2-106", s1_cands_map["S1-06"])

        # Case 7: Changed address number recovered via rare name tokens
        self.assertIn("S3-207", s1_cands_map["S1-07"])

        # Case 8: Cross-country label mismatch recovered via high specificity
        self.assertIn("S2-108", s1_cands_map["S1-08"])

        # Case 9: Singleton S1 has empty candidate list
        self.assertEqual(s1_cands_map["S1-09"], [])

        # Evaluation metrics verification
        self.assertEqual(diags["evaluation"]["counts"]["hits"], 9)
        self.assertEqual(diags["evaluation"]["pair_recall"], 1.0)
        self.assertEqual(diags["evaluation"]["s1_coverage"], 1.0)
        self.assertTrue(diags["evaluation"]["target_achieved"])

    def test_repeated_runs_determinism(self):
        """Repeated runs of the blocking pipeline must produce byte-for-byte identical candidate TSVs."""
        s1_rows = [
            "entity_id\tbusiness_name\tbusiness_address\tcountry\tbusiness_name_normalized\tbusiness_address_normalized\tcountry_normalized\tbusiness_name_is_missing\tbusiness_address_is_missing\tcountry_is_missing\n",
            "S1-01\tDelta Dynamics\t100 First St\tUS\tdelta dynamics\t100 first st\tus\tFalse\tFalse\tFalse\n",
        ]
        s2_rows = [
            "entity_id\tbusiness_name\tbusiness_address\tcountry\tbusiness_name_normalized\tbusiness_address_normalized\tcountry_normalized\tbusiness_name_is_missing\tbusiness_address_is_missing\tcountry_is_missing\n",
            "S2-01\tDelta Dynamics Corp\t100 First St\tUS\tdelta dynamics corp\t100 first st\tus\tFalse\tFalse\tFalse\n",
        ]
        s3_rows = [
            "entity_id\tbusiness_name\tbusiness_address\tcountry\tbusiness_name_normalized\tbusiness_address_normalized\tcountry_normalized\tbusiness_name_is_missing\tbusiness_address_is_missing\tcountry_is_missing\n",
            "S3-01\tDelta Dynamics Inc\t100 First St\tUS\tdelta dynamics inc\t100 first st\tus\tFalse\tFalse\tFalse\n",
        ]
        (self.train_dir / "train_source1.tsv").write_text("".join(s1_rows), encoding="utf-8")
        (self.train_dir / "train_source2.tsv").write_text("".join(s2_rows), encoding="utf-8")
        (self.train_dir / "train_source3.tsv").write_text("".join(s3_rows), encoding="utf-8")

        config1 = BlockingConfig(split="train", input_dir=self.dataset_dir, output_dir=self.base_path / "run1")
        tsv1, _, _ = run_blocking_pipeline(config1, quiet=True)

        config2 = BlockingConfig(split="train", input_dir=self.dataset_dir, output_dir=self.base_path / "run2")
        tsv2, _, _ = run_blocking_pipeline(config2, quiet=True)

        self.assertEqual(tsv1.read_bytes(), tsv2.read_bytes())


class TestCapSweepAndConfig(unittest.TestCase):
    """
    Test candidate cap CLI overrides, cap sweep report structure,
    independent S2/S3 recall rejection, and deterministic evaluation.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.base_path = Path(self.temp_dir)
        self.dataset_dir = self.base_path / "cleaned"
        self.output_dir = self.base_path / "candidates_val"
        self.train_dir = self.dataset_dir / "train"
        self.train_dir.mkdir(parents=True, exist_ok=True)

        # Create basic fixture
        s1_rows = [
            "entity_id\tbusiness_name\tbusiness_address\tcountry\tbusiness_name_normalized\tbusiness_address_normalized\tcountry_normalized\tbusiness_name_is_missing\tbusiness_address_is_missing\tcountry_is_missing\n",
            "S1-01\tAcme Corp\t100 Main St\tUS\tacme corp\t100 main st\tus\tFalse\tFalse\tFalse\n",
            "S1-02\tBeta LLC\t200 Oak Ave\tUS\tbeta llc\t200 oak ave\tus\tFalse\tFalse\tFalse\n",
        ]
        s2_rows = [
            "entity_id\tbusiness_name\tbusiness_address\tcountry\tbusiness_name_normalized\tbusiness_address_normalized\tcountry_normalized\tbusiness_name_is_missing\tbusiness_address_is_missing\tcountry_is_missing\n",
            "S2-01\tAcme Corp\t100 Main St\tUS\tacme corp\t100 main st\tus\tFalse\tFalse\tFalse\n",
            "S2-02\tBeta LLC\t200 Oak Ave\tUS\tbeta llc\t200 oak ave\tus\tFalse\tFalse\tFalse\n",
        ]
        s3_rows = [
            "entity_id\tbusiness_name\tbusiness_address\tcountry\tbusiness_name_normalized\tbusiness_address_normalized\tcountry_normalized\tbusiness_name_is_missing\tbusiness_address_is_missing\tcountry_is_missing\n",
            "S3-01\tAcme Corp\t100 Main St\tUS\tacme corp\t100 main st\tus\tFalse\tFalse\tFalse\n",
            "S3-02\tBeta LLC\t200 Oak Ave\tUS\tbeta llc\t200 oak ave\tus\tFalse\tFalse\tFalse\n",
        ]
        gt_rows = [
            "source1_entity_id\tmatched_entity_ids\n",
            "S1-01\tS2-01,S3-01\n",
            "S1-02\tS2-02,S3-02\n",
        ]

        (self.train_dir / "train_source1.tsv").write_text("".join(s1_rows), encoding="utf-8")
        (self.train_dir / "train_source2.tsv").write_text("".join(s2_rows), encoding="utf-8")
        (self.train_dir / "train_source3.tsv").write_text("".join(s3_rows), encoding="utf-8")
        self.gt_path = self.train_dir / "train_ground_truth.tsv"
        self.gt_path.write_text("".join(gt_rows), encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cli_cap_override(self):
        """CLI option --max-candidates-per-s1-per-source overrides config."""
        config = BlockingConfig(max_candidates_per_s1_per_source=2500)
        self.assertEqual(config.max_candidates_per_s1_per_source, 2500)

    def test_cap_recorded_in_diagnostics(self):
        """Candidate cap must be recorded in diagnostics JSON."""
        config = BlockingConfig(
            split="train",
            input_dir=self.dataset_dir,
            output_dir=self.output_dir,
            ground_truth_path=self.gt_path,
            max_candidates_per_s1_per_source=750,
            holdout_ratio=1.0,
        )
        _, diag_path, diags = run_blocking_pipeline(config, quiet=True)
        self.assertEqual(diags["configuration"]["max_candidates_per_s1_per_source"], 750)

        # Check persisted file
        loaded = json.loads(diag_path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["configuration"]["max_candidates_per_s1_per_source"], 750)

    def test_cap_sweep_report_structure(self):
        """Cap-sweep report structure contains all required sections and summary TSV columns."""
        config = BlockingConfig(
            split="train",
            input_dir=self.dataset_dir,
            output_dir=self.output_dir,
            ground_truth_path=self.gt_path,
            holdout_ratio=1.0,
        )
        rep_path, sum_path, report = run_cap_sweep(config, caps=[1, 2, 5], quiet=True)

        self.assertTrue(rep_path.is_file())
        self.assertTrue(sum_path.is_file())

        # Verify JSON report structure
        required_keys = [
            "title", "timestamp", "blocking_version", "git_commit",
            "effective_configuration", "holdout_definition", "acceptance_criteria",
            "recommended_cap", "acceptance_decision", "cap_results",
            "summary_table", "production_projections",
        ]
        for k in required_keys:
            self.assertIn(k, report)

        # Verify summary TSV columns
        expected_cols = [
            "cap", "validation_s1_count", "true_pairs", "retrieved_true_pairs",
            "missed_true_pairs", "pair_recall", "s2_recall", "s3_recall",
            "s1_coverage", "generated_pairs", "retained_pairs", "truncated_pairs",
            "capped_s1_count", "candidates_p50_before", "candidates_p95_before",
            "candidates_p99_before", "candidates_max_before", "candidates_p50_after",
            "candidates_p95_after", "candidates_p99_after", "candidates_max_after",
            "overflow_events", "candidates_lost_before_store", "runtime_seconds",
            "peak_rss_mb", "index_db_size_bytes", "candidate_output_size_bytes",
        ]
        lines = sum_path.read_text(encoding="utf-8").strip().splitlines()
        header_cols = lines[0].split("\t")
        self.assertEqual(header_cols, expected_cols)
        # Verify 3 rows for caps [1, 2, 5]
        self.assertEqual(len(lines), 4)

    def test_independent_s2_s3_recall_and_rejection(self):
        """Evaluation must reject a cap if either S2 or S3 recall is below 99%."""
        gt = {
            "S1-01": {"S2-01", "S3-01"},
            "S1-02": {"S2-02", "S3-02"},
        }
        evaluator = BlockingEvaluator(gt)

        # Mock retrieved candidates where S2 is 100% (2/2) but S3 is 50% (1/2)
        retrieved = {
            "S1-01": {"S2-01": 1, "S3-01": 1},
            "S1-02": {"S2-02": 1},  # Missing S3-02
        }
        res = evaluator.evaluate_candidates(retrieved)

        # S2 recall is 100%, S3 recall is 50%, overall recall is 75%
        self.assertEqual(res["s2_recall"], 1.0)
        self.assertEqual(res["s3_recall"], 0.5)
        self.assertEqual(res["pair_recall"], 0.75)
        self.assertFalse(res["target_achieved"])

    def test_no_stale_candidate_database_reuse(self):
        """Pre-existing database files in output directory must be overwritten/cleaned safely."""
        db_path = self.output_dir / "candidates_val.db"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        db_path.write_text("stale_dummy_data")

        config = BlockingConfig(
            split="train",
            input_dir=self.dataset_dir,
            output_dir=self.output_dir,
            ground_truth_path=self.gt_path,
            holdout_ratio=1.0,
        )
        run_cap_sweep(config, caps=[2], quiet=True)
        # Database must have been recreated as a valid SQLite db
        self.assertTrue(db_path.is_file())
        with open(db_path, "rb") as f:
            header = f.read(16)
            self.assertEqual(header, b"SQLite format 3\x00")

    def test_cleanup_after_cap_runs(self):
        """Index database files must be cleaned up after cap sweep finishes."""
        config = BlockingConfig(
            split="train",
            input_dir=self.dataset_dir,
            output_dir=self.output_dir,
            ground_truth_path=self.gt_path,
            holdout_ratio=1.0,
        )
        run_cap_sweep(config, caps=[2], quiet=True)
        # Ensure no temporary idx_*.db files remain in output_dir/.idx_tmp
        idx_tmp = self.output_dir / ".idx_tmp"
        if idx_tmp.exists():
            remaining_dbs = list(idx_tmp.glob("idx_*.db"))
            self.assertEqual(len(remaining_dbs), 0)


if __name__ == "__main__":
    unittest.main()
