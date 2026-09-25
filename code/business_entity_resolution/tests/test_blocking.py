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
- Country fallback rules
- Cross-country high-specificity rules
- Pair deduplication
- Provenance union
- Deterministic candidate ordering
- Overflow reporting
- Final budget behavior
- One-to-many ground-truth evaluation
- Singleton handling
- Synthetic end-to-end fixture covering all 10 edge cases
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Set

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from blocking.candidate_store import CandidateStore, compute_candidate_rank
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
from blocking.run_blocking import run_blocking_pipeline


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
        # Create 10 S2 candidates with different evidence
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
        self.store.export_tsv(tsv_path, [s1], max_candidates_per_source=2)

        content = tsv_path.read_text().strip().split("\t")
        retained = content[1].split(",")
        self.assertEqual(len(retained), 2)
        # Exact name hits (S2-002 and S2-004) must be prioritized over weak hits
        self.assertIn("S2-002", retained)
        self.assertIn("S2-004", retained)


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
        self.assertAlmostEqual(results["s1_coverage"], 1.0, places=4)  # Both non-singletons got >= 1 hit


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


class TestBlockingPassesAndOverflow(unittest.TestCase):
    """Test rare-token filtering, country fallback rules, and overflow metadata tracking."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.base_path = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_rare_token_frequency_filtering(self):
        """Tokens with DF exceeding max_name_df must be filtered and not indexed."""
        tsv_path = self.base_path / "cand.tsv"
        # 10 rows sharing 'popular', 1 row with 'rare'
        rows = [
            "entity_id\tbusiness_name\tbusiness_address\tcountry\tbusiness_name_normalized\tbusiness_address_normalized\tcountry_normalized\tbusiness_name_is_missing\tbusiness_address_is_missing\tcountry_is_missing\n",
        ]
        for i in range(10):
            rows.append(f"S2-{i}\tPopular Biz\t100 Main St\tUS\tpopular biz\t100 main st\tus\tFalse\tFalse\tFalse\n")
        rows.append("S2-99\tRare Biz\t100 Main St\tUS\trare biz\t100 main st\tus\tFalse\tFalse\tFalse\n")
        tsv_path.write_text("".join(rows), encoding="utf-8")

        config = BlockingConfig(max_name_df_absolute=3)
        index = CandidateSourceIndex("test_s2", config)
        index.build_from_tsv(tsv_path)

        # 'popular' must be in filtered_name_tokens, not in name_token_index
        self.assertIn(("us", "popular"), index.filtered_name_tokens)
        self.assertNotIn(("us", "popular"), index.name_token_index)

        # 'rare' must be in name_token_index
        self.assertIn(("us", "rare"), index.name_token_index)

    def test_overflow_event_structure(self):
        """Pathological block overflows must record complete metadata dictionaries."""
        config = BlockingConfig(max_block_size=2)
        index = CandidateSourceIndex("test_s2", config)
        index.cand_ids = ["S2-1", "S2-2", "S2-3"]
        index.cand_countries = ["us", "us", "us"]
        index.name_token_index[("us", "overflowing")] = [0, 1, 2]  # 3 postings > max_block_size=2

        # Query with single token
        cands = index.query_name_tokens("S1-001", "us", ["overflowing"])
        self.assertEqual(len(cands), 0)
        self.assertEqual(len(index.overflow_events), 1)

        event = index.overflow_events[0]
        self.assertEqual(event["s1_id"], "S1-001")
        self.assertEqual(event["source"], "test_s2")
        self.assertEqual(event["block_family"], "rare_name_token")
        self.assertEqual(event["key"], "overflowing")
        self.assertEqual(event["generated_count"], 3)
        self.assertEqual(event["retained_count"], 0)
        self.assertEqual(event["truncated_count"], 3)
        self.assertFalse(event["is_filtered"])
        self.assertTrue(event["refinement_attempted"])

    def test_country_fallback_rules(self):
        """Missing-country records must route through missing-country fallback."""
        config = BlockingConfig(allow_missing_country_fallback=True)
        index = CandidateSourceIndex("test_s2", config)
        index.cand_ids = ["S2-missing"]
        index.cand_countries = [""]
        index.missing_country_exact["omni tech"] = [0]

        # S1 has missing country
        cands = generate_candidates_for_s1(
            s1_id="S1-001",
            name_norm="omni tech",
            addr_norm="123 Road",
            country_norm="",
            index=index,
            config=config,
        )
        self.assertIn("S2-missing", cands)
        self.assertTrue(cands["S2-missing"] & PROV_MISSING_COUNTRY_FALLBACK)

    def test_cross_country_high_specificity_rules(self):
        """Cross-country candidate must be allowed only through high-specificity evidence."""
        config = BlockingConfig(allow_cross_country_high_specificity=True)
        index = CandidateSourceIndex("test_s2", config)
        index.cand_ids = ["S2-fr"]
        index.cand_countries = ["france"]
        index.exact_name_global["hyperion robotics"] = [0]

        # S1 is US, Candidate is France
        cands = generate_candidates_for_s1(
            s1_id="S1-001",
            name_norm="hyperion robotics",
            addr_norm="100 Main St",
            country_norm="us",
            index=index,
            config=config,
        )
        self.assertIn("S2-fr", cands)
        self.assertTrue(cands["S2-fr"] & PROV_CROSS_COUNTRY_HIGH_SPECIFICITY)


if __name__ == "__main__":
    unittest.main()
