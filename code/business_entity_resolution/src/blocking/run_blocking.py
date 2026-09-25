"""
Multi-Pass Blocking & Candidate Generation Pipeline Runner
Amazon ML Challenge 2026 - Business Entity Resolution

Orchestrates:
1. Indexing Candidate Source 2 and Candidate Source 3 separately using disk-backed SQLite.
2. Streaming Source 1 entities and generating multi-pass candidate unions.
3. Disk-backed SQLite candidate storage with compact provenance tracking.
4. Deterministic streaming export to candidate_pairs.tsv.
5. Entity-level validation holdout evaluation (if ground truth available).
6. Comprehensive versioned reproducibility diagnostics JSON report.
7. Guaranteed cleanup of temporary databases and resources.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from blocking.candidate_store import CandidateStore
from blocking.config import BlockingConfig, PROVENANCE_NAMES
from blocking.evaluate import BlockingEvaluator, is_validation_entity, load_ground_truth
from blocking.index import CandidateSourceIndex
from blocking.passes import generate_candidates_for_s1

BLOCKING_VERSION = "2.0.0"


def stream_s1_ids(
    s1_path: Path,
    eval_validation_only: bool,
    holdout_ratio: float,
    holdout_salt: str,
    max_s1: Optional[int] = None,
) -> Generator[str, None, None]:
    """Generator that streams Source 1 IDs in their exact input order with bounded memory."""
    count = 0
    with open(s1_path, "r", encoding="utf-8") as f:
        f.readline()  # Skip header
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if not parts or not parts[0]:
                continue
            s1_id = parts[0]
            if eval_validation_only:
                if not is_validation_entity(s1_id, holdout_ratio, holdout_salt):
                    continue
            yield s1_id
            count += 1
            if max_s1 and count >= max_s1:
                break


def run_blocking_pipeline(
    config: BlockingConfig,
    eval_validation_only: bool = False,
    max_s1: Optional[int] = None,
    max_candidates: Optional[int] = None,
    quiet: bool = False,
) -> Tuple[Path, Path, Dict[str, Any]]:
    """
    Run complete blocking pipeline for a dataset split.

    Returns:
        (candidate_tsv_path, diagnostics_json_path, diagnostics_dict)
    """
    start_time = time.time()
    split = config.split
    input_dir = config.input_dir
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not quiet:
        print("=" * 80)
        print("AMAZON ML CHALLENGE 2026 - BLOCKING & CANDIDATE GENERATION")
        print("=" * 80)
        print(f"Version          : {BLOCKING_VERSION}")
        print(f"Split            : {split}")
        print(f"Input Directory  : {input_dir}")
        print(f"Output Directory : {output_dir}")
        print(f"Validation Only  : {eval_validation_only}")
        if max_s1:
            print(f"Max S1 Records   : {max_s1:,}")
        if max_candidates:
            print(f"Max Cand Records : {max_candidates:,}")
        print("=" * 80)

    # 1. Locate split input files
    s1_path = input_dir / split / f"{split}_source1.tsv"
    s2_path = input_dir / split / f"{split}_source2.tsv"
    s3_path = input_dir / split / f"{split}_source3.tsv"

    assert s1_path.is_file(), f"Missing Source 1 file: {s1_path}"
    assert s2_path.is_file(), f"Missing Source 2 file: {s2_path}"
    assert s3_path.is_file(), f"Missing Source 3 file: {s3_path}"

    # Read manifest if available
    manifest_path = input_dir / "reports" / "manifest.json"
    manifest_data: Optional[Dict[str, Any]] = None
    if manifest_path.is_file():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
        except Exception:
            manifest_data = None

    index_s2: Optional[CandidateSourceIndex] = None
    index_s3: Optional[CandidateSourceIndex] = None
    store: Optional[CandidateStore] = None

    try:
        # 2. Build Inverted Index for Source 2
        if not quiet:
            print(f"\n>>> Step 1/4: Indexing Candidate Source 2 ({s2_path.name})...")
        t0 = time.time()
        index_s2 = CandidateSourceIndex(f"{split}_source2", config)
        index_s2.build_from_tsv(s2_path, max_rows=max_candidates)
        t_index_s2 = time.time() - t0
        if not quiet:
            print(f"  [DONE] Indexed {index_s2.total_records:,} S2 records in {t_index_s2:.2f}s")
            print(f"         Filtered high-DF keys: name_tokens={len(index_s2.filtered_name_tokens)}, "
                  f"addr_tokens={len(index_s2.filtered_addr_tokens)}, ngrams={len(index_s2.filtered_ngrams)}")

        # 3. Build Inverted Index for Source 3
        if not quiet:
            print(f"\n>>> Step 2/4: Indexing Candidate Source 3 ({s3_path.name})...")
        t0 = time.time()
        index_s3 = CandidateSourceIndex(f"{split}_source3", config)
        index_s3.build_from_tsv(s3_path, max_rows=max_candidates)
        t_index_s3 = time.time() - t0
        if not quiet:
            print(f"  [DONE] Indexed {index_s3.total_records:,} S3 records in {t_index_s3:.2f}s")
            print(f"         Filtered high-DF keys: name_tokens={len(index_s3.filtered_name_tokens)}, "
                  f"addr_tokens={len(index_s3.filtered_addr_tokens)}, ngrams={len(index_s3.filtered_ngrams)}")

        # 4. Stream Source 1 and Query Multi-Pass Candidates
        if not quiet:
            print(f"\n>>> Step 3/4: Querying Multi-Pass Candidates for Source 1 ({s1_path.name})...")
        t0 = time.time()
        db_path = output_dir / f"candidates_{split}.db"
        store = CandidateStore(db_path, cache_size_kb=config.sqlite_cache_size_kb, batch_size=config.batch_size)

        s1_count = 0
        with open(s1_path, "r", encoding="utf-8") as f:
            f.readline()  # Skip header
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 10:
                    continue

                s1_id = parts[0]
                name_norm = parts[4]
                addr_norm = parts[5]
                ctry_norm = parts[6]

                is_val = is_validation_entity(s1_id, config.holdout_ratio, config.holdout_salt)
                if eval_validation_only and not is_val:
                    continue

                s1_count += 1

                # Query Source 2
                c2 = generate_candidates_for_s1(s1_id, name_norm, addr_norm, ctry_norm, index_s2, config)
                # Query Source 3
                c3 = generate_candidates_for_s1(s1_id, name_norm, addr_norm, ctry_norm, index_s3, config)

                # Union candidates across Source 2 and Source 3
                s1_cands: Dict[str, int] = {}
                for cid, prov in c2.items():
                    s1_cands[cid] = s1_cands.get(cid, 0) | prov
                for cid, prov in c3.items():
                    s1_cands[cid] = s1_cands.get(cid, 0) | prov

                store.add_candidates(s1_id, s1_cands)

                if max_s1 and s1_count >= max_s1:
                    break

        store.flush()
        t_query_s1 = time.time() - t0
        if not quiet:
            print(f"  [DONE] Processed {s1_count:,} S1 entities in {t_query_s1:.2f}s")

        # 5. Export Deterministic Candidate TSV
        if not quiet:
            print("\n>>> Step 4/4: Exporting Deterministic candidate_pairs.tsv...")
        t0 = time.time()
        tsv_filename = "candidate_pairs.tsv" if not eval_validation_only else "candidate_pairs_val.tsv"
        tsv_path = output_dir / tsv_filename

        s1_iterator = stream_s1_ids(
            s1_path,
            eval_validation_only=eval_validation_only,
            holdout_ratio=config.holdout_ratio,
            holdout_salt=config.holdout_salt,
            max_s1=max_s1,
        )
        export_stats = store.export_tsv(
            tsv_path,
            s1_iterator,
            max_candidates_per_source=config.max_candidates_per_s1_per_source,
        )
        t_export = time.time() - t0
        if not quiet:
            print(f"  [DONE] Exported {export_stats['total_s1_entities']:,} rows to {tsv_path.name} in {t_export:.2f}s")
            print(f"         Total pairs generated: {export_stats['total_candidate_pairs_generated']:,}")
            print(f"         Total pairs retained : {export_stats['total_candidate_pairs_retained']:,}")
            print(f"         Pairs capped         : {export_stats['total_candidate_pairs_truncated_by_caps']:,}")
            print(f"         Median candidates/S1 : {export_stats['candidate_counts_after_capping']['median']}")
            print(f"         P95 candidates/S1    : {export_stats['candidate_counts_after_capping']['p95']}")
            print(f"         Max candidates/S1    : {export_stats['candidate_counts_after_capping']['max']}")

        # Collect provenance diagnostics
        provenance_stats = store.get_provenance_diagnostics()

        # 6. Evaluate against Ground Truth if available (train split)
        eval_results: Optional[Dict[str, Any]] = None
        if config.ground_truth_path and config.ground_truth_path.is_file() and split == "train":
            if not quiet:
                print("\n>>> Evaluating Validation Holdout Recall against Ground Truth...")
            gt = load_ground_truth(
                config.ground_truth_path,
                holdout_ratio=config.holdout_ratio,
                salt=config.holdout_salt,
                validation_only=True,
            )
            # Filter ground truth down to S1 entities processed in this run
            if max_s1 or eval_validation_only:
                # Re-stream the S1 IDs processed
                processed_s1_set = set(stream_s1_ids(
                    s1_path,
                    eval_validation_only=eval_validation_only,
                    holdout_ratio=config.holdout_ratio,
                    holdout_salt=config.holdout_salt,
                    max_s1=max_s1,
                ))
                eval_gt = {s1: matched for s1, matched in gt.items() if s1 in processed_s1_set}
            else:
                eval_gt = gt

            total_universe = len(eval_gt) * (index_s2.total_records + index_s3.total_records)
            evaluator = BlockingEvaluator(eval_gt)
            eval_results = evaluator.evaluate_from_store(
                store,
                overflow_events=index_s2.overflow_events + index_s3.overflow_events,
                total_candidate_universe=total_universe,
            )
            if not quiet:
                print(f"  [EVALUATION] Validation Pair Recall: {eval_results['pair_recall'] * 100:.2f}% "
                      f"(Target >= 99%: {'PASSED' if eval_results['target_achieved'] else 'FAILED'})")
                print(f"               Union Recall          : {eval_results['union_recall'] * 100:.2f}%")
                print(f"               S1 Coverage           : {eval_results['s1_coverage'] * 100:.2f}%")
                print(f"               S2 Recall             : {eval_results['s2_recall'] * 100:.2f}%")
                print(f"               S3 Recall             : {eval_results['s3_recall'] * 100:.2f}%")
                print(f"               True Pairs Hit        : {eval_results['counts']['hits']:,} / "
                      f"{eval_results['counts']['total_true_pairs']:,}")

        total_duration = time.time() - start_time
        peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

        # Compute overflow breakdown
        all_overflows = index_s2.overflow_events + index_s3.overflow_events
        refined_overflows = sum(1 for e in all_overflows if e.get("refined"))
        unrefined_retained = sum(1 for e in all_overflows if e.get("refinement_attempted") and not e.get("refined") and not e.get("is_filtered"))
        filtered_overflows = sum(1 for e in all_overflows if e.get("is_filtered"))

        # 7. Construct Versioned Reproducibility Diagnostics JSON
        diagnostics: Dict[str, Any] = {
            "title": "Amazon ML Challenge 2026 - Blocking & Candidate Generation Report",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "blocking_version": BLOCKING_VERSION,
            "split": split,
            "is_validation_holdout_only": eval_validation_only,
            "sources": {
                "source1": s1_path.name,
                "source2": s2_path.name,
                "source3": s3_path.name,
            },
            "input_manifest": manifest_data.get("files") if manifest_data else None,
            "configuration": {
                "max_name_df_absolute": config.max_name_df_absolute,
                "max_name_df_relative": config.max_name_df_relative,
                "max_addr_df_absolute": config.max_addr_df_absolute,
                "max_addr_df_relative": config.max_addr_df_relative,
                "max_ngram_df_absolute": config.max_ngram_df_absolute,
                "max_ngram_df_relative": config.max_ngram_df_relative,
                "min_token_length": config.min_token_length,
                "min_addr_token_length": config.min_addr_token_length,
                "ngram_size": config.ngram_size,
                "min_ngram_length": config.min_ngram_length,
                "max_block_size": config.max_block_size,
                "max_candidates_per_s1_per_source": config.max_candidates_per_s1_per_source,
                "allow_missing_country_fallback": config.allow_missing_country_fallback,
                "allow_cross_country_high_specificity": config.allow_cross_country_high_specificity,
                "cross_country_min_rare_tokens": config.cross_country_min_rare_tokens,
                "holdout_ratio": config.holdout_ratio,
                "holdout_salt": config.holdout_salt,
                "legal_stoplist": sorted(config.legal_stoplist),
                "ranking_policy": "Deterministic evidence-tier priority: exact_name > popcount(provenance) > rare_name > num_loc > addr_token > ngram > missing_country > cross_country, tie-breaker: cand_id ascending",
            },
            "performance": {
                "total_runtime_seconds": round(total_duration, 2),
                "index_s2_time_seconds": round(t_index_s2, 2),
                "index_s3_time_seconds": round(t_index_s3, 2),
                "query_s1_time_seconds": round(t_query_s1, 2),
                "export_tsv_time_seconds": round(t_export, 2),
                "peak_rss_mb": round(peak_rss_mb, 2),
                "sqlite_db_size_bytes": os.path.getsize(db_path) if db_path.exists() else 0,
                "candidate_tsv_size_bytes": os.path.getsize(tsv_path) if tsv_path.exists() else 0,
            },
            "candidate_volume_summary": export_stats,
            "provenance_summary": provenance_stats,
            "overflow_summary": {
                "total_overflow_events": len(all_overflows),
                "refined_overflow_events": refined_overflows,
                "unrefined_retained_events": unrefined_retained,
                "filtered_high_df_events": filtered_overflows,
                "candidates_lost_before_store": False,
                "s2_overflows": len(index_s2.overflow_events),
                "s3_overflows": len(index_s3.overflow_events),
            },
            "evaluation": eval_results,
        }

        diag_filename = f"blocking_diagnostics_{split}.json" if not eval_validation_only else "blocking_diagnostics_val.json"
        diag_path = output_dir / diag_filename
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, indent=2)

        if not quiet:
            print("\n" + "=" * 80)
            print(f"Blocking pipeline completed in {total_duration:.2f}s!")
            print(f"Peak RSS Memory    : {peak_rss_mb:.2f} MB")
            print(f"Candidate pairs TSV: {tsv_path}")
            print(f"Diagnostics Report : {diag_path}")
            print("=" * 80 + "\n")

        return tsv_path, diag_path, diagnostics

    finally:
        if store:
            store.close()
        if index_s2:
            index_s2.close()
        if index_s3:
            index_s3.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-Pass Blocking & Candidate Generation")
    parser.add_argument("--split", choices=["train", "test", "val"], default="train", help="Dataset split")
    parser.add_argument("--input-dir", type=Path, default=Path("cleaned"), help="Cleaned inputs directory")
    parser.add_argument("--output-dir", type=Path, default=Path("candidates"), help="Outputs directory")
    parser.add_argument("--ground-truth", type=Path, default=Path("student_resource/dataset/train/train_ground_truth.tsv"))
    parser.add_argument("--max-s1", type=int, default=None, help="Limit number of Source 1 records")
    parser.add_argument("--max-cands", type=int, default=None, help="Limit candidate records indexed")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode")

    args = parser.parse_args()

    actual_split = "train" if args.split == "val" else args.split
    val_only = (args.split == "val")

    config = BlockingConfig(
        split=actual_split,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        ground_truth_path=args.ground_truth,
    )

    try:
        run_blocking_pipeline(
            config,
            eval_validation_only=val_only,
            max_s1=args.max_s1,
            max_candidates=args.max_cands,
            quiet=args.quiet,
        )
        return 0
    except Exception as e:
        print(f"\n[FATAL ERROR] Blocking failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
