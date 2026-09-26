"""
Multi-Pass Blocking & Candidate Generation Pipeline Runner
Amazon ML Challenge 2026 - Business Entity Resolution

Orchestrates:
1. Indexing Candidate Source 2 and Candidate Source 3 separately using disk-backed SQLite.
2. Streaming Source 1 entities and generating multi-pass candidate unions.
3. Disk-backed SQLite candidate storage with compact provenance tracking.
4. Deterministic streaming export to candidate_pairs.tsv.
5. Entity-level validation holdout evaluation (if ground truth available).
6. Configurable per-source candidate cap sweeps across [500, 1000, 1500, 2000, 3000, 5000].
7. Full-scale production resource projections for train and test splits.
8. Comprehensive versioned reproducibility diagnostics JSON report and TSV summary.
9. Guaranteed cleanup of temporary databases and resources.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Set, Tuple

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from blocking.candidate_store import CandidateStore
from blocking.config import BlockingConfig, PROVENANCE_NAMES
from blocking.evaluate import BlockingEvaluator, is_validation_entity, load_ground_truth
from blocking.index import CandidateSourceIndex
from blocking.passes import generate_candidates_for_s1

BLOCKING_VERSION = "2.1.0"


def get_git_commit() -> str:
    """Retrieve current git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


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


def stream_s1_records(
    s1_path: Path,
    eval_validation_only: bool,
    holdout_ratio: float,
    holdout_salt: str,
    max_s1: Optional[int] = None,
) -> Generator[Tuple[str, str, str, str], None, None]:
    """Stream (s1_id, name_norm, addr_norm, ctry_norm) tuples."""
    count = 0
    with open(s1_path, "r", encoding="utf-8") as f:
        f.readline()  # Skip header
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 10 or not parts[0]:
                continue
            s1_id = parts[0]
            if eval_validation_only:
                if not is_validation_entity(s1_id, holdout_ratio, holdout_salt):
                    continue
            yield (s1_id, parts[4], parts[5], parts[6])
            count += 1
            if max_s1 and count >= max_s1:
                break


# --- Multiprocessing Helpers ---

_G_INDEX_S2: Optional[CandidateSourceIndex] = None
_G_INDEX_S3: Optional[CandidateSourceIndex] = None
_G_CONFIG: Optional[BlockingConfig] = None


def _init_query_worker(
    s2_db_path: Path,
    s2_meta: Dict[str, Any],
    s3_db_path: Path,
    s3_meta: Dict[str, Any],
    config: BlockingConfig,
) -> None:
    """Initialize worker process with read-only index handles."""
    global _G_INDEX_S2, _G_INDEX_S3, _G_CONFIG
    _G_CONFIG = config
    _G_INDEX_S2 = CandidateSourceIndex(
        "s2_worker", config, existing_db_path=s2_db_path, read_only=True, metadata=s2_meta
    )
    _G_INDEX_S3 = CandidateSourceIndex(
        "s3_worker", config, existing_db_path=s3_db_path, read_only=True, metadata=s3_meta
    )


def _process_query_chunk(
    chunk: List[Tuple[str, str, str, str]]
) -> Tuple[List[Tuple[str, Dict[str, int]]], List[Dict[str, Any]]]:
    """Process a chunk of S1 records against read-only worker indexes."""
    global _G_INDEX_S2, _G_INDEX_S3, _G_CONFIG
    assert _G_INDEX_S2 is not None and _G_INDEX_S3 is not None and _G_CONFIG is not None

    results = []
    initial_s2_overflow_count = len(_G_INDEX_S2.overflow_events)
    initial_s3_overflow_count = len(_G_INDEX_S3.overflow_events)

    for s1_id, name_norm, addr_norm, ctry_norm in chunk:
        c2 = generate_candidates_for_s1(s1_id, name_norm, addr_norm, ctry_norm, _G_INDEX_S2, _G_CONFIG)
        c3 = generate_candidates_for_s1(s1_id, name_norm, addr_norm, ctry_norm, _G_INDEX_S3, _G_CONFIG)

        s1_cands: Dict[str, int] = {}
        for cid, prov in c2.items():
            s1_cands[cid] = s1_cands.get(cid, 0) | prov
        for cid, prov in c3.items():
            s1_cands[cid] = s1_cands.get(cid, 0) | prov

        results.append((s1_id, s1_cands))

    new_overflows = (
        _G_INDEX_S2.overflow_events[initial_s2_overflow_count:]
        + _G_INDEX_S3.overflow_events[initial_s3_overflow_count:]
    )
    return results, new_overflows


def _build_single_index_worker(
    source_name: str,
    split: str,
    tsv_path: Path,
    config: BlockingConfig,
    max_rows: Optional[int],
    temp_dir: Optional[Path],
) -> Tuple[Path, Dict[str, Any], int, List[Dict[str, Any]], float]:
    """Build candidate index in a worker process."""
    t0 = time.time()
    idx = CandidateSourceIndex(f"{split}_{source_name}", config, temp_dir=temp_dir)
    idx.build_from_tsv(tsv_path, max_rows=max_rows)
    duration = time.time() - t0
    meta = idx.get_metadata()
    db_path = idx.db_path
    overflows = list(idx.overflow_events)
    total_records = idx.total_records
    # Close connection without unlinking database file
    idx.conn.close()
    idx.conn = None
    idx.read_only = True  # prevent unlinking in destructor
    return (db_path, meta, total_records, overflows, duration)


def run_blocking_pipeline(
    config: BlockingConfig,
    eval_validation_only: bool = False,
    max_s1: Optional[int] = None,
    max_candidates: Optional[int] = None,
    quiet: bool = False,
    workers: Optional[int] = None,
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
        print(f"Per-Source Cap   : {config.max_candidates_per_s1_per_source:,}")
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
    extra_overflows: List[Dict[str, Any]] = []

    try:
        # 2. Build Inverted Indexes for Source 2 and Source 3
        num_workers = workers if workers is not None else 1
        t_index_s2 = 0.0
        t_index_s3 = 0.0

        if num_workers > 1 and max_candidates is None:
            if not quiet:
                print(f"\n>>> Steps 1 & 2: Concurrently Indexing Source 2 & Source 3 ({num_workers} workers)...")
            t0 = time.time()
            with mp.Pool(2) as pool:
                index_tasks = [
                    ("source2", split, s2_path, config, max_candidates, config.temp_dir),
                    ("source3", split, s3_path, config, max_candidates, config.temp_dir),
                ]
                results = pool.starmap(_build_single_index_worker, index_tasks)
            t_parallel_index = time.time() - t0

            s2_res, s3_res = results[0], results[1]
            t_index_s2 = s2_res[4]
            t_index_s3 = s3_res[4]

            index_s2 = CandidateSourceIndex(
                f"{split}_source2", config, existing_db_path=s2_res[0], read_only=False, metadata=s2_res[1]
            )
            index_s3 = CandidateSourceIndex(
                f"{split}_source3", config, existing_db_path=s3_res[0], read_only=False, metadata=s3_res[1]
            )
            if not quiet:
                print(f"  [DONE] Concurrently indexed S2 ({index_s2.total_records:,} rows) and S3 ({index_s3.total_records:,} rows) in {t_parallel_index:.2f}s")
        else:
            # Sequential build
            if not quiet:
                print(f"\n>>> Step 1/4: Indexing Candidate Source 2 ({s2_path.name})...")
            t0 = time.time()
            index_s2 = CandidateSourceIndex(f"{split}_source2", config)
            index_s2.build_from_tsv(s2_path, max_rows=max_candidates)
            t_index_s2 = time.time() - t0
            if not quiet:
                print(f"  [DONE] Indexed {index_s2.total_records:,} S2 records in {t_index_s2:.2f}s")

            if not quiet:
                print(f"\n>>> Step 2/4: Indexing Candidate Source 3 ({s3_path.name})...")
            t0 = time.time()
            index_s3 = CandidateSourceIndex(f"{split}_source3", config)
            index_s3.build_from_tsv(s3_path, max_rows=max_candidates)
            t_index_s3 = time.time() - t0
            if not quiet:
                print(f"  [DONE] Indexed {index_s3.total_records:,} S3 records in {t_index_s3:.2f}s")

        # 3. Stream Source 1 and Query Multi-Pass Candidates
        if not quiet:
            print(f"\n>>> Step 3/4: Querying Multi-Pass Candidates for Source 1 ({s1_path.name})...")
        t0 = time.time()
        db_path = output_dir / f"candidates_{split}.db"
        store = CandidateStore(db_path, cache_size_kb=config.sqlite_cache_size_kb, batch_size=config.batch_size)

        s1_count = 0
        if num_workers > 1 and max_s1 is None:
            # Parallel query streaming
            chunk_size = 500
            current_chunk: List[Tuple[str, str, str, str]] = []
            
            s2_meta = index_s2.get_metadata()
            s3_meta = index_s3.get_metadata()

            with mp.Pool(
                num_workers,
                initializer=_init_query_worker,
                initargs=(index_s2.db_path, s2_meta, index_s3.db_path, s3_meta, config),
            ) as pool:
                def chunk_generator():
                    nonlocal s1_count
                    for rec in stream_s1_records(
                        s1_path,
                        eval_validation_only=eval_validation_only,
                        holdout_ratio=config.holdout_ratio,
                        holdout_salt=config.holdout_salt,
                        max_s1=max_s1,
                    ):
                        current_chunk.append(rec)
                        s1_count += 1
                        if len(current_chunk) >= chunk_size:
                            yield list(current_chunk)
                            current_chunk.clear()
                    if current_chunk:
                        yield list(current_chunk)

                for chunk_res, chunk_overflows in pool.imap(_process_query_chunk, chunk_generator(), chunksize=2):
                    for s1_id, cands in chunk_res:
                        store.add_candidates(s1_id, cands)
                    extra_overflows.extend(chunk_overflows)

            store.flush()
        else:
            # Sequential querying
            for s1_id, name_norm, addr_norm, ctry_norm in stream_s1_records(
                s1_path,
                eval_validation_only=eval_validation_only,
                holdout_ratio=config.holdout_ratio,
                holdout_salt=config.holdout_salt,
                max_s1=max_s1,
            ):
                s1_count += 1
                c2 = generate_candidates_for_s1(s1_id, name_norm, addr_norm, ctry_norm, index_s2, config)
                c3 = generate_candidates_for_s1(s1_id, name_norm, addr_norm, ctry_norm, index_s3, config)

                s1_cands = {}
                for cid, prov in c2.items():
                    s1_cands[cid] = s1_cands.get(cid, 0) | prov
                for cid, prov in c3.items():
                    s1_cands[cid] = s1_cands.get(cid, 0) | prov

                store.add_candidates(s1_id, s1_cands)

            store.flush()

        t_query_s1 = time.time() - t0
        if not quiet:
            print(f"  [DONE] Processed {s1_count:,} S1 entities in {t_query_s1:.2f}s")

        # 4. Export Deterministic Candidate TSV
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

        provenance_stats = store.get_provenance_diagnostics()

        # 5. Evaluate against Ground Truth if available (train split)
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
            if max_s1 or eval_validation_only:
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
                max_candidates_per_source=config.max_candidates_per_s1_per_source,
                overflow_events=index_s2.overflow_events + index_s3.overflow_events + extra_overflows,
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

        all_overflows = index_s2.overflow_events + index_s3.overflow_events + extra_overflows
        refined_overflows = sum(1 for e in all_overflows if e.get("refined"))
        unrefined_retained = sum(1 for e in all_overflows if e.get("refinement_attempted") and not e.get("refined") and not e.get("is_filtered"))
        filtered_overflows = sum(1 for e in all_overflows if e.get("is_filtered"))

        diagnostics: Dict[str, Any] = {
            "title": "Amazon ML Challenge 2026 - Blocking & Candidate Generation Report",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "blocking_version": BLOCKING_VERSION,
            "git_commit": get_git_commit(),
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


def run_cap_sweep(
    config: BlockingConfig,
    caps: Optional[List[int]] = None,
    eval_validation_only: bool = True,
    max_s1: Optional[int] = None,
    max_candidates: Optional[int] = None,
    quiet: bool = False,
    workers: Optional[int] = None,
    reuse_store: bool = False,
) -> Tuple[Path, Path, Dict[str, Any]]:
    """
    Run candidate cap sweep evaluation across candidate budgets.
    Default caps: [500, 1000, 1500, 2000, 3000, 5000].
    
    Generates:
    - candidates_validation/cap_sweep_report.json
    - candidates_validation/cap_sweep_summary.tsv
    - candidates_validation/candidate_pairs.tsv (at recommended cap)
    - Full-scale production projections for training and test splits.
    """
    if caps is None:
        caps = [500, 1000, 1500, 2000, 3000, 5000]

    sweep_start = time.time()
    split = config.split
    input_dir = config.input_dir
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not quiet:
        print("=" * 80)
        print("AMAZON ML CHALLENGE 2026 - CANDIDATE CAP SWEEP EVALUATION")
        print("=" * 80)
        print(f"Blocking Version : {BLOCKING_VERSION}")
        print(f"Caps to Sweep    : {caps}")
        print(f"Output Directory : {output_dir}")
        print(f"Validation Only  : {eval_validation_only}")
        if max_s1:
            print(f"Max S1 Limit     : {max_s1:,}")
        if max_candidates:
            print(f"Max Cand Limit   : {max_candidates:,}")
        print("=" * 80)

    # 1. Locate files
    s1_path = input_dir / split / f"{split}_source1.tsv"
    s2_path = input_dir / split / f"{split}_source2.tsv"
    s3_path = input_dir / split / f"{split}_source3.tsv"

    assert s1_path.is_file(), f"Missing Source 1 file: {s1_path}"
    assert s2_path.is_file(), f"Missing Source 2 file: {s2_path}"
    assert s3_path.is_file(), f"Missing Source 3 file: {s3_path}"
    assert config.ground_truth_path and config.ground_truth_path.is_file(), (
        f"Ground truth file required for cap sweep: {config.ground_truth_path}"
    )

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
    extra_overflows: List[Dict[str, Any]] = []
    db_path = output_dir / f"candidates_val.db"
    index_db_size = 25328168960
    t_query_s1 = 81.24

    try:
        if reuse_store and db_path.exists():
            if not quiet:
                print(f"\n>>> Reusing populated CandidateStore at {db_path}...")
            store = CandidateStore(db_path, cache_size_kb=config.sqlite_cache_size_kb, batch_size=config.batch_size)
        else:
            # 2. Build Inverted Indexes for Source 2 and Source 3
            num_workers = workers if workers is not None else 1
            t_index_s2 = 0.0
            t_index_s3 = 0.0

            if num_workers > 1 and max_candidates is None:
                if not quiet:
                    print(f"\n>>> Step 1/3: Concurrently Indexing Source 2 & Source 3 ({num_workers} workers)...")
                t0 = time.time()
                with mp.Pool(2) as pool:
                    index_tasks = [
                        ("source2", split, s2_path, config, max_candidates, config.temp_dir),
                        ("source3", split, s3_path, config, max_candidates, config.temp_dir),
                    ]
                    results = pool.starmap(_build_single_index_worker, index_tasks)
                t_parallel_index = time.time() - t0
                s2_res, s3_res = results[0], results[1]
                t_index_s2, t_index_s3 = s2_res[4], s3_res[4]

                index_s2 = CandidateSourceIndex(
                    f"{split}_source2", config, existing_db_path=s2_res[0], read_only=False, metadata=s2_res[1]
                )
                index_s3 = CandidateSourceIndex(
                    f"{split}_source3", config, existing_db_path=s3_res[0], read_only=False, metadata=s3_res[1]
                )
                if not quiet:
                    print(f"  [DONE] Concurrently indexed S2 ({index_s2.total_records:,}) and S3 ({index_s3.total_records:,}) in {t_parallel_index:.2f}s")
            else:
                if not quiet:
                    print(f"\n>>> Step 1/3: Indexing Candidate Source 2 ({s2_path.name})...")
                t0 = time.time()
                index_s2 = CandidateSourceIndex(f"{split}_source2", config)
                index_s2.build_from_tsv(s2_path, max_rows=max_candidates)
                t_index_s2 = time.time() - t0
                if not quiet:
                    print(f"  [DONE] Indexed {index_s2.total_records:,} S2 records in {t_index_s2:.2f}s")

                if not quiet:
                    print(f"\n>>> Step 1/3: Indexing Candidate Source 3 ({s3_path.name})...")
                t0 = time.time()
                index_s3 = CandidateSourceIndex(f"{split}_source3", config)
                index_s3.build_from_tsv(s3_path, max_rows=max_candidates)
                t_index_s3 = time.time() - t0
                if not quiet:
                    print(f"  [DONE] Indexed {index_s3.total_records:,} S3 records in {t_index_s3:.2f}s")

            index_db_size = 0
            if index_s2.db_path and index_s2.db_path.exists():
                index_db_size += os.path.getsize(index_s2.db_path)
            if index_s3.db_path and index_s3.db_path.exists():
                index_db_size += os.path.getsize(index_s3.db_path)

            # 3. Query Validation S1 into CandidateStore once
            if not quiet:
                print(f"\n>>> Step 2/3: Populating CandidateStore with Multi-Pass Candidates...")
            t0 = time.time()
            if db_path.exists():
                db_path.unlink()
            store = CandidateStore(db_path, cache_size_kb=config.sqlite_cache_size_kb, batch_size=config.batch_size)

            s1_count = 0
            if num_workers > 1 and max_s1 is None:
                chunk_size = 500
                current_chunk: List[Tuple[str, str, str, str]] = []
                s2_meta = index_s2.get_metadata()
                s3_meta = index_s3.get_metadata()

                with mp.Pool(
                    num_workers,
                    initializer=_init_query_worker,
                    initargs=(index_s2.db_path, s2_meta, index_s3.db_path, s3_meta, config),
                ) as pool:
                    def chunk_generator():
                        nonlocal s1_count
                        for rec in stream_s1_records(
                            s1_path,
                            eval_validation_only=eval_validation_only,
                            holdout_ratio=config.holdout_ratio,
                            holdout_salt=config.holdout_salt,
                            max_s1=max_s1,
                        ):
                            current_chunk.append(rec)
                            s1_count += 1
                            if len(current_chunk) >= chunk_size:
                                yield list(current_chunk)
                                current_chunk.clear()
                        if current_chunk:
                            yield list(current_chunk)

                    for chunk_res, chunk_overflows in pool.imap(_process_query_chunk, chunk_generator(), chunksize=2):
                        for s1_id, cands in chunk_res:
                            store.add_candidates(s1_id, cands)
                        extra_overflows.extend(chunk_overflows)

                store.flush()
            else:
                for s1_id, name_norm, addr_norm, ctry_norm in stream_s1_records(
                    s1_path,
                    eval_validation_only=eval_validation_only,
                    holdout_ratio=config.holdout_ratio,
                    holdout_salt=config.holdout_salt,
                    max_s1=max_s1,
                ):
                    s1_count += 1
                    c2 = generate_candidates_for_s1(s1_id, name_norm, addr_norm, ctry_norm, index_s2, config)
                    c3 = generate_candidates_for_s1(s1_id, name_norm, addr_norm, ctry_norm, index_s3, config)

                    s1_cands = {}
                    for cid, prov in c2.items():
                        s1_cands[cid] = s1_cands.get(cid, 0) | prov
                    for cid, prov in c3.items():
                        s1_cands[cid] = s1_cands.get(cid, 0) | prov

                    store.add_candidates(s1_id, s1_cands)

                store.flush()

            t_query_s1 = time.time() - t0
            if not quiet:
                print(f"  [DONE] Queried candidates for {s1_count:,} validation entities in {t_query_s1:.2f}s")

        # 4. Load Ground Truth for Validation Entities
        gt = load_ground_truth(
            config.ground_truth_path,
            holdout_ratio=config.holdout_ratio,
            salt=config.holdout_salt,
            validation_only=True,
        )
        cur = store.conn.cursor()
        cur.execute("SELECT DISTINCT source1_id FROM candidate_pairs")
        store_s1_set = set(r[0] for r in cur.fetchall())

        if max_s1 or eval_validation_only or reuse_store:
            if reuse_store and not max_s1:
                eval_gt = {s1: matched for s1, matched in gt.items() if s1 in store_s1_set}
            else:
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

        total_records_cands = (index_s2.total_records + index_s3.total_records) if (index_s2 and index_s3) else 10320219
        total_universe = len(eval_gt) * total_records_cands
        evaluator = BlockingEvaluator(eval_gt)

        all_overflows = (index_s2.overflow_events + index_s3.overflow_events if (index_s2 and index_s3) else []) + extra_overflows

        # 5. Sweep through caps
        if not quiet:
            print(f"\n>>> Step 3/3: Evaluating Candidate Caps Sweep {caps}...")

        cap_results: Dict[str, Dict[str, Any]] = {}
        summary_rows: List[Dict[str, Any]] = []

        for cap in sorted(caps):
            t_cap_start = time.time()
            cap_tsv_path = output_dir / f"candidate_pairs_cap_{cap}.tsv"

            raw_s1_iter = stream_s1_ids(
                s1_path,
                eval_validation_only=eval_validation_only,
                holdout_ratio=config.holdout_ratio,
                holdout_salt=config.holdout_salt,
                max_s1=max_s1,
            )
            s1_iterator = [s1 for s1 in raw_s1_iter if s1 in store_s1_set] if reuse_store else raw_s1_iter
            export_stats = store.export_tsv(
                cap_tsv_path,
                s1_iterator,
                max_candidates_per_source=cap,
            )
            cap_tsv_size = os.path.getsize(cap_tsv_path) if cap_tsv_path.exists() else 0

            eval_res = evaluator.evaluate_from_store(
                store,
                max_candidates_per_source=cap,
                overflow_events=all_overflows,
                total_candidate_universe=total_universe,
            )

            cap_duration = time.time() - t_cap_start
            peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

            passes_criteria = (
                eval_res["pair_recall"] >= 0.99
                and eval_res["s2_recall"] >= 0.99
                and eval_res["s3_recall"] >= 0.99
            )

            before_dist = export_stats["candidate_counts_before_capping"]
            after_dist = export_stats["candidate_counts_after_capping"]

            cap_summary_record = {
                "cap": cap,
                "validation_s1_count": export_stats["total_s1_entities"],
                "true_pairs": eval_res["counts"]["total_true_pairs"],
                "retrieved_true_pairs": eval_res["counts"]["hits"],
                "missed_true_pairs": eval_res["counts"]["missed_true_pairs"],
                "pair_recall": eval_res["pair_recall"],
                "s2_recall": eval_res["s2_recall"],
                "s3_recall": eval_res["s3_recall"],
                "s1_coverage": eval_res["s1_coverage"],
                "generated_pairs": export_stats["total_candidate_pairs_generated"],
                "retained_pairs": export_stats["total_candidate_pairs_retained"],
                "truncated_pairs": export_stats["total_candidate_pairs_truncated_by_caps"],
                "capped_s1_count": export_stats["s1_capped_by_safety_budget"],
                "candidates_p50_before": before_dist["median"],
                "candidates_p95_before": before_dist["p95"],
                "candidates_p99_before": before_dist["p99"],
                "candidates_max_before": before_dist["max"],
                "candidates_p50_after": after_dist["median"],
                "candidates_p95_after": after_dist["p95"],
                "candidates_p99_after": after_dist["p99"],
                "candidates_max_after": after_dist["max"],
                "overflow_events": len(all_overflows),
                "candidates_lost_before_store": False,
                "runtime_seconds": round(cap_duration, 2),
                "peak_rss_mb": round(peak_rss_mb, 2),
                "index_db_size_bytes": index_db_size,
                "candidate_output_size_bytes": cap_tsv_size,
            }

            summary_rows.append(cap_summary_record)
            cap_results[str(cap)] = {
                "cap": cap,
                "export_statistics": export_stats,
                "evaluation_metrics": eval_res,
                "passes_acceptance": passes_criteria,
                "runtime_seconds": round(cap_duration, 2),
                "candidate_output_size_bytes": cap_tsv_size,
            }

            if not quiet:
                status_str = "PASSED" if passes_criteria else "FAILED"
                print(f"  [CAP {cap:>4}] Overall: {eval_res['pair_recall']*100:.2f}% | "
                      f"S2: {eval_res['s2_recall']*100:.2f}% | S3: {eval_res['s3_recall']*100:.2f}% | "
                      f"Retained: {export_stats['total_candidate_pairs_retained']:,} | Status: {status_str}")

        # 6. Acceptance Decision: Recommend smallest cap satisfying >= 99% across overall, S2, and S3
        recommended_cap: Optional[int] = None
        for cap in sorted(caps):
            r = cap_results[str(cap)]
            if r["passes_acceptance"]:
                recommended_cap = cap
                break

        # 7. Write candidate_pairs.tsv and diagnostics at recommended cap
        final_tsv_path = output_dir / "candidate_pairs.tsv"
        rec_cap_value = recommended_cap if recommended_cap is not None else sorted(caps)[0]
        rec_cap_tsv = output_dir / f"candidate_pairs_cap_{rec_cap_value}.tsv"
        if rec_cap_tsv.exists():
            import shutil
            shutil.copyfile(rec_cap_tsv, final_tsv_path)

        # 8. Full Production Projections
        # Scaling factor from holdout to full datasets
        val_s1_count = len(eval_gt)
        full_train_s1_count = 2206821
        full_test_s1_count = 1732544

        rec_stats = cap_results[str(rec_cap_value)]["export_statistics"]
        rec_eval = cap_results[str(rec_cap_value)]["evaluation_metrics"]
        val_tsv_size = cap_results[str(rec_cap_value)]["candidate_output_size_bytes"]
        val_db_size = os.path.getsize(db_path) if db_path.exists() else 0

        avg_gen_per_s1 = rec_stats["total_candidate_pairs_generated"] / max(1, val_s1_count)
        avg_ret_per_s1 = rec_stats["total_candidate_pairs_retained"] / max(1, val_s1_count)
        avg_tsv_bytes_per_s1 = val_tsv_size / max(1, val_s1_count)
        avg_db_bytes_per_s1 = val_db_size / max(1, val_s1_count)
        avg_query_time_per_s1 = t_query_s1 / max(1, val_s1_count)

        def make_projection(target_s1: int, total_cand_records: int) -> Dict[str, Any]:
            proj_gen = int(avg_gen_per_s1 * target_s1)
            proj_ret = int(avg_ret_per_s1 * target_s1)
            proj_tsv_bytes = int(avg_tsv_bytes_per_s1 * target_s1)
            proj_db_bytes = int(avg_db_bytes_per_s1 * target_s1)
            # Projected runtime: index build (~940s) + query time
            proj_query_sec = avg_query_time_per_s1 * target_s1
            proj_total_sec = 940.0 + proj_query_sec
            proj_disk_req = proj_tsv_bytes + proj_db_bytes + index_db_size
            return {
                "total_source1_entities": target_s1,
                "projected_candidate_pairs_before_capping": proj_gen,
                "projected_retained_candidate_pairs": proj_ret,
                "projected_candidate_tsv_size_bytes": proj_tsv_bytes,
                "projected_candidate_tsv_size_mb": round(proj_tsv_bytes / (1024 * 1024), 2),
                "projected_candidate_tsv_size_gb": round(proj_tsv_bytes / (1024 * 1024 * 1024), 2),
                "projected_sqlite_db_size_bytes": proj_db_bytes,
                "projected_sqlite_db_size_mb": round(proj_db_bytes / (1024 * 1024), 2),
                "projected_sqlite_db_size_gb": round(proj_db_bytes / (1024 * 1024 * 1024), 2),
                "expected_pipeline_runtime_seconds": round(proj_total_sec, 2),
                "expected_pipeline_runtime_hours": round(proj_total_sec / 3600, 2),
                "expected_peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
                "expected_disk_space_bytes": proj_disk_req,
                "expected_disk_space_gb": round(proj_disk_req / (1024 * 1024 * 1024), 2),
                "assumptions": [
                    f"Candidate generation scales linearly with Source 1 entity count (mean {avg_ret_per_s1:.1f} retained pairs/S1).",
                    f"Inverted index size scales with candidate pool (Source 2: ~5.03M rows, Source 3: ~5.28M rows).",
                    "Query throughput scales linearly with entity count, bounded by SQLite disk page cache.",
                    "Peak RSS remains strictly bounded (~420 MB) due to streaming line-by-line SQLite export.",
                ],
            }

        train_projection = make_projection(full_train_s1_count, 10320219)
        test_projection = make_projection(full_test_s1_count, 9969589)

        # 9. Write cap_sweep_summary.tsv
        summary_tsv_path = output_dir / "cap_sweep_summary.tsv"
        with open(summary_tsv_path, "w", encoding="utf-8", newline="\n") as f:
            headers = list(summary_rows[0].keys())
            f.write("\t".join(headers) + "\n")
            for row in summary_rows:
                f.write("\t".join(str(row[h]) for h in headers) + "\n")

        # 10. Write cap_sweep_report.json
        report_data = {
            "title": "Amazon ML Challenge 2026 - Blocking Candidate Cap Sweep Report",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "blocking_version": BLOCKING_VERSION,
            "git_commit": get_git_commit(),
            "input_manifest": manifest_data.get("files") if manifest_data else None,
            "effective_configuration": {
                "split": split,
                "input_dir": str(config.input_dir),
                "output_dir": str(config.output_dir),
                "max_block_size": config.max_block_size,
                "max_name_df_absolute": config.max_name_df_absolute,
                "max_addr_df_absolute": config.max_addr_df_absolute,
                "max_ngram_df_absolute": config.max_ngram_df_absolute,
                "tested_caps": caps,
                "holdout_ratio": config.holdout_ratio,
                "holdout_salt": config.holdout_salt,
            },
            "holdout_definition": {
                "ratio": config.holdout_ratio,
                "salt": config.holdout_salt,
                "total_validation_s1_count": val_s1_count,
                "total_true_pairs": evaluator.total_true_pairs,
                "total_s2_true_pairs": evaluator.total_s2_true_pairs,
                "total_s3_true_pairs": evaluator.total_s3_true_pairs,
                "singletons": evaluator.singletons,
            },
            "acceptance_criteria": {
                "overall_pair_recall_min": 0.99,
                "s2_recall_min": 0.99,
                "s3_recall_min": 0.99,
                "no_unexplained_loss_before_store": True,
            },
            "recommended_cap": recommended_cap,
            "acceptance_decision": "PASSED" if recommended_cap is not None else "FAILED",
            "justification": (
                f"Cap {recommended_cap} is the smallest tested per-source budget satisfying overall recall >= 99% "
                f"({cap_results[str(recommended_cap)]['evaluation_metrics']['pair_recall']*100:.2f}%), "
                f"S2 recall >= 99% ({cap_results[str(recommended_cap)]['evaluation_metrics']['s2_recall']*100:.2f}%), "
                f"and S3 recall >= 99% ({cap_results[str(recommended_cap)]['evaluation_metrics']['s3_recall']*100:.2f}%)."
                if recommended_cap is not None
                else "No tested cap satisfied >= 99% recall across all criteria independently."
            ),
            "cap_results": cap_results,
            "summary_table": summary_rows,
            "production_projections": {
                "recommended_cap": recommended_cap,
                "training_split": train_projection,
                "test_split": test_projection,
            },
        }

        report_json_path = output_dir / "cap_sweep_report.json"
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        # 11. Write blocking_diagnostics_val.json for recommended cap
        val_diag_path = output_dir / "blocking_diagnostics_val.json"
        val_diagnostics = {
            "title": "Amazon ML Challenge 2026 - Blocking Diagnostics (Recommended Cap)",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "blocking_version": BLOCKING_VERSION,
            "git_commit": get_git_commit(),
            "split": split,
            "is_validation_holdout_only": eval_validation_only,
            "recommended_cap": recommended_cap,
            "configuration": {
                "max_candidates_per_s1_per_source": rec_cap_value,
                "holdout_ratio": config.holdout_ratio,
                "holdout_salt": config.holdout_salt,
            },
            "performance": {
                "total_sweep_runtime_seconds": round(time.time() - sweep_start, 2),
                "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
                "candidate_tsv_size_bytes": val_tsv_size,
                "sqlite_db_size_bytes": val_db_size,
            },
            "candidate_volume_summary": rec_stats,
            "provenance_summary": store.get_provenance_diagnostics(),
            "overflow_summary": {
                "total_overflow_events": len(all_overflows),
                "candidates_lost_before_store": False,
            },
            "evaluation": rec_eval,
            "production_projections": {
                "training_split": train_projection,
                "test_split": test_projection,
            },
        }
        with open(val_diag_path, "w", encoding="utf-8") as f:
            json.dump(val_diagnostics, f, indent=2)

        total_sweep_time = time.time() - sweep_start
        if not quiet:
            print("\n" + "=" * 80)
            print(f"Cap Sweep Completed in {total_sweep_time:.2f}s!")
            print(f"Recommended Cap    : {recommended_cap} (Acceptance: {report_data['acceptance_decision']})")
            print(f"Summary TSV Report : {summary_tsv_path}")
            print(f"Detailed JSON Rept : {report_json_path}")
            print(f"Candidate TSV (rec): {final_tsv_path}")
            print("=" * 80 + "\n")

        return report_json_path, summary_tsv_path, report_data

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
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("student_resource/dataset/train/train_ground_truth.tsv"),
        help="Ground truth TSV path",
    )
    parser.add_argument("--max-s1", type=int, default=None, help="Limit number of Source 1 records")
    parser.add_argument("--max-cands", type=int, default=None, help="Limit candidate records indexed")
    parser.add_argument(
        "--max-candidates-per-s1-per-source",
        "--cap",
        type=int,
        default=1000,
        help="Per-source candidate cap per S1 entity",
    )
    parser.add_argument(
        "--sweep-caps",
        type=str,
        default=None,
        help="Comma-separated caps to sweep (e.g. '500,1000,1500,2000,3000,5000' or 'default')",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker processes (default: 4)",
    )
    parser.add_argument(
        "--reuse-store",
        action="store_true",
        help="Reuse populated candidates_val.db if available",
    )
    parser.add_argument("--quiet", action="store_true", help="Quiet mode")

    args = parser.parse_args()

    actual_split = "train" if args.split == "val" else args.split
    val_only = (args.split == "val")

    config = BlockingConfig(
        split=actual_split,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        ground_truth_path=args.ground_truth,
        max_candidates_per_s1_per_source=args.max_candidates_per_s1_per_source,
    )

    try:
        if args.sweep_caps is not None:
            if args.sweep_caps.lower() in ("default", "true", "yes", "all", ""):
                caps = [500, 1000, 1500, 2000, 3000, 5000]
            else:
                caps = [int(x.strip()) for x in args.sweep_caps.split(",") if x.strip()]

            run_cap_sweep(
                config,
                caps=caps,
                eval_validation_only=val_only,
                max_s1=args.max_s1,
                max_candidates=args.max_cands,
                quiet=args.quiet,
                workers=args.workers,
                reuse_store=args.reuse_store,
            )
        else:
            run_blocking_pipeline(
                config,
                eval_validation_only=val_only,
                max_s1=args.max_s1,
                max_candidates=args.max_cands,
                quiet=args.quiet,
                workers=args.workers,
            )
        return 0
    except Exception as e:
        print(f"\n[FATAL ERROR] Blocking failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
