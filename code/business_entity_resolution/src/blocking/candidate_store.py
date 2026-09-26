"""
Disk-Backed SQLite Candidate Store & Deterministic Exporter
Amazon ML Challenge 2026 - Business Entity Resolution

Features:
- Composite PRIMARY KEY (source1_id, cand_id) WITHOUT ROWID
- Bitmask provenance tracking with bitwise OR on conflict
- Deterministic evidence ranking with stable candidate ID tie-breaking
- Streaming TSV export: source1_entity_id <TAB> candidate_entity_ids
- Strict S2-first (sorted), then S3-second (sorted) ordering
- Exactly one row per Source 1 entity, strictly preserving S1 input order
- Transactional atomic export with rollback cleanup on failure
- Comprehensive capping and percentile distributions before and after capping
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

from blocking.config import (
    PROV_ADDRESS_NUMBER_LOCATION,
    PROV_ADDRESS_TOKEN,
    PROV_COMPACT_CHAR_NGRAM,
    PROV_CROSS_COUNTRY_HIGH_SPECIFICITY,
    PROV_EXACT_NAME,
    PROV_MISSING_COUNTRY_FALLBACK,
    PROV_RARE_NAME_TOKEN,
    PROVENANCE_NAMES,
)


def compute_candidate_evidence_tier(provenance: int) -> Tuple[int, int, int, int, int, int, int, int]:
    """
    Deterministic candidate priority ranking based strictly on blocking evidence:
    1. Exact normalized-name evidence
    2. Multiple independent block-family hits (count of active bits)
    3. Rare name token evidence
    4. Address-number plus location evidence
    5. Informative address-token evidence
    6. Compact n-gram evidence
    7. Missing country fallback evidence
    8. Cross-country high-specificity evidence
    """
    return (
        1 if (provenance & PROV_EXACT_NAME) else 0,
        bin(provenance).count("1"),
        1 if (provenance & PROV_RARE_NAME_TOKEN) else 0,
        1 if (provenance & PROV_ADDRESS_NUMBER_LOCATION) else 0,
        1 if (provenance & PROV_ADDRESS_TOKEN) else 0,
        1 if (provenance & PROV_COMPACT_CHAR_NGRAM) else 0,
        1 if (provenance & PROV_MISSING_COUNTRY_FALLBACK) else 0,
        1 if (provenance & PROV_CROSS_COUNTRY_HIGH_SPECIFICITY) else 0,
    )


def rank_and_cap_candidates(
    candidates: List[Tuple[str, int]],
    max_cap: int,
) -> Tuple[List[str], int]:
    """
    Deterministic ranking:
    1. Primary sort: Candidate ID ascending (for stable tie-breaking).
    2. Secondary sort: Evidence tier descending (using Python's stable timsort).
    Returns (retained_candidate_ids, truncated_count).
    """
    if len(candidates) <= max_cap:
        return sorted(cid for cid, _ in candidates), 0

    truncated_count = len(candidates) - max_cap
    # Deterministic tie-breaking: ID ascending first
    candidates.sort(key=lambda x: x[0])
    # Rank by evidence tier descending
    candidates.sort(key=lambda x: compute_candidate_evidence_tier(x[1]), reverse=True)
    retained = candidates[:max_cap]
    return sorted(cid for cid, _ in retained), truncated_count


def calculate_distribution(counts: List[int]) -> Dict[str, Any]:
    """Calculate min, median, mean, p95, p99, and max from a list of counts."""
    if not counts:
        return {"min": 0, "median": 0, "mean": 0.0, "p95": 0, "p99": 0, "max": 0}
    s = sorted(counts)
    n = len(s)
    return {
        "min": s[0],
        "median": s[n // 2],
        "mean": round(sum(s) / n, 2),
        "p95": s[int(n * 0.95)],
        "p99": s[int(n * 0.99)],
        "max": s[-1],
    }


class CandidateStore:
    """
    Disk-backed SQLite candidate store managing candidate pair deduplication,
    provenance aggregation, and deterministic export.
    """

    def __init__(
        self,
        db_path: Path,
        cache_size_kb: int = 64000,
        batch_size: int = 50000,
    ) -> None:
        self.db_path = db_path
        self.batch_size = batch_size
        self.batch: List[Tuple[str, str, int]] = []

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.cur = self.conn.cursor()

        # Performance tuning pragmas
        self.cur.execute(f"PRAGMA cache_size = -{cache_size_kb}")
        self.cur.execute("PRAGMA journal_mode = WAL")
        self.cur.execute("PRAGMA synchronous = OFF")

        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS candidate_pairs (
                source1_id TEXT NOT NULL,
                cand_id TEXT NOT NULL,
                provenance INTEGER NOT NULL,
                PRIMARY KEY (source1_id, cand_id)
            ) WITHOUT ROWID;
        """)
        self.conn.commit()

    def add_candidates(self, s1_id: str, candidates: Dict[str, int]) -> None:
        """Add candidate IDs with their provenance bitmask for an S1 entity."""
        for cid, prov in candidates.items():
            self.batch.append((s1_id, cid, prov))
            if len(self.batch) >= self.batch_size:
                self.flush()

    def flush(self) -> None:
        """Flush in-memory batch to SQLite with ON CONFLICT bitwise OR update."""
        if not self.batch:
            return
        self.cur.executemany("""
            INSERT INTO candidate_pairs (source1_id, cand_id, provenance)
            VALUES (?, ?, ?)
            ON CONFLICT(source1_id, cand_id) DO UPDATE SET
                provenance = provenance | excluded.provenance
        """, self.batch)
        self.conn.commit()
        self.batch.clear()

    def get_candidates_for_s1(self, s1_id: str) -> Dict[str, int]:
        """Fetch candidates and provenance bitmasks for a specific Source 1 ID."""
        self.flush()
        self.cur.execute("""
            SELECT cand_id, provenance
            FROM candidate_pairs
            WHERE source1_id = ?
        """, (s1_id,))
        return {cid: prov for cid, prov in self.cur.fetchall()}

    def get_capped_candidates_for_s1(
        self,
        s1_id: str,
        max_candidates_per_source: int,
    ) -> Dict[str, int]:
        """
        Fetch candidates for S1 and apply deterministic ranking and capping per source.
        Returns Dict[cand_id, provenance].
        """
        all_cands = self.get_candidates_for_s1(s1_id)
        if not all_cands:
            return {}

        s2_cands = [(cid, prov) for cid, prov in all_cands.items() if cid.startswith("S2-")]
        s3_cands = [(cid, prov) for cid, prov in all_cands.items() if cid.startswith("S3-")]

        s2_retained, _ = rank_and_cap_candidates(s2_cands, max_candidates_per_source)
        s3_retained, _ = rank_and_cap_candidates(s3_cands, max_candidates_per_source)

        retained_set = set(s2_retained + s3_retained)
        return {cid: prov for cid, prov in all_cands.items() if cid in retained_set}

    def close(self) -> None:
        """Flush any pending rows and close database connection."""
        self.flush()
        if hasattr(self, "conn") and self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None

    def export_tsv(
        self,
        tsv_path: Path,
        s1_ids_or_source: Union[List[str], Path, Iterable[str]],
        max_candidates_per_source: int = 1000,
    ) -> Dict[str, Any]:
        """
        Export final candidate_pairs.tsv following strict requirements:
        1. Streaming export: line-by-line streaming without buffering millions of rows in memory.
        2. One row per Source 1 entity, in original input order.
        3. Candidate IDs comma-separated, empty string if none.
        4. Deterministic order: all S2 IDs first (sorted), then all S3 IDs (sorted).
        5. Configurable per-source safety budget with deterministic evidence ranking.
        6. Transactional atomic write: writes to temporary file and renames on success.
        """
        self.flush()
        tsv_path.parent.mkdir(parents=True, exist_ok=True)
        temp_tsv = tsv_path.with_suffix(f".tmp_{os.getpid()}")

        capped_s1_count = 0
        total_pairs_generated = 0
        total_pairs_retained = 0

        total_s2_generated = 0
        total_s2_retained = 0
        total_s2_truncated = 0

        total_s3_generated = 0
        total_s3_retained = 0
        total_s3_truncated = 0

        counts_before_capping: List[int] = []
        counts_after_capping: List[int] = []

        try:
            # Open streaming S1 generator
            if isinstance(s1_ids_or_source, Path):
                def s1_stream():
                    with open(s1_ids_or_source, "r", encoding="utf-8") as f:
                        f.readline()  # Skip header
                        for line in f:
                            parts = line.rstrip("\r\n").split("\t")
                            if parts and parts[0]:
                                yield parts[0]
                s1_iterator = s1_stream()
            else:
                s1_iterator = iter(s1_ids_or_source)

            with open(temp_tsv, "w", encoding="utf-8", newline="\n") as out_f:
                for s1_id in s1_iterator:
                    self.cur.execute("""
                        SELECT cand_id, provenance
                        FROM candidate_pairs
                        WHERE source1_id = ?
                    """, (s1_id,))
                    rows = self.cur.fetchall()

                    if not rows:
                        out_f.write(f"{s1_id}\t\n")
                        counts_before_capping.append(0)
                        counts_after_capping.append(0)
                        continue

                    # Separate candidates by source
                    s2_cands: List[Tuple[str, int]] = []
                    s3_cands: List[Tuple[str, int]] = []

                    for cid, prov in rows:
                        if cid.startswith("S2-"):
                            s2_cands.append((cid, prov))
                        elif cid.startswith("S3-"):
                            s3_cands.append((cid, prov))

                    s2_gen = len(s2_cands)
                    s3_gen = len(s3_cands)
                    total_s1_gen = s2_gen + s3_gen
                    total_pairs_generated += total_s1_gen
                    total_s2_generated += s2_gen
                    total_s3_generated += s3_gen
                    counts_before_capping.append(total_s1_gen)

                    # Rank and cap deterministically
                    s2_retained_ids, s2_trunc = rank_and_cap_candidates(s2_cands, max_candidates_per_source)
                    s3_retained_ids, s3_trunc = rank_and_cap_candidates(s3_cands, max_candidates_per_source)

                    total_s2_retained += len(s2_retained_ids)
                    total_s2_truncated += s2_trunc
                    total_s3_retained += len(s3_retained_ids)
                    total_s3_truncated += s3_trunc

                    if s2_trunc > 0 or s3_trunc > 0:
                        capped_s1_count += 1

                    # Deterministic final order: all S2 IDs sorted, then all S3 IDs sorted
                    final_ids = s2_retained_ids + s3_retained_ids
                    total_pairs_retained += len(final_ids)
                    counts_after_capping.append(len(final_ids))

                    out_f.write(f"{s1_id}\t{','.join(final_ids)}\n")

            # Transactional atomic replacement
            temp_tsv.replace(tsv_path)

        except Exception:
            if temp_tsv.exists():
                temp_tsv.unlink(missing_ok=True)
            raise

        dist_before = calculate_distribution(counts_before_capping)
        dist_after = calculate_distribution(counts_after_capping)

        return {
            "total_s1_entities": len(counts_after_capping),
            "total_candidate_pairs_generated": total_pairs_generated,
            "total_candidate_pairs_retained": total_pairs_retained,
            "total_candidate_pairs_truncated_by_caps": total_s2_truncated + total_s3_truncated,
            "s1_capped_by_safety_budget": capped_s1_count,
            "source_breakdown": {
                "S2": {
                    "generated": total_s2_generated,
                    "retained": total_s2_retained,
                    "truncated": total_s2_truncated,
                },
                "S3": {
                    "generated": total_s3_generated,
                    "retained": total_s3_retained,
                    "truncated": total_s3_truncated,
                },
            },
            "candidate_counts_before_capping": dist_before,
            "candidate_counts_after_capping": dist_after,
            # For backward compatibility with existing diagnostic keys
            "candidates_per_s1": dist_after,
        }

    def get_provenance_diagnostics(self) -> Dict[str, Any]:
        """Aggregate provenance bitmask counts across all stored candidate pairs."""
        self.flush()
        self.cur.execute("SELECT provenance, count(*) FROM candidate_pairs GROUP BY provenance")
        prov_rows = self.cur.fetchall()

        total_pairs = sum(cnt for _, cnt in prov_rows)
        bit_counts: Dict[str, int] = {name: 0 for name in PROVENANCE_NAMES.values()}

        for prov_mask, cnt in prov_rows:
            for flag, name in PROVENANCE_NAMES.items():
                if prov_mask & flag:
                    bit_counts[name] += cnt

        # Breakdown by source
        self.cur.execute("SELECT substr(cand_id, 1, 2), count(*) FROM candidate_pairs GROUP BY substr(cand_id, 1, 2)")
        source_counts = {src: cnt for src, cnt in self.cur.fetchall()}

        return {
            "total_candidate_pairs": total_pairs,
            "candidate_pairs_by_source": source_counts,
            "provenance_breakdown": bit_counts,
        }
