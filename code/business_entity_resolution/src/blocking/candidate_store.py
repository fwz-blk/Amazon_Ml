"""
Disk-Backed SQLite Candidate Store & Deterministic Exporter
Amazon ML Challenge 2026 - Business Entity Resolution

Features:
- Composite PRIMARY KEY (source1_id, cand_id) WITHOUT ROWID
- Bitmask provenance tracking with bitwise OR on conflict
- Deterministic ranking and safety budget truncation
- TSV export: source1_entity_id <TAB> candidate_entity_ids (S2 sorted, then S3 sorted)
- One row per Source 1 entity, strictly preserving S1 input order
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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


def compute_candidate_rank(cand_id: str, provenance: int) -> Tuple[int, int, int, int, int, int, str]:
    """
    Deterministic candidate priority ranking based strictly on blocking evidence:
    1. Exact normalized-name evidence
    2. Multiple independent block-family hits (count of active bits)
    3. Rare name token evidence
    4. Address-number plus location evidence
    5. Informative address-token evidence
    6. Compact n-gram evidence
    7. Deterministic candidate ID order (tie-breaker)
    """
    return (
        1 if (provenance & PROV_EXACT_NAME) else 0,
        bin(provenance).count("1"),
        1 if (provenance & PROV_RARE_NAME_TOKEN) else 0,
        1 if (provenance & PROV_ADDRESS_NUMBER_LOCATION) else 0,
        1 if (provenance & PROV_ADDRESS_TOKEN) else 0,
        1 if (provenance & PROV_COMPACT_CHAR_NGRAM) else 0,
        cand_id,  # tie-breaker
    )


class CandidateStore:
    """
    Disk-backed SQLite candidate store managing candidate pair deduplication,
    provenance aggregation, and deterministic export.
    """

    def __init__(
        self,
        db_path: Path,
        cache_size_kb: int = 4000,
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

    def close(self) -> None:
        """Flush any pending rows and close database connection."""
        self.flush()
        self.conn.close()

    def export_tsv(
        self,
        tsv_path: Path,
        all_s1_ids: List[str],
        max_candidates_per_source: int = 500,
    ) -> Dict[str, Any]:
        """
        Export final candidate_pairs.tsv following strict requirements:
        1. One row per Source 1 entity, in original input order.
        2. Candidate IDs comma-separated, empty string if none.
        3. Deterministic order: all S2 IDs first (sorted), then all S3 IDs (sorted).
        4. Configurable per-source safety budget with deterministic evidence ranking.

        Returns diagnostics on candidate volumes and cap triggers.
        """
        self.flush()
        tsv_path.parent.mkdir(parents=True, exist_ok=True)

        # Build in-memory index or query by S1
        # For memory efficiency, query sorted by source1_id
        capped_s1_count = 0
        total_pairs_retained = 0
        s1_candidate_counts: List[int] = []

        # Stream candidate rows grouped by source1_id
        self.cur.execute("""
            SELECT source1_id, cand_id, provenance
            FROM candidate_pairs
            ORDER BY source1_id
        """)

        # Buffer candidates per S1
        grouped_candidates: Dict[str, List[Tuple[str, int]]] = {}
        row = self.cur.fetchone()

        temp_tsv = tsv_path.with_suffix(".tmp")
        with open(temp_tsv, "w", encoding="utf-8", newline="\n") as out_f:
            for s1_id in all_s1_ids:
                # Collect candidates for this s1_id from SQL stream if matched
                # Since SQL stream is ordered by source1_id and all_s1_ids might have a different order,
                # we query per-s1 or populate a lookup. For maximum reliability across millions of S1,
                # let's fetch by chunks or indexing.
                pass

        # To handle any arbitrary S1 input order, create an index on source1_id
        # In SQLite WITHOUT ROWID on (source1_id, cand_id), the table is ALREADY indexed by source1_id!
        with open(temp_tsv, "w", encoding="utf-8", newline="\n") as out_f:
            for s1_id in all_s1_ids:
                self.cur.execute("""
                    SELECT cand_id, provenance
                    FROM candidate_pairs
                    WHERE source1_id = ?
                """, (s1_id,))
                rows = self.cur.fetchall()

                if not rows:
                    out_f.write(f"{s1_id}\t\n")
                    s1_candidate_counts.append(0)
                    continue

                # Separate by candidate source
                s2_cands: List[Tuple[str, int]] = []
                s3_cands: List[Tuple[str, int]] = []

                for cid, prov in rows:
                    if cid.startswith("S2-"):
                        s2_cands.append((cid, prov))
                    elif cid.startswith("S3-"):
                        s3_cands.append((cid, prov))

                # Apply deterministic safety budget if needed
                s1_was_capped = False
                if len(s2_cands) > max_candidates_per_source:
                    s2_cands.sort(
                        key=lambda x: (
                            compute_candidate_rank(x[0], x[1])[:6]
                        ),
                        reverse=True,
                    )
                    # For tie-breaking sort ascending by ID for same rank
                    # Python's timsort is stable, so sort by ID ascending first, then rank descending
                    s2_cands.sort(key=lambda x: x[0])
                    s2_cands.sort(
                        key=lambda x: compute_candidate_rank(x[0], x[1])[:6],
                        reverse=True,
                    )
                    s2_cands = s2_cands[:max_candidates_per_source]
                    s1_was_capped = True

                if len(s3_cands) > max_candidates_per_source:
                    s3_cands.sort(key=lambda x: x[0])
                    s3_cands.sort(
                        key=lambda x: compute_candidate_rank(x[0], x[1])[:6],
                        reverse=True,
                    )
                    s3_cands = s3_cands[:max_candidates_per_source]
                    s1_was_capped = True

                if s1_was_capped:
                    capped_s1_count += 1

                # Deterministic final order: all S2 IDs sorted, then all S3 IDs sorted
                s2_sorted_ids = sorted(c[0] for c in s2_cands)
                s3_sorted_ids = sorted(c[0] for c in s3_cands)
                final_ids = s2_sorted_ids + s3_sorted_ids

                total_pairs_retained += len(final_ids)
                s1_candidate_counts.append(len(final_ids))
                out_f.write(f"{s1_id}\t{','.join(final_ids)}\n")

        # Atomic replace
        if tsv_path.exists():
            tsv_path.unlink()
        temp_tsv.rename(tsv_path)

        s1_candidate_counts.sort()
        count_len = len(s1_candidate_counts)
        median = s1_candidate_counts[count_len // 2] if count_len else 0
        p95 = s1_candidate_counts[int(count_len * 0.95)] if count_len else 0
        max_c = s1_candidate_counts[-1] if count_len else 0
        mean = total_pairs_retained / count_len if count_len else 0

        return {
            "total_s1_entities": count_len,
            "total_candidate_pairs_retained": total_pairs_retained,
            "s1_capped_by_safety_budget": capped_s1_count,
            "candidates_per_s1": {
                "min": s1_candidate_counts[0] if count_len else 0,
                "median": median,
                "mean": round(mean, 2),
                "p95": p95,
                "max": max_c,
            },
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
