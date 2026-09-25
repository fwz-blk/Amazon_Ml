"""
Candidate Source Inverted Index (Disk-Backed SQLite)
Amazon ML Challenge 2026 - Business Entity Resolution

Features:
- Disk-backed SQLite indexing with bounded page cache memory
- Strict placement of database and temporary sort files on workspace disk
- Covering B-tree indexes for fast zero-heap streaming queries
- Open-set country partitioning with observed countries tracking
- High-DF key filtering (absolute and relative thresholds)
- Pathological block overflow tracking with multi-token refinement
- Retained overflow postings (never silently dropped to zero candidates)
- Missing-country and cross-country high-specificity fallbacks
- Guaranteed cleanup of temporary databases on success and failure
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from blocking.config import BlockingConfig
from blocking.keys import (
    extract_address_number_location_keys,
    extract_compact_ngrams,
    tokenize_address,
    tokenize_name,
)


class CandidateSourceIndex:
    """
    Disk-backed SQLite inverted index over a candidate source (Source 2 or Source 3).
    Supports country-partitioned queries across all 6 blocking pass families with
    bounded memory usage.
    """

    def __init__(
        self,
        source_name: str,
        config: BlockingConfig,
        temp_dir: Optional[Path] = None,
        in_memory: bool = False,
    ) -> None:
        self.source_name = source_name
        self.config = config
        self.total_records = 0
        self.observed_countries: Set[str] = set()

        # Overflow tracking
        self.overflow_events: List[Dict[str, Any]] = []

        # High-DF filtered keys
        self.filtered_name_tokens: Set[Tuple[str, str]] = set()
        self.filtered_addr_tokens: Set[Tuple[str, str]] = set()
        self.filtered_num_loc: Set[Tuple[str, str, str]] = set()
        self.filtered_ngrams: Set[Tuple[str, str]] = set()

        # Database setup
        self.in_memory = in_memory
        if in_memory:
            self.db_path: Optional[Path] = None
            self.conn = sqlite3.connect(":memory:")
        else:
            base_temp = temp_dir or config.temp_dir or (config.output_dir / ".idx_tmp")
            base_temp.mkdir(parents=True, exist_ok=True)
            # Ensure SQLite temporary files (sort files during CREATE INDEX) are placed on disk
            os.environ["SQLITE_TMPDIR"] = str(base_temp.resolve())
            os.environ["TMPDIR"] = str(base_temp.resolve())

            self.db_path = base_temp / f"idx_{source_name}_{uuid.uuid4().hex[:8]}.db"
            self.conn = sqlite3.connect(str(self.db_path))

        self.cur = self.conn.cursor()

        # Memory bounded PRAGMAs
        self.cur.execute("PRAGMA synchronous = OFF")
        self.cur.execute("PRAGMA journal_mode = OFF")
        self.cur.execute(f"PRAGMA cache_size = -{config.sqlite_cache_size_kb}")
        self.cur.execute("PRAGMA temp_store = FILE")

        # Create unindexed staging tables for fast bulk insertion
        self._create_tables()

        # Streaming batch buffers
        self._batch_exact: List[Tuple[str, str, str]] = []
        self._batch_name_tok: List[Tuple[str, str, str]] = []
        self._batch_addr_tok: List[Tuple[str, str, str]] = []
        self._batch_num_loc: List[Tuple[str, str, str, str]] = []
        self._batch_ngram: List[Tuple[str, str, str]] = []

    def _create_tables(self) -> None:
        """Create postings tables."""
        self.cur.execute("""
            CREATE TABLE postings_exact_name (
                name_norm TEXT NOT NULL,
                country TEXT NOT NULL,
                cand_id TEXT NOT NULL
            );
        """)
        self.cur.execute("""
            CREATE TABLE postings_name_token (
                token TEXT NOT NULL,
                country TEXT NOT NULL,
                cand_id TEXT NOT NULL
            );
        """)
        self.cur.execute("""
            CREATE TABLE postings_addr_token (
                token TEXT NOT NULL,
                country TEXT NOT NULL,
                cand_id TEXT NOT NULL
            );
        """)
        self.cur.execute("""
            CREATE TABLE postings_num_loc (
                num TEXT NOT NULL,
                token TEXT NOT NULL,
                country TEXT NOT NULL,
                cand_id TEXT NOT NULL
            );
        """)
        self.cur.execute("""
            CREATE TABLE postings_ngram (
                ngram TEXT NOT NULL,
                country TEXT NOT NULL,
                cand_id TEXT NOT NULL
            );
        """)
        self.conn.commit()

    def add_candidate(
        self,
        cand_id: str,
        country_norm: str,
        name_norm: str,
        addr_norm: str,
    ) -> None:
        """Extract keys for a candidate entity and buffer postings for batch insertion."""
        self.total_records += 1
        if country_norm:
            self.observed_countries.add(country_norm)

        # 1. Exact normalized name
        if name_norm:
            self._batch_exact.append((name_norm, country_norm, cand_id))

            # Name tokens
            tokens = tokenize_name(name_norm, self.config.legal_stoplist, self.config.min_token_length)
            for tok in set(tokens):
                self._batch_name_tok.append((tok, country_norm, cand_id))

            # Compact character n-grams
            if country_norm:
                ngrams = extract_compact_ngrams(name_norm, self.config.ngram_size, self.config.min_ngram_length)
                for ng in set(ngrams):
                    self._batch_ngram.append((ng, country_norm, cand_id))

        # 2. Address tokens & number-location keys
        if addr_norm:
            atoks = tokenize_address(
                addr_norm, self.config.common_address_terms, self.config.min_addr_token_length
            )
            for atok in set(atoks):
                self._batch_addr_tok.append((atok, country_norm, cand_id))

            num_locs = extract_address_number_location_keys(addr_norm, self.config.min_addr_token_length)
            for num, tok in num_locs:
                self._batch_num_loc.append((num, tok, country_norm, cand_id))

        total_buffered = (
            len(self._batch_exact)
            + len(self._batch_name_tok)
            + len(self._batch_addr_tok)
            + len(self._batch_num_loc)
            + len(self._batch_ngram)
        )
        if total_buffered >= self.config.batch_size:
            self._flush_batches()

    def _flush_batches(self) -> None:
        """Flush in-memory buffers into SQLite."""
        if self._batch_exact:
            self.cur.executemany("INSERT INTO postings_exact_name VALUES (?, ?, ?)", self._batch_exact)
            self._batch_exact.clear()
        if self._batch_name_tok:
            self.cur.executemany("INSERT INTO postings_name_token VALUES (?, ?, ?)", self._batch_name_tok)
            self._batch_name_tok.clear()
        if self._batch_addr_tok:
            self.cur.executemany("INSERT INTO postings_addr_token VALUES (?, ?, ?)", self._batch_addr_tok)
            self._batch_addr_tok.clear()
        if self._batch_num_loc:
            self.cur.executemany("INSERT INTO postings_num_loc VALUES (?, ?, ?, ?)", self._batch_num_loc)
            self._batch_num_loc.clear()
        if self._batch_ngram:
            self.cur.executemany("INSERT INTO postings_ngram VALUES (?, ?, ?)", self._batch_ngram)
            self._batch_ngram.clear()
        self.conn.commit()

    def commit_index(self) -> None:
        """
        Finalize index build:
        1. Flush remaining buffers.
        2. Create covering B-tree indexes.
        3. Identify and prune high Document Frequency (DF) keys.
        """
        self._flush_batches()

        # Create covering indexes (prefix ordered for dual same-country and global scans)
        self.cur.execute("CREATE INDEX idx_en ON postings_exact_name(name_norm, country, cand_id);")
        self.cur.execute("CREATE INDEX idx_nt ON postings_name_token(token, country, cand_id);")
        self.cur.execute("CREATE INDEX idx_at ON postings_addr_token(token, country, cand_id);")
        self.cur.execute("CREATE INDEX idx_nl ON postings_num_loc(num, token, country, cand_id);")
        self.cur.execute("CREATE INDEX idx_ng ON postings_ngram(ngram, country, cand_id);")
        self.conn.commit()

        # Compute Document Frequency (DF) filtering thresholds
        max_name_df = self.config.get_max_name_df(self.total_records)
        max_addr_df = self.config.get_max_addr_df(self.total_records)
        max_ngram_df = self.config.get_max_ngram_df(self.total_records)

        # High-DF name tokens
        self.cur.execute(
            "SELECT country, token FROM postings_name_token GROUP BY token, country HAVING count(*) > ?",
            (max_name_df,),
        )
        self.filtered_name_tokens = set(self.cur.fetchall())

        # High-DF address tokens
        self.cur.execute(
            "SELECT country, token FROM postings_addr_token GROUP BY token, country HAVING count(*) > ?",
            (max_addr_df,),
        )
        self.filtered_addr_tokens = set(self.cur.fetchall())

        # High-DF number-location keys
        self.cur.execute(
            "SELECT country, num, token FROM postings_num_loc GROUP BY num, token, country HAVING count(*) > ?",
            (max_addr_df,),
        )
        self.filtered_num_loc = set(self.cur.fetchall())

        # High-DF n-grams
        self.cur.execute(
            "SELECT country, ngram FROM postings_ngram GROUP BY ngram, country HAVING count(*) > ?",
            (max_ngram_df,),
        )
        self.filtered_ngrams = set(self.cur.fetchall())

    def build_from_tsv(self, tsv_path: Path, max_rows: Optional[int] = None) -> None:
        """Build inverted index from a cleaned TSV file in a single streaming pass."""
        with open(tsv_path, "r", encoding="utf-8") as f:
            header_line = f.readline()
            headers = [h.strip() for h in header_line.rstrip("\r\n").split("\t")]
            assert len(headers) >= 10, f"Expected 10 cleaned columns in {tsv_path}, got {len(headers)}"

            count = 0
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 10:
                    continue

                eid = parts[0]
                name_norm = parts[4]
                addr_norm = parts[5]
                ctry_norm = parts[6]

                self.add_candidate(eid, ctry_norm, name_norm, addr_norm)
                count += 1
                if max_rows and count >= max_rows:
                    break

        self.commit_index()

    # --- Query Methods ---

    def query_exact_name(self, country: str, name_norm: str) -> List[str]:
        """Query same-country exact normalized name."""
        if not name_norm:
            return []
        self.cur.execute("""
            SELECT cand_id FROM postings_exact_name
            WHERE name_norm = ? AND country = ?
            ORDER BY cand_id
        """, (name_norm, country))
        return [r[0] for r in self.cur.fetchall()]

    def query_name_tokens(
        self,
        s1_id: str,
        country: str,
        tokens: List[str],
    ) -> List[str]:
        """
        Query rare name tokens with pathological block overflow tracking and refinement.
        Oversized blocks are refined using deterministic 2-token intersections where possible,
        and retained (never discarded to zero candidates) for final safety budget ranking.
        """
        cands: Set[str] = set()
        distinct_tokens = list(dict.fromkeys(tokens))

        for i, tok in enumerate(distinct_tokens):
            key = (country, tok)
            if key in self.filtered_name_tokens:
                self.overflow_events.append({
                    "s1_id": s1_id,
                    "source": self.source_name,
                    "block_family": "rare_name_token",
                    "key": tok,
                    "generated_count": -1,
                    "retained_count": 0,
                    "truncated_count": 0,
                    "is_filtered": True,
                    "refinement_attempted": False,
                    "refined": False,
                })
                continue

            self.cur.execute("""
                SELECT cand_id FROM postings_name_token
                WHERE token = ? AND country = ?
                ORDER BY cand_id
            """, (tok, country))
            postings = [r[0] for r in self.cur.fetchall()]
            if not postings:
                continue

            if len(postings) <= self.config.max_block_size:
                cands.update(postings)
            else:
                # Pathological block overflow: attempt refinement with a second token
                refined = False
                for j, tok2 in enumerate(distinct_tokens):
                    if i == j or (country, tok2) in self.filtered_name_tokens:
                        continue
                    self.cur.execute("""
                        SELECT a.cand_id FROM postings_name_token a
                        JOIN postings_name_token b ON a.cand_id = b.cand_id
                        WHERE a.token = ? AND a.country = ? AND b.token = ? AND b.country = ?
                        ORDER BY a.cand_id
                    """, (tok, country, tok2, country))
                    intersect = [r[0] for r in self.cur.fetchall()]
                    if 0 < len(intersect) <= self.config.max_block_size:
                        cands.update(intersect)
                        refined = True
                        self.overflow_events.append({
                            "s1_id": s1_id,
                            "source": self.source_name,
                            "block_family": "rare_name_token",
                            "key": f"{tok}+{tok2}",
                            "generated_count": len(postings),
                            "retained_count": len(intersect),
                            "truncated_count": len(postings) - len(intersect),
                            "is_filtered": False,
                            "refinement_attempted": True,
                            "refined": True,
                        })
                        break

                if not refined:
                    # Crucial: retain oversized postings rather than discarding to zero!
                    cands.update(postings)
                    self.overflow_events.append({
                        "s1_id": s1_id,
                        "source": self.source_name,
                        "block_family": "rare_name_token",
                        "key": tok,
                        "generated_count": len(postings),
                        "retained_count": len(postings),
                        "truncated_count": 0,
                        "is_filtered": False,
                        "refinement_attempted": True,
                        "refined": False,
                    })

        return sorted(cands)

    def query_address_tokens(
        self,
        s1_id: str,
        country: str,
        tokens: List[str],
    ) -> List[str]:
        """Query distinctive address tokens with overflow refinement and retention."""
        cands: Set[str] = set()
        distinct_tokens = list(dict.fromkeys(tokens))

        for i, tok in enumerate(distinct_tokens):
            key = (country, tok)
            if key in self.filtered_addr_tokens:
                continue

            self.cur.execute("""
                SELECT cand_id FROM postings_addr_token
                WHERE token = ? AND country = ?
                ORDER BY cand_id
            """, (tok, country))
            postings = [r[0] for r in self.cur.fetchall()]
            if not postings:
                continue

            if len(postings) <= self.config.max_block_size:
                cands.update(postings)
            else:
                # Attempt refinement with a second address token
                refined = False
                for j, tok2 in enumerate(distinct_tokens):
                    if i == j or (country, tok2) in self.filtered_addr_tokens:
                        continue
                    self.cur.execute("""
                        SELECT a.cand_id FROM postings_addr_token a
                        JOIN postings_addr_token b ON a.cand_id = b.cand_id
                        WHERE a.token = ? AND a.country = ? AND b.token = ? AND b.country = ?
                        ORDER BY a.cand_id
                    """, (tok, country, tok2, country))
                    intersect = [r[0] for r in self.cur.fetchall()]
                    if 0 < len(intersect) <= self.config.max_block_size:
                        cands.update(intersect)
                        refined = True
                        self.overflow_events.append({
                            "s1_id": s1_id,
                            "source": self.source_name,
                            "block_family": "address_token",
                            "key": f"{tok}+{tok2}",
                            "generated_count": len(postings),
                            "retained_count": len(intersect),
                            "truncated_count": len(postings) - len(intersect),
                            "is_filtered": False,
                            "refinement_attempted": True,
                            "refined": True,
                        })
                        break

                if not refined:
                    cands.update(postings)
                    self.overflow_events.append({
                        "s1_id": s1_id,
                        "source": self.source_name,
                        "block_family": "address_token",
                        "key": tok,
                        "generated_count": len(postings),
                        "retained_count": len(postings),
                        "truncated_count": 0,
                        "is_filtered": False,
                        "refinement_attempted": True,
                        "refined": False,
                    })

        return sorted(cands)

    def query_num_loc(
        self,
        s1_id: str,
        country: str,
        num_loc_keys: List[Tuple[str, str]],
    ) -> List[str]:
        """Query address number + location pairs with overflow refinement and retention."""
        cands: Set[str] = set()
        for i, (num, tok) in enumerate(num_loc_keys):
            key = (country, num, tok)
            if key in self.filtered_num_loc:
                continue

            self.cur.execute("""
                SELECT cand_id FROM postings_num_loc
                WHERE num = ? AND token = ? AND country = ?
                ORDER BY cand_id
            """, (num, tok, country))
            postings = [r[0] for r in self.cur.fetchall()]
            if not postings:
                continue

            if len(postings) <= self.config.max_block_size:
                cands.update(postings)
            else:
                # Attempt refinement with a second num_loc key
                refined = False
                for j, (num2, tok2) in enumerate(num_loc_keys):
                    if i == j or (country, num2, tok2) in self.filtered_num_loc:
                        continue
                    self.cur.execute("""
                        SELECT a.cand_id FROM postings_num_loc a
                        JOIN postings_num_loc b ON a.cand_id = b.cand_id
                        WHERE a.num = ? AND a.token = ? AND a.country = ?
                          AND b.num = ? AND b.token = ? AND b.country = ?
                        ORDER BY a.cand_id
                    """, (num, tok, country, num2, tok2, country))
                    intersect = [r[0] for r in self.cur.fetchall()]
                    if 0 < len(intersect) <= self.config.max_block_size:
                        cands.update(intersect)
                        refined = True
                        self.overflow_events.append({
                            "s1_id": s1_id,
                            "source": self.source_name,
                            "block_family": "address_number_location",
                            "key": f"{num}:{tok}+{num2}:{tok2}",
                            "generated_count": len(postings),
                            "retained_count": len(intersect),
                            "truncated_count": len(postings) - len(intersect),
                            "is_filtered": False,
                            "refinement_attempted": True,
                            "refined": True,
                        })
                        break

                if not refined:
                    cands.update(postings)
                    self.overflow_events.append({
                        "s1_id": s1_id,
                        "source": self.source_name,
                        "block_family": "address_number_location",
                        "key": f"{num}:{tok}",
                        "generated_count": len(postings),
                        "retained_count": len(postings),
                        "truncated_count": 0,
                        "is_filtered": False,
                        "refinement_attempted": True,
                        "refined": False,
                    })

        return sorted(cands)

    def query_compact_ngrams(
        self,
        s1_id: str,
        country: str,
        ngrams: List[str],
    ) -> List[str]:
        """Query compact character n-grams with non-overlapping refinement and retention."""
        cands: Set[str] = set()
        distinct_ngrams = list(dict.fromkeys(ngrams))

        for i, ng in enumerate(distinct_ngrams):
            key = (country, ng)
            if key in self.filtered_ngrams:
                continue

            self.cur.execute("""
                SELECT cand_id FROM postings_ngram
                WHERE ngram = ? AND country = ?
                ORDER BY cand_id
            """, (ng, country))
            postings = [r[0] for r in self.cur.fetchall()]
            if not postings:
                continue

            if len(postings) <= self.config.max_block_size:
                cands.update(postings)
            else:
                # Refine with a second non-overlapping n-gram
                refined = False
                for j, ng2 in enumerate(distinct_ngrams):
                    if abs(i - j) >= self.config.ngram_size and (country, ng2) not in self.filtered_ngrams:
                        self.cur.execute("""
                            SELECT a.cand_id FROM postings_ngram a
                            JOIN postings_ngram b ON a.cand_id = b.cand_id
                            WHERE a.ngram = ? AND a.country = ? AND b.ngram = ? AND b.country = ?
                            ORDER BY a.cand_id
                        """, (ng, country, ng2, country))
                        intersect = [r[0] for r in self.cur.fetchall()]
                        if 0 < len(intersect) <= self.config.max_block_size:
                            cands.update(intersect)
                            refined = True
                            self.overflow_events.append({
                                "s1_id": s1_id,
                                "source": self.source_name,
                                "block_family": "compact_char_ngram",
                                "key": f"{ng}+{ng2}",
                                "generated_count": len(postings),
                                "retained_count": len(intersect),
                                "truncated_count": len(postings) - len(intersect),
                                "is_filtered": False,
                                "refinement_attempted": True,
                                "refined": True,
                            })
                            break

                if not refined:
                    cands.update(postings)
                    self.overflow_events.append({
                        "s1_id": s1_id,
                        "source": self.source_name,
                        "block_family": "compact_char_ngram",
                        "key": ng,
                        "generated_count": len(postings),
                        "retained_count": len(postings),
                        "truncated_count": 0,
                        "is_filtered": False,
                        "refinement_attempted": True,
                        "refined": False,
                    })

        return sorted(cands)

    def query_missing_country_candidates(
        self,
        name_norm: str,
        name_tokens: List[str],
        num_loc_keys: List[Tuple[str, str]],
    ) -> List[str]:
        """
        Query candidates where the candidate record has missing country (country = '').
        Invoked when Source 1 has a known country, but candidate side is missing country.
        """
        cands: Set[str] = set()

        # 1. Exact normalized name on candidate records with country = ''
        if name_norm:
            self.cur.execute("""
                SELECT cand_id FROM postings_exact_name
                WHERE name_norm = ? AND country = ''
                ORDER BY cand_id
            """, (name_norm,))
            cands.update(r[0] for r in self.cur.fetchall())

        # 2. Rare name tokens on candidate records with country = ''
        for tok in set(name_tokens):
            if ("", tok) not in self.filtered_name_tokens:
                self.cur.execute("""
                    SELECT cand_id FROM postings_name_token
                    WHERE token = ? AND country = ''
                    ORDER BY cand_id
                """, (tok,))
                cands.update(r[0] for r in self.cur.fetchall())

        # 3. Address number + location on candidate records with country = ''
        for num, tok in num_loc_keys:
            if ("", num, tok) not in self.filtered_num_loc:
                self.cur.execute("""
                    SELECT cand_id FROM postings_num_loc
                    WHERE num = ? AND token = ? AND country = ''
                    ORDER BY cand_id
                """, (num, tok))
                cands.update(r[0] for r in self.cur.fetchall())

        return sorted(cands)

    def query_global_fallback_for_missing_s1(
        self,
        name_norm: str,
        name_tokens: List[str],
        num_loc_keys: List[Tuple[str, str]],
    ) -> List[str]:
        """
        Query candidate records across all countries when Source 1 has missing country.
        Restricted to high-specificity fallback keys: exact name, rare tokens, and num+location.
        """
        cands: Set[str] = set()

        # 1. Exact normalized name across all candidates
        if name_norm:
            self.cur.execute("""
                SELECT cand_id FROM postings_exact_name
                WHERE name_norm = ?
                ORDER BY cand_id
            """, (name_norm,))
            cands.update(r[0] for r in self.cur.fetchall())

        # 2. Rare name tokens across all candidates
        for tok in set(name_tokens):
            self.cur.execute("""
                SELECT cand_id FROM postings_name_token
                WHERE token = ?
                ORDER BY cand_id
            """, (tok,))
            postings = [r[0] for r in self.cur.fetchall()]
            if postings:
                cands.update(postings)

        # 3. Address number + location across all candidates
        for num, tok in num_loc_keys:
            self.cur.execute("""
                SELECT cand_id FROM postings_num_loc
                WHERE num = ? AND token = ?
                ORDER BY cand_id
            """, (num, tok))
            postings = [r[0] for r in self.cur.fetchall()]
            if postings:
                cands.update(postings)

        return sorted(cands)

    def query_cross_country_high_specificity(
        self,
        s1_country: str,
        name_norm: str,
        name_tokens: List[str],
        num_loc_keys: List[Tuple[str, str]],
    ) -> List[str]:
        """
        Cross-country fallback allows candidates ONLY through high-specificity evidence:
        1. Exact normalized name across different known countries
        2. >= 2 rare name tokens matching across different known countries
        3. Rare address-number + location across different known countries
        """
        cands: Set[str] = set()

        # 1. Exact normalized name match across different known countries
        if name_norm:
            self.cur.execute("""
                SELECT cand_id FROM postings_exact_name
                WHERE name_norm = ? AND country != ? AND country != ''
                ORDER BY cand_id
            """, (name_norm, s1_country))
            cands.update(r[0] for r in self.cur.fetchall())

        # 2. Multiple rare name tokens (>= min_rare_tokens) matching across different known countries
        distinct_tokens = list(dict.fromkeys(name_tokens))
        if len(distinct_tokens) >= self.config.cross_country_min_rare_tokens:
            placeholders = ",".join("?" for _ in distinct_tokens)
            self.cur.execute(f"""
                SELECT cand_id
                FROM postings_name_token
                WHERE token IN ({placeholders}) AND country != ? AND country != ''
                GROUP BY cand_id
                HAVING count(DISTINCT token) >= ?
                ORDER BY cand_id
            """, distinct_tokens + [s1_country, self.config.cross_country_min_rare_tokens])
            cands.update(r[0] for r in self.cur.fetchall())

        # 3. Rare address-number + location across different known countries
        for num, tok in num_loc_keys:
            self.cur.execute("""
                SELECT cand_id FROM postings_num_loc
                WHERE num = ? AND token = ? AND country != ? AND country != ''
                ORDER BY cand_id
            """, (num, tok, s1_country))
            postings = [r[0] for r in self.cur.fetchall()]
            if postings and len(postings) <= self.config.max_block_size:
                cands.update(postings)

        return sorted(cands)

    def close(self) -> None:
        """Close connection and clean up temporary database file."""
        if hasattr(self, "conn") and self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None

        if self.db_path and self.db_path.exists():
            try:
                self.db_path.unlink(missing_ok=True)
                wal = self.db_path.with_name(self.db_path.name + "-wal")
                wal.unlink(missing_ok=True)
                shm = self.db_path.with_name(self.db_path.name + "-shm")
                shm.unlink(missing_ok=True)
            except Exception:
                pass

    def __enter__(self) -> "CandidateSourceIndex":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
