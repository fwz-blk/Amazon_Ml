"""
Candidate Source Inverted Index
Amazon ML Challenge 2026 - Business Entity Resolution

Features:
- Separate indexing for Source 2 and Source 3
- Split-specific (train candidate sources vs test candidate sources)
- Country partitioning with missing-country and cross-country fallback indices
- Data-driven Document Frequency (DF) counting with absolute and relative thresholding
- Pathological block overflow tracking and refinement
"""

from __future__ import annotations

from collections import Counter
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
    Inverted index over a candidate source (Source 2 or Source 3).
    Supports country-partitioned queries across all 6 blocking pass families.
    """

    def __init__(self, source_name: str, config: BlockingConfig) -> None:
        self.source_name = source_name
        self.config = config

        # Candidate records
        self.cand_ids: List[str] = []
        self.cand_countries: List[str] = []
        self.cand_names: List[str] = []
        self.cand_addrs: List[str] = []
        self.observed_countries: Set[str] = set()
        self.total_records = 0

        # Inverted indices (partitioned by (country, key))
        self.exact_name_index: Dict[Tuple[str, str], List[int]] = {}
        self.exact_name_global: Dict[str, List[int]] = {}

        self.name_token_index: Dict[Tuple[str, str], List[int]] = {}
        self.addr_token_index: Dict[Tuple[str, str], List[int]] = {}
        self.num_loc_index: Dict[Tuple[str, str, str], List[int]] = {}
        self.compact_ngram_index: Dict[Tuple[str, str], List[int]] = {}

        # Missing country indices
        self.missing_country_exact: Dict[str, List[int]] = {}
        self.missing_country_name_tokens: Dict[str, List[int]] = {}
        self.missing_country_num_loc: Dict[Tuple[str, str], List[int]] = {}

        # High-DF filtered keys
        self.filtered_name_tokens: Set[Tuple[str, str]] = set()
        self.filtered_addr_tokens: Set[Tuple[str, str]] = set()
        self.filtered_num_loc: Set[Tuple[str, str, str]] = set()
        self.filtered_ngrams: Set[Tuple[str, str]] = set()

        # Overflow tracking
        self.overflow_events: List[Dict[str, Any]] = []

    def build_from_tsv(self, tsv_path: Path, max_rows: Optional[int] = None) -> None:
        """
        Build inverted index from a cleaned TSV file in a single streaming pass,
        followed by deterministic DF threshold pruning.
        """
        self.cand_ids.clear()
        self.cand_countries.clear()
        self.cand_names.clear()
        self.cand_addrs.clear()
        self.observed_countries.clear()

        # Temporary full dictionaries before pruning
        raw_name_tokens: Dict[Tuple[str, str], List[int]] = {}
        raw_addr_tokens: Dict[Tuple[str, str], List[int]] = {}
        raw_num_loc: Dict[Tuple[str, str, str], List[int]] = {}
        raw_ngrams: Dict[Tuple[str, str], List[int]] = {}

        count = 0
        with open(tsv_path, "r", encoding="utf-8") as f:
            header_line = f.readline()
            headers = [h.strip() for h in header_line.rstrip("\r\n").split("\t")]

            # Verify cleaned TSV structure
            assert len(headers) >= 10, f"Expected 10 cleaned columns in {tsv_path}, got {len(headers)}"

            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 10:
                    continue

                eid = parts[0]
                name_norm = parts[4]
                addr_norm = parts[5]
                ctry_norm = parts[6]

                idx = count
                count += 1
                self.cand_ids.append(eid)
                self.cand_countries.append(ctry_norm)
                self.cand_names.append(name_norm)
                self.cand_addrs.append(addr_norm)
                if ctry_norm:
                    self.observed_countries.add(ctry_norm)

                is_missing_ctry = not ctry_norm

                # Exact normalized name
                if name_norm:
                    if is_missing_ctry:
                        self.missing_country_exact.setdefault(name_norm, []).append(idx)
                    else:
                        self.exact_name_index.setdefault((ctry_norm, name_norm), []).append(idx)
                    self.exact_name_global.setdefault(name_norm, []).append(idx)

                    # Name tokens
                    tokens = tokenize_name(name_norm, self.config.legal_stoplist, self.config.min_token_length)
                    for tok in set(tokens):
                        if is_missing_ctry:
                            self.missing_country_name_tokens.setdefault(tok, []).append(idx)
                        else:
                            raw_name_tokens.setdefault((ctry_norm, tok), []).append(idx)

                    # Compact character n-grams
                    ngrams = extract_compact_ngrams(name_norm, self.config.ngram_size, self.config.min_ngram_length)
                    for ng in set(ngrams):
                        if not is_missing_ctry:
                            raw_ngrams.setdefault((ctry_norm, ng), []).append(idx)

                # Address tokens & number-location keys
                if addr_norm:
                    atoks = tokenize_address(
                        addr_norm, self.config.common_address_terms, self.config.min_addr_token_length
                    )
                    for atok in set(atoks):
                        if not is_missing_ctry:
                            raw_addr_tokens.setdefault((ctry_norm, atok), []).append(idx)

                    num_locs = extract_address_number_location_keys(addr_norm, self.config.min_addr_token_length)
                    for pair in num_locs:
                        if is_missing_ctry:
                            self.missing_country_num_loc.setdefault(pair, []).append(idx)
                        else:
                            raw_num_loc.setdefault((ctry_norm, pair[0], pair[1]), []).append(idx)

                if max_rows and count >= max_rows:
                    break

        self.total_records = count

        # Apply Document Frequency (DF) thresholds
        max_name_df = self.config.get_max_name_df(self.total_records)
        max_addr_df = self.config.get_max_addr_df(self.total_records)
        max_ngram_df = self.config.get_max_ngram_df(self.total_records)

        # Prune name tokens
        for k, postings in raw_name_tokens.items():
            if len(postings) <= max_name_df:
                self.name_token_index[k] = postings
            else:
                self.filtered_name_tokens.add(k)

        # Prune address tokens
        for k, postings in raw_addr_tokens.items():
            if len(postings) <= max_addr_df:
                self.addr_token_index[k] = postings
            else:
                self.filtered_addr_tokens.add(k)

        # Prune number-location keys
        for k, postings in raw_num_loc.items():
            if len(postings) <= max_addr_df:
                self.num_loc_index[k] = postings
            else:
                self.filtered_num_loc.add(k)

        # Prune character n-grams
        for k, postings in raw_ngrams.items():
            if len(postings) <= max_ngram_df:
                self.compact_ngram_index[k] = postings
            else:
                self.filtered_ngrams.add(k)

    def query_exact_name(self, country: str, name_norm: str) -> List[int]:
        """Query same-country exact normalized name."""
        if not name_norm:
            return []
        return self.exact_name_index.get((country, name_norm), [])

    def query_name_tokens(
        self,
        s1_id: str,
        country: str,
        tokens: List[str],
    ) -> List[int]:
        """
        Query rare name tokens with pathological block overflow tracking and refinement.
        """
        cands: Set[int] = set()
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
                })
                continue

            postings = self.name_token_index.get(key)
            if not postings:
                continue

            if len(postings) <= self.config.max_block_size:
                cands.update(postings)
            else:
                # Pathological block overflow: attempt refinement with a second token
                refined = False
                for j, tok2 in enumerate(distinct_tokens):
                    if i == j:
                        continue
                    postings2 = self.name_token_index.get((country, tok2))
                    if postings2:
                        intersect = set(postings) & set(postings2)
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
                            })
                            break
                if not refined:
                    self.overflow_events.append({
                        "s1_id": s1_id,
                        "source": self.source_name,
                        "block_family": "rare_name_token",
                        "key": tok,
                        "generated_count": len(postings),
                        "retained_count": 0,
                        "truncated_count": len(postings),
                        "is_filtered": False,
                        "refinement_attempted": True,
                    })

        return list(cands)

    def query_address_tokens(
        self,
        s1_id: str,
        country: str,
        tokens: List[str],
    ) -> List[int]:
        """Query distinctive address tokens with overflow handling."""
        cands: Set[int] = set()
        distinct_tokens = list(dict.fromkeys(tokens))

        for tok in distinct_tokens:
            key = (country, tok)
            if key in self.filtered_addr_tokens:
                continue

            postings = self.addr_token_index.get(key)
            if not postings:
                continue

            if len(postings) <= self.config.max_block_size:
                cands.update(postings)
            else:
                self.overflow_events.append({
                    "s1_id": s1_id,
                    "source": self.source_name,
                    "block_family": "address_token",
                    "key": tok,
                    "generated_count": len(postings),
                    "retained_count": 0,
                    "truncated_count": len(postings),
                    "is_filtered": False,
                    "refinement_attempted": False,
                })

        return list(cands)

    def query_num_loc(
        self,
        s1_id: str,
        country: str,
        num_loc_keys: List[Tuple[str, str]],
    ) -> List[int]:
        """Query address number + location pairs."""
        cands: Set[int] = set()
        for num, tok in num_loc_keys:
            key = (country, num, tok)
            if key in self.filtered_num_loc:
                continue

            postings = self.num_loc_index.get(key)
            if not postings:
                continue

            if len(postings) <= self.config.max_block_size:
                cands.update(postings)
            else:
                self.overflow_events.append({
                    "s1_id": s1_id,
                    "source": self.source_name,
                    "block_family": "address_number_location",
                    "key": f"{num}:{tok}",
                    "generated_count": len(postings),
                    "retained_count": 0,
                    "truncated_count": len(postings),
                    "is_filtered": False,
                    "refinement_attempted": False,
                })

        return list(cands)

    def query_compact_ngrams(
        self,
        s1_id: str,
        country: str,
        ngrams: List[str],
    ) -> List[int]:
        """Query compact character n-grams with overflow refinement."""
        cands: Set[int] = set()
        distinct_ngrams = list(dict.fromkeys(ngrams))

        for i, ng in enumerate(distinct_ngrams):
            key = (country, ng)
            if key in self.filtered_ngrams:
                continue

            postings = self.compact_ngram_index.get(key)
            if not postings:
                continue

            if len(postings) <= self.config.max_block_size:
                cands.update(postings)
            else:
                # Refine with a second non-overlapping n-gram
                refined = False
                for j, ng2 in enumerate(distinct_ngrams):
                    if abs(i - j) >= self.config.ngram_size:
                        postings2 = self.compact_ngram_index.get((country, ng2))
                        if postings2:
                            intersect = set(postings) & set(postings2)
                            if 0 < len(intersect) <= self.config.max_block_size:
                                cands.update(intersect)
                                refined = True
                                break
                if not refined:
                    self.overflow_events.append({
                        "s1_id": s1_id,
                        "source": self.source_name,
                        "block_family": "compact_char_ngram",
                        "key": ng,
                        "generated_count": len(postings),
                        "retained_count": 0,
                        "truncated_count": len(postings),
                        "is_filtered": False,
                        "refinement_attempted": True,
                    })

        return list(cands)

    def query_missing_country_fallback(
        self,
        name_norm: str,
        name_tokens: List[str],
        num_loc_keys: List[Tuple[str, str]],
    ) -> List[int]:
        """Query candidates when Source 1 has missing country."""
        cands: Set[int] = set()
        if name_norm and name_norm in self.exact_name_global:
            cands.update(self.exact_name_global[name_norm])

        # Also match missing country candidates
        if name_norm in self.missing_country_exact:
            cands.update(self.missing_country_exact[name_norm])

        for tok in name_tokens:
            if tok in self.missing_country_name_tokens:
                cands.update(self.missing_country_name_tokens[tok])

        for pair in num_loc_keys:
            if pair in self.missing_country_num_loc:
                cands.update(self.missing_country_num_loc[pair])

        return list(cands)

    def query_cross_country_high_specificity(
        self,
        s1_country: str,
        name_norm: str,
        name_tokens: List[str],
        num_loc_keys: List[Tuple[str, str]],
        max_block_size: Optional[int] = None,
    ) -> List[int]:
        """
        Cross-country fallback allows candidates ONLY through high-specificity evidence:
        1. Exact normalized name
        2. >= 2 rare name tokens
        3. Rare address-number + location
        """
        cands: Set[int] = set()

        # 1. Exact normalized name match across countries
        if name_norm and name_norm in self.exact_name_global:
            for idx in self.exact_name_global[name_norm]:
                cand_ctry = self.cand_countries[idx]
                if cand_ctry and cand_ctry != s1_country:
                    cands.add(idx)

        # 2. Multiple rare name tokens matching across countries
        if len(name_tokens) >= self.config.cross_country_min_rare_tokens:
            token_matches: Counter[int] = Counter()
            for tok in name_tokens:
                for ctry in self.observed_countries:
                    if ctry != s1_country:
                        postings = self.name_token_index.get((ctry, tok))
                        if postings:
                            token_matches.update(postings)

            for idx, match_cnt in token_matches.items():
                if match_cnt >= self.config.cross_country_min_rare_tokens:
                    cands.add(idx)

        # 3. Rare address-number + location across countries
        for num, tok in num_loc_keys:
            for ctry in self.observed_countries:
                if ctry != s1_country:
                    postings = self.num_loc_index.get((ctry, num, tok))
                    if postings and len(postings) <= self.config.max_block_size:
                        cands.update(postings)

        return list(cands)
