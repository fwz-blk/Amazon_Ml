"""
Multi-Pass Blocking Orchestrator
Amazon ML Challenge 2026 - Business Entity Resolution

Orchestrates the 6 blocking pass families:
1. Country partition (default)
2. Exact normalized-name block
3. Rare name-token block
4. Address-token/location block
5. Address-number plus location block
6. Compact Unicode character n-gram block
Plus Missing-Country and Cross-Country High-Specificity fallbacks.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from blocking.config import (
    BlockingConfig,
    PROV_ADDRESS_NUMBER_LOCATION,
    PROV_ADDRESS_TOKEN,
    PROV_COMPACT_CHAR_NGRAM,
    PROV_CROSS_COUNTRY_HIGH_SPECIFICITY,
    PROV_EXACT_NAME,
    PROV_MISSING_COUNTRY_FALLBACK,
    PROV_RARE_NAME_TOKEN,
)
from blocking.index import CandidateSourceIndex
from blocking.keys import (
    extract_address_number_location_keys,
    extract_compact_ngrams,
    tokenize_address,
    tokenize_name,
)


def generate_candidates_for_s1(
    s1_id: str,
    name_norm: str,
    addr_norm: str,
    country_norm: str,
    index: CandidateSourceIndex,
    config: BlockingConfig,
) -> Dict[str, int]:
    """
    Generate candidate IDs and provenance bitmasks for a single Source 1 record
    against a CandidateSourceIndex (Source 2 or Source 3).

    Returns:
        Dict[candidate_entity_id, provenance_bitmask]
    """
    candidates: Dict[str, int] = {}
    is_missing_country = not country_norm

    # 1. Key extraction from Source 1 fields
    name_tokens = (
        tokenize_name(name_norm, config.legal_stoplist, config.min_token_length)
        if name_norm else []
    )
    addr_tokens = (
        tokenize_address(addr_norm, config.common_address_terms, config.min_addr_token_length)
        if addr_norm else []
    )
    num_loc_keys = (
        extract_address_number_location_keys(addr_norm, config.min_addr_token_length)
        if addr_norm else []
    )
    ngrams = (
        extract_compact_ngrams(name_norm, config.ngram_size, config.min_ngram_length)
        if name_norm else []
    )

    # 2. Known Country S1: Same-Country Blocking Passes (Default Partition)
    if not is_missing_country:
        # Pass 2: Exact normalized name
        if name_norm:
            for cid in index.query_exact_name(country_norm, name_norm):
                candidates[cid] = candidates.get(cid, 0) | PROV_EXACT_NAME

        # Pass 3: Rare name tokens
        if name_tokens:
            for cid in index.query_name_tokens(s1_id, country_norm, name_tokens):
                candidates[cid] = candidates.get(cid, 0) | PROV_RARE_NAME_TOKEN

        # Pass 4: Address tokens
        if addr_tokens:
            for cid in index.query_address_tokens(s1_id, country_norm, addr_tokens):
                candidates[cid] = candidates.get(cid, 0) | PROV_ADDRESS_TOKEN

        # Pass 5: Address number + location keys
        if num_loc_keys:
            for cid in index.query_num_loc(s1_id, country_norm, num_loc_keys):
                candidates[cid] = candidates.get(cid, 0) | PROV_ADDRESS_NUMBER_LOCATION

        # Pass 6: Compact Unicode character n-grams
        if ngrams:
            for cid in index.query_compact_ngrams(s1_id, country_norm, ngrams):
                candidates[cid] = candidates.get(cid, 0) | PROV_COMPACT_CHAR_NGRAM

        # Cross-country fallback (high specificity only)
        if config.allow_cross_country_high_specificity:
            for cid in index.query_cross_country_high_specificity(
                country_norm, name_norm, name_tokens, num_loc_keys
            ):
                candidates[cid] = candidates.get(cid, 0) | PROV_CROSS_COUNTRY_HIGH_SPECIFICITY

        # Missing-country candidate records fallback (known-country S1 against missing-country candidates)
        if config.allow_missing_country_fallback:
            for cid in index.query_missing_country_candidates(name_norm, name_tokens, num_loc_keys):
                candidates[cid] = candidates.get(cid, 0) | PROV_MISSING_COUNTRY_FALLBACK

    # 3. Missing Country S1: Global Fallback across all candidate records
    else:
        if config.allow_missing_country_fallback:
            for cid in index.query_global_fallback_for_missing_s1(name_norm, name_tokens, num_loc_keys):
                candidates[cid] = candidates.get(cid, 0) | PROV_MISSING_COUNTRY_FALLBACK

    return candidates
