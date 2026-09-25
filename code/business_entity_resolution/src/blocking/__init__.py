"""
Amazon ML Challenge 2026 - Business Entity Resolution
Multi-Pass Blocking & Candidate Generation Package
"""

from blocking.config import (
    DEFAULT_LEGAL_STOPLIST,
    PROV_ADDRESS_NUMBER_LOCATION,
    PROV_ADDRESS_TOKEN,
    PROV_COMPACT_CHAR_NGRAM,
    PROV_CROSS_COUNTRY_HIGH_SPECIFICITY,
    PROV_EXACT_NAME,
    PROV_MISSING_COUNTRY_FALLBACK,
    PROV_RARE_NAME_TOKEN,
    PROVENANCE_NAMES,
    BlockingConfig,
)
from blocking.candidate_store import CandidateStore
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

__all__ = [
    "BlockingConfig",
    "CandidateSourceIndex",
    "CandidateStore",
    "BlockingEvaluator",
    "generate_candidates_for_s1",
    "tokenize_unicode",
    "tokenize_name",
    "tokenize_address",
    "extract_address_numbers",
    "extract_address_number_location_keys",
    "extract_compact_ngrams",
    "load_ground_truth",
    "is_validation_entity",
    "PROV_EXACT_NAME",
    "PROV_RARE_NAME_TOKEN",
    "PROV_ADDRESS_TOKEN",
    "PROV_ADDRESS_NUMBER_LOCATION",
    "PROV_COMPACT_CHAR_NGRAM",
    "PROV_MISSING_COUNTRY_FALLBACK",
    "PROV_CROSS_COUNTRY_HIGH_SPECIFICITY",
    "PROVENANCE_NAMES",
    "DEFAULT_LEGAL_STOPLIST",
]
