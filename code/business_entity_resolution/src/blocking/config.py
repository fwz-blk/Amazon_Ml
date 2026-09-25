"""
Blocking and Candidate Generation Configuration
Amazon ML Challenge 2026 - Business Entity Resolution
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Set

# Bitmask Provenance Flags
PROV_EXACT_NAME = 1 << 0                   # 1
PROV_RARE_NAME_TOKEN = 1 << 1              # 2
PROV_ADDRESS_TOKEN = 1 << 2                # 4
PROV_ADDRESS_NUMBER_LOCATION = 1 << 3      # 8
PROV_COMPACT_CHAR_NGRAM = 1 << 4           # 16
PROV_MISSING_COUNTRY_FALLBACK = 1 << 5     # 32
PROV_CROSS_COUNTRY_HIGH_SPECIFICITY = 1 << 6 # 64

PROVENANCE_NAMES: Dict[int, str] = {
    PROV_EXACT_NAME: "exact_name",
    PROV_RARE_NAME_TOKEN: "rare_name_token",
    PROV_ADDRESS_TOKEN: "address_token",
    PROV_ADDRESS_NUMBER_LOCATION: "address_number_location",
    PROV_COMPACT_CHAR_NGRAM: "compact_char_ngram",
    PROV_MISSING_COUNTRY_FALLBACK: "missing_country_fallback",
    PROV_CROSS_COUNTRY_HIGH_SPECIFICITY: "cross_country_high_specificity",
}

# Documented universal generic legal terms stoplist
DEFAULT_LEGAL_STOPLIST: Set[str] = {
    "inc", "incorporated", "corp", "corporation", "ltd", "limited",
    "llc", "llp", "pvt", "private", "gmbh", "sa", "srl", "co", "company"
}

# Documented universal common address terms (prevented from acting as standalone block keys)
DEFAULT_COMMON_ADDRESS_TERMS: Set[str] = {
    "st", "street", "rd", "road", "ave", "avenue", "dr", "drive",
    "lane", "ln", "blvd", "boulevard", "way", "ct", "court", "cir", "circle",
    "hwy", "highway", "pkwy", "parkway", "pl", "place",
    "suite", "ste", "apt", "apartment", "unit", "floor", "fl", "bldg", "building",
    "no", "nr", "near", "opp", "opposite", "behind",
}


@dataclass
class BlockingConfig:
    """Configuration parameters for multi-pass blocking."""
    # Data paths
    input_dir: Path = field(default_factory=lambda: Path("cleaned"))
    output_dir: Path = field(default_factory=lambda: Path("candidates"))
    ground_truth_path: Optional[Path] = field(
        default_factory=lambda: Path("student_resource/dataset/train/train_ground_truth.tsv")
    )
    temp_dir: Optional[Path] = None

    # Target split
    split: str = "train"  # 'train' or 'test'

    # Document frequency filtering thresholds (absolute and relative)
    max_name_df_absolute: int = 10000
    max_name_df_relative: float = 0.003
    max_addr_df_absolute: int = 5000
    max_addr_df_relative: float = 0.002
    max_ngram_df_absolute: int = 5000
    max_ngram_df_relative: float = 0.002

    # Key derivation parameters
    min_token_length: int = 2
    min_addr_token_length: int = 2
    ngram_size: int = 4
    min_ngram_length: int = 3
    legal_stoplist: Set[str] = field(default_factory=lambda: set(DEFAULT_LEGAL_STOPLIST))
    common_address_terms: Set[str] = field(default_factory=lambda: set(DEFAULT_COMMON_ADDRESS_TERMS))

    # Pathological block overflow & refinement
    max_block_size: int = 1000

    # Deterministic safety budget per S1 per candidate source
    max_candidates_per_s1_per_source: int = 1000

    # Country partition & fallback policy
    allow_missing_country_fallback: bool = True
    allow_cross_country_high_specificity: bool = True
    cross_country_min_rare_tokens: int = 2

    # Deterministic entity holdout evaluation
    holdout_ratio: float = 0.05
    holdout_salt: str = "amazon_ml_2026_val_salt"

    # SQLite performance tuning
    sqlite_cache_size_kb: int = 64000
    batch_size: int = 50000

    def get_max_name_df(self, total_records: int) -> int:
        return min(self.max_name_df_absolute, max(5, int(total_records * self.max_name_df_relative)))

    def get_max_addr_df(self, total_records: int) -> int:
        return min(self.max_addr_df_absolute, max(5, int(total_records * self.max_addr_df_relative)))

    def get_max_ngram_df(self, total_records: int) -> int:
        return min(self.max_ngram_df_absolute, max(5, int(total_records * self.max_ngram_df_relative)))
