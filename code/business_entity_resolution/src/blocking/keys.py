"""
Key Extraction Functions for Multi-Pass Blocking
Amazon ML Challenge 2026 - Business Entity Resolution

Requirements:
- Unicode-aware tokenization across all scripts (Latin, Devanagari, Tamil, Cyrillic, Arabic, etc.)
- Combining mark preservation (accents, matras, diacritics)
- Documented universal generic legal stoplist
- Address numbers and location combinations
- Compact Unicode character n-grams
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Set, Tuple

# Precompute character category cache for fast Unicode tokenization
_CHAR_IS_WORD: List[bool] = [False] * 65536
for i in range(65536):
    ch = chr(i)
    cat = unicodedata.category(ch)
    if cat.startswith(("L", "N", "M")) or ch == "_":
        _CHAR_IS_WORD[i] = True

RE_DIGITS = re.compile(r"\d+", flags=re.UNICODE)


def is_unicode_word_char(ch: str) -> bool:
    """Check if character is a Unicode Letter, Number, Combining Mark, or underscore."""
    cp = ord(ch)
    if cp < 65536:
        return _CHAR_IS_WORD[cp]
    cat = unicodedata.category(ch)
    return cat.startswith(("L", "N", "M")) or ch == "_"


def tokenize_unicode(text: str) -> List[str]:
    """
    Tokenize string into Unicode word tokens, preserving letters, numbers,
    and combining marks across all scripts without ASCII-folding or transliteration.
    """
    if not text:
        return []
    tokens: List[str] = []
    current: List[str] = []
    for ch in text:
        if is_unicode_word_char(ch):
            current.append(ch)
        else:
            if current:
                tokens.append("".join(current))
                current = []
    if current:
        tokens.append("".join(current))
    return tokens


def tokenize_name(
    name: str,
    stoplist: Set[str],
    min_length: int = 2,
) -> List[str]:
    """
    Extract informative name tokens from business_name_normalized.
    Excludes universally generic legal stoplist and tokens shorter than min_length.
    Never alters raw text.
    """
    if not name:
        return []
    tokens = tokenize_unicode(name)
    return [
        t for t in tokens
        if len(t) >= min_length and t.casefold() not in stoplist
    ]


def tokenize_address(
    address: str,
    stoplist: Optional[Set[str]] = None,
    min_length: int = 2,
) -> List[str]:
    """
    Extract address word tokens from business_address_normalized.
    Excludes pure digit sequences (handled by address number extractor),
    optional generic street/structural stoplist, and tokens shorter than min_length.
    """
    if not address:
        return []
    tokens = tokenize_unicode(address)
    return [
        t for t in tokens
        if len(t) >= min_length
        and not t.isdigit()
        and (not stoplist or t.casefold() not in stoplist)
    ]


def extract_address_numbers(address: str) -> List[str]:
    """
    Extract all distinct digit sequences from business_address_normalized.
    Preserves exact digit sequences without rewriting or assuming postal codes.
    """
    if not address:
        return []
    return RE_DIGITS.findall(address)


def extract_address_number_location_keys(
    address: str,
    min_token_length: int = 2,
    max_tokens_per_num: int = 10,
) -> List[Tuple[str, str]]:
    """
    Combine each observed address number with informative address locality tokens.
    Produces (number, token) pairs to tolerate reordered address components.
    """
    if not address:
        return []
    nums = extract_address_numbers(address)
    if not nums:
        return []
    tokens = tokenize_address(address, min_length=min_token_length)
    if not tokens:
        return []

    # Filter out extremely common structural keywords like 'no', 'apt', 'suite', 'floor'
    structural_noise = {"no", "nr", "apt", "suite", "ste", "floor", "fl", "unit", "bldg"}
    informative_tokens = [t for t in tokens if t not in structural_noise]
    if not informative_tokens:
        informative_tokens = tokens

    keys: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()

    for num in nums:
        for tok in informative_tokens[:max_tokens_per_num]:
            pair = (num, tok)
            if pair not in seen:
                seen.add(pair)
                keys.append(pair)

    return keys


def extract_compact_ngrams(
    name: str,
    n: int = 4,
    min_length: int = 3,
) -> List[str]:
    """
    Generate character n-grams from a compact Unicode alphanumeric view of business_name_normalized.
    Removes whitespace and punctuation only in this derived signature view.
    Does not transliterate or change cleaned TSV values.
    """
    if not name:
        return []
    compact = "".join(ch for ch in name if is_unicode_word_char(ch) and ch != "_")
    if len(compact) < min_length:
        return []
    if len(compact) < n:
        return [compact]
    return [compact[i:i + n] for i in range(len(compact) - n + 1)]
