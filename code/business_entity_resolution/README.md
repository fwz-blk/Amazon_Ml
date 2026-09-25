# Business Entity Resolution Pipeline — Stage 1: Data Cleaning & Profiling

## 1. Purpose and Scope

This module implements **Stage 1 (Conservative Data Cleaning, Validation, and Profiling)** of the Business Entity Resolution pipeline for the Amazon ML Challenge 2026.

### Explicit Scope Constraints
- **What this stage does**:
  - Validates raw input TSV files for UTF-8 readability, exact expected headers, 4-column TSV structure, non-empty and unique entity IDs, and source prefix consistency.
  - Performs bounded-memory, streaming conservative normalization to produce derived columns (`business_name_normalized`, `business_address_normalized`, `country_normalized`) and boolean missingness indicators (`business_name_is_missing`, `business_address_is_missing`, `country_is_missing`).
  - Preserves every input row, every entity ID, and all four original columns untouched and in exact deterministic input order.
  - Enforces atomic file publication and post-cleaning quality checks to verify 12 strict integrity invariants before publishing any output.
  - Produces diagnostic profiling (`profile.json`) and invariant verification reports (`quality_checks.json`).
- **What this stage does NOT do**:
  - **No blocking or candidate generation**.
  - **No bucketing, canopy clustering, or indexing**.
  - **No similarity feature computation**.
  - **No entity matching, ML scoring, clustering, or classification**.
  - **No validation scoring, prediction, or submission generation**.
  - **No use of external databases, APIs, geocoding, translation, or web lookups**.

---

## 2. Execution Instructions

### Exact Command to Run the Cleaner
From the repository root (`6ab10eb3b23ba_student_resource/`):

```bash
python3 code/business_entity_resolution/src/clean_dataset.py \
    --input-dir student_resource/dataset \
    --output-dir cleaned
```

Default arguments allow simply running:
```bash
python3 code/business_entity_resolution/src/clean_dataset.py
```

### Running Specific Splits
To clean only the training split:
```bash
python3 code/business_entity_resolution/src/clean_dataset.py --split train
```

To clean only the test split:
```bash
python3 code/business_entity_resolution/src/clean_dataset.py --split test
```

### Running the Test Suite
Run the test suite using Python's standard library test runner:
```bash
python3 code/business_entity_resolution/tests/test_clean_dataset.py
```

---

## 3. Input and Output Paths

### Raw Input Files (Read-Only)
```text
student_resource/dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv   (Untouched)
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

Each source file contains exactly 4 tab-delimited columns:
1. `entity_id`
2. `business_name`
3. `business_address`
4. `country`

### Cleaned Output Files
```text
cleaned/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   └── train_source3.tsv
├── test/
│   ├── test_source1.tsv
│   ├── test_source2.tsv
│   └── test_source3.tsv
└── reports/
    ├── profile.json
    └── quality_checks.json
```

Each cleaned source file contains exactly 10 tab-delimited columns in the following order:
1. `entity_id` (original parsed value)
2. `business_name` (original parsed value)
3. `business_address` (original parsed value)
4. `country` (original parsed value)
5. `business_name_normalized` (conservative derived normalized string)
6. `business_address_normalized` (conservative derived normalized string)
7. `country_normalized` (conservative derived normalized string)
8. `business_name_is_missing` (`True` / `False`)
9. `business_address_is_missing` (`True` / `False`)
10. `country_is_missing` (`True` / `False`)

---

## 4. Conservative Normalization Rules

All normalization operations are applied **strictly to derived columns**. The original four columns remain unaltered.

### Unicode
- **Unicode NFC Normalization**: Applies `unicodedata.normalize('NFC', text)` to unify canonical decompositions (e.g., precomposed `é` vs `e` + combining acute accent `\u0301`).
- **Full Script Preservation**: Preserves non-Latin scripts (e.g., Devanagari / Hindi, Arabic, Cyrillic, CJK, etc.) and accented characters.
- **No Transliteration**: No phonetic, romanized, or ASCII-only conversion is applied (e.g., `unidecode` is explicitly avoided).
- **No Translation**: All multilingual text remains in its original language and script.

### Case
- **Unicode-Aware Casefolding**: Uses Python's built-in `.casefold()`, which adheres to the Unicode standard for full case mapping (including German `ß` $\rightarrow$ `ss`, Greek, Cyrillic, etc.).
- Original casing is strictly preserved in original columns.

### Whitespace
- **Noise Treatment**: Unicode whitespace characters, tabs (`\t`), carriage returns (`\r`), and embedded line breaks (`\n`) in text fields are treated as formatting noise in derived values.
- **Run Collapse**: Consecutive whitespace sequences (`\s+`) are collapsed into a single ASCII space (`' '`).
- **Stripping**: Leading and trailing whitespace is stripped.
- Guaranteed structural TSV validity with zero embedded raw tab or newline characters in derived fields.

### Safe Punctuation Equivalences
Only safe, explicitly documented equivalences are mapped in derived fields:
- **Curly Single Quotes $\rightarrow$ Straight Single Quote (`'`)**:
  - `‘` (`\u2018`), `’` (`\u2019`), `‚` (`\u201A`), `‛` (`\u201B`), `ʻ` (`\u02BB`), `ʼ` (`\u02BC`), `′` (`\u2032`), `‵` (`\u2035`) $\rightarrow$ `'`
- **Curly Double Quotes $\rightarrow$ Straight Double Quote (`"`)**:
  - `“` (`\u201C`), `”` (`\u201D`), `„` (`\u201E`), `‟` (`\u201F`), `″` (`\u2033`), `‶` (`\u2036`) $\rightarrow$ `"`
- **Unicode Dash Variants $\rightarrow$ ASCII Hyphen (`-`)**:
  - `‐` (`\u2010`), `‑` (`\u2011`), `‒` (`\u2012`), `–` (`\u2013`), `—` (`\u2014`), `―` (`\u2015`), `−` (`\u2212`), `﹘` (`\uFE58`), `﹣` (`\uFE63`), `－` (`\uFF0D`) $\rightarrow$ `-`
- **Unicode Space Variants $\rightarrow$ ASCII Space (`' '`)**:
  - `\u00A0` (non-breaking space), `\u2000`–`\u200A` (quads/spaces), `\u202F`, `\u3000` (ideographic space), `\uFEFF` (zero-width no-break space) $\rightarrow$ `' '`
- **Meaningful Punctuation Strictly Preserved**:
  - Punctuation such as `&`, `-`, `.`, `'`, `/`, `@`, `+`, `#`, commas `,`, and parentheses `(` `)` is **NOT** globally stripped or deleted.

### Business Names & Addresses
- **No legal suffix stripping**: Does not remove suffixes like `LLC`, `Corp`, `Inc`, `Pvt Ltd`, `LLP`.
- **No abbreviation expansion**: Does not expand `St` to `Street`, `Rd` to `Road`, `Corp` to `Corporation`.
- **No token reordering**: Token order is preserved exactly.
- **No deduplication of tokens**: Repeated words (e.g. `Pizza Pizza`) remain untouched.
- **No typo correction**: Suspected typos are not modified.
- **No geocoding or address reconstruction**: Postal codes, state names, landmark references (e.g., `Near SBI ATM`) are kept untouched.

### Country
- Created using **only**:
  1. Unicode NFC
  2. Unicode `.casefold()`
  3. Whitespace normalization
- Does **not** hardcode country sets, filter countries, or map aliases (e.g. `US`, `India`, and `France` are preserved generically).

### Missing Value Handling
- Empty strings `""` and whitespace-only fields are identified as missing.
- Derived normalized columns for missing fields are set to empty string `""`.
- No synthetic placeholders (e.g., `"Unknown"`, `"Missing"`, `"N/A"`, `"None"`) are fabricated.
- Boolean flags `business_name_is_missing`, `business_address_is_missing`, `country_is_missing` are explicitly emitted (`True` / `False`).
- **No record is ever dropped due to missing fields**.

---

## 5. Raw Data Preservation Guarantees

1. **Original Files Untouched**: Raw files under `student_resource/dataset/` are opened read-only and never modified.
2. **Deterministic Sequence & 100% Row Retention**: Output rows strictly match input rows 1-to-1 in the exact same sequence. No filtering, merging, deduplication, or reordering is performed.
3. **Exact Original Values Retained**: The original four columns in positions 0..3 of the cleaned TSV files contain exact parsed byte values from the input.
4. **Equal Treatment Across Sources**: Source 1 is kept semantically untouched and receives only the identical conservative base normalization applied to Source 2 and Source 3.

---

## 6. Structural Integrity & Quality Gates

The pipeline enforces a multi-tier gatekeeper architecture:

```
[Raw TSV Files]
       │
       ▼
Phase 1: Input Validation Gate
  ├── UTF-8 readability
  ├── Exact header check
  ├── 4-column TSV structure
  ├── Non-empty entity_id
  ├── Consistent source prefix (S1-, S2-, S3-)
  └── File-level entity_id uniqueness
       │ (Fails before writing any output if violated)
       ▼
Phase 2: Bounded-Memory Streaming Clean & Profile
  └── Writes to hidden temporary files (.tmp_*)
       │
       ▼
Phase 3: Post-Cleaning Quality Check Gate
  ├── Verifies 12 strict invariants against raw input line-by-line:
  │   1. row_count_match
  │   2. entity_id_match_and_order
  │   3. original_values_unaltered
  │   4. source_columns_present
  │   5. no_unexpected_nulls
  │   6. normalized_missing_not_fabricated
  │   7. output_headers_correct
  │   8. valid_utf8_tsv
  │   9. no_malformed_rows
  │  10. normalized_columns_present
  │  11. source_prefix_validation
  │  12. boolean_flags_valid
       │
       ▼
Phase 4: Atomic Publication & Reports
  ├── Generates reports/profile.json and reports/quality_checks.json
  └── Atomically renames temporary files to target paths via os.replace
```

If any invariant fails:
- Temporary files are immediately deleted.
- The pipeline aborts with exit code 1, reporting file path, line number, and exact reason.
- No partial or corrupted files are ever left in the destination folder.

---

## 7. Profiling Report Format (`profile.json`)

The bounded-memory profiling report (`cleaned/reports/profile.json`) summarizes dataset characteristics without unconstrained memory growth:

- **File Metadata**: File path, row count, column count, headers.
- **Data Types**: Observed types for all 10 columns.
- **Missing & Empty Counts**: Missing counts and raw empty-string counts per field.
- **Unicode & Script Indicators**: Non-ASCII counts, character category breakdowns (Latin, Latin-Extended, Devanagari, Arabic, Cyrillic, CJK, etc.).
- **Suspicious Symbols**: Control character counts, replacement character (`\uFFFD`) occurrences, private use codepoints.
- **Text Length Summaries**: Exact min, max, mean, and quantiles (p25, p50, p75, p90, p95, p99) computed via bounded integer histograms.
- **Deterministic Examples**: Bounded top-5 shortest non-empty and top-5 longest examples per field.
- **Duplicate Metrics**:
  - Unique normalized names with >1 occurrences and total duplicate rows.
  - Unique normalized addresses with >1 occurrences and total duplicate rows.
  - Unique full normalized tuples `(name, address, country)` with >1 occurrences.
- **Integrity Findings**: Empty list `[]` for duplicate entity IDs and malformed rows.
- **Country Distribution**: Frequency counts for each observed country label.
- **Preservation Verification**: Explicit verification flags confirming row count, order, IDs, and raw fields were preserved.

---

## 8. External Data & Fairness Statement

- **No external data used**: This pipeline strictly does not use external databases, public entity registries, postal databases, geocoding APIs, web search engines, or translation models.
- **No blocking or matching performed**: This stage is strictly confined to data hygiene, character normalization, validation, and profiling. Downstream candidate generation and matching stages operate on the validated outputs of this module.
