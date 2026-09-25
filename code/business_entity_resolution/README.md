# Business Entity Resolution Pipeline — Stage 1: Data Cleaning & Profiling

## 1. Purpose and Scope

This module implements **Stage 1 (Conservative Data Cleaning, Validation, and Profiling)** of the Business Entity Resolution pipeline for the Amazon ML Challenge 2026.

### Explicit Scope Constraints
- **What this stage does**:
  - Validates raw input TSV files for UTF-8 readability, exact expected headers, 4-column TSV structure, non-empty and unique entity IDs, and source prefix consistency.
  - Performs bounded-memory streaming conservative normalization to produce derived columns (`business_name_normalized`, `business_address_normalized`, `country_normalized`) and boolean missingness indicators (`business_name_is_missing`, `business_address_is_missing`, `country_is_missing`).
  - Ensures **raw input files are never modified**, **original parsed field values are retained in the first four output columns**, and **output files are deterministic UTF-8 TSV serializations**.
  - Enforces atomic file publication with generation staging and automatic rollback protection, verifying 12 strict integrity invariants (including exact derived value recomputation) before publishing any output.
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
    ├── quality_checks.json
    └── manifest.json
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

### Whitespace & Missingness Policy
- **Noise Treatment**: Unicode whitespace characters, tabs (`\t`), carriage returns (`\r`), embedded line breaks (`\n`), and BOM / zero-width characters (`\uFEFF`, `\u200B`) are treated as formatting noise.
- **Run Collapse**: Consecutive whitespace sequences (`\s+`) are collapsed into a single ASCII space (`' '`).
- **Stripping**: Leading and trailing whitespace is stripped.
- **Missing Value Handling**:
  - Fields consisting solely of empty strings, Unicode whitespace, or BOM (`\uFEFF`) are treated as missing.
  - Normalized value for missing fields is empty string `""`.
  - Boolean flags `business_name_is_missing`, `business_address_is_missing`, `country_is_missing` are emitted (`True` / `False`).
  - No synthetic placeholders (e.g., `"Unknown"`, `"Missing"`, `"N/A"`) are fabricated.
  - Real business names such as `"NAN"`, `"Null"`, or `"None"` are preserved and NOT treated as missing.
  - **No record is ever dropped due to missing fields**.

### Narrowed Safe Punctuation Equivalences
Only clearly safe, explicitly documented equivalences are mapped in derived fields:
- **Curly Single Quotes $\rightarrow$ Straight Single Quote (`'`)**:
  - `‘` (`\u2018`), `’` (`\u2019`), `‚` (`\u201A`), `‛` (`\u201B`) $\rightarrow$ `'`
- **Curly Double Quotes $\rightarrow$ Straight Double Quote (`"`)**:
  - `“` (`\u201C`), `”` (`\u201D`), `„` (`\u201E`), `‟` (`\u201F`) $\rightarrow$ `"`
- **Documented Unicode Dash Variants $\rightarrow$ ASCII Hyphen (`-`)**:
  - `‐` (`\u2010`), `‑` (`\u2011`), `‒` (`\u2012`), `–` (`\u2013`), `—` (`\u2014`), `―` (`\u2015`), `−` (`\u2212`), `﹘` (`\uFE58`), `﹣` (`\uFE63`), `－` (`\uFF0D`) $\rightarrow$ `-`
- **Unicode Space Variants $\rightarrow$ ASCII Space (`' '`)**:
  - `\u00A0` (non-breaking space), `\u2000`–`\u200A` (quads/spaces), `\u202F`, `\u3000` (ideographic space), `\uFEFF` $\rightarrow$ `' '`
- **Punctuation Strictly Preserved**:
  - Primes (`\u2032` / `\u2033`) and language-specific modifier letters (`\u02BB` / `\u02BC`) are **NOT** converted to quotes.
  - Punctuation such as `&`, `-`, `.`, `'`, `/`, `@`, `+`, `#`, commas `,`, and parentheses `(` `)` is **NOT** stripped or deleted.

### Business Names & Addresses
- **No legal suffix stripping**: Does not remove suffixes like `LLC`, `Corp`, `Inc`, `Pvt Ltd`, `LLP`.
- **No abbreviation expansion**: Does not expand abbreviations (e.g. `St` to `Street`, `Rd` to `Road`).
- **No token reordering**: Token order is preserved exactly.
- **No deduplication of tokens**: Repeated words (e.g. `Pizza Pizza`) remain untouched.
- **No typo correction**: Suspected typos are not modified.
- **No geocoding or address reconstruction**: Postal codes, state names, landmark references (e.g., `Near SBI ATM`) are kept untouched.

### Country
- Created using **only**:
  1. Unicode NFC
  2. Unicode `.casefold()`
  3. Whitespace normalization
- Does **not** hardcode country sets, filter countries, or map aliases (`US`, `India`, and `France` are preserved generically).

---

## 5. Raw Data Preservation Guarantees

1. **Raw input files under student_resource/dataset are never modified**.
2. **Original parsed field values are retained in the first four output columns** in exact original order.
3. **Output files are deterministic UTF-8 TSV serializations**.
4. **Equal Treatment Across Sources**: Source 1 receives only the identical conservative base normalization applied to Source 2 and Source 3.

---

## 6. Structural Integrity, Memory Model & Quality Gates

### Actual Memory Model
- **Streaming Row Transformation**: Streaming row-by-row reading and writing with bounded I/O buffers.
- **Disk-Backed SQLite Aggregation**: Uniqueness validation, duplicate profiling, and length frequency distributions are offloaded to ephemeral SQLite databases on disk with small fixed page cache (`PRAGMA cache_size = -4000`, ~4MB RAM).
- **True Bounded-Memory Aggregations**: Length distributions are stored and aggregated via disk-backed SQLite frequency tables rather than unbounded in-memory dictionaries or Counters, enabling exact quantile calculation (p25, p50, p75, p90, p95, p99) with strictly bounded memory.
- **Chunked Manifest Hashing**: All file digests in `manifest.json` are computed using streaming 1MB chunks (`sha256_file`), never loading multi-gigabyte TSVs into memory.
- **Collision-Resistant Deterministic Digests**: Duplicate metrics are calculated using deterministic SHA-256 digests (`hashlib.sha256`) rather than process-randomized Python `hash()`.

### Multi-Tier Gatekeeper & Failure-Safe Publication
```
[Raw TSV Files]
       │
       ▼
Phase 1: Mandatory Structural Validation Gate (Disk-backed uniqueness check)
  ├── UTF-8 readability
  ├── Exact header check
  ├── 4-column TSV structure
  ├── Non-empty entity_id
  ├── Consistent source prefix (S1-, S2-, S3-)
  └── File-level entity_id uniqueness (disk-backed SQLite PRIMARY KEY)
       │ (Mandatory; cannot be skipped)
       ▼
Phase 2: Streaming Clean & Profile into Generation Staging
  └── Writes to <output_dir>/.generation_staging_<timestamp>_<pid>/
       │
       ▼
Phase 3: Post-Cleaning Quality Check Gate
  ├── Recomputes normalize_text and normalize_country for every row
  └── Verifies 12 strict invariants against raw input line-by-line:
      1. row_count_match
      2. entity_id_match_and_order
      3. original_values_unaltered
      4. source_columns_present
      5. no_unexpected_nulls
      6. normalized_missing_not_fabricated
      7. output_headers_correct
      8. valid_utf8_tsv
      9. no_malformed_rows
     10. normalized_columns_present (exact derived value match)
     11. source_prefix_validation
     12. boolean_flags_valid
       │
       ▼
Phase 4: Rollback-Safe Multi-Directory Publication Transaction
  ├── Stages reports/profile.json, reports/quality_checks.json, reports/manifest.json
  ├── Step 1: Backs up existing target directories
  ├── Step 2: Publishes staged directories to destination
  ├── Step 3: On success, removes backups and staging
  └── On any failure, automatically rolls back restored state byte-identically
```

If any invariant fails:
- Staging directory is immediately cleaned up.
- The destination `output_dir` is completely untouched.
- The pipeline aborts with exit code 1, reporting file path, line number, and exact reason.

---

## 7. Profiling Report Format (`profile.json`)

The profiling report (`cleaned/reports/profile.json`) summarizes dataset characteristics:

- **File Metadata**: File path, row count, column count, headers.
- **Data Types**: Observed types for all 10 columns.
- **Missing & Empty Counts**: Missing counts (including BOM/zero-width space) and empty-string counts.
- **Unicode & Script Indicators**: Non-ASCII counts, character category breakdowns (Latin, Latin-Extended, Devanagari, Arabic, Cyrillic, CJK, etc.).
- **Suspicious Symbols**: Control character counts, replacement character (`\uFFFD`) occurrences, private use codepoints.
- **Text Length Summaries**: Exact min, max, mean, and quantiles (p25, p50, p75, p90, p95, p99) computed via disk-backed SQLite length frequencies.
- **Deterministic Examples**: Bounded top-5 shortest non-empty and top-5 longest examples per field.
- **Duplicate Metrics (SHA-256)**:
  - Unique normalized names with >1 occurrences and total duplicate rows.
  - Unique normalized addresses with >1 occurrences and total duplicate rows.
  - Unique full normalized tuples `(name, address, country)` with >1 occurrences.
- **Integrity Findings**: Confirmation of duplicate entity IDs (`{"status": "none_detected_during_mandatory_validation"}`) and malformed rows (`{"status": "none_detected_during_mandatory_validation"}`).
- **Country Distribution**: Frequency counts for each observed country label.
- **Preservation Verification**: Explicit verification flags confirming row count, order, IDs, and raw fields were preserved.

---

## 8. External Data & Fairness Statement

- **No external data used**: This pipeline strictly does not use external databases, public entity registries, postal databases, geocoding APIs, web search engines, or translation models.
- **No blocking or matching performed**: This stage is strictly confined to data hygiene, character normalization, validation, and profiling. Downstream candidate generation and matching stages operate on the validated outputs of this module.
