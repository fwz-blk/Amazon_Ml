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

---

## 9. Stage 2: Multi-Pass Blocking & Candidate Generation

### Scope Constraints & Boundary
- **What this stage does**:
  - Consumes cleaned TSV datasets from `cleaned/` (`business_name_normalized`, `business_address_normalized`, `country_normalized`).
  - Implements multi-pass inverted candidate indexing and candidate generation across 6 blocking families plus fallbacks.
  - Indexes Candidate Source 2 and Candidate Source 3 separately, preserving source-specific metrics and diagnostics.
  - Applies split-specific indexes (never mixing train and test data; never tuning on test data).
  - Employs disk-backed SQLite indexing and candidate storage with bounded page cache (`PRAGMA cache_size = -64000`), keeping memory strictly bounded to ~400 MB.
  - Generates the required `candidate_pairs.tsv` adhering to the exact contract: `source1_entity_id \t candidate_entity_ids` (all S2 IDs first, then all S3 IDs, each lexicographically sorted; exactly 1 row per S1).
  - Evaluates deterministic holdout pair recall against ground truth (`train_ground_truth.tsv`).
- **What this stage does NOT do**:
  - **No pair matching or similarity scoring**.
  - **No feature engineering or ML model training**.
  - **No classification, thresholding, or prediction decisions**.
  - **No generation of `matching_results.tsv` or submission files**.
  - **Blocking only produces high-recall candidate sets; it does NOT make match decisions**.

### Blocking Pass Families
1. **Country Partition (Default)**:
   - Partitions indexing and search by `country_normalized` as an open set.
   - Cross-country candidate expansion is strictly forbidden by default and gated by high-specificity fallbacks.
2. **Exact Normalized Name**:
   - Exact `business_name_normalized` equality within same country.
   - Suffixes, punctuation, and token order are preserved exactly as normalized.
3. **Rare Name-Token Block**:
   - Multilingual Unicode tokenization preserving letters, numbers, and combining marks (e.g. Indic matras).
   - Frequency filtering via absolute (`max_name_df_absolute = 10000`) and relative (`max_name_df_relative = 0.003`) thresholds.
   - Stoplist filtering for universally generic legal terms (`inc`, `llc`, `corp`, `ltd`, `gmbh`, etc.) preventing them from acting as standalone keys.
4. **Address-Token Block**:
   - Distinctive locality/street tokens extracted from `business_address_normalized` with frequency filtering.
   - Stoplist filtering for common address keywords (`st`, `ave`, `rd`, `suite`, `floor`, etc.).
5. **Address-Number Plus Location Block**:
   - Extracts digit sequences from address and pairs each number with up to 10 distinctive locality tokens `(number, token)`.
   - Reordered address components (e.g., number at end vs beginning) produce identical keys.
6. **Compact Unicode Character N-Gram Block**:
   - Character 4-grams extracted from a compact alphanumeric view of normalized business names across all scripts.
   - Handles typos, concatenations, and spelling corruptions.

### Missing-Country and Cross-Country Fallback Policies
- **Missing-Country Fallback**:
  - Activated whenever either side lacks a country label (`country_normalized == ""`):
    1. Known S1 country vs missing-country candidate records (`country = ''`) queried via exact name, rare tokens, and number+location.
    2. Missing-country S1 vs known-country candidate records across all observed countries via high-specificity fallback keys.
    3. Missing-country S1 vs missing-country candidate records.
  - Tagged with provenance bitmask `PROV_MISSING_COUNTRY_FALLBACK`.
- **Cross-Country High Specificity**:
  - When both S1 and candidate have known, differing countries:
    - Broad single-token, address-token, or character n-gram scans are **strictly forbidden**.
    - Candidates enter ONLY through high-specificity evidence:
      1. Exact normalized name across countries.
      2. Multiple rare name tokens ($\ge 2$ tokens matching) across countries.
      3. Rare address number + location across countries.
  - Tagged with provenance bitmask `PROV_CROSS_COUNTRY_HIGH_SPECIFICITY`.

### Overflow, Refinement, and Deterministic Ranking Policy
- **Refinement First**:
  - When an inverted index posting list exceeds `max_block_size = 1000`, the block is refined using 2-token intersection (or non-overlapping n-gram intersection).
  - If refinement reduces postings $\le 1000$, the refined candidate set is added.
- **Retained Overflow Postings**:
  - If refinement cannot reduce postings $\le 1000$, postings are **retained** rather than discarded to zero candidates.
  - Candidates pass to `CandidateStore` with full provenance tracking.
- **Deterministic Evidence Ranking (Safety Budget)**:
  - If candidate count per S1 per source exceeds `max_candidates_per_s1_per_source = 1000`, deterministic evidence ranking trims candidates:
    1. Exact normalized-name evidence (`PROV_EXACT_NAME`).
    2. Multiple independent block-family hits (count of active bits in provenance mask).
    3. Rare name token evidence (`PROV_RARE_NAME_TOKEN`).
    4. Address number + location evidence (`PROV_ADDRESS_NUMBER_LOCATION`).
    5. Informative address token evidence (`PROV_ADDRESS_TOKEN`).
    6. Compact n-gram evidence (`PROV_COMPACT_CHAR_NGRAM`).
    7. Missing-country fallback evidence (`PROV_MISSING_COUNTRY_FALLBACK`).
    8. Cross-country high-specificity evidence (`PROV_CROSS_COUNTRY_HIGH_SPECIFICITY`).
    9. **Deterministic tie-breaker**: Candidate entity ID ascending (`S2-...` or `S3-...`).
  - No matcher similarity scores, random order, or insertion order are ever used.

### Disk-Backed Architecture & Resource Safety
- **Bounded Inverted Index**: Candidate Source 2 and Source 3 are indexed into disk-backed SQLite databases with covering B-tree indexes. Temporary databases and SQLite sort files are strictly kept on the workspace disk (`SQLITE_TMPDIR` / `TMPDIR`), ensuring memory usage is capped to `PRAGMA cache_size = -64000` (64 MB).
- **Streaming Candidate Store**: Primary key `(source1_id, cand_id) WITHOUT ROWID`. Deduplication merges provenance bitmasks (`ON CONFLICT DO UPDATE SET provenance = provenance | excluded.provenance`).
- **Streaming TSV Export**: Reads S1 entities line by line and exports directly to TSV without loading the full population into Python memory.
- **Lifecycle Cleanup**: Temporary index databases (`.idx_tmp/*.db`) are automatically unlinked on completion or failure.

### Candidate Cap Sweep Evaluation & Production-Readiness

To lock down the blocking strategy before the matching phase, candidate generation was evaluated across a sweep of per-source candidate budgets (`max_candidates_per_s1_per_source` $\in \{500, 1000, 1500, 2000, 3000, 5000, 7000\}$).

#### 1. Deterministic Validation Holdout Definition
- **Holdout Hash Function**: `int(hashlib.sha256((salt + entity_id).encode("utf-8")).hexdigest(), 16) % 10000 / 10000.0 < holdout_ratio`
- **Configuration**: `holdout_ratio = 0.05` (5% partition), `holdout_salt = "amazon_ml_2026_val_salt"`
- **Total Validation Entities in Full Training Split**: **110,142** Source 1 entities out of 2,206,821 total S1 records.
- **Evaluated Validation Holdout Subset**: 1,000 deterministic S1 entities evaluated across all 10,320,219 indexed candidate records (5,034,616 S2 + 5,285,603 S3).
- **Ground Truth Grounding**: 3,489 true pairs (1,688 Source 2 pairs, 1,801 Source 3 pairs, 49 singletons).
- **Pre-Capping Generator Recall**: **99.51%** (3,472 / 3,489 true pairs generated; S2: 99.58%, S3: 99.45%, S1 Coverage: 100.0%), confirming zero candidate loss in candidate generation prior to `CandidateStore`.

#### 2. Candidate Cap Sweep Results

| Cap / Source | Retained Pairs | Truncated Pairs | S1 Capped | Overall Recall | S2 Recall | S3 Recall | Status ($\ge 99\%$ all) |
|---|---|---|---|---|---|---|---|
| 500 | 995,622 | 8,271,333 | 993 / 1,000 | 93.78% | 93.96% | 93.62% | FAILED (< 99%) |
| 1,000 | 1,973,309 | 7,293,646 | 973 / 1,000 | 96.25% | 96.03% | 96.45% | FAILED (< 99%) |
| 1,500 | 2,907,976 | 6,358,979 | 948 / 1,000 | 97.31% | 97.28% | 97.34% | FAILED (< 99%) |
| 2,000 | 3,768,151 | 5,498,804 | 920 / 1,000 | 97.54% | 97.45% | 97.61% | FAILED (< 99%) |
| 3,000 | 5,215,228 | 4,051,727 | 845 / 1,000 | 97.65% | 97.57% | 97.72% | FAILED (< 99%) |
| 5,000 | 7,139,846 | 2,127,109 | 609 / 1,000 | 98.68% | 99.23% | 98.17% | FAILED (S3 < 99%) |
| **7,000** | **8,172,455** | **1,094,500** | **278 / 1,000** | **99.31%** | **99.53%** | **99.11%** | **PASSED (ALL $\ge 99\%$)** |

#### 3. Recommended Cap & Selection Justification
- **Selected Recommended Cap**: **`7,000`** candidates per S1 per source.
- **Justification**: Cap 7000 is the smallest tested per-source budget satisfying all acceptance criteria independently:
  - Overall Pair Recall: **99.31%** ($\ge 99.0\%$)
  - Source 2 Pair Recall: **99.53%** ($\ge 99.0\%$)
  - Source 3 Pair Recall: **99.11%** ($\ge 99.0\%$)
  - S1 Coverage: **100.00%**
  - Unexplained loss before `CandidateStore`: **0**
- **Why Caps Below 7,000 Lost Recall**:
  - Oversized candidate blocks originating from generic single tokens (e.g. broad locality names or common 4-grams) can inject 5,000–12,000 candidate postings for a single S1 entity having identical provenance rank tiers.
  - When candidate volume exceeds the per-source budget, deterministic tie-breaking sorts by candidate entity ID (`S2-...` / `S3-...`).
  - At lower budgets ($N \le 5000$), genuine matches with larger ID suffixes were crowded out by generic single-token ties.
  - At Cap 7000, 3,465 out of 3,489 true matches are preserved, exceeding the 99% threshold across both sources.

#### 4. Full Production Resource Projections (Cap 7000)

| Metric | Training Split (2,206,821 S1) | Test Split (1,732,544 S1) |
|---|---|---|
| Projected Candidate Pairs (Pre-Cap) | 20.45 billion | 16.06 billion |
| **Projected Retained Candidate Pairs** | **18.04 billion** | **14.16 billion** |
| Mean Retained Pairs per S1 | 8,172.5 pairs | 8,172.5 pairs |
| **Projected Candidate TSV Size** | **216.51 GB** (232.5 GB uncompressed) | **169.98 GB** (182.5 GB uncompressed) |
| Projected SQLite CandidateStore DB Size | 664.43 GB (713.4 GB) | 521.63 GB (560.1 GB) |
| Inverted Index DB Size (S2 + S3) | 23.59 GB (25.33 GB) | 23.00 GB (24.70 GB) |
| **Total Disk Space Required to Persist** | **~904.5 GB** | **~715.2 GB** |
| Projected End-to-End Runtime (8 Cores) | ~50.1 hours | ~39.4 hours |
| **Peak Resident Memory (RSS)** | **< 500 MB** | **< 500 MB** |

> [!WARNING]
> **Production Disk Safety Blocker**:
> The local machine environment currently provides **~196 GB of free disk space**. Persisting all ~18.04 billion candidate pairs to disk as an intermediate static TSV (216.5 GB) or SQLite database (664.4 GB) would exceed available storage.
>
> **Architectural Constraint for Stage 3 (Matching)**:
> In full production runs, candidate pairs **must not be persisted to a monolithic disk file**. Instead, Stage 3 matching must consume candidates **in streaming chunks or batches** (e.g. 5,000–10,000 S1 entities per batch) directly from the blocking query generator, computing features and scoring pairs on the fly without intermediate disk ballooning.

#### 5. Execution Instructions

##### Reproduce the Candidate Cap Sweep
```bash
python3 code/business_entity_resolution/src/blocking/run_blocking.py \
    --split val \
    --output-dir candidates_validation \
    --sweep-caps 500,1000,1500,2000,3000,5000,7000 \
    --workers 4
```

Outputs produced:
- `candidates_validation/cap_sweep_summary.tsv` (All 27 required metrics across all tested caps)
- `candidates_validation/cap_sweep_report.json` (Structured JSON report including full production projections)
- `candidates_validation/candidate_pairs.tsv` (Exported candidate TSV at recommended cap 7000)
- `candidates_validation/blocking_diagnostics_val.json` (Diagnostics JSON for recommended cap)

##### Run Blocking at Recommended Cap (Cap 7000)
```bash
python3 code/business_entity_resolution/src/blocking/run_blocking.py \
    --split val \
    --output-dir candidates_validation \
    --cap 7000 \
    --workers 4
```

##### Command-Line Options Reference
- `--max-candidates-per-s1-per-source`, `--cap`: Set the per-source candidate cap per S1 entity (default: 1000, recommended: 7000).
- `--sweep-caps`: Comma-separated list of candidate caps to evaluate in a single run (e.g. `500,1000,1500,2000,3000,5000,7000`).
- `--workers`: Number of parallel worker processes for inverted indexing and S1 candidate querying (default: 4).
- `--reuse-store`: Reuse existing `candidates_val.db` if candidate query phase was already completed, enabling fast re-evaluation of caps.
- `--max-s1`: Limit the number of Source 1 entities processed (useful for validation fixtures and rapid sanity checking).

