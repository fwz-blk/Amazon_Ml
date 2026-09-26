"""
Blocking Evaluation & Validation Holdout Metrics
Amazon ML Challenge 2026 - Business Entity Resolution

Requirements:
- Read train_ground_truth.tsv with comma-separated matched_entity_ids
- Preserve one-to-many matches and singleton Source 1 rows
- Deterministic entity-level train/validation holdout (SHA-256 hash modulo)
- Evaluate pair-level recall, S1 coverage, S2 recall, S3 recall, union recall, per-block recall
- Measure unique recovery per block family
- Track candidate count distribution (mean, median, p95, max)
- Track candidate reduction ratio, block overflows, cap triggers, and lost true pairs
- Minimum acceptance target: >= 99% pair recall on held-out validation set
- Support memory-bounded evaluation streaming directly from CandidateStore
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

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


def is_validation_entity(s1_id: str, holdout_ratio: float, salt: str = "val_salt") -> bool:
    """Deterministic entity-level holdout partition via SHA-256 hash modulo."""
    h = hashlib.sha256(f"{s1_id}_{salt}".encode("utf-8")).hexdigest()
    val = int(h[:8], 16) % 10000
    return val < int(holdout_ratio * 10000)


def load_ground_truth(
    gt_path: Path,
    holdout_ratio: Optional[float] = None,
    salt: str = "val_salt",
    validation_only: bool = False,
) -> Dict[str, Set[str]]:
    """
    Load ground truth mapping from train_ground_truth.tsv:
        source1_entity_id -> set of matched candidate IDs.
    Preserves one-to-many relationships and empty sets for singletons.
    """
    ground_truth: Dict[str, Set[str]] = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if not parts:
                continue
            s1_id = parts[0]
            matched_str = parts[1] if len(parts) > 1 else ""

            if holdout_ratio is not None:
                in_val = is_validation_entity(s1_id, holdout_ratio, salt)
                if validation_only and not in_val:
                    continue
                if not validation_only and in_val:
                    continue

            matched_ids = set(filter(None, matched_str.split(","))) if matched_str else set()
            ground_truth[s1_id] = matched_ids

    return ground_truth


class BlockingEvaluator:
    """Evaluates candidate generation recall, diagnostics, and capacity."""

    def __init__(self, ground_truth: Dict[str, Set[str]]) -> None:
        self.ground_truth = ground_truth
        self.total_s1 = len(ground_truth)
        self.total_true_pairs = sum(len(cands) for cands in ground_truth.values())
        self.total_s2_true_pairs = sum(
            len([c for c in cands if c.startswith("S2-")]) for cands in ground_truth.values()
        )
        self.total_s3_true_pairs = sum(
            len([c for c in cands if c.startswith("S3-")]) for cands in ground_truth.values()
        )
        self.singletons = sum(1 for cands in ground_truth.values() if not cands)

    def evaluate_candidates(
        self,
        retrieved_candidates: Dict[str, Dict[str, int]],
        overflow_events: Optional[List[Dict[str, Any]]] = None,
        total_candidate_universe: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate in-memory retrieved candidates against ground truth.
        """
        return self._evaluate_stream(
            lambda s1_id: retrieved_candidates.get(s1_id, {}),
            overflow_events=overflow_events,
            total_candidate_universe=total_candidate_universe,
        )

    def evaluate_from_store(
        self,
        store: Any,
        max_candidates_per_source: Optional[int] = None,
        overflow_events: Optional[List[Dict[str, Any]]] = None,
        total_candidate_universe: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate candidate retrieval streaming directly from CandidateStore,
        optionally applying deterministic ranking and capping per source.
        """
        if max_candidates_per_source is not None:
            candidate_getter = lambda s1_id: store.get_capped_candidates_for_s1(
                s1_id, max_candidates_per_source
            )
        else:
            candidate_getter = lambda s1_id: store.get_candidates_for_s1(s1_id)

        return self._evaluate_stream(
            candidate_getter,
            overflow_events=overflow_events,
            total_candidate_universe=total_candidate_universe,
        )

    def _evaluate_stream(
        self,
        candidate_getter: Callable[[str], Dict[str, int]],
        overflow_events: Optional[List[Dict[str, Any]]] = None,
        total_candidate_universe: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Core streaming evaluation logic."""
        hits = 0
        s2_hits = 0
        s3_hits = 0
        s1_with_at_least_one_hit = 0
        no_usable_key_count = 0

        # Per-block true pair recovery
        per_block_hits: Dict[str, int] = {name: 0 for name in PROVENANCE_NAMES.values()}
        # For unique recovery: pair -> mask of blocks that hit it
        pair_prov_masks: Dict[Tuple[str, str], int] = {}

        candidate_counts: List[int] = []
        source_counts: Dict[str, int] = defaultdict(int)

        for s1_id, true_cands in self.ground_truth.items():
            retrieved = candidate_getter(s1_id)
            candidate_counts.append(len(retrieved))

            if not retrieved:
                no_usable_key_count += 1

            for cid, prov in retrieved.items():
                src = cid[:2]
                source_counts[src] += 1

            # Match against ground truth
            if true_cands:
                matched_in_gt = true_cands & set(retrieved.keys())
                hit_count = len(matched_in_gt)
                hits += hit_count

                if hit_count > 0:
                    s1_with_at_least_one_hit += 1

                for cid in matched_in_gt:
                    prov = retrieved[cid]
                    pair_prov_masks[(s1_id, cid)] = prov
                    if cid.startswith("S2-"):
                        s2_hits += 1
                    elif cid.startswith("S3-"):
                        s3_hits += 1

                    for flag, name in PROVENANCE_NAMES.items():
                        if prov & flag:
                            per_block_hits[name] += 1

        # Unique recovery per block family (hits recovered ONLY by this block family)
        unique_recovery: Dict[str, int] = {name: 0 for name in PROVENANCE_NAMES.values()}
        for (s1_id, cid), prov in pair_prov_masks.items():
            active_flags = [flag for flag in PROVENANCE_NAMES if (prov & flag)]
            if len(active_flags) == 1:
                unique_recovery[PROVENANCE_NAMES[active_flags[0]]] += 1

        candidate_counts.sort()
        count_len = len(candidate_counts)
        median = candidate_counts[count_len // 2] if count_len else 0
        p95 = candidate_counts[int(count_len * 0.95)] if count_len else 0
        p99 = candidate_counts[int(count_len * 0.99)] if count_len else 0
        max_c = candidate_counts[-1] if count_len else 0
        total_retrieved = sum(candidate_counts)
        mean = total_retrieved / count_len if count_len else 0

        pair_recall = hits / self.total_true_pairs if self.total_true_pairs else 1.0
        s2_recall = s2_hits / self.total_s2_true_pairs if self.total_s2_true_pairs else 1.0
        s3_recall = s3_hits / self.total_s3_true_pairs if self.total_s3_true_pairs else 1.0

        non_singletons = self.total_s1 - self.singletons
        s1_coverage = s1_with_at_least_one_hit / non_singletons if non_singletons else 1.0

        # Candidate reduction ratio
        reduction_ratio = (
            1.0 - (total_retrieved / total_candidate_universe)
            if total_candidate_universe and total_candidate_universe > 0
            else None
        )

        return {
            "pair_recall": round(pair_recall, 5),
            # Union recall: defined as fraction of true ground-truth pairs captured across
            # the full multi-pass candidate union
            "union_recall": round(pair_recall, 5),
            "union_recall_definition": "Proportion of all true matching candidate pairs retrieved across the union of all blocking passes",
            "s1_coverage": round(s1_coverage, 5),
            "s2_recall": round(s2_recall, 5),
            "s3_recall": round(s3_recall, 5),
            "target_achieved": pair_recall >= 0.99,
            "counts": {
                "total_true_pairs": self.total_true_pairs,
                "hits": hits,
                "missed_true_pairs": self.total_true_pairs - hits,
                "s2_true_pairs": self.total_s2_true_pairs,
                "s2_hits": s2_hits,
                "s3_true_pairs": self.total_s3_true_pairs,
                "s3_hits": s3_hits,
                "total_s1_entities": self.total_s1,
                "singletons": self.singletons,
                "s1_with_hits": s1_with_at_least_one_hit,
                "no_usable_key_count": no_usable_key_count,
                "total_candidate_pairs_retrieved": total_retrieved,
            },
            "candidate_counts_per_s1": {
                "min": candidate_counts[0] if count_len else 0,
                "median": median,
                "mean": round(mean, 2),
                "p95": p95,
                "p99": p99,
                "max": max_c,
            },
            "candidate_reduction_ratio": round(reduction_ratio, 6) if reduction_ratio else None,
            "candidates_by_source": dict(source_counts),
            "per_block_recall": {
                name: round(cnt / self.total_true_pairs, 5) if self.total_true_pairs else 0
                for name, cnt in per_block_hits.items()
            },
            "per_block_hits": per_block_hits,
            "unique_recovery_by_block": unique_recovery,
            "overflow_event_count": len(overflow_events) if overflow_events else 0,
        }
