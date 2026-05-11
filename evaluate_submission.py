"""
evaluate_submission.py

Evaluates a submission against the consolidated ground truth.
Computes nDCG@5 (Exp) per split and updates leaderboard.json.

Usage:
    python evaluate_submission.py --submission_dir submissions/<model_name> \
                                  --annotated_dir annotated/unified_<consolidation_model> \
                                  --leaderboard leaderboard.json
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# nDCG@5 with exponential gain
# ---------------------------------------------------------------------------

def dcg_at_k(ranked_labels: list[int], k: int = 5) -> float:
    """Compute DCG@k using exponential gain: sum (2^rel - 1) / log2(i+2)."""
    score = 0.0
    for i, rel in enumerate(ranked_labels[:k]):
        score += (2 ** rel - 1) / math.log2(i + 2)
    return score


def ndcg_at_k(ranked_labels: list[int], all_labels: list[int], k: int = 5) -> float:
    """
    Compute nDCG@k.
    ranked_labels : relevance of model's ranked steps (in ranked order)
    all_labels    : relevance of ALL steps (to build ideal ranking)
    """
    ideal = sorted(all_labels, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0.0:
        # No positive labels in this example; skip (return None)
        return None
    return dcg_at_k(ranked_labels, k) / idcg


# ---------------------------------------------------------------------------
# Load ground truth from annotated unified directory
# ---------------------------------------------------------------------------

def load_ground_truth(annotated_dir: Path, split: str) -> dict[str, list[int]]:
    """
    Returns { file_id -> list of per-step binary labels (0/1) }
    from annotated/<unified>/<split>/*.json files.

    Each file is expected to have:
      {
        "history": [
          { "step": "0", "fail_annotation": "0" | "1", ... },
          ...
        ]
      }
    """
    split_dir = annotated_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Ground truth directory not found: {split_dir}\n"
            "Make sure you ran data_processing.sh and reasoning_consolidation.py first."
        )

    gt = {}
    for path in sorted(split_dir.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        file_id = path.stem
        labels = [int(step.get("fail_annotation", 0)) for step in data.get("history", [])]
        gt[file_id] = labels

    if not gt:
        raise ValueError(f"No ground truth files found in {split_dir}")

    return gt


# ---------------------------------------------------------------------------
# Load and validate predictions
# ---------------------------------------------------------------------------

def load_predictions(pred_path: Path) -> dict[str, list[int]]:
    """
    Load predictions JSON: { file_id -> [ranked step indices] }
    """
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    with open(pred_path) as f:
        preds = json.load(f)
    if not isinstance(preds, dict):
        raise ValueError(f"{pred_path} must be a JSON object mapping file_id -> ranked step list")
    return preds


# ---------------------------------------------------------------------------
# Evaluate one split
# ---------------------------------------------------------------------------

def evaluate_split(
    preds: dict[str, list[int]],
    ground_truth: dict[str, list[int]],
    k: int = 5,
) -> dict:
    """
    Returns {
        "ndcg5": float,          # mean nDCG@5 (only over examples with ≥1 positive label)
        "num_examples": int,     # total files in GT
        "num_scored": int,       # files with ≥1 positive label
        "num_missing": int,      # GT files absent from submission
        "per_file": { file_id: float | null }
    }
    """
    scores = {}
    missing = 0

    for file_id, gt_labels in ground_truth.items():
        if file_id not in preds:
            missing += 1
            scores[file_id] = None
            continue

        ranked_steps = preds[file_id]

        # Validate step indices
        n_steps = len(gt_labels)
        ranked_steps = [s for s in ranked_steps if 0 <= s < n_steps]

        # Build ranked_labels (relevance in ranked order, pad with 0 if < k steps)
        seen = set()
        ranked_labels = []
        for s in ranked_steps:
            if s not in seen:
                ranked_labels.append(gt_labels[s])
                seen.add(s)

        # Any unranked steps default to 0 relevance — no need to pad explicitly
        # (dcg_at_k handles truncation at k)

        score = ndcg_at_k(ranked_labels, gt_labels, k=k)
        scores[file_id] = score  # None if no positive labels in this example

    scored = [v for v in scores.values() if v is not None]
    mean_ndcg = sum(scored) / len(scored) if scored else 0.0

    return {
        "ndcg5": round(mean_ndcg, 4),
        "num_examples": len(ground_truth),
        "num_scored": len(scored),
        "num_missing": missing,
        "per_file": scores,
    }


# ---------------------------------------------------------------------------
# Update leaderboard.json
# ---------------------------------------------------------------------------

def update_leaderboard(
    leaderboard_path: Path,
    model_name: str,
    metadata: dict,
    results: dict,
) -> None:
    if leaderboard_path.exists():
        with open(leaderboard_path) as f:
            lb = json.load(f)
    else:
        lb = []

    # Remove existing entry for this model (upsert)
    lb = [e for e in lb if e["model_name"] != model_name]

    entry = {
        "model_name": model_name,
        "organization": metadata.get("organization", ""),
        "paper_url": metadata.get("paper_url", ""),
        "code_url": metadata.get("code_url", ""),
        "description": metadata.get("description", ""),
        "manual_ndcg5": results.get("manual", {}).get("ndcg5"),
        "automatic_ndcg5": results.get("automatic", {}).get("ndcg5"),
        "manual_num_scored": results.get("manual", {}).get("num_scored"),
        "automatic_num_scored": results.get("automatic", {}).get("num_scored"),
        "submitted_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }
    lb.append(entry)

    # Sort by manual nDCG@5 descending (None sorts last)
    lb.sort(key=lambda e: e.get("manual_ndcg5") or -1, reverse=True)

    with open(leaderboard_path, "w") as f:
        json.dump(lb, f, indent=2)

    print(f"Leaderboard updated: {leaderboard_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission_dir", required=True, help="Path to submissions/<model_name>/")
    parser.add_argument("--annotated_dir", required=True, help="Path to annotated/unified_<model>/")
    parser.add_argument("--leaderboard", default="leaderboard.json", help="Path to leaderboard.json")
    parser.add_argument("--k", type=int, default=5, help="Cutoff for nDCG (default: 5)")
    args = parser.parse_args()

    submission_dir = Path(args.submission_dir)
    annotated_dir = Path(args.annotated_dir)
    leaderboard_path = Path(args.leaderboard)

    # Load metadata
    metadata_path = submission_dir / "metadata.json"
    if not metadata_path.exists():
        print(f"ERROR: metadata.json not found in {submission_dir}", file=sys.stderr)
        sys.exit(1)
    with open(metadata_path) as f:
        metadata = json.load(f)
    model_name = metadata.get("model_name") or submission_dir.name

    print(f"\n=== Evaluating: {model_name} ===")

    results = {}
    for split in ["manual", "automatic"]:
        pred_path = submission_dir / f"predictions_{split}.json"
        if not pred_path.exists():
            print(f"  [{split}] No predictions file found — skipping")
            continue

        try:
            gt = load_ground_truth(annotated_dir, split)
            preds = load_predictions(pred_path)
            result = evaluate_split(preds, gt, k=args.k)
            results[split] = result

            print(f"  [{split}]  nDCG@{args.k} = {result['ndcg5']:.4f}  "
                  f"({result['num_scored']}/{result['num_examples']} examples scored, "
                  f"{result['num_missing']} missing)")

            if result["num_missing"] > 0:
                print(f"  WARNING: {result['num_missing']} ground truth files have no prediction.")

        except Exception as e:
            print(f"  ERROR evaluating {split}: {e}", file=sys.stderr)
            sys.exit(1)

    if not results:
        print("No splits could be evaluated.", file=sys.stderr)
        sys.exit(1)

    update_leaderboard(leaderboard_path, model_name, metadata, results)

    # Print summary for GitHub Actions PR comment
    print("\n--- SUMMARY ---")
    print(f"Model: **{model_name}**")
    for split, res in results.items():
        print(f"  {split}: nDCG@{args.k} = **{res['ndcg5']:.4f}**")


if __name__ == "__main__":
    main()
