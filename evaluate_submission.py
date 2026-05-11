"""
evaluate_submission.py

Evaluates a submission against ground truth built from raw MP-Bench annotations.
Consolidates multi-annotator labels via majority vote on-the-fly.
Computes nDCG@5 (Exp) per split and updates leaderboard.json.

Usage:
    python evaluate_submission.py \
        --submission_dir submissions/<model_name> \
        --raw_bench_dir  multi-agent-eval-bench/MP-Bench \
        --leaderboard    leaderboard.json
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# nDCG@5 with exponential gain
# ---------------------------------------------------------------------------

def dcg_at_k(ranked_labels, k=5):
    score = 0.0
    for i, rel in enumerate(ranked_labels[:k]):
        score += (2 ** rel - 1) / math.log2(i + 2)
    return score


def ndcg_at_k(ranked_labels, all_labels, k=5):
    ideal = sorted(all_labels, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0.0:
        return None  # no positive labels -> skip
    return dcg_at_k(ranked_labels, k) / idcg


# ---------------------------------------------------------------------------
# Build ground truth via majority vote from raw MP-Bench annotations
#
# Raw layout (adobe-research/multi-agent-eval-bench MP-Bench/):
#   <annotator_id>/    e.g. 1, 2, 3
#     <split>/         manual | automatic
#       <file_id>.json
#
# Each file has "annotation" or "history" list of steps with "fail_annotation"
# ---------------------------------------------------------------------------

def build_ground_truth(raw_bench_dir, split):
    """Returns { file_id -> [per-step binary label] } via majority vote."""
    votes = defaultdict(lambda: defaultdict(list))

    annotator_dirs = sorted(
        [d for d in raw_bench_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name
    )
    if not annotator_dirs:
        raise FileNotFoundError(f"No annotator dirs found in {raw_bench_dir}")

    for ann_dir in annotator_dirs:
        split_dir = ann_dir / split
        if not split_dir.exists():
            continue
        for path in sorted(split_dir.glob("*.json")):
            file_id = path.stem
            with open(path) as f:
                data = json.load(f)
            steps = data.get("annotation") or data.get("history") or []
            for step in steps:
                try:
                    idx   = int(step.get("step", 0))
                    label = int(step.get("fail_annotation", 0))
                    votes[file_id][idx].append(label)
                except (ValueError, TypeError):
                    continue

    if not votes:
        raise ValueError(f"No annotation files found for split '{split}' in {raw_bench_dir}")

    gt = {}
    for file_id, step_votes in votes.items():
        n_steps = max(step_votes.keys()) + 1
        labels = []
        for i in range(n_steps):
            v = step_votes.get(i, [0])
            labels.append(1 if sum(v) * 2 >= len(v) else 0)  # tie -> 1
        gt[file_id] = labels
    return gt


# ---------------------------------------------------------------------------
# Load predictions
# ---------------------------------------------------------------------------

def load_predictions(pred_path):
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

def evaluate_split(preds, ground_truth, k=5):
    scores = {}
    missing = 0

    for file_id, gt_labels in ground_truth.items():
        if file_id not in preds:
            missing += 1
            scores[file_id] = None
            continue

        ranked_steps = [s for s in preds[file_id] if 0 <= s < len(gt_labels)]
        seen, ranked_labels = set(), []
        for s in ranked_steps:
            if s not in seen:
                ranked_labels.append(gt_labels[s])
                seen.add(s)

        scores[file_id] = ndcg_at_k(ranked_labels, gt_labels, k=k)

    scored = [v for v in scores.values() if v is not None]
    return {
        "ndcg5":        round(sum(scored) / len(scored), 4) if scored else 0.0,
        "num_examples": len(ground_truth),
        "num_scored":   len(scored),
        "num_missing":  missing,
        "per_file":     scores,
    }


# ---------------------------------------------------------------------------
# Update leaderboard.json
# ---------------------------------------------------------------------------

def update_leaderboard(leaderboard_path, model_name, metadata, results):
    lb = json.loads(leaderboard_path.read_text()) if leaderboard_path.exists() else []
    lb = [e for e in lb if e["model_name"] != model_name]
    lb.append({
        "model_name":          model_name,
        "organization":        metadata.get("organization", ""),
        "paper_url":           metadata.get("paper_url", ""),
        "code_url":            metadata.get("code_url", ""),
        "description":         metadata.get("description", ""),
        "manual_ndcg5":        results.get("manual",    {}).get("ndcg5"),
        "automatic_ndcg5":     results.get("automatic", {}).get("ndcg5"),
        "manual_num_scored":   results.get("manual",    {}).get("num_scored"),
        "automatic_num_scored":results.get("automatic", {}).get("num_scored"),
        "submitted_at":        datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    })
    lb.sort(key=lambda e: e.get("manual_ndcg5") or -1, reverse=True)
    leaderboard_path.write_text(json.dumps(lb, indent=2))
    print(f"Leaderboard updated: {leaderboard_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission_dir", required=True,
                        help="Path to submissions/<model_name>/")
    parser.add_argument("--raw_bench_dir",  required=True,
                        help="Path to MP-Bench/ folder from adobe-research/multi-agent-eval-bench")
    parser.add_argument("--leaderboard", default="leaderboard.json")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    submission_dir   = Path(args.submission_dir)
    raw_bench_dir    = Path(args.raw_bench_dir)
    leaderboard_path = Path(args.leaderboard)

    metadata_path = submission_dir / "metadata.json"
    if not metadata_path.exists():
        print(f"ERROR: metadata.json not found in {submission_dir}", file=sys.stderr)
        sys.exit(1)
    metadata   = json.loads(metadata_path.read_text())
    model_name = metadata.get("model_name") or submission_dir.name

    print(f"\n=== Evaluating: {model_name} ===")

    results = {}
    for split in ["manual", "automatic"]:
        pred_path = submission_dir / f"predictions_{split}.json"
        if not pred_path.exists():
            print(f"  [{split}] No predictions file — skipping")
            continue
        try:
            gt    = build_ground_truth(raw_bench_dir, split)
            preds = load_predictions(pred_path)
            res   = evaluate_split(preds, gt, k=args.k)
            results[split] = res
            print(f"  [{split}]  nDCG@{args.k} = {res['ndcg5']:.4f}  "
                  f"({res['num_scored']}/{res['num_examples']} examples, "
                  f"{res['num_missing']} missing)")
            if res["num_missing"] > 0:
                print(f"  WARNING: {res['num_missing']} files have no prediction.")
        except Exception as e:
            print(f"  ERROR [{split}]: {e}", file=sys.stderr)
            sys.exit(1)

    if not results:
        print("No splits evaluated.", file=sys.stderr)
        sys.exit(1)

    update_leaderboard(leaderboard_path, model_name, metadata, results)

    print("\n--- SUMMARY ---")
    print(f"Model: **{model_name}**")
    for split, res in results.items():
        print(f"  {split}: nDCG@{args.k} = **{res['ndcg5']:.4f}**")


if __name__ == "__main__":
    main()