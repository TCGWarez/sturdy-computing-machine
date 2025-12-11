"""
src/eval.py: Evaluation harness for card recognition system
Following PRD.md Task 12 specifications

Computes metrics:
- Precision@1: % of queries where top-1 match is correct
- Precision@5: % of queries where correct match is in top-5
- MRR (Mean Reciprocal Rank): Average of 1/rank for correct matches
- avg_time_per_image: Average processing time per image

Input:
--pred matches.csv: Predicted matches (scanned_path, match_card_id, confidence, candidates_json)
--gold data/fixtures/gold.csv: Ground truth (scanned_path, correct_card_id)

Output:
JSON metrics file with all computed metrics
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """Single prediction record"""
    scanned_path: str
    match_card_id: str
    confidence: float
    candidates: List[str]  # List of candidate card IDs in order
    processing_time: float = 0.0


@dataclass
class GroundTruthRecord:
    """Single ground truth record"""
    scanned_path: str
    correct_card_id: str


@dataclass
class EvaluationMetrics:
    """Evaluation metrics"""
    precision_at_1: float
    precision_at_5: float
    mean_reciprocal_rank: float
    avg_time_per_image: float
    total_queries: int
    correct_at_1: int
    correct_at_5: int

    # Per-confidence-threshold metrics
    high_confidence_accuracy: float = 0.0  # Accuracy for predictions with confidence >= 0.85
    high_confidence_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'precision_at_1': self.precision_at_1,
            'precision_at_5': self.precision_at_5,
            'mean_reciprocal_rank': self.mean_reciprocal_rank,
            'avg_time_per_image': self.avg_time_per_image,
            'total_queries': self.total_queries,
            'correct_at_1': self.correct_at_1,
            'correct_at_5': self.correct_at_5,
            'high_confidence_accuracy': self.high_confidence_accuracy,
            'high_confidence_count': self.high_confidence_count
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)


def load_predictions(pred_file: Path) -> Dict[str, PredictionRecord]:
    """
    Load predictions from CSV file

    Args:
        pred_file: Path to predictions CSV

    Returns:
        Dict mapping scanned_path to PredictionRecord
    """
    predictions = {}

    with open(pred_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            scanned_path = row['scanned_path']

            # Parse candidates from JSON if available
            candidates = []
            if 'candidates' in row and row['candidates']:
                try:
                    candidates_data = json.loads(row['candidates'])
                    # Extract card IDs from candidates list
                    if isinstance(candidates_data, list):
                        candidates = [
                            c.get('card_id', c) if isinstance(c, dict) else c
                            for c in candidates_data
                        ]
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse candidates JSON for {scanned_path}")

            # Create prediction record
            predictions[scanned_path] = PredictionRecord(
                scanned_path=scanned_path,
                match_card_id=row.get('match_card_id', row.get('matched_card_id', '')),
                confidence=float(row.get('confidence', 0.0)),
                candidates=candidates,
                processing_time=float(row.get('processing_time', 0.0))
            )

    logger.info(f"Loaded {len(predictions)} predictions from {pred_file}")
    return predictions


def load_ground_truth(gold_file: Path) -> Dict[str, GroundTruthRecord]:
    """
    Load ground truth from CSV file

    Args:
        gold_file: Path to ground truth CSV

    Returns:
        Dict mapping scanned_path to GroundTruthRecord
    """
    ground_truth = {}

    with open(gold_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            scanned_path = row['scanned_path']

            ground_truth[scanned_path] = GroundTruthRecord(
                scanned_path=scanned_path,
                correct_card_id=row['correct_card_id']
            )

    logger.info(f"Loaded {len(ground_truth)} ground truth records from {gold_file}")
    return ground_truth


def compute_metrics(
    predictions: Dict[str, PredictionRecord],
    ground_truth: Dict[str, GroundTruthRecord],
    high_confidence_threshold: float = 0.85
) -> EvaluationMetrics:
    """
    Compute evaluation metrics

    Args:
        predictions: Dict of predictions
        ground_truth: Dict of ground truth
        high_confidence_threshold: Threshold for high-confidence predictions

    Returns:
        EvaluationMetrics
    """
    # Find common scanned paths
    common_paths = set(predictions.keys()) & set(ground_truth.keys())

    if not common_paths:
        logger.error("No common paths between predictions and ground truth")
        return EvaluationMetrics(
            precision_at_1=0.0,
            precision_at_5=0.0,
            mean_reciprocal_rank=0.0,
            avg_time_per_image=0.0,
            total_queries=0,
            correct_at_1=0,
            correct_at_5=0
        )

    logger.info(f"Evaluating {len(common_paths)} queries")

    correct_at_1 = 0
    correct_at_5 = 0
    reciprocal_ranks = []
    processing_times = []

    high_confidence_correct = 0
    high_confidence_count = 0

    for scanned_path in common_paths:
        pred = predictions[scanned_path]
        gt = ground_truth[scanned_path]

        # Track processing time
        processing_times.append(pred.processing_time)

        # Check Precision@1
        if pred.match_card_id == gt.correct_card_id:
            correct_at_1 += 1

            # High confidence accuracy
            if pred.confidence >= high_confidence_threshold:
                high_confidence_correct += 1

        # Check Precision@5 and compute MRR
        if pred.candidates:
            # Ensure top match is in candidates
            if pred.match_card_id and pred.match_card_id not in pred.candidates:
                candidates_with_top = [pred.match_card_id] + pred.candidates[:4]
            else:
                candidates_with_top = pred.candidates[:5]

            # Check if correct card is in top 5
            if gt.correct_card_id in candidates_with_top:
                correct_at_5 += 1

                # Compute rank (1-indexed)
                rank = candidates_with_top.index(gt.correct_card_id) + 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                # Not found in top 5
                reciprocal_ranks.append(0.0)
        else:
            # No candidates - only check top match
            if pred.match_card_id == gt.correct_card_id:
                correct_at_5 += 1
                reciprocal_ranks.append(1.0)
            else:
                reciprocal_ranks.append(0.0)

        # Count high confidence predictions
        if pred.confidence >= high_confidence_threshold:
            high_confidence_count += 1

    # Compute metrics
    total_queries = len(common_paths)

    precision_at_1 = correct_at_1 / total_queries if total_queries > 0 else 0.0
    precision_at_5 = correct_at_5 / total_queries if total_queries > 0 else 0.0
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    avg_time = sum(processing_times) / len(processing_times) if processing_times else 0.0

    high_conf_acc = (high_confidence_correct / high_confidence_count
                     if high_confidence_count > 0 else 0.0)

    return EvaluationMetrics(
        precision_at_1=precision_at_1,
        precision_at_5=precision_at_5,
        mean_reciprocal_rank=mrr,
        avg_time_per_image=avg_time,
        total_queries=total_queries,
        correct_at_1=correct_at_1,
        correct_at_5=correct_at_5,
        high_confidence_accuracy=high_conf_acc,
        high_confidence_count=high_confidence_count
    )


def print_metrics(metrics: EvaluationMetrics):
    """
    Print metrics in formatted output

    Args:
        metrics: EvaluationMetrics to print
    """
    print("\n" + "="*80)
    print("EVALUATION METRICS")
    print("="*80)
    print(f"Total Queries: {metrics.total_queries}")
    print(f"\nPrecision@1: {metrics.precision_at_1:.4f} ({metrics.correct_at_1}/{metrics.total_queries})")
    print(f"Precision@5: {metrics.precision_at_5:.4f} ({metrics.correct_at_5}/{metrics.total_queries})")
    print(f"Mean Reciprocal Rank (MRR): {metrics.mean_reciprocal_rank:.4f}")
    print(f"\nAvg Time per Image: {metrics.avg_time_per_image:.3f}s")
    print(f"\nHigh Confidence Accuracy (≥0.85): {metrics.high_confidence_accuracy:.4f} "
          f"({metrics.high_confidence_count} predictions)")
    print("="*80)

    # Check against PRD acceptance criteria
    print("\nPRD Acceptance Criteria:")
    print(f"  Precision@1 ≥ 0.95: {'✓ PASS' if metrics.precision_at_1 >= 0.95 else '✗ FAIL'}")
    print(f"  Precision@5 ≥ 0.99: {'✓ PASS' if metrics.precision_at_5 >= 0.99 else '✗ FAIL'}")
    print(f"  Avg Time ≤ 1.0s:    {'✓ PASS' if metrics.avg_time_per_image <= 1.0 else '✗ FAIL'}")
    print()


def save_metrics(metrics: EvaluationMetrics, output_file: Path):
    """
    Save metrics to JSON file

    Args:
        metrics: EvaluationMetrics to save
        output_file: Path to output JSON file
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(metrics.to_json())

    logger.info(f"Saved metrics to {output_file}")


def generate_error_report(
    predictions: Dict[str, PredictionRecord],
    ground_truth: Dict[str, GroundTruthRecord],
    output_file: Path,
    top_n: int = 20
):
    """
    Generate error analysis report

    Args:
        predictions: Dict of predictions
        ground_truth: Dict of ground truth
        output_file: Path to output CSV file
        top_n: Number of top errors to report
    """
    common_paths = set(predictions.keys()) & set(ground_truth.keys())

    errors = []

    for scanned_path in common_paths:
        pred = predictions[scanned_path]
        gt = ground_truth[scanned_path]

        if pred.match_card_id != gt.correct_card_id:
            # Find rank of correct card in candidates
            rank = -1
            if gt.correct_card_id in pred.candidates:
                rank = pred.candidates.index(gt.correct_card_id) + 1

            errors.append({
                'scanned_path': scanned_path,
                'predicted_card_id': pred.match_card_id,
                'correct_card_id': gt.correct_card_id,
                'confidence': pred.confidence,
                'correct_rank': rank,
                'processing_time': pred.processing_time
            })

    # Sort by confidence (descending) - highest confidence errors first
    errors.sort(key=lambda e: e['confidence'], reverse=True)

    # Write to CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['scanned_path', 'predicted_card_id', 'correct_card_id',
                     'confidence', 'correct_rank', 'processing_time']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for error in errors[:top_n]:
            writer.writerow(error)

    logger.info(f"Saved error report ({len(errors)} errors, showing top {top_n}) to {output_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Evaluate MTG card recognition predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python src/eval.py --pred out/matches.csv --gold data/fixtures/gold.csv
  python src/eval.py --pred out/matches.csv --gold data/fixtures/gold.csv --output metrics.json
  python src/eval.py --pred out/matches.csv --gold data/fixtures/gold.csv --errors errors.csv
        """
    )

    parser.add_argument('--pred', required=True, type=Path,
                       help='Path to predictions CSV file')
    parser.add_argument('--gold', required=True, type=Path,
                       help='Path to ground truth CSV file')
    parser.add_argument('--output', type=Path, default=None,
                       help='Path to output metrics JSON file (default: metrics.json)')
    parser.add_argument('--errors', type=Path, default=None,
                       help='Path to output error report CSV file')
    parser.add_argument('--confidence-threshold', type=float, default=0.85,
                       help='High confidence threshold (default: 0.85)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate input files
    if not args.pred.exists():
        logger.error(f"Predictions file not found: {args.pred}")
        return 1

    if not args.gold.exists():
        logger.error(f"Ground truth file not found: {args.gold}")
        return 1

    # Load data
    try:
        predictions = load_predictions(args.pred)
        ground_truth = load_ground_truth(args.gold)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return 1

    # Compute metrics
    metrics = compute_metrics(
        predictions,
        ground_truth,
        high_confidence_threshold=args.confidence_threshold
    )

    # Print metrics
    print_metrics(metrics)

    # Save metrics to file
    output_file = args.output if args.output else Path('metrics.json')
    save_metrics(metrics, output_file)

    # Generate error report if requested
    if args.errors:
        generate_error_report(predictions, ground_truth, args.errors)

    # Return exit code based on PRD criteria
    if metrics.precision_at_1 >= 0.95 and metrics.precision_at_5 >= 0.99:
        logger.info("✓ PRD acceptance criteria met")
        return 0
    else:
        logger.warning("✗ PRD acceptance criteria not met")
        return 1


if __name__ == '__main__':
    exit(main())
