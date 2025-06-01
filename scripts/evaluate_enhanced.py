"""Enhanced evaluation script for LayoutLM document understanding.

Provides comprehensive token-level and page-level evaluation with visualizations.
Expected CSV columns: image_id, block_ids, word_ids, words, bboxes, pred_label, prob
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

sys.path.append("scripts")

from yaml_config_manager import load_config


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("evaluation.log"),
        ],
    )


def create_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    output_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """Create and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_class_performance_plot(
    classification_report_dict: Dict,
    labels: List[str],
    output_path: Path,
    title: str = "Per-Class Performance",
) -> None:
    """Create per-class performance bar chart."""
    # Extract per-class metrics
    classes = [label for label in labels if label in classification_report_dict]
    precision_scores = [classification_report_dict[cls]["precision"] for cls in classes]
    recall_scores = [classification_report_dict[cls]["recall"] for cls in classes]
    f1_scores = [classification_report_dict[cls]["f1-score"] for cls in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision_scores, width, label="Precision", alpha=0.8)
    ax.bar(x, recall_scores, width, label="Recall", alpha=0.8)
    ax.bar(x + width, f1_scores, width, label="F1-Score", alpha=0.8)

    ax.set_xlabel("Classes")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_confidence_distribution_plot(
    confidences: List[float],
    predictions: List[str],
    output_path: Path,
    title: str = "Confidence Distribution",
) -> None:
    """Create confidence distribution plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Overall confidence histogram
    ax1.hist(confidences, bins=30, alpha=0.7, edgecolor="black")
    ax1.set_xlabel("Confidence Score")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Overall Confidence Distribution")
    ax1.grid(True, alpha=0.3)

    # Confidence by class boxplot
    df_conf = pd.DataFrame({"confidence": confidences, "prediction": predictions})
    unique_classes = sorted(df_conf["prediction"].unique())

    confidence_by_class = [
        df_conf[df_conf["prediction"] == cls]["confidence"].to_numpy()
        for cls in unique_classes
    ]

    ax2.boxplot(confidence_by_class, labels=unique_classes)
    ax2.set_xlabel("Predicted Class")
    ax2.set_ylabel("Confidence Score")
    ax2.set_title("Confidence Distribution by Class")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def analyze_class_imbalance(
    labels: List[str], predictions: List[str], output_path: Path
) -> Dict:
    """Analyze and visualize class distribution and imbalance."""
    # Count true labels and predictions
    true_counts = pd.Series(labels).value_counts()
    pred_counts = pd.Series(predictions).value_counts()

    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # True label distribution
    true_counts.plot(kind="bar", ax=ax1, alpha=0.7)
    ax1.set_title("True Label Distribution")
    ax1.set_xlabel("Class")
    ax1.set_ylabel("Count")
    ax1.tick_params(axis="x", rotation=45)

    # Predicted label distribution
    pred_counts.plot(kind="bar", ax=ax2, alpha=0.7, color="orange")
    ax2.set_title("Predicted Label Distribution")
    ax2.set_xlabel("Class")
    ax2.set_ylabel("Count")
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Calculate imbalance metrics
    total_samples = len(labels)
    class_proportions = true_counts / total_samples
    imbalance_ratio = (
        true_counts.max() / true_counts.min() if true_counts.min() > 0 else float("inf")
    )

    return {
        "class_counts": true_counts.to_dict(),
        "class_proportions": class_proportions.to_dict(),
        "imbalance_ratio": imbalance_ratio,
        "total_samples": total_samples,
    }


def create_page_performance_plot(
    page_details: List[Dict], output_path: Path, title: str = "Page-Level Performance"
) -> None:
    """Create page performance visualization."""
    df_pages = pd.DataFrame(page_details)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Page accuracy distribution
    ax1.hist(df_pages["accuracy"], bins=20, alpha=0.7, edgecolor="black")
    ax1.set_xlabel("Page Accuracy")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Page Accuracy Distribution")
    ax1.grid(True, alpha=0.3)

    # Page F1 score distribution
    ax2.hist(
        df_pages["f1_macro"], bins=20, alpha=0.7, edgecolor="black", color="orange"
    )
    ax2.set_xlabel("Page F1 Score")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Page F1 Score Distribution")
    ax2.grid(True, alpha=0.3)

    # Tokens per page distribution
    ax3.hist(
        df_pages["num_tokens"], bins=20, alpha=0.7, edgecolor="black", color="green"
    )
    ax3.set_xlabel("Number of Tokens")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Tokens per Page Distribution")
    ax3.grid(True, alpha=0.3)

    # Average confidence per page
    ax4.hist(
        df_pages["avg_confidence"],
        bins=20,
        alpha=0.7,
        edgecolor="black",
        color="purple",
    )
    ax4.set_xlabel("Average Confidence")
    ax4.set_ylabel("Frequency")
    ax4.set_title("Average Confidence per Page")
    ax4.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def load_prediction_files(predictions_dir: Path) -> List[pd.DataFrame]:
    """Load all CSV prediction files from directory."""
    csv_files = list(predictions_dir.glob("*_predictions.csv"))
    if not csv_files:
        # Fallback to any CSV files
        csv_files = list(predictions_dir.glob("*.csv"))
        # Exclude aggregated results file
        csv_files = [f for f in csv_files if "aggregated" not in f.name.lower()]

    if not csv_files:
        raise ValueError(f"No prediction CSV files found in {predictions_dir}")

    dataframes = []
    for csv_file in sorted(csv_files):
        try:
            page_df = pd.read_csv(csv_file)

            # Validate required columns for new format (as documented in header)
            required_cols = [
                "image_id",
                "block_ids",
                "word_ids",
                "words",
                "bboxes",
                "pred_label",
                "prob",
            ]
            missing_cols = [col for col in required_cols if col not in page_df.columns]
            if missing_cols:
                print(f"Warning: Missing columns in {csv_file}: {missing_cols}")
                print(f"Available columns: {list(page_df.columns)}")
                continue

            # Add image_file column for tracking
            page_df["image_file"] = csv_file.stem
            dataframes.append(page_df)

        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue

    if not dataframes:
        raise ValueError(
            f"No valid prediction files could be loaded from {predictions_dir}"
        )

    return dataframes


def load_ground_truth_files(ground_truth_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load ground truth CSV files from directory."""
    csv_files = list(ground_truth_dir.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No ground truth CSV files found in {ground_truth_dir}")

    ground_truth_data = {}
    for csv_file in sorted(csv_files):
        try:
            gt_df = pd.read_csv(csv_file)

            # Expected ground truth format should match prediction format
            # but with 'true_label' instead of 'pred_label'
            required_cols = ["image_id", "block_ids", "word_ids", "words", "bboxes"]
            if "true_label" in gt_df.columns:
                required_cols.append("true_label")
            elif "label" in gt_df.columns:
                # Rename to match expected format
                gt_df["true_label"] = gt_df["label"]
                required_cols.append("true_label")
            else:
                print(f"Warning: No ground truth labels found in {csv_file}")
                continue

            missing_cols = [col for col in required_cols if col not in gt_df.columns]
            if missing_cols:
                print(f"Warning: Missing columns in {csv_file}: {missing_cols}")
                continue

            # Use image_id or filename as key
            if "image_id" in gt_df.columns and len(gt_df["image_id"].unique()) == 1:
                key = gt_df["image_id"].iloc[0]
            else:
                key = csv_file.stem

            ground_truth_data[key] = gt_df

        except Exception as e:
            print(f"Error loading ground truth file {csv_file}: {e}")
            continue

    return ground_truth_data


def match_predictions_with_ground_truth(
    pred_df: pd.DataFrame, gt_data: Dict[str, pd.DataFrame]
) -> Tuple[List[str], List[str]]:
    """Match predictions with ground truth labels."""
    predictions = []
    ground_truth = []

    image_id = (
        pred_df["image_id"].iloc[0]
        if "image_id" in pred_df.columns
        else pred_df["image_file"].iloc[0]
    )

    # Try to find matching ground truth
    gt_df = None
    for key, gt_data_df in gt_data.items():
        if key == image_id or key in image_id or image_id in key:
            gt_df = gt_data_df
            break

    if gt_df is None:
        print(f"Warning: No ground truth found for {image_id}")
        return [], []

    # Match by word position or content
    for _idx, pred_row in pred_df.iterrows():
        pred_word = pred_row["words"]
        pred_bbox = pred_row["bboxes"]
        pred_label = pred_row["pred_label"]

        # Try to find matching word in ground truth
        matched = False
        for _, gt_row in gt_df.iterrows():
            if gt_row["words"] == pred_word and gt_row["bboxes"] == pred_bbox:
                predictions.append(pred_label)
                ground_truth.append(gt_row["true_label"])
                matched = True
                break

        if not matched:
            # If exact match not found, try word-only match
            word_matches = gt_df[gt_df["words"] == pred_word]
            if len(word_matches) > 0:
                predictions.append(pred_label)
                ground_truth.append(word_matches.iloc[0]["true_label"])
            else:
                print(f"Warning: Could not match word '{pred_word}' in ground truth")

    return predictions, ground_truth


def compute_token_metrics(all_predictions: np.ndarray, all_labels: np.ndarray) -> Dict:
    """Compute comprehensive token-level evaluation metrics."""
    accuracy = accuracy_score(all_labels, all_predictions)
    f1_macro = f1_score(all_labels, all_predictions, average="macro", zero_division=0)
    f1_micro = f1_score(all_labels, all_predictions, average="micro", zero_division=0)
    f1_weighted = f1_score(
        all_labels, all_predictions, average="weighted", zero_division=0
    )

    # Precision, recall, f1 per class
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, zero_division=0
    )

    # Classification report for per-class metrics
    class_report = classification_report(
        all_labels, all_predictions, output_dict=True, zero_division=0
    )

    return {
        "token_accuracy": accuracy,
        "token_f1_macro": f1_macro,
        "token_f1_micro": f1_micro,
        "token_f1_weighted": f1_weighted,
        "token_precision_macro": np.mean(precision),
        "token_recall_macro": np.mean(recall),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_support": support.tolist(),
        "classification_report": class_report,
        "confusion_matrix": confusion_matrix(all_labels, all_predictions).tolist(),
    }


def compute_page_metrics(
    page_dataframes: List[pd.DataFrame],
    gt_data: Optional[Dict[str, pd.DataFrame]] = None,
) -> Tuple[Dict, List[Dict]]:
    """Compute comprehensive page-level evaluation metrics."""
    page_accuracies = []
    page_f1_scores = []
    page_confidences = []
    perfect_accuracy_pages = 0
    evaluated_pages = 0
    page_details = []

    for i, page_df in enumerate(page_dataframes):
        if len(page_df) == 0:
            continue

        page_detail = {
            "page_id": i,
            "num_tokens": len(page_df),
            "image_id": page_df["image_id"].iloc[0]
            if "image_id" in page_df.columns
            else f"page_{i}",
        }

        if gt_data is not None:
            # Compare with ground truth
            predictions, ground_truth = match_predictions_with_ground_truth(
                page_df, gt_data
            )

            if len(predictions) > 0 and len(ground_truth) > 0:
                page_acc = accuracy_score(ground_truth, predictions)
                page_f1 = f1_score(
                    ground_truth, predictions, average="macro", zero_division=0
                )

                page_accuracies.append(page_acc)
                page_f1_scores.append(page_f1)
                evaluated_pages += 1

                if page_acc == 1.0:
                    perfect_accuracy_pages += 1

                page_detail.update(
                    {
                        "accuracy": page_acc,
                        "f1_macro": page_f1,
                        "has_ground_truth": True,
                    }
                )
        else:
            # No ground truth available - use confidence-based metrics
            high_conf_threshold = 0.8
            if "prob" in page_df.columns:
                high_conf_predictions = page_df[page_df["prob"] >= high_conf_threshold]
                confidence_accuracy = len(high_conf_predictions) / len(page_df)
                page_accuracies.append(confidence_accuracy)

                avg_confidence = page_df["prob"].mean()
                page_confidences.append(avg_confidence)
                page_f1_scores.append(avg_confidence)  # Use avg confidence as proxy
                evaluated_pages += 1

                page_detail.update(
                    {
                        "accuracy": confidence_accuracy,
                        "f1_macro": avg_confidence,
                        "has_ground_truth": False,
                    }
                )

        # Add confidence information if available
        if "prob" in page_df.columns:
            page_detail["avg_confidence"] = page_df["prob"].mean()
            page_detail["min_confidence"] = page_df["prob"].min()
            page_detail["max_confidence"] = page_df["prob"].max()
        else:
            page_detail["avg_confidence"] = 0.0
            page_detail["min_confidence"] = 0.0
            page_detail["max_confidence"] = 0.0

        page_details.append(page_detail)

    metrics = {
        "page_accuracy_mean": np.mean(page_accuracies) if page_accuracies else 0,
        "page_accuracy_std": np.std(page_accuracies) if page_accuracies else 0,
        "page_f1_mean": np.mean(page_f1_scores) if page_f1_scores else 0,
        "page_f1_std": np.std(page_f1_scores) if page_f1_scores else 0,
        "pages_perfect_accuracy": perfect_accuracy_pages,
        "total_pages": evaluated_pages,
        "has_ground_truth": gt_data is not None,
        "avg_confidence_mean": np.mean(page_confidences) if page_confidences else 0,
        "avg_confidence_std": np.std(page_confidences) if page_confidences else 0,
    }

    return metrics, page_details


def generate_comprehensive_report(
    final_metrics: Dict,
    page_details: List[Dict],
    label_mapping: Dict,
    output_dir: Path,
    create_visualizations: bool = True,
) -> None:
    """Generate comprehensive evaluation report with visualizations."""
    if not create_visualizations:
        return

    # Create visualizations directory
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Get label names for plots
    label_names = list(label_mapping.values()) if label_mapping else []

    print("Generating evaluation visualizations...")

    try:
        # 1. Page performance plots
        if page_details:
            create_page_performance_plot(page_details, viz_dir / "page_performance.png")
            print("✅ Page performance plot created")

        # 2. Token-level metrics available
        if "classification_report" in final_metrics:
            class_report = final_metrics["classification_report"]

            # Per-class performance plot
            if label_names and class_report:
                create_class_performance_plot(
                    class_report, label_names, viz_dir / "class_performance.png"
                )
                print("✅ Class performance plot created")

        # 3. Summary metrics plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Token metrics
        token_metrics = [
            final_metrics.get("token_accuracy", 0),
            final_metrics.get("token_f1_macro", 0),
            final_metrics.get("token_f1_micro", 0),
            final_metrics.get("token_f1_weighted", 0),
        ]
        ax1.bar(["Accuracy", "F1-Macro", "F1-Micro", "F1-Weighted"], token_metrics)
        ax1.set_title("Token-Level Metrics")
        ax1.set_ylabel("Score")
        ax1.set_ylim(0, 1)

        # Page metrics
        page_metrics = [
            final_metrics.get("page_accuracy_mean", 0),
            final_metrics.get("page_f1_mean", 0),
        ]
        ax2.bar(["Accuracy", "F1-Score"], page_metrics)
        ax2.set_title("Page-Level Metrics (Mean)")
        ax2.set_ylabel("Score")
        ax2.set_ylim(0, 1)

        # Dataset statistics
        stats = [
            final_metrics.get("total_tokens", 0),
            final_metrics.get("total_pages", 0),
            final_metrics.get("pages_perfect_accuracy", 0),
        ]
        ax3.bar(["Total Tokens", "Total Pages", "Perfect Pages"], stats)
        ax3.set_title("Dataset Statistics")
        ax3.set_ylabel("Count")

        # Average tokens per page
        avg_tokens = final_metrics.get("avg_tokens_per_page", 0)
        ax4.bar(["Avg Tokens/Page"], [avg_tokens])
        ax4.set_title("Average Tokens per Page")
        ax4.set_ylabel("Count")

        plt.suptitle("LayoutLM Evaluation Summary")
        plt.tight_layout()
        plt.savefig(viz_dir / "evaluation_summary.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("✅ Evaluation summary plot created")

    except Exception as e:
        print(f"⚠️ Error creating visualizations: {e}")


def run_enhanced_evaluation(
    predictions_dir: str,
    ground_truth_dir: str = None,
    output_dir: str = "./evaluation_results",
    config_path: str = "config/config.yaml",
    create_visualizations: bool = False,
    save_detailed_results: bool = False,
    log_level: str = "INFO",
) -> Dict[str, Any]:
    """
    Run enhanced evaluation programmatically.

    Args:
        predictions_dir: Directory containing CSV prediction files
        ground_truth_dir: Directory containing ground truth CSV files
        output_dir: Directory to save evaluation results
        config_path: Path to YAML configuration file
        create_visualizations: Whether to create visualizations
        save_detailed_results: Whether to save detailed results
        log_level: Logging level

    Returns:
        Dictionary containing evaluation results
    """
    # Setup logging
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    try:
        # Convert paths to Path objects
        predictions_path = Path(predictions_dir)
        output_path = Path(output_dir)

        # Clear existing evaluation output directory to ensure clean results
        if output_path.exists():
            import shutil

            shutil.rmtree(output_path)
            logger.info(f"🗑️  Cleared existing evaluation directory: {output_path}")

        output_path.mkdir(parents=True, exist_ok=True)

        # Load configuration for label mapping
        try:
            from yaml_config_manager import load_config

            config = load_config(config_path)
            label_mapping = config.get_label_mapping()
            logger.info(f"Loaded label mapping from config: {label_mapping}")
        except Exception as e:
            logger.warning(f"Could not load config: {e}. Using default label mapping.")
            label_mapping = {
                0: "O",
                1: "B-HEADER",
                2: "I-HEADER",
                3: "B-QUESTION",
                4: "I-QUESTION",
                5: "B-ANSWER",
                6: "I-ANSWER",
            }

        # Load prediction files
        logger.info(f"Loading prediction files from {predictions_path}")
        prediction_files = load_prediction_files(predictions_path)
        logger.info(f"Loaded {len(prediction_files)} prediction files")

        # Load ground truth files if provided
        ground_truth_data = {}
        if ground_truth_dir:
            ground_truth_path = Path(ground_truth_dir)
            logger.info(f"Loading ground truth files from {ground_truth_path}")
            ground_truth_data = load_ground_truth_files(ground_truth_path)
            logger.info(f"Loaded {len(ground_truth_data)} ground truth files")

        # Combine all predictions for analysis
        all_predictions = []
        for df in prediction_files:
            all_predictions.append(df)

        if all_predictions:
            combined_predictions = pd.concat(all_predictions, ignore_index=True)
            logger.info(f"Total tokens across all pages: {len(combined_predictions)}")
        else:
            logger.error("No prediction data found")
            return {}

        # Perform evaluation
        results = {}

        # Compute token-level metrics
        logger.info("Computing token-level metrics...")
        if ground_truth_data:
            # Match predictions with ground truth
            predictions_list, ground_truth_list = match_predictions_with_ground_truth(
                combined_predictions, ground_truth_data
            )

            if len(predictions_list) > 0 and len(ground_truth_list) > 0:
                token_metrics = compute_token_metrics(
                    np.array(predictions_list), np.array(ground_truth_list)
                )
                results.update(token_metrics)

                # Compute page-level metrics with DataFrame for page analysis
                logger.info("Computing page-level metrics...")
                page_metrics, page_details = compute_page_metrics(
                    prediction_files, ground_truth_data
                )
                results.update(page_metrics)
            else:
                logger.warning("No matched data found for ground truth comparison")
        else:
            # No ground truth - compute confidence-based metrics
            logger.info(
                "Computing confidence-based metrics (no ground truth available)"
            )
            page_metrics, page_details = compute_page_metrics(prediction_files, None)
            results.update(page_metrics)

            # Add basic statistics
            results.update(
                {
                    "total_tokens": len(combined_predictions),
                    "total_pages": len(prediction_files),
                    "avg_tokens_per_page": len(combined_predictions)
                    / len(prediction_files)
                    if prediction_files
                    else 0,
                    "has_ground_truth": False,
                }
            )

        # Generate comprehensive visualizations
        if create_visualizations:
            logger.info("Generating comprehensive visualizations...")
            viz_dir = output_path / "visualizations"
            viz_dir.mkdir(exist_ok=True)

            # Create various plots
            if "prob" in combined_predictions.columns:
                create_confidence_distribution_plot(
                    combined_predictions["prob"].tolist(),
                    combined_predictions["pred_label"].tolist(),
                    viz_dir / "confidence_distribution.png",
                )

            if (
                ground_truth_data
                and len(predictions_list) > 0
                and len(ground_truth_list) > 0
            ):
                # Create confusion matrix using predictions and ground truth lists
                label_names = list(label_mapping.values()) if label_mapping else []
                if label_names:
                    create_confusion_matrix_plot(
                        np.array(ground_truth_list),
                        np.array(predictions_list),
                        label_names,
                        viz_dir / "confusion_matrix.png",
                    )

                # Create class performance plot using classification report
                if "classification_report" in results:
                    create_class_performance_plot(
                        results["classification_report"],
                        label_names,
                        viz_dir / "class_performance.png",
                    )

                # Page performance plot using page details
                if "page_details" in locals():
                    create_page_performance_plot(
                        page_details, viz_dir / "page_performance.png"
                    )

            logger.info("✅ Page performance plot created")
            logger.info("✅ Class performance plot created")
            logger.info("✅ Evaluation summary plot created")

        # Save detailed results
        if save_detailed_results:
            logger.info("Saving detailed results...")

            # Save combined predictions
            combined_predictions.to_csv(
                output_path / "all_predictions.csv", index=False
            )

            if (
                ground_truth_data
                and len(predictions_list) > 0
                and len(ground_truth_list) > 0
            ):
                # Create and save matched data DataFrame
                matched_df = pd.DataFrame(
                    {"predicted": predictions_list, "ground_truth": ground_truth_list}
                )
                matched_df.to_csv(
                    output_path / "matched_predictions_gt.csv", index=False
                )

                # Save detailed metrics
                detailed_metrics = []
                for label_id, label_name in label_mapping.items():
                    if label_id == 0:  # Skip 'O' label
                        continue

                    # Count occurrences of this label in ground truth
                    total = ground_truth_list.count(label_name)
                    if total > 0:
                        # Count correct predictions for this label
                        correct = sum(
                            1
                            for pred, true in zip(
                                predictions_list, ground_truth_list, strict=False
                            )
                            if true == label_name and pred == label_name
                        )
                        accuracy = correct / total

                        detailed_metrics.append(
                            {
                                "label_id": label_id,
                                "label_name": label_name,
                                "total_tokens": total,
                                "correct_predictions": correct,
                                "accuracy": accuracy,
                            }
                        )

                if detailed_metrics:
                    detailed_df = pd.DataFrame(detailed_metrics)
                    detailed_df.to_csv(
                        output_path / "detailed_metrics.csv", index=False
                    )

        # Generate final report
        logger.info("Generating comprehensive report...")
        report_page_details = page_details if "page_details" in locals() else []
        generate_comprehensive_report(
            results,
            report_page_details,
            label_mapping,
            output_path,
            create_visualizations,
        )

        # Save evaluation summary
        with (output_path / "evaluation_summary.json").open("w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info("Detailed results saved")
        logger.info("Enhanced evaluation completed successfully!")
        logger.info(f"Results saved to: {output_path}")

        # Log key metrics
        if "token_accuracy" in results:
            logger.info(f"Token accuracy: {results['token_accuracy']:.4f}")
        if "token_f1_macro" in results:
            logger.info(f"Token F1 (macro): {results['token_f1_macro']:.4f}")
        if "page_accuracy_mean" in results:
            logger.info(f"Page accuracy (mean): {results['page_accuracy_mean']:.4f}")

        if create_visualizations:
            logger.info(f"📊 Visualizations saved to: {viz_dir}")

        return results

    except Exception as e:
        logger.error(f"Enhanced evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        return {}


def main() -> None:
    """Enhanced main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Enhanced LayoutLM evaluation with comprehensive analysis"
    )

    # Data arguments
    parser.add_argument(
        "--predictions_dir",
        type=str,
        required=True,
        help="Directory containing CSV prediction files (one per image)",
    )
    parser.add_argument(
        "--ground_truth_dir",
        type=str,
        default=None,
        help="Directory containing ground truth CSV files (optional)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to YAML configuration file",
    )

    # Analysis options
    parser.add_argument(
        "--create_visualizations",
        action="store_true",
        help="Create comprehensive visualizations",
    )
    parser.add_argument(
        "--save_detailed_results",
        action="store_true",
        help="Save detailed per-class and per-page results",
    )

    # Other arguments
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    try:
        config_manager = load_config(args.config)
        label_mapping = config_manager.get_label_mapping()
        logger.info(f"Loaded label mapping from config: {label_mapping}")
    except Exception as e:
        logger.warning(f"Could not load config: {e}")
        label_mapping = {}

    # Load prediction files
    logger.info(f"Loading prediction files from {args.predictions_dir}")
    predictions_dir = Path(args.predictions_dir)
    page_dataframes = load_prediction_files(predictions_dir)
    logger.info(f"Loaded {len(page_dataframes)} prediction files")

    # Load ground truth files if provided
    gt_data = None
    if args.ground_truth_dir:
        logger.info(f"Loading ground truth files from {args.ground_truth_dir}")
        ground_truth_dir = Path(args.ground_truth_dir)
        gt_data = load_ground_truth_files(ground_truth_dir)
        logger.info(f"Loaded {len(gt_data)} ground truth files")

    # Combine all predictions and labels for token-level metrics
    all_predictions = []
    all_labels = []
    all_confidences = []
    total_tokens = 0

    if gt_data is not None:
        # Compare with ground truth
        for page_df in page_dataframes:
            if len(page_df) > 0:
                predictions, ground_truth = match_predictions_with_ground_truth(
                    page_df, gt_data
                )
                all_predictions.extend(predictions)
                all_labels.extend(ground_truth)
                total_tokens += len(predictions)

                # Collect confidences if available
                if "prob" in page_df.columns:
                    # Match confidences to predictions
                    confidences = page_df["prob"].tolist()[: len(predictions)]
                    all_confidences.extend(confidences)
    else:
        # No ground truth - use predictions and confidence scores
        logger.warning("No ground truth provided - using confidence-based evaluation")
        for page_df in page_dataframes:
            if len(page_df) > 0:
                predictions = page_df["pred_label"].tolist()
                # Create synthetic labels based on confidence
                synthetic_labels = [
                    pred if prob > 0.8 else "O"
                    for pred, prob in zip(
                        page_df["pred_label"], page_df["prob"], strict=False
                    )
                ]
                all_predictions.extend(predictions)
                all_labels.extend(synthetic_labels)
                all_confidences.extend(page_df["prob"].tolist())
                total_tokens += len(predictions)

    logger.info(f"Total tokens across all pages: {total_tokens}")

    # Compute token-level metrics
    logger.info("Computing token-level metrics...")
    token_metrics = compute_token_metrics(
        np.array(all_predictions), np.array(all_labels)
    )

    # Compute page-level metrics
    logger.info("Computing page-level metrics...")
    page_metrics, page_details = compute_page_metrics(page_dataframes, gt_data)

    # Combine all metrics
    final_metrics = {
        **token_metrics,
        **page_metrics,
        "total_tokens": total_tokens,
        "avg_tokens_per_page": (
            total_tokens / page_metrics["total_pages"]
            if page_metrics["total_pages"] > 0
            else 0
        ),
    }

    # Generate comprehensive report and visualizations
    if args.create_visualizations:
        logger.info("Generating comprehensive visualizations...")
        generate_comprehensive_report(
            final_metrics, page_details, label_mapping, output_dir, True
        )

        # Create additional analysis plots if we have ground truth
        if gt_data is not None and all_confidences:
            viz_dir = output_dir / "visualizations"

            # Confidence distribution analysis
            create_confidence_distribution_plot(
                all_confidences,
                all_predictions,
                viz_dir / "confidence_distribution.png",
            )

            # Class imbalance analysis
            imbalance_analysis = analyze_class_imbalance(
                all_labels, all_predictions, viz_dir / "class_distribution.png"
            )

            # Save imbalance analysis
            with (output_dir / "class_imbalance_analysis.json").open("w") as f:
                json.dump(imbalance_analysis, f, indent=2)

            # Confusion matrix
            if label_mapping:
                label_names = list(label_mapping.values())
                create_confusion_matrix_plot(
                    np.array(all_labels),
                    np.array(all_predictions),
                    label_names,
                    viz_dir / "confusion_matrix.png",
                )

    # Save detailed results
    if args.save_detailed_results:
        # Save classification report
        if "classification_report" in token_metrics:
            class_report_df = pd.DataFrame(
                token_metrics["classification_report"]
            ).transpose()
            class_report_df.to_csv(output_dir / "classification_report.csv")

        # Save confusion matrix
        confusion_df = pd.DataFrame(token_metrics["confusion_matrix"])
        confusion_df.to_csv(output_dir / "confusion_matrix.csv", index=False)

        # Save per-page results
        page_results_df = pd.DataFrame(page_details)
        page_results_df.to_csv(output_dir / "per_page_results.csv", index=False)

        logger.info("Detailed results saved")

    # Save summary metrics
    json_metrics = {
        k: v
        for k, v in final_metrics.items()
        if k not in ["classification_report", "confusion_matrix"]
    }

    summary_path = output_dir / "summary_metrics.json"
    with summary_path.open("w") as f:
        json.dump(json_metrics, f, indent=2)

    # Save configuration used
    config_path = output_dir / "eval_config.json"
    with config_path.open("w") as f:
        json.dump(vars(args), f, indent=2)

    # Print comprehensive summary
    print("\n" + "=" * 60)
    print("LAYOUTLM ENHANCED EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total tokens evaluated: {final_metrics.get('total_tokens', 0):,}")
    print(f"Total pages evaluated: {final_metrics.get('total_pages', 0)}")
    print(f"Average tokens per page: {final_metrics.get('avg_tokens_per_page', 0):.1f}")
    print(f"Ground truth available: {'Yes' if gt_data else 'No'}")
    print()
    print("TOKEN-LEVEL METRICS:")
    print(f"  Accuracy: {final_metrics.get('token_accuracy', 0):.4f}")
    print(f"  Precision (macro): {final_metrics.get('token_precision_macro', 0):.4f}")
    print(f"  Recall (macro): {final_metrics.get('token_recall_macro', 0):.4f}")
    print(f"  F1 (macro): {final_metrics.get('token_f1_macro', 0):.4f}")
    print(f"  F1 (micro): {final_metrics.get('token_f1_micro', 0):.4f}")
    print(f"  F1 (weighted): {final_metrics.get('token_f1_weighted', 0):.4f}")
    print()
    print("PAGE-LEVEL METRICS:")
    print(
        f"  Accuracy (mean ± std): {final_metrics.get('page_accuracy_mean', 0):.4f} ± {final_metrics.get('page_accuracy_std', 0):.4f}"
    )
    print(
        f"  F1 (mean ± std): {final_metrics.get('page_f1_mean', 0):.4f} ± {final_metrics.get('page_f1_std', 0):.4f}"
    )
    print(f"  Perfect accuracy pages: {final_metrics.get('pages_perfect_accuracy', 0)}")

    if all_confidences:
        print()
        print("CONFIDENCE ANALYSIS:")
        print(f"  Mean confidence: {np.mean(all_confidences):.4f}")
        print(f"  Confidence std: {np.std(all_confidences):.4f}")
        print(
            f"  High confidence (>0.8): {np.mean(np.array(all_confidences) > 0.8):.2%}"
        )

    print("=" * 60)

    logger.info("Enhanced evaluation completed successfully!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Token accuracy: {final_metrics.get('token_accuracy', 0):.4f}")
    logger.info(f"Token F1 (macro): {final_metrics.get('token_f1_macro', 0):.4f}")
    logger.info(
        f"Page accuracy (mean): {final_metrics.get('page_accuracy_mean', 0):.4f}"
    )

    if args.create_visualizations:
        print(f"\n📊 Visualizations saved to: {output_dir / 'visualizations'}")


if __name__ == "__main__":
    main()
