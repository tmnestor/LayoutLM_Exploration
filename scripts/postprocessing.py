#!/usr/bin/env python3
"""
Postprocessing module for LayoutLM outputs.
Converts model predictions to CSV format with required columns.
"""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


class LayoutLMPostprocessor:
    """Postprocesses LayoutLM model outputs to CSV format."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the postprocessor.

        Args:
            config: Configuration dictionary containing postprocessing settings
        """
        self.config = config
        self.label_mapping = config.get_label_mapping()

        # Use the configured output directory from environment settings
        self.output_dir = Path(config.get("environment.output_dir"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # CSV output settings - use configured csv_output_dir or fall back to subdirectory
        csv_output_config = config.get("postprocessing.csv_output_dir")
        if csv_output_config:
            self.csv_output_dir = Path(csv_output_config)
        else:
            self.csv_output_dir = self.output_dir / "csv_results"

        # Clear existing CSV output directory to ensure clean results
        if self.csv_output_dir.exists():
            import shutil

            shutil.rmtree(self.csv_output_dir)
            print(f"🗑️  Cleared existing CSV output directory: {self.csv_output_dir}")

        self.csv_output_dir.mkdir(parents=True, exist_ok=True)

    def process_predictions(
        self,
        image_path: str,
        words: List[str],
        bboxes: List[List[int]],
        word_ids: List[int],
        block_ids: List[int],
        predictions: List[int],
        probabilities: List[float],
        image_id: Optional[str] = None,
    ) -> str:
        """
        Process model predictions and save to CSV format.

        Args:
            image_path: Path to the input image
            words: List of extracted words
            bboxes: List of bounding boxes [x0, y0, x1, y1]
            word_ids: List of word identifiers
            block_ids: List of block identifiers
            predictions: List of predicted label indices
            probabilities: List of prediction probabilities
            image_id: Optional custom image identifier

        Returns:
            Path to the generated CSV file
        """
        # Generate image_id if not provided
        if image_id is None:
            image_id = Path(image_path).stem

        # Prepare data for CSV
        csv_data = []

        for _i, (word, bbox, word_id, block_id, pred_idx, prob) in enumerate(
            zip(
                words,
                bboxes,
                word_ids,
                block_ids,
                predictions,
                probabilities,
                strict=False,
            )
        ):
            # Convert prediction index to label
            pred_label = self.label_mapping.get(pred_idx, f"LABEL_{pred_idx}")

            # Format bounding box as "(ul_x, ul_y, lr_x, lr_y)"
            bbox_str = f"({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})"

            csv_row = {
                "image_id": image_id,
                "block_ids": block_id,
                "word_ids": word_id,
                "words": word,
                "bboxes": bbox_str,
                "pred_label": pred_label,
                "prob": round(prob, 4),
            }
            csv_data.append(csv_row)

        # Save to CSV file
        csv_filename = f"{image_id}_predictions.csv"
        csv_filepath = self.csv_output_dir / csv_filename

        with csv_filepath.open("w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "image_id",
                "block_ids",
                "word_ids",
                "words",
                "bboxes",
                "pred_label",
                "prob",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(csv_data)

        print(f"✅ CSV saved: {csv_filepath}")
        return str(csv_filepath)

    def process_batch_predictions(
        self, batch_results: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Process a batch of predictions and save multiple CSV files.

        Args:
            batch_results: List of dictionaries containing prediction results

        Returns:
            List of paths to generated CSV files
        """
        csv_files = []

        for result in batch_results:
            csv_file = self.process_predictions(
                image_path=result["image_path"],
                words=result["words"],
                bboxes=result["bboxes"],
                word_ids=result.get("word_ids", list(range(len(result["words"])))),
                block_ids=result.get("block_ids", [0] * len(result["words"])),
                predictions=result["predictions"],
                probabilities=result["probabilities"],
                image_id=result.get("image_id"),
            )
            csv_files.append(csv_file)

        return csv_files

    def aggregate_results(
        self, csv_files: List[str], output_filename: str = "aggregated_results.csv"
    ) -> str:
        """
        Aggregate multiple CSV files into a single file.

        Args:
            csv_files: List of CSV file paths to aggregate
            output_filename: Name of the aggregated output file

        Returns:
            Path to the aggregated CSV file
        """
        all_data = []

        for csv_file in csv_files:
            try:
                data_frame = pd.read_csv(csv_file)
                all_data.append(data_frame)
            except Exception as e:
                print(f"⚠️  Error reading {csv_file}: {e}")
                continue

        if not all_data:
            raise ValueError("No valid CSV files to aggregate")

        # Combine all dataframes
        aggregated_df = pd.concat(all_data, ignore_index=True)

        # Save aggregated results
        aggregated_path = self.csv_output_dir / output_filename
        aggregated_df.to_csv(aggregated_path, index=False)

        print(f"✅ Aggregated CSV saved: {aggregated_path}")
        print(f"📊 Total records: {len(aggregated_df)}")
        print(f"📁 Unique images: {aggregated_df['image_id'].nunique()}")

        return str(aggregated_path)

    def generate_summary_report(
        self, csv_files: List[str], output_filename: str = "processing_summary.txt"
    ) -> str:
        """
        Generate a summary report of the processing results.

        Args:
            csv_files: List of processed CSV files
            output_filename: Name of the summary report file

        Returns:
            Path to the summary report file
        """
        report_path = self.csv_output_dir / output_filename

        total_words = 0
        total_images = len(csv_files)
        label_counts = {}

        for csv_file in csv_files:
            try:
                data_frame = pd.read_csv(csv_file)
                total_words += len(data_frame)

                # Count labels
                for label in data_frame["pred_label"]:
                    label_counts[label] = label_counts.get(label, 0) + 1

            except Exception as e:
                print(f"⚠️  Error processing {csv_file}: {e}")
                continue

        # Generate report
        with report_path.open("w") as f:
            f.write("LayoutLM Postprocessing Summary Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Images Processed: {total_images}\n")
            f.write(f"Total Words Extracted: {total_words}\n")
            f.write(
                f"Average Words per Image: {total_words / max(total_images, 1):.1f}\n\n"
            )

            f.write("Label Distribution:\n")
            f.write("-" * 20 + "\n")
            for label, count in sorted(label_counts.items()):
                percentage = (count / total_words) * 100
                f.write(f"{label}: {count} ({percentage:.1f}%)\n")

            f.write(f"\nOutput Directory: {self.csv_output_dir}\n")
            f.write(f"CSV Files Generated: {len(csv_files)}\n")

        print(f"📋 Summary report saved: {report_path}")
        return str(report_path)

    def validate_csv_format(self, csv_file: str) -> bool:
        """
        Validate that a CSV file has the required format.

        Args:
            csv_file: Path to the CSV file to validate

        Returns:
            True if format is valid, False otherwise
        """
        required_columns = [
            "image_id",
            "block_ids",
            "word_ids",
            "words",
            "bboxes",
            "pred_label",
            "prob",
        ]

        try:
            data_frame = pd.read_csv(csv_file)

            # Check if all required columns are present
            missing_columns = set(required_columns) - set(data_frame.columns)
            if missing_columns:
                print(f"❌ Missing columns in {csv_file}: {missing_columns}")
                return False

            # Check data types and formats
            if not data_frame["image_id"].dtype == "object":
                print(f"❌ Invalid image_id type in {csv_file}")
                return False

            # Check bounding box format
            bbox_pattern = r"^\(\d+, \d+, \d+, \d+\)$"
            if not data_frame["bboxes"].str.match(bbox_pattern).all():
                print(f"❌ Invalid bbox format in {csv_file}")
                return False

            # Check probability range
            if not ((data_frame["prob"] >= 0) & (data_frame["prob"] <= 1)).all():
                print(f"❌ Invalid probability values in {csv_file}")
                return False

            print(f"✅ CSV format valid: {csv_file}")
            return True

        except Exception as e:
            print(f"❌ Error validating {csv_file}: {e}")
            return False


def create_postprocessor(config: Dict[str, Any]) -> LayoutLMPostprocessor:
    """
    Factory function to create a LayoutLM postprocessor.

    Args:
        config: Configuration dictionary

    Returns:
        LayoutLMPostprocessor instance
    """
    return LayoutLMPostprocessor(config)


def format_bbox_string(bbox: List[int]) -> str:
    """
    Format bounding box coordinates as required string format.

    Args:
        bbox: Bounding box coordinates [x0, y0, x1, y1]

    Returns:
        Formatted string "(ul_x, ul_y, lr_x, lr_y)"
    """
    return f"({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})"


def parse_bbox_string(bbox_str: str) -> List[int]:
    """
    Parse bounding box string back to coordinates.

    Args:
        bbox_str: Bounding box string "(ul_x, ul_y, lr_x, lr_y)"

    Returns:
        List of coordinates [x0, y0, x1, y1]
    """
    # Remove parentheses and split by comma
    coords_str = bbox_str.strip("()").split(", ")
    return [int(coord.strip()) for coord in coords_str]


if __name__ == "__main__":
    # Example usage
    print("LayoutLM Postprocessing Module")
    print("This module provides postprocessing functionality for LayoutLM predictions.")
    print("Import this module to use the LayoutLMPostprocessor class.")
