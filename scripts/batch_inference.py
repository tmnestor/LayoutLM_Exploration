#!/usr/bin/env python3
"""
Batch inference script for processing large datasets through LayoutLM.
Processes all images in a directory and generates CSV outputs for evaluation.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List

from tqdm import tqdm

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))

from inference import LayoutLMInference
from postprocessing import LayoutLMPostprocessor
from yaml_config_manager import load_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def batch_inference(
    model_dir: str,
    images_dir: str,
    config_path: str = "config/config.yaml",
    max_images: int = None,
) -> List[str]:
    """
    Run inference on all images in a directory.

    Args:
        model_dir: Path to trained model
        images_dir: Directory containing images to process
        config_path: Path to configuration file
        max_images: Maximum number of images to process (None for all)
        batch_size: Number of images to process in parallel (currently 1)

    Returns:
        List of generated CSV file paths
    """
    # Load configuration
    config = load_config(config_path)
    logger.info(f"Loaded configuration from {config_path}")

    # Find all image files
    images_path = Path(images_dir)
    image_extensions = config.get(
        "data.image_extensions", [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]
    )

    image_files = []
    for ext in image_extensions:
        image_files.extend(images_path.glob(f"*{ext}"))
        image_files.extend(images_path.glob(f"*{ext.upper()}"))

    # Sort and limit if requested
    image_files = sorted(image_files)
    if max_images:
        image_files = image_files[:max_images]

    logger.info(f"Found {len(image_files)} images to process")

    if not image_files:
        logger.error(f"No images found in {images_dir}")
        return []

    # Initialize inference engine and postprocessor
    logger.info("Initializing inference engine...")
    engine = LayoutLMInference(
        model_dir=model_dir, config_manager=config, device="auto"
    )

    processor = LayoutLMPostprocessor(config)
    logger.info(f"CSV outputs will be saved to: {processor.csv_output_dir}")

    # Process images
    csv_files = []
    failed_files = []
    start_time = time.time()

    logger.info(f"Starting batch inference on {len(image_files)} images...")

    for i, image_file in enumerate(tqdm(image_files, desc="Processing images")):
        try:
            # Get document name
            doc_name = image_file.stem

            # Run prediction
            result = engine.predict_document(str(image_file))

            # Generate CSV
            csv_file = processor.process_predictions(
                image_path=str(image_file),
                words=result["words"],
                bboxes=result["bboxes"],
                word_ids=result["word_ids"],
                block_ids=result["block_ids"],
                predictions=result["predictions"],
                probabilities=result["probabilities"],
                image_id=doc_name,
            )

            csv_files.append(csv_file)

            # Progress update every 100 files
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(image_files) - i - 1) / rate
                logger.info(
                    f"Processed {i + 1}/{len(image_files)} images. "
                    f"Rate: {rate:.1f} img/s. ETA: {remaining / 60:.1f} min"
                )

        except Exception as e:
            logger.error(f"Failed to process {image_file}: {e}")
            failed_files.append(str(image_file))
            continue

    elapsed_time = time.time() - start_time

    # Generate summary statistics
    logger.info("=" * 50)
    logger.info("BATCH INFERENCE COMPLETED")
    logger.info("=" * 50)
    logger.info(f"Total images processed: {len(csv_files)}")
    logger.info(f"Failed images: {len(failed_files)}")
    logger.info(f"Success rate: {len(csv_files) / len(image_files) * 100:.1f}%")
    logger.info(f"Total time: {elapsed_time / 60:.1f} minutes")
    logger.info(f"Average rate: {len(csv_files) / elapsed_time:.1f} images/second")

    if failed_files:
        logger.warning("Failed files:")
        for failed_file in failed_files[:10]:  # Show first 10
            logger.warning(f"  {failed_file}")
        if len(failed_files) > 10:
            logger.warning(f"  ... and {len(failed_files) - 10} more")

    # Generate aggregated results if we have multiple files
    if len(csv_files) > 1:
        logger.info("Generating aggregated results...")
        aggregated_path = processor.aggregate_results(csv_files)
        summary_path = processor.generate_summary_report(csv_files)

        logger.info(f"Aggregated CSV: {aggregated_path}")
        logger.info(f"Summary report: {summary_path}")

    logger.info(f"All CSV files saved to: {processor.csv_output_dir}")

    return csv_files


def main():
    """Main function for batch inference."""
    parser = argparse.ArgumentParser(
        description="Batch inference on LayoutLM documents"
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing trained model",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing images to process",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to process (default: all)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for processing (currently must be 1)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not Path(args.model_dir).exists():
        logger.error(f"Model directory not found: {args.model_dir}")
        return 1

    if not Path(args.images_dir).exists():
        logger.error(f"Images directory not found: {args.images_dir}")
        return 1

    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        return 1

    # Run batch inference
    try:
        csv_files = batch_inference(
            model_dir=args.model_dir,
            images_dir=args.images_dir,
            config_path=args.config,
            max_images=args.max_images,
            batch_size=args.batch_size,
        )

        if csv_files:
            logger.info(f"SUCCESS: Generated {len(csv_files)} CSV files")
            return 0
        else:
            logger.error("FAILED: No CSV files generated")
            return 1

    except Exception as e:
        logger.error(f"Batch inference failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
