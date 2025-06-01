"""
Inference utilities for LayoutLM document understanding with YAML configuration.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from layoutlm_model import LayoutLMTrainer
from PIL import Image
from postprocessing import create_postprocessor
from preprocessing import extract_text_and_boxes, normalize_boxes
from yaml_config_manager import load_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LayoutLMInference:
    """Inference class for LayoutLM document understanding."""

    def __init__(
        self, model_dir: str, config_manager=None, device: Optional[str] = None
    ):
        """
        Initialize inference class.

        Args:
            model_dir: Directory containing trained model
            config_manager: Optional config manager for label mapping
            device: Device to run inference on
        """
        self.model_dir = Path(model_dir)
        self.config = config_manager  # Store config manager

        # Handle device selection properly
        if device == "auto" or device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"  # Apple Silicon GPU
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.config_manager = config_manager

        # Initialize postprocessor if config is available
        self.postprocessor = None
        if config_manager:
            self.postprocessor = create_postprocessor(config_manager)

        # Load model
        self.trainer = LayoutLMTrainer(device=self.device)
        self.trainer.load_model(str(self.model_dir))

        # Load label mapping - try config first, then file, then default
        if config_manager and hasattr(config_manager, "get_label_mapping"):
            self.label_mapping = config_manager.get_label_mapping()
        else:
            label_mapping_path = self.model_dir / "label_mapping.json"
            if label_mapping_path.exists():
                with label_mapping_path.open("r") as f:
                    self.label_mapping = json.load(f)
                    # Convert string keys to integers
                    self.label_mapping = {
                        int(k): v for k, v in self.label_mapping.items()
                    }
            else:
                # Default label mapping if no file exists
                self.label_mapping = {
                    0: "O",
                    1: "B-HEADER",
                    2: "I-HEADER",
                    3: "B-QUESTION",
                    4: "I-QUESTION",
                    5: "B-ANSWER",
                    6: "I-ANSWER",
                }

        logger.info(f"Loaded model from {model_dir}")
        logger.info(f"Label mapping: {self.label_mapping}")

    def predict_document(
        self, image_path: str, confidence_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Predict labels for a document image.

        Args:
            image_path: Path to document image
            confidence_threshold: Minimum confidence for predictions

        Returns:
            Prediction results dictionary
        """
        # Extract text and bounding boxes using OCR
        logger.info(f"Processing document: {image_path}")
        ocr_result = extract_text_and_boxes(image_path, config_manager=self.config)

        if not ocr_result["words"]:
            logger.warning("No text found in document")
            return {
                "image_path": image_path,
                "words": [],
                "boxes": [],
                "predictions": [],
                "confidences": [],
            }

        # Prepare input for model
        words = ocr_result["words"]
        boxes = normalize_boxes(ocr_result["boxes"], ocr_result["image_size"])

        # Tokenize input
        input_data = self._prepare_input(words, boxes)

        # Move to device
        input_data = {k: v.to(self.device) for k, v in input_data.items()}

        # Get predictions
        self.trainer.model.eval()
        with torch.no_grad():
            outputs = self.trainer.model(**input_data)
            logits = outputs.logits

            # Get probabilities and predictions
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)

            # Get confidence scores
            max_probs = torch.max(probabilities, dim=-1)[0]

        # Process predictions
        attention_mask = input_data["attention_mask"][0]
        active_indices = (attention_mask == 1).nonzero(as_tuple=True)[0]

        # Skip [CLS] and [SEP] tokens
        token_predictions = predictions[0][active_indices][1:-1]
        token_confidences = max_probs[0][active_indices][1:-1]

        # Map tokens back to words
        word_predictions, word_confidences = self._map_tokens_to_words(
            words, token_predictions, token_confidences
        )

        # Filter by confidence threshold
        filtered_predictions = []
        filtered_confidences = []

        for pred, conf in zip(word_predictions, word_confidences, strict=False):
            if conf >= confidence_threshold:
                filtered_predictions.append(pred)
                filtered_confidences.append(conf)
            else:
                filtered_predictions.append(0)  # Default to 'O' label
                filtered_confidences.append(conf)

        # Convert prediction IDs to labels
        predicted_labels = [
            self.label_mapping.get(pred, "O") for pred in filtered_predictions
        ]

        result = {
            "image_path": image_path,
            "image_size": ocr_result["image_size"],
            "words": words,
            "boxes": boxes,
            "bboxes": ocr_result["boxes"],  # Original pixel coordinates
            "predictions": filtered_predictions,
            "predicted_labels": predicted_labels,
            "confidences": [float(conf) for conf in filtered_confidences],
            "probabilities": [
                float(conf) for conf in filtered_confidences
            ],  # Alias for postprocessing
            "word_ids": list(range(len(words))),  # Generate word IDs
            "block_ids": [0] * len(words),  # Default block IDs (can be enhanced later)
            "label_mapping": self.label_mapping,
        }

        return result

    def _prepare_input(
        self, words: List[str], boxes: List[List[int]]
    ) -> Dict[str, torch.Tensor]:
        """Prepare input for the model."""
        # Tokenize words
        tokens = []
        token_boxes = []

        for word, box in zip(words, boxes, strict=False):
            word_tokens = self.trainer.tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            token_boxes.extend([box] * len(word_tokens))

        # Add special tokens
        tokens = ["[CLS]"] + tokens[: self.trainer.max_seq_length - 2] + ["[SEP]"]
        token_boxes = (
            [[0, 0, 0, 0]]
            + token_boxes[: self.trainer.max_seq_length - 2]
            + [[0, 0, 0, 0]]
        )

        # Convert to IDs
        input_ids = self.trainer.tokenizer.convert_tokens_to_ids(tokens)

        # Pad sequences
        padding_length = self.trainer.max_seq_length - len(input_ids)
        input_ids += [self.trainer.tokenizer.pad_token_id] * padding_length
        token_boxes += [[0, 0, 0, 0]] * padding_length

        # Create attention mask
        attention_mask = [1] * len(tokens) + [0] * padding_length

        return {
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
            "bbox": torch.tensor([token_boxes], dtype=torch.long),
        }

    def _map_tokens_to_words(
        self,
        words: List[str],
        token_predictions: torch.Tensor,
        token_confidences: torch.Tensor,
    ) -> Tuple[List[int], List[float]]:
        """Map token-level predictions back to word-level predictions."""
        word_predictions = []
        word_confidences = []

        token_idx = 0
        for word in words:
            # Get tokens for this word
            word_tokens = self.trainer.tokenizer.tokenize(word)
            num_tokens = len(word_tokens)

            if token_idx + num_tokens <= len(token_predictions):
                # Take the prediction of the first token for the word
                word_pred = token_predictions[token_idx].item()
                word_conf = token_confidences[token_idx].item()

                word_predictions.append(word_pred)
                word_confidences.append(word_conf)

                token_idx += num_tokens
            else:
                # Fallback for edge cases
                word_predictions.append(0)
                word_confidences.append(0.0)
                break

        return word_predictions, word_confidences

    def visualize_predictions(
        self,
        prediction_result: Dict[str, Any],
        output_path: Optional[str] = None,
        show_confidence: bool = True,
    ) -> None:
        """
        Visualize predictions on the document image.

        Args:
            prediction_result: Result from predict_document
            output_path: Optional path to save visualization
            show_confidence: Whether to show confidence scores
        """
        # Load image
        image = Image.open(prediction_result["image_path"])

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 16))
        ax.imshow(image)

        # Define colors for different labels
        colors = {
            "O": "gray",
            "B-HEADER": "red",
            "I-HEADER": "orange",
            "B-QUESTION": "blue",
            "I-QUESTION": "lightblue",
            "B-ANSWER": "green",
            "I-ANSWER": "lightgreen",
        }

        # Get image dimensions for denormalization
        img_width, img_height = prediction_result["image_size"]

        # Draw bounding boxes and labels
        for _i, (_word, box, label, confidence) in enumerate(
            zip(
                prediction_result["words"],
                prediction_result["boxes"],
                prediction_result["predicted_labels"],
                prediction_result["confidences"],
                strict=False,
            )
        ):
            # Denormalize bounding box coordinates
            x1 = (box[0] / 1000) * img_width
            y1 = (box[1] / 1000) * img_height
            x2 = (box[2] / 1000) * img_width
            y2 = (box[3] / 1000) * img_height

            width = x2 - x1
            height = y2 - y1

            # Get color for label
            color = colors.get(label, "gray")

            # Draw bounding box
            rect = patches.Rectangle(
                (x1, y1), width, height, linewidth=2, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)

            # Add text label
            label_text = f"{label}"
            if show_confidence:
                label_text += f" ({confidence:.2f})"

            ax.text(
                x1,
                y1 - 5,
                label_text,
                color=color,
                fontsize=8,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            )

        ax.set_xlim(0, img_width)
        ax.set_ylim(img_height, 0)
        ax.set_title(
            f"LayoutLM Predictions: {Path(prediction_result['image_path']).name}"
        )
        ax.axis("off")

        # Add legend
        legend_elements = [
            patches.Patch(facecolor=color, label=label)
            for label, color in colors.items()
            if label in prediction_result["predicted_labels"]
        ]
        if legend_elements:
            ax.legend(handles=legend_elements, loc="upper right")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Visualization saved to {output_path}")

        plt.show()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run LayoutLM inference on documents")

    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to YAML configuration file",
    )

    # Required arguments
    parser.add_argument(
        "--model_dir", type=str, help="Directory containing trained model"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input document image or directory",
    )

    # Override options
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        help="Override minimum confidence threshold",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Create visualization of predictions"
    )
    parser.add_argument(
        "--batch_process",
        action="store_true",
        help="Process all images in input directory",
    )

    # Environment options
    parser.add_argument(
        "--offline", action="store_true", help="Force offline mode (use local models)"
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Force online mode (download models from HF)",
    )

    return parser.parse_args()


def main():
    """Main inference function."""
    args = parse_args()

    # Load configuration
    try:
        config_manager = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return

    # Set up environment variables for Hugging Face
    hf_home = config_manager.get_hf_cache_dir()
    if hf_home:
        os.environ["HF_HOME"] = hf_home
        logger.info(f"Set HF_HOME to {hf_home}")

    # Override config with command line arguments
    if args.offline:
        config_manager.set("production.offline_mode", True)
        config_manager.set("model.use_local_model", True)
    elif args.online:
        config_manager.set("production.offline_mode", False)
        config_manager.set("model.use_local_model", False)

    # Get configuration values
    inference_config = config_manager.get_inference_config()
    data_config = config_manager.get_data_config()
    model_config = config_manager.get_model_config()

    # Determine model directory
    model_dir = args.model_dir or model_config.get("final_model_dir")
    if not model_dir:
        logger.error("Model directory not specified and not found in config")
        return

    # Determine output directory
    output_dir = args.output_dir or config_manager.get("environment.output_dir")

    # Determine confidence threshold
    confidence_threshold = args.confidence_threshold or inference_config.get(
        "confidence_threshold"
    )

    # Initialize inference
    inference = LayoutLMInference(model_dir, config_manager)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input_path)

    if args.batch_process and input_path.is_dir():
        # Process all images in directory
        image_extensions = data_config.get("image_extensions")
        image_files = []

        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))

        logger.info(f"Found {len(image_files)} images to process")

        batch_results = []
        csv_files = []

        for image_file in image_files:
            logger.info(f"Processing {image_file.name}...")

            # Run prediction
            result = inference.predict_document(
                str(image_file), confidence_threshold=confidence_threshold
            )

            # Save results
            result_file = output_path / f"{image_file.stem}_predictions.json"
            with result_file.open("w") as f:
                json.dump(result, f, indent=2)

            # Generate CSV if postprocessor is available
            if inference.postprocessor:
                csv_file = inference.postprocessor.process_predictions(
                    image_path=str(image_file),
                    words=result["words"],
                    bboxes=result["bboxes"],
                    word_ids=result["word_ids"],
                    block_ids=result["block_ids"],
                    predictions=result["predictions"],
                    probabilities=result["probabilities"],
                    image_id=image_file.stem,
                )
                csv_files.append(csv_file)
                batch_results.append(result)

            # Create visualization if requested
            if args.visualize or inference_config.get("save_visualizations"):
                viz_file = output_path / f"{image_file.stem}_visualization.png"
                inference.visualize_predictions(result, str(viz_file))

        # Generate aggregated results and summary if we processed multiple files
        if inference.postprocessor and len(csv_files) > 1:
            logger.info("Generating aggregated results...")
            inference.postprocessor.aggregate_results(csv_files)
            inference.postprocessor.generate_summary_report(csv_files)

    else:
        # Process single image
        if not input_path.exists():
            logger.error(f"Input path {input_path} does not exist!")
            return

        logger.info(f"Processing {input_path.name}...")

        # Run prediction
        result = inference.predict_document(
            str(input_path), confidence_threshold=confidence_threshold
        )

        # Save results
        result_file = output_path / f"{input_path.stem}_predictions.json"
        with result_file.open("w") as f:
            json.dump(result, f, indent=2)

        logger.info(f"Results saved to {result_file}")

        # Generate CSV if postprocessor is available
        if inference.postprocessor:
            csv_file = inference.postprocessor.process_predictions(
                image_path=str(input_path),
                words=result["words"],
                bboxes=result["bboxes"],
                word_ids=result["word_ids"],
                block_ids=result["block_ids"],
                predictions=result["predictions"],
                probabilities=result["probabilities"],
                image_id=input_path.stem,
            )
            logger.info(f"CSV results saved to {csv_file}")

        # Create visualization if requested
        if args.visualize or inference_config.get("save_visualizations"):
            viz_file = output_path / f"{input_path.stem}_visualization.png"
            inference.visualize_predictions(result, str(viz_file))

    logger.info("Inference completed!")


if __name__ == "__main__":
    main()
