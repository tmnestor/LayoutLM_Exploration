"""
Data loading and preprocessing utilities for LayoutLM document understanding.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from transformers import LayoutLMTokenizer

# Explicitly define what can be imported from this module
__all__ = ["DocumentDataset", "create_sample_data", "load_funsd_data"]


class DocumentDataset(Dataset):
    """Dataset for document understanding with LayoutLM."""

    def __init__(
        self,
        data_dir: str,
        tokenizer: LayoutLMTokenizer,
        max_seq_length: int = 512,
        image_size: Tuple[int, int] = (224, 224),
        max_samples: int = None,
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.image_size = image_size
        self.max_samples = max_samples
        self.examples = self._load_examples()

    def _load_examples(self) -> List[Dict[str, Any]]:
        """Load examples from data directory."""
        examples = []

        # Look for annotation files
        annotation_files = list(self.data_dir.glob("*.json"))

        # Limit files if max_samples is specified
        if self.max_samples and len(annotation_files) > self.max_samples:
            annotation_files = annotation_files[: self.max_samples]

        for ann_file in annotation_files:
            with ann_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            # Handle different annotation formats
            if isinstance(data, list):
                examples.extend(data)
            elif isinstance(data, dict):
                examples.append(data)

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]

        # Extract text and bounding boxes
        words = example.get("words", [])
        boxes = example.get("boxes", [])
        labels = example.get("labels", [0] * len(words))

        # Tokenize
        tokens = []
        token_boxes = []
        token_labels = []

        for word, box, label in zip(words, boxes, labels, strict=False):
            word_tokens = self.tokenizer.tokenize(word)
            tokens.extend(word_tokens)

            # Assign same box and label to all word tokens
            token_boxes.extend([box] * len(word_tokens))
            token_labels.extend([label] * len(word_tokens))

        # Add special tokens
        tokens = ["[CLS]"] + tokens[: self.max_seq_length - 2] + ["[SEP]"]
        token_boxes = (
            [[0, 0, 0, 0]] + token_boxes[: self.max_seq_length - 2] + [[0, 0, 0, 0]]
        )
        token_labels = [0] + token_labels[: self.max_seq_length - 2] + [0]

        # Convert to IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # Pad sequences
        padding_length = self.max_seq_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_length
        token_boxes += [[0, 0, 0, 0]] * padding_length
        token_labels += [0] * padding_length

        # Create attention mask
        attention_mask = [1] * len(tokens) + [0] * padding_length

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "bbox": torch.tensor(token_boxes, dtype=torch.long),
            "labels": torch.tensor(token_labels, dtype=torch.long),
        }


def create_sample_data(output_dir: str, num_samples: int = 10) -> None:
    """Create sample data for testing."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sample_data = []

    for i in range(num_samples):
        # Generate sample document data
        words = [
            "INVOICE",
            "Invoice",
            "Number:",
            f"INV-{i:04d}",
            "Date:",
            "2024-01-15",
            "Customer:",
            "John Doe",
            "Amount:",
            "$1,234.56",
            "Total:",
            "$1,234.56",
        ]

        # Generate bounding boxes (normalized coordinates)
        boxes = [
            [100, 50, 200, 80],  # INVOICE
            [100, 100, 180, 120],  # Invoice
            [100, 140, 170, 160],  # Number:
            [180, 140, 280, 160],  # INV-0001
            [100, 180, 140, 200],  # Date:
            [150, 180, 250, 200],  # 2024-01-15
            [100, 220, 180, 240],  # Customer:
            [190, 220, 270, 240],  # John Doe
            [100, 260, 170, 280],  # Amount:
            [180, 260, 280, 280],  # $1,234.56
            [100, 300, 150, 320],  # Total:
            [160, 300, 260, 320],  # $1,234.56
        ]

        # Generate labels (0: O, 1: B-INVOICE_NO, 2: B-DATE, 3: B-CUSTOMER, 4: B-AMOUNT)
        labels = [0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 4]

        sample_data.append(
            {"id": f"sample_{i}", "words": words, "boxes": boxes, "labels": labels}
        )

    # Save sample data
    sample_file = output_path / "sample_data.json"
    with sample_file.open("w") as f:
        json.dump(sample_data, f, indent=2)

    print(f"Created {num_samples} sample documents in {output_path}")


def load_funsd_data(data_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """Load FUNSD dataset if available."""
    data_path = Path(data_dir)

    train_data = []
    test_data = []

    # Check for FUNSD format
    train_dir = data_path / "training_data" / "annotations"
    test_dir = data_path / "testing_data" / "annotations"

    if train_dir.exists():
        for ann_file in train_dir.glob("*.json"):
            with ann_file.open("r") as f:
                data = json.load(f)
                train_data.append(data)

    if test_dir.exists():
        for ann_file in test_dir.glob("*.json"):
            with ann_file.open("r") as f:
                data = json.load(f)
                test_data.append(data)

    return train_data, test_data


if __name__ == "__main__":
    # Create sample data for testing
    create_sample_data("data/raw", num_samples=20)
    print("Sample data created successfully!")
