"""
LayoutLM model implementation for document understanding.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from sklearn.metrics import accuracy_score, classification_report
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    LayoutLMForTokenClassification,
    LayoutLMTokenizer,
    get_linear_schedule_with_warmup,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LayoutLMTrainer:
    """Trainer class for LayoutLM fine-tuning."""

    def __init__(
        self,
        model_name: str = "microsoft/layoutlm-base-uncased",
        num_labels: int = 5,
        max_seq_length: int = 512,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_seq_length = max_seq_length

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

        # Initialize tokenizer and model
        self.tokenizer = LayoutLMTokenizer.from_pretrained(model_name)
        self.model = LayoutLMForTokenClassification.from_pretrained(
            model_name, num_labels=num_labels
        ).to(self.device)

        logger.info(f"Initialized LayoutLM model on {self.device}")
        logger.info(
            f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}"
        )

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        num_epochs: int = 3,
        learning_rate: float = 5e-5,
        warmup_steps: int = 500,
        output_dir: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the LayoutLM model.

        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            warmup_steps: Number of warmup steps for scheduler
            output_dir: Directory to save model checkpoints

        Returns:
            Training history dictionary
        """
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_dataloader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        # Training history
        history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

        # Training loop
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")

            # Training phase
            train_loss = self._train_epoch(train_dataloader, optimizer, scheduler)
            history["train_loss"].append(train_loss)

            # Validation phase
            if val_dataloader is not None:
                val_loss, val_accuracy = self._validate_epoch(val_dataloader)
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_accuracy)

                logger.info(
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
                )
            else:
                logger.info(f"Train Loss: {train_loss:.4f}")

            # Save checkpoint
            if output_dir:
                self._save_checkpoint(output_dir, epoch)

        return history

    def _train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0

        progress_bar = tqdm(dataloader, desc="Training")

        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        return total_loss / len(dataloader)

    def _validate_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                logits = outputs.logits

                total_loss += loss.item()

                # Get predictions
                predictions = torch.argmax(logits, dim=-1)

                # Filter out padding tokens AND special tokens (CLS, SEP)
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]

                # Only evaluate on real word tokens (labels > 0, not padding, not special tokens)
                active_mask = (attention_mask == 1) & (labels != -100) & (labels > 0)

                if active_mask.sum() > 0:  # Only add if we have valid tokens
                    active_predictions = predictions[active_mask]
                    active_labels = labels[active_mask]

                    all_predictions.extend(active_predictions.cpu().numpy())
                    all_labels.extend(active_labels.cpu().numpy())

        # Calculate accuracy only if we have predictions
        if len(all_predictions) > 0:
            accuracy = accuracy_score(all_labels, all_predictions)
        else:
            accuracy = 0.0

        avg_loss = total_loss / len(dataloader)

        return avg_loss, accuracy

    def predict(self, dataloader: DataLoader) -> Tuple[List[int], List[int]]:
        """Make predictions on a dataset."""
        self.model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(**batch)
                logits = outputs.logits

                # Get predictions
                predictions = torch.argmax(logits, dim=-1)

                # Flatten and filter out padding tokens
                attention_mask = batch["attention_mask"]
                active_predictions = predictions[attention_mask == 1]
                active_labels = batch["labels"][attention_mask == 1]

                all_predictions.extend(active_predictions.cpu().numpy())
                all_labels.extend(active_labels.cpu().numpy())

        return all_predictions, all_labels

    def evaluate(
        self, dataloader: DataLoader, label_names: Optional[List[str]] = None
    ) -> Dict:
        """Evaluate the model and return metrics."""
        predictions, labels = self.predict(dataloader)

        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)

        # Create classification report
        if label_names is None:
            label_names = [f"Label_{i}" for i in range(self.num_labels)]

        report = classification_report(
            labels,
            predictions,
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        )

        results = {
            "accuracy": accuracy,
            "classification_report": report,
            "predictions": predictions,
            "labels": labels,
        }

        return results

    def _save_checkpoint(self, save_dir: str, epoch: int) -> None:
        """Save model checkpoint."""
        save_path = Path(save_dir)

        # Clear existing checkpoint directory only on first epoch
        if epoch == 1 and save_path.exists():
            import shutil

            shutil.rmtree(save_path)
            logger.info(f"🗑️  Cleared existing checkpoint directory: {save_path}")

        save_path.mkdir(parents=True, exist_ok=True)

        # Save model and tokenizer
        model_path = save_path / f"epoch_{epoch}"
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)

        logger.info(f"Checkpoint saved to {model_path}")

    def save_model(self, save_dir: str) -> None:
        """Save the final trained model."""
        save_path = Path(save_dir)

        # Clear existing model directory to ensure clean results
        if save_path.exists():
            import shutil

            shutil.rmtree(save_path)
            logger.info(f"🗑️  Cleared existing model directory: {save_path}")

        save_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        # Save training configuration
        config = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "max_seq_length": self.max_seq_length,
            "device": self.device,
        }

        config_file = save_path / "training_config.json"
        with config_file.open("w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Model saved to {save_path}")

    def load_model(self, model_dir: str) -> None:
        """Load a trained model."""
        model_path = Path(model_dir)

        # Load model and tokenizer
        self.model = LayoutLMForTokenClassification.from_pretrained(model_path).to(
            self.device
        )
        self.tokenizer = LayoutLMTokenizer.from_pretrained(model_path)

        # Load configuration if available
        config_path = model_path / "training_config.json"
        if config_path.exists():
            with config_path.open("r") as f:
                config = json.load(f)
                self.num_labels = config["num_labels"]
                self.max_seq_length = config["max_seq_length"]

        logger.info(f"Model loaded from {model_path}")


def create_label_mapping() -> Dict[int, str]:
    """Create default label mapping for document understanding."""
    return {
        0: "O",  # Outside any entity
        1: "B-HEADER",  # Beginning of header
        2: "I-HEADER",  # Inside header
        3: "B-QUESTION",  # Beginning of question
        4: "I-QUESTION",  # Inside question
        5: "B-ANSWER",  # Beginning of answer
        6: "I-ANSWER",  # Inside answer
    }


if __name__ == "__main__":
    # Example usage
    print("LayoutLM model implementation ready!")
    print("Use LayoutLMTrainer class to train and evaluate models")

    # Print label mapping
    labels = create_label_mapping()
    print("\nDefault label mapping:")
    for idx, label in labels.items():
        print(f"  {idx}: {label}")
