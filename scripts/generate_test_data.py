#!/usr/bin/env python3
"""
Generate synthetic test dataset with gold labels for LayoutLM evaluation.
This creates UNSEEN test data that is different from training data.
"""

import argparse
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont
from preprocessing import extract_text_and_boxes
from tqdm import tqdm
from yaml_config_manager import load_config


class TestDataGenerator:
    """Generate synthetic test documents with ground truth labels."""

    def __init__(self, config_manager, output_dir: str):
        self.config = config_manager
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "test_images"
        self.ground_truth_dir = self.output_dir / "ground_truth"

        # Clear existing test data directories to ensure clean results
        if self.images_dir.exists():
            import shutil

            shutil.rmtree(self.images_dir)
            print(f"🗑️  Cleared existing test images directory: {self.images_dir}")

        if self.ground_truth_dir.exists():
            import shutil

            shutil.rmtree(self.ground_truth_dir)
            print(
                f"🗑️  Cleared existing ground truth directory: {self.ground_truth_dir}"
            )

        # Create directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.ground_truth_dir.mkdir(parents=True, exist_ok=True)

        # Get label mapping
        self.label_mapping = config_manager.get_label_mapping()
        self.reverse_mapping = {v: k for k, v in self.label_mapping.items()}

        # Load fonts
        self._load_fonts()

        # Set random seed for reproducible test data
        random.seed(42)

    def _load_fonts(self):
        """Load fonts for document generation."""
        try:
            self.font_large = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 32)
            self.font_medium = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 22)
            self.font_small = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 18)
        except OSError:
            self.font_large = self.font_medium = self.font_small = (
                ImageFont.load_default()
            )

    def _create_test_invoice(
        self, doc_id: str
    ) -> Tuple[List[str], List[List[int]], List[int]]:
        """Create a test invoice with DIFFERENT characteristics from training data."""
        # Make test invoices DIFFERENT from training:
        # - Different layouts
        # - Different company names
        # - Different field arrangements
        # - Different fonts/sizes

        width, height = 850, 1100  # Different size from training
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)

        # Test-specific data (different from training)
        invoice_num = random.randint(200000, 299999)  # Different range
        test_companies = [
            "TestCorp Solutions",
            "Beta Industries",
            "Gamma Services",
            "Delta Consulting",
            "Epsilon LLC",
            "Zeta Corporation",
        ]
        test_services = [
            "System Integration",
            "Data Analytics",
            "Cloud Migration",
            "Security Audit",
            "Performance Testing",
            "Code Review",
        ]

        # Different layout: right-aligned header
        company = random.choice(test_companies)
        service = random.choice(test_services)
        amount = random.uniform(1000, 25000)  # Different range
        tax_rate = random.choice([0.06, 0.075, 0.095, 0.13])  # Different rates
        tax = amount * tax_rate
        total = amount + tax

        # Varied positioning for test data
        x_offset = random.randint(30, 80)
        y_start = random.randint(40, 80)

        content_with_labels = [
            ("TEST INVOICE", width - 200, y_start, self.font_large, "B-HEADER"),
            ("Invoice Number:", x_offset, y_start + 80, self.font_medium, "B-QUESTION"),
            (
                f"TEST-{invoice_num}",
                x_offset + 160,
                y_start + 80,
                self.font_medium,
                "B-ANSWER",
            ),
            ("Issue Date:", x_offset, y_start + 120, self.font_medium, "B-QUESTION"),
            (
                f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
                x_offset + 110,
                y_start + 120,
                self.font_medium,
                "B-ANSWER",
            ),
            (
                "Client Company:",
                x_offset,
                y_start + 160,
                self.font_medium,
                "B-QUESTION",
            ),
            (company, x_offset + 150, y_start + 160, self.font_medium, "B-ANSWER"),
            (
                "Service Provided:",
                x_offset,
                y_start + 220,
                self.font_medium,
                "B-QUESTION",
            ),
            (service, x_offset + 170, y_start + 220, self.font_medium, "B-ANSWER"),
            ("Base Amount:", x_offset, y_start + 400, self.font_medium, "B-QUESTION"),
            (
                f"${amount:.2f}",
                x_offset + 130,
                y_start + 400,
                self.font_medium,
                "B-ANSWER",
            ),
            ("Tax Rate:", x_offset, y_start + 440, self.font_medium, "B-QUESTION"),
            (
                f"{tax_rate * 100:.1f}%",
                x_offset + 100,
                y_start + 440,
                self.font_medium,
                "B-ANSWER",
            ),
            ("Tax Amount:", x_offset, y_start + 480, self.font_medium, "B-QUESTION"),
            (
                f"${tax:.2f}",
                x_offset + 120,
                y_start + 480,
                self.font_medium,
                "B-ANSWER",
            ),
            ("TOTAL DUE:", x_offset, y_start + 540, self.font_large, "B-QUESTION"),
            (
                f"${total:.2f}",
                x_offset + 140,
                y_start + 540,
                self.font_large,
                "B-ANSWER",
            ),
        ]

        return self._draw_and_extract_data(image, draw, content_with_labels, doc_id)

    def _create_test_receipt(
        self, doc_id: str
    ) -> Tuple[List[str], List[List[int]], List[int]]:
        """Create a test receipt with different characteristics."""
        width, height = 600, 800  # Smaller receipt format
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)

        # Test-specific receipt data
        receipt_num = random.randint(50000, 59999)
        test_stores = [
            "QuickMart #789",
            "SuperShop Beta",
            "MegaStore Gamma",
            "RetailHub Delta",
            "ShopCenter Epsilon",
        ]
        test_items = [
            "Premium Coffee Blend",
            "Artisan Sandwich",
            "Technical Manual",
            "Organic Groceries",
            "Premium Fuel",
            "Healthy Snacks",
        ]

        store = random.choice(test_stores)
        item = random.choice(test_items)
        price = random.uniform(15, 150)

        y_pos = 60
        content_with_labels = [
            ("RECEIPT", 250, y_pos, self.font_large, "B-HEADER"),
            ("Store Location:", 50, y_pos + 80, self.font_small, "B-QUESTION"),
            (store, 170, y_pos + 80, self.font_small, "B-ANSWER"),
            ("Receipt ID:", 50, y_pos + 110, self.font_small, "B-QUESTION"),
            (f"RCP-{receipt_num}", 140, y_pos + 110, self.font_small, "B-ANSWER"),
            ("Transaction Date:", 50, y_pos + 140, self.font_small, "B-QUESTION"),
            (
                f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
                180,
                y_pos + 140,
                self.font_small,
                "B-ANSWER",
            ),
            ("Item Description:", 50, y_pos + 180, self.font_small, "B-QUESTION"),
            (item, 180, y_pos + 180, self.font_small, "B-ANSWER"),
            ("Unit Price:", 50, y_pos + 210, self.font_small, "B-QUESTION"),
            (f"${price:.2f}", 130, y_pos + 210, self.font_small, "B-ANSWER"),
            ("TOTAL PAID:", 50, y_pos + 300, self.font_medium, "B-QUESTION"),
            (f"${price:.2f}", 160, y_pos + 300, self.font_medium, "B-ANSWER"),
        ]

        return self._draw_and_extract_data(image, draw, content_with_labels, doc_id)

    def _create_test_form(
        self, doc_id: str
    ) -> Tuple[List[str], List[List[int]], List[int]]:
        """Create a test form with different layout."""
        width, height = 800, 1000
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)

        # Test-specific form data
        test_names = [
            "Alexandra Thompson",
            "Benjamin Rodriguez",
            "Catherine Walsh",
            "David Kim",
            "Elena Petrov",
            "Francisco Garcia",
        ]
        test_addresses = [
            "789 Test Avenue",
            "456 Sample Street",
            "123 Demo Boulevard",
            "321 Trial Road",
            "654 Example Lane",
        ]
        test_phones = [f"555-{random.randint(2000, 9999)}" for _ in range(10)]
        test_emails = [
            "alex@testmail.com",
            "ben@sampleemail.com",
            "cat@democorp.com",
            "david@trialbiz.com",
            "elena@examplefirm.com",
        ]

        name = random.choice(test_names)
        address = random.choice(test_addresses)
        phone = random.choice(test_phones)
        email = random.choice(test_emails)

        # Center-aligned header
        content_with_labels = [
            ("REGISTRATION FORM", 250, 60, self.font_large, "B-HEADER"),
            ("Full Name:", 100, 150, self.font_medium, "B-QUESTION"),
            (name, 200, 150, self.font_medium, "B-ANSWER"),
            ("Street Address:", 100, 190, self.font_medium, "B-QUESTION"),
            (address, 230, 190, self.font_medium, "B-ANSWER"),
            ("Phone Number:", 100, 230, self.font_medium, "B-QUESTION"),
            (phone, 220, 230, self.font_medium, "B-ANSWER"),
            ("Email Address:", 100, 270, self.font_medium, "B-QUESTION"),
            (email, 220, 270, self.font_medium, "B-ANSWER"),
            ("Submission Date:", 100, 310, self.font_medium, "B-QUESTION"),
            (
                f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
                240,
                310,
                self.font_medium,
                "B-ANSWER",
            ),
        ]

        return self._draw_and_extract_data(image, draw, content_with_labels, doc_id)

    def _create_test_report(
        self, doc_id: str
    ) -> Tuple[List[str], List[List[int]], List[int]]:
        """Create a test report - NEW document type not in training."""
        width, height = 900, 1200
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)

        # Report-specific data (NEW type)
        report_num = random.randint(300000, 399999)
        test_departments = [
            "Quality Assurance",
            "Research & Development",
            "Operations Analysis",
            "Strategic Planning",
            "Risk Management",
        ]
        test_metrics = [
            "Performance Score",
            "Efficiency Rating",
            "Quality Index",
            "Compliance Level",
            "Satisfaction Rate",
        ]

        department = random.choice(test_departments)
        metric = random.choice(test_metrics)
        score = random.uniform(75, 98)

        content_with_labels = [
            ("QUARTERLY REPORT", 300, 80, self.font_large, "B-HEADER"),
            ("Report ID:", 80, 160, self.font_medium, "B-QUESTION"),
            (f"RPT-{report_num}", 180, 160, self.font_medium, "B-ANSWER"),
            ("Department:", 80, 200, self.font_medium, "B-QUESTION"),
            (department, 180, 200, self.font_medium, "B-ANSWER"),
            ("Reporting Period:", 80, 240, self.font_medium, "B-QUESTION"),
            ("Q4 2024", 220, 240, self.font_medium, "B-ANSWER"),
            ("Key Metric:", 80, 300, self.font_medium, "B-QUESTION"),
            (metric, 180, 300, self.font_medium, "B-ANSWER"),
            ("Current Score:", 80, 340, self.font_medium, "B-QUESTION"),
            (f"{score:.1f}%", 200, 340, self.font_medium, "B-ANSWER"),
            ("Status:", 80, 400, self.font_large, "B-QUESTION"),
            (
                "APPROVED" if score > 85 else "PENDING",
                160,
                400,
                self.font_large,
                "B-ANSWER",
            ),
        ]

        return self._draw_and_extract_data(image, draw, content_with_labels, doc_id)

    def _draw_and_extract_data(
        self,
        image: Image.Image,
        draw: ImageDraw.ImageDraw,
        content_with_labels: List[Tuple],
        doc_id: str,
    ) -> Tuple[List[str], List[List[int]], List[int]]:
        """Draw content and extract data using OCR for perfect alignment."""
        # First, draw all content on the image
        label_regions = []  # Store regions with their intended labels

        for text, x, y, font, label_name in content_with_labels:
            # Draw text
            draw.text((x, y), text, fill="black", font=font)

            # Calculate approximate region for this label
            bbox = draw.textbbox((x, y), text, font=font)
            x1, y1, x2, y2 = bbox
            label_regions.append(
                {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "label": label_name,
                    "center": ((x1 + x2) / 2, (y1 + y2) / 2),
                }
            )

        # Save image first
        image_path = self.images_dir / f"{doc_id}.png"
        image.save(image_path, "PNG")

        # Extract text using OCR (same process as inference)
        ocr_result = extract_text_and_boxes(str(image_path), config_manager=self.config)

        # Match OCR words to intended labels based on proximity
        words = []
        boxes = []
        labels = []

        for word, box in zip(ocr_result["words"], ocr_result["boxes"], strict=False):
            # Find the closest label region
            word_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

            closest_label = "O"  # Default
            min_distance = float("inf")

            for region in label_regions:
                # Calculate distance between word center and region center
                region_center = region["center"]
                distance = (
                    (word_center[0] - region_center[0]) ** 2
                    + (word_center[1] - region_center[1]) ** 2
                ) ** 0.5

                # Check if word is within region bounds (with some tolerance)
                if (
                    box[0] >= region["bbox"][0] - 10
                    and box[2] <= region["bbox"][2] + 10
                    and box[1] >= region["bbox"][1] - 10
                    and box[3] <= region["bbox"][3] + 10
                ):
                    if distance < min_distance:
                        min_distance = distance
                        closest_label = region["label"]

            words.append(word)
            boxes.append(box)
            labels.append(self.reverse_mapping.get(closest_label, 0))

        return words, boxes, labels

    def generate_test_dataset(
        self, num_documents: int = 100, doc_type_distribution: Dict[str, float] = None
    ) -> Dict[str, int]:
        """Generate the complete test dataset."""
        if doc_type_distribution is None:
            # Different distribution from training
            doc_type_distribution = {
                "invoice": 0.30,  # 30 docs
                "receipt": 0.25,  # 25 docs
                "form": 0.20,  # 20 docs
                "report": 0.25,  # 25 docs (NEW type)
            }

        print(f"🔄 Generating {num_documents} TEST documents...")
        print("📊 Document type distribution:")
        for doc_type, ratio in doc_type_distribution.items():
            count = int(num_documents * ratio)
            print(f"  {doc_type}: {count} documents ({ratio * 100:.0f}%)")

        generated_counts = {}
        total_words = 0
        start_time = time.time()

        # Generate documents by type
        doc_counter = 0
        for doc_type, ratio in doc_type_distribution.items():
            count = int(num_documents * ratio)
            generated_counts[doc_type] = count

            print(f"\n📝 Creating {count} test {doc_type}s...")

            for i in tqdm(range(count), desc=f"Creating {doc_type}s"):
                doc_id = f"test_{doc_type}_{i + 1:04d}"

                try:
                    # Generate document based on type
                    if doc_type == "invoice":
                        words, boxes, labels = self._create_test_invoice(doc_id)
                    elif doc_type == "receipt":
                        words, boxes, labels = self._create_test_receipt(doc_id)
                    elif doc_type == "form":
                        words, boxes, labels = self._create_test_form(doc_id)
                    elif doc_type == "report":
                        words, boxes, labels = self._create_test_report(doc_id)
                    else:
                        raise ValueError(f"Unknown document type: {doc_type}")

                    # Create ground truth file in CSV format
                    self._create_ground_truth_csv(doc_id, words, boxes, labels)

                    total_words += len(words)
                    doc_counter += 1

                except Exception as e:
                    print(f"    ❌ Failed to create {doc_id}: {e}")
                    continue

        elapsed_time = time.time() - start_time

        print("\n🎉 TEST DATASET CREATED!")
        print("📊 Final Statistics:")
        print(f"  Total test documents: {doc_counter}")
        print(f"  Total words: {total_words}")
        print(f"  Average words per doc: {total_words / doc_counter:.1f}")
        print(f"  Generation time: {elapsed_time:.1f} seconds")
        print(f"  Images saved to: {self.images_dir}")
        print(f"  Ground truth saved to: {self.ground_truth_dir}")

        return generated_counts

    def _create_ground_truth_csv(
        self, doc_id: str, words: List[str], boxes: List[List[int]], labels: List[int]
    ):
        """Create ground truth CSV file matching prediction format."""
        import csv

        # Convert to expected CSV format
        csv_data = []
        for i, (word, box, label) in enumerate(zip(words, boxes, labels, strict=False)):
            # Format bounding box as string
            bbox_str = f"({box[0]}, {box[1]}, {box[2]}, {box[3]})"

            # Convert label ID to label name
            true_label = self.label_mapping.get(label, "O")

            csv_row = {
                "image_id": doc_id,
                "block_ids": 0,  # Default block ID
                "word_ids": i,
                "words": word,
                "bboxes": bbox_str,
                "true_label": true_label,  # Ground truth label
                "confidence": 1.0,  # Ground truth has 100% confidence
            }
            csv_data.append(csv_row)

        # Save ground truth CSV
        csv_file = self.ground_truth_dir / f"{doc_id}_ground_truth.csv"
        with csv_file.open("w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "image_id",
                "block_ids",
                "word_ids",
                "words",
                "bboxes",
                "true_label",
                "confidence",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)


def main():
    """Main function for test data generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic test dataset with gold labels"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for test data"
    )
    parser.add_argument(
        "--num_documents",
        type=int,
        default=100,
        help="Number of test documents to generate",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducible test data"
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Load configuration
    config = load_config(args.config)
    print(f"✅ Configuration loaded from {args.config}")

    # Create test data generator
    generator = TestDataGenerator(config, args.output_dir)

    # Generate test dataset
    try:
        generated_counts = generator.generate_test_dataset(args.num_documents)

        print("\n✅ TEST DATA GENERATION COMPLETED!")
        print(f"📁 Test images: {generator.images_dir}")
        print(f"📁 Ground truth: {generator.ground_truth_dir}")
        print("\n📊 Generated document types:")
        for doc_type, count in generated_counts.items():
            print(f"  {doc_type}: {count}")

        print(
            f"\n🚀 Ready for evaluation with {sum(generated_counts.values())} test documents!"
        )

        return 0

    except Exception as e:
        print(f"❌ Test data generation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
