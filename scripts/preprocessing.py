"""
Document preprocessing utilities for LayoutLM.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from PIL import Image


def extract_text_and_boxes(
    image_path: str, output_dir: Optional[str] = None, config_manager=None
) -> Dict[str, Any]:
    """
    Extract text and bounding boxes from document image using OCR.

    Args:
        image_path: Path to the input image
        output_dir: Optional directory to save extracted data
        config_manager: Configuration manager for OCR settings

    Returns:
        Dictionary containing words, bounding boxes, and confidence scores
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Convert to RGB for PIL
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)

    # Get OCR configuration from config manager or use defaults
    if config_manager:
        ocr_config = config_manager.get("data.ocr.config", "--oem 3 --psm 6")
        confidence_threshold = config_manager.get("data.ocr.confidence_threshold", 30)
        language = config_manager.get("data.ocr.language", "eng")

        # Add language to OCR config if specified
        if language != "eng":
            ocr_config += f" -l {language}"
    else:
        ocr_config = "--oem 3 --psm 6"
        confidence_threshold = 30

    # Extract text with bounding boxes using Tesseract
    ocr_data = pytesseract.image_to_data(
        pil_image, output_type=pytesseract.Output.DICT, config=ocr_config
    )

    # Filter and process OCR results
    words = []
    boxes = []
    confidences = []

    for i in range(len(ocr_data["text"])):
        # Skip empty text and low confidence
        text = ocr_data["text"][i].strip()
        confidence = int(ocr_data["conf"][i])

        if (
            text and confidence > confidence_threshold
        ):  # Filter low confidence detections
            words.append(text)

            # Extract bounding box coordinates
            x = ocr_data["left"][i]
            y = ocr_data["top"][i]
            w = ocr_data["width"][i]
            h = ocr_data["height"][i]

            # Convert to [x1, y1, x2, y2] format
            box = [x, y, x + w, y + h]
            boxes.append(box)
            confidences.append(confidence)

    result = {
        "image_path": str(image_path),
        "image_size": pil_image.size,
        "words": words,
        "boxes": boxes,
        "confidences": confidences,
    }

    # Save results if output directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        image_name = Path(image_path).stem
        json_path = output_path / f"{image_name}_ocr.json"

        with json_path.open("w") as f:
            json.dump(result, f, indent=2)

        print(f"OCR results saved to {json_path}")

    return result


def normalize_boxes(
    boxes: List[List[int]], image_size: Tuple[int, int]
) -> List[List[int]]:
    """
    Normalize bounding boxes to LayoutLM format (0-1000 scale).

    Args:
        boxes: List of bounding boxes in [x1, y1, x2, y2] format
        image_size: Image dimensions (width, height)

    Returns:
        Normalized bounding boxes
    """
    width, height = image_size
    normalized_boxes = []

    for box in boxes:
        x1, y1, x2, y2 = box

        # Normalize to 0-1000 scale (LayoutLM standard)
        norm_x1 = int((x1 / width) * 1000)
        norm_y1 = int((y1 / height) * 1000)
        norm_x2 = int((x2 / width) * 1000)
        norm_y2 = int((y2 / height) * 1000)

        # Ensure coordinates are within bounds
        norm_x1 = max(0, min(1000, norm_x1))
        norm_y1 = max(0, min(1000, norm_y1))
        norm_x2 = max(0, min(1000, norm_x2))
        norm_y2 = max(0, min(1000, norm_y2))

        normalized_boxes.append([norm_x1, norm_y1, norm_x2, norm_y2])

    return normalized_boxes


def preprocess_image(
    image_path: str, target_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Preprocess image for LayoutLM vision encoder.

    Args:
        image_path: Path to the input image
        target_size: Target image size for resizing

    Returns:
        Preprocessed image array
    """
    # Load and resize image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize while maintaining aspect ratio
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize image
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create canvas and center the image
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized_image

    # Normalize pixel values
    canvas = canvas.astype(np.float32) / 255.0

    return canvas


def assign_entity_labels(words: List[str], boxes: List[List[int]], image_size: Tuple[int, int]) -> List[int]:
    """
    Assign entity labels based on document structure and content patterns.
    
    Label mapping:
    0: "O" - Outside any entity
    1: "B-HEADER" - Beginning of header
    2: "I-HEADER" - Inside header  
    3: "B-QUESTION" - Beginning of question
    4: "I-QUESTION" - Inside question
    5: "B-ANSWER" - Beginning of answer
    6: "I-ANSWER" - Inside answer
    
    Args:
        words: List of OCR words
        boxes: List of bounding boxes
        image_size: Image dimensions
        
    Returns:
        List of entity labels for each word
    """
    import re
    
    labels = [0] * len(words)  # Default to 'O'
    width, height = image_size
    
    for i, (word, box) in enumerate(zip(words, boxes, strict=False)):
        x1, y1, x2, y2 = box
        
        # Calculate relative position in document
        y_position = y1 / height  # 0.0 = top, 1.0 = bottom
        
        # Header detection patterns
        header_patterns = [
            r'^(INVOICE|RECEIPT|FORM|STATEMENT|ORDER|APPLICATION)$',
            r'^(Invoice|Receipt|Form|Statement|Order|Application)$',
            r'^(#:|Number:|ID:|Reference:?)$',
            r'^(Date:|Time:|Created:?)$',
        ]
        
        # Question/Field label patterns  
        question_patterns = [
            r'^(Name:?|Customer:?|Client:?)$',
            r'^(Address:?|Location:?)$', 
            r'^(Phone:?|Tel:?|Mobile:?)$',
            r'^(Email:?|E-mail:?)$',
            r'^(Description:?|Details:?)$',
            r'^(Amount:?|Total:?|Price:?)$',
            r'^(Tax:?|VAT:?)$',
            r'^(Quantity:?|Qty:?)$',
            r'^(Payment:?|Method:?)$',
            r'^(Balance:?|Due:?)$',
            r'^(Date:?|Time:?|Created:?)$',
            # Handle compound words like "Name:Alice", "Phone555-123"
            r'^(Name:|Address\d*|Phone\d*):?',
        ]
        
        # Answer/Value patterns
        answer_patterns = [
            r'^\$[\d,]+\.?\d*$',  # Money amounts
            r'^\d{4}-\d{2}-\d{2}$',  # Dates
            r'^\d{3}-\d{3}-\d{4}$',  # Phone numbers
            r'^\w+@\w+\.\w+$',  # Email addresses
            r'^INV-\d+$',  # Invoice numbers
            r'^[A-Z]{2,}\s+(Corp|LLC|Inc|Ltd|Co|Solutions)$',  # Company names
            # Names and addresses in compound format
            r'^Name:[A-Z][a-z]+$',  # Name:Alice
            r'^Address\d*[A-Z][a-z]+$',  # Address654Maple
            r'^Phone\d*[-\d]+$',  # Phone555-9291
        ]
        
        # Check for headers (typically at top of document or large text)
        is_header = False
        if y_position < 0.2:  # Top 20% of document
            for pattern in header_patterns:
                if re.match(pattern, word, re.IGNORECASE):
                    is_header = True
                    break
                    
        # Check for questions/field labels (typically followed by colon or answer)
        is_question = False
        for pattern in question_patterns:
            if re.match(pattern, word, re.IGNORECASE):
                is_question = True
                break
                
        # Check for answers/values
        is_answer = False
        for pattern in answer_patterns:
            if re.match(pattern, word):
                is_answer = True
                break
                
        # Additional heuristics for answers
        if not is_answer and i > 0:
            prev_word = words[i-1]
            # If previous word was a question pattern, this might be an answer
            for pattern in question_patterns:
                if re.match(pattern, prev_word, re.IGNORECASE):
                    # Check if this looks like answer data
                    if (len(word) > 1 and 
                        (word.isalnum() or '@' in word or '.' in word or '-' in word)):
                        is_answer = True
                        break
        
        # Assign labels with BIO tagging
        if is_header:
            # Check if previous word was also a header
            if i > 0 and labels[i-1] in [1, 2]:  # Previous was B-HEADER or I-HEADER
                labels[i] = 2  # I-HEADER
            else:
                labels[i] = 1  # B-HEADER
                
        elif is_question:
            # Check if previous word was also a question
            if i > 0 and labels[i-1] in [3, 4]:  # Previous was B-QUESTION or I-QUESTION
                labels[i] = 4  # I-QUESTION
            else:
                labels[i] = 3  # B-QUESTION
                
        elif is_answer:
            # Check if previous word was also an answer
            if i > 0 and labels[i-1] in [5, 6]:  # Previous was B-ANSWER or I-ANSWER
                labels[i] = 6  # I-ANSWER
            else:
                labels[i] = 5  # B-ANSWER
                
    return labels


def create_training_annotations(
    ocr_results: Dict[str, Any],
    labels: Optional[List[int]] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create training annotations in LayoutLM format with intelligent entity labeling.

    Args:
        ocr_results: OCR results from extract_text_and_boxes
        labels: Optional labels for each word (for supervised training)
        output_path: Optional path to save annotations

    Returns:
        Training annotation dictionary
    """
    words = ocr_results["words"]
    boxes = ocr_results["boxes"]
    image_size = ocr_results["image_size"]

    # Normalize bounding boxes
    normalized_boxes = normalize_boxes(boxes, image_size)

    # Create intelligent labels if not provided
    if labels is None:
        labels = assign_entity_labels(words, boxes, image_size)

    # Ensure labels match words count
    if len(labels) != len(words):
        print(
            f"Warning: Labels count ({len(labels)}) doesn't match words count ({len(words)})"
        )
        labels = labels[: len(words)] + [0] * max(0, len(words) - len(labels))

    annotation = {
        "id": Path(ocr_results["image_path"]).stem,
        "image_path": ocr_results["image_path"],
        "image_size": image_size,
        "words": words,
        "boxes": normalized_boxes,
        "labels": labels,
    }

    # Save annotation if path provided
    if output_path:
        output_file = Path(output_path)
        with output_file.open("w") as f:
            json.dump(annotation, f, indent=2)
        print(f"Annotation saved to {output_file}")

    return annotation


def batch_process_images(
    image_dir: str, output_dir: str, image_extensions: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Process multiple images in a directory.

    Args:
        image_dir: Directory containing images
        output_dir: Directory to save processed results
        image_extensions: List of valid image file extensions

    Returns:
        List of processing results
    """
    if image_extensions is None:
        image_extensions = [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]
    image_path = Path(image_dir)
    output_path = Path(output_dir)

    # Clear existing output directory to ensure clean results
    if output_path.exists():
        import shutil

        shutil.rmtree(output_path)
        print(f"🗑️  Cleared existing preprocessing output directory: {output_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    results = []

    # Find all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_path.glob(f"*{ext}"))
        image_files.extend(image_path.glob(f"*{ext.upper()}"))

    print(f"Found {len(image_files)} images to process")

    for image_file in image_files:
        try:
            print(f"Processing {image_file.name}...")

            # Extract text and boxes
            ocr_result = extract_text_and_boxes(str(image_file))

            # Create annotation
            annotation = create_training_annotations(
                ocr_result,
                output_path=str(output_path / f"{image_file.stem}_annotation.json"),
            )

            results.append(annotation)

        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            continue

    print(f"Successfully processed {len(results)} images")
    return results


if __name__ == "__main__":
    # Example usage
    print("Document preprocessing utilities ready!")
    print("Use extract_text_and_boxes() to process individual images")
    print("Use batch_process_images() to process multiple images")
