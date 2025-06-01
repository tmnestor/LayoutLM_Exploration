# LayoutLM Label Schema Modification Guide

A comprehensive guide for adapting the LayoutLM model to your production dataset by modifying the label schema and entity recognition patterns.

## Table of Contents

1. [Overview](#overview)
2. [Understanding the Current Schema](#understanding-the-current-schema)
3. [Analyzing Your Production Data](#analyzing-your-production-data)
4. [Designing Your Custom Schema](#designing-your-custom-schema)
5. [Implementation Steps](#implementation-steps)
6. [Pattern Recognition Customization](#pattern-recognition-customization)
7. [Configuration Updates](#configuration-updates)
8. [Testing and Validation](#testing-and-validation)
9. [Advanced Customization](#advanced-customization)
10. [Troubleshooting](#troubleshooting)

## Overview

The LayoutLM pipeline uses an intelligent entity labeling system that automatically assigns labels based on document structure and content patterns. To adapt this system to your production data, you'll need to:

1. **Analyze your document types** and identify key entities
2. **Design a custom label schema** that matches your use case
3. **Update the pattern recognition logic** to identify your entities
4. **Modify the configuration** to support your schema
5. **Retrain the model** with your custom labels

## Understanding the Current Schema

### Default Label Schema

The current system uses a 7-class BIO (Beginning-Inside-Outside) tagging scheme:

```yaml
labels:
  mapping:
    0: "O"          # Outside any entity
    1: "B-HEADER"   # Beginning of header
    2: "I-HEADER"   # Inside header
    3: "B-QUESTION" # Beginning of question/field label
    4: "I-QUESTION" # Inside question
    5: "B-ANSWER"   # Beginning of answer/field value
    6: "I-ANSWER"   # Inside answer
```

### Current Document Types

- **Invoices**: Business invoices with amounts, dates, customer info
- **Receipts**: Store receipts with items and pricing
- **Forms**: Application forms with personal information fields
- **Statements**: Account statements with balances and payments
- **Orders**: Purchase orders with product details

## Analyzing Your Production Data

### Step 1: Document Type Inventory

Create an inventory of your production document types:

```markdown
## Production Document Types

1. **Medical Records**
   - Patient information forms
   - Lab reports
   - Prescription forms
   - Insurance claims

2. **Legal Documents**
   - Contracts
   - Court filings
   - Legal briefs
   - Witness statements

3. **Financial Documents**
   - Bank statements
   - Tax forms
   - Investment reports
   - Loan applications
```

### Step 2: Entity Identification

For each document type, identify the key entities you want to extract:

```markdown
## Entity Analysis Example: Medical Records

### Patient Information Forms
- **Patient Details**: Name, DOB, SSN, Address, Phone
- **Medical History**: Conditions, medications, allergies
- **Insurance**: Provider, policy number, group number
- **Emergency Contact**: Name, relationship, phone

### Lab Reports
- **Test Information**: Test name, date, reference number
- **Patient Identity**: Name, DOB, medical record number
- **Results**: Test values, normal ranges, abnormal flags
- **Provider**: Ordering physician, lab facility
```

### Step 3: Pattern Recognition

Analyze how entities appear in your documents:

```markdown
## Pattern Examples

### Headers
- "LABORATORY REPORT"
- "PATIENT INFORMATION"
- "INSURANCE CLAIM FORM"

### Field Labels
- "Patient Name:"
- "Date of Birth:"
- "Test Results:"
- "Normal Range:"

### Values
- "John Smith" (names)
- "01/15/1985" (dates)
- "123-45-6789" (SSN)
- "150 mg/dL" (lab values)
```

## Designing Your Custom Schema

### Step 1: Define Entity Categories

Based on your analysis, define broad entity categories:

```yaml
# Example: Medical Records Schema
labels:
  mapping:
    0: "O"                    # Outside any entity
    1: "B-HEADER"            # Document headers
    2: "I-HEADER"            # Header continuation
    3: "B-PATIENT"           # Patient information
    4: "I-PATIENT"           # Patient info continuation
    5: "B-MEDICAL"           # Medical data (tests, conditions)
    6: "I-MEDICAL"           # Medical data continuation
    7: "B-PROVIDER"          # Healthcare provider info
    8: "I-PROVIDER"          # Provider info continuation
    9: "B-INSURANCE"         # Insurance information
    10: "I-INSURANCE"        # Insurance continuation
    11: "B-DATE"             # Dates and times
    12: "I-DATE"             # Date continuation
    13: "B-IDENTIFIER"       # IDs, numbers, codes
    14: "I-IDENTIFIER"       # Identifier continuation
```

### Step 2: Detailed Entity Mapping

Create detailed mappings for each entity type:

```yaml
# Example: Legal Documents Schema
labels:
  mapping:
    0: "O"                    # Outside any entity
    1: "B-DOC_TYPE"          # Document type headers
    2: "I-DOC_TYPE"          # Document type continuation
    3: "B-PARTY"             # Legal parties (plaintiff, defendant)
    4: "I-PARTY"             # Party information continuation
    5: "B-ATTORNEY"          # Attorney information
    6: "I-ATTORNEY"          # Attorney info continuation
    7: "B-CASE_INFO"         # Case numbers, court info
    8: "I-CASE_INFO"         # Case info continuation
    9: "B-DATE_LEGAL"        # Legal dates (filing, hearing)
    10: "I-DATE_LEGAL"       # Legal date continuation
    11: "B-MONETARY"         # Amounts, damages, fees
    12: "I-MONETARY"         # Monetary continuation
    13: "B-CITATION"         # Legal citations, statutes
    14: "I-CITATION"         # Citation continuation
```

### Step 3: Schema Validation

Ensure your schema follows best practices:

- **Balanced Distribution**: Aim for relatively balanced entity types
- **Clear Boundaries**: Entities should have distinct, non-overlapping patterns
- **BIO Consistency**: Every B- label should have a corresponding I- label
- **Manageable Size**: Keep total labels under 20 for training efficiency

## Implementation Steps

### Step 1: Update Configuration

Modify `config/config.yaml` with your custom schema:

```yaml
# config/config.yaml
model:
  num_labels: 15  # Update based on your schema size

labels:
  mapping:
    0: "O"
    1: "B-PATIENT"
    2: "I-PATIENT"
    3: "B-MEDICAL"
    4: "I-MEDICAL"
    5: "B-PROVIDER"
    6: "I-PROVIDER"
    7: "B-INSURANCE"
    8: "I-INSURANCE"
    9: "B-DATE"
    10: "I-DATE"
    11: "B-IDENTIFIER"
    12: "I-IDENTIFIER"
    13: "B-HEADER"
    14: "I-HEADER"

  colors:
    "O": "gray"
    "B-PATIENT": "blue"
    "I-PATIENT": "lightblue"
    "B-MEDICAL": "red"
    "I-MEDICAL": "lightcoral"
    "B-PROVIDER": "green"
    "I-PROVIDER": "lightgreen"
    "B-INSURANCE": "purple"
    "I-INSURANCE": "plum"
    "B-DATE": "orange"
    "I-DATE": "moccasin"
    "B-IDENTIFIER": "brown"
    "I-IDENTIFIER": "tan"
    "B-HEADER": "darkblue"
    "I-HEADER": "lightsteelblue"
```

### Step 2: Create Pattern Files

Create domain-specific pattern files for easy maintenance:

```python
# scripts/patterns/medical_patterns.py

MEDICAL_PATTERNS = {
    "headers": [
        r'^(LABORATORY|LAB)\s+(REPORT|RESULTS?)$',
        r'^PATIENT\s+INFORMATION$',
        r'^MEDICAL\s+(RECORD|HISTORY)$',
        r'^INSURANCE\s+(CLAIM|INFORMATION)$',
        r'^PRESCRIPTION\s+(FORM|ORDER)$',
    ],
    
    "patient_labels": [
        r'^(Patient\s+)?Name:?$',
        r'^(Date\s+of\s+)?Birth:?$',
        r'^DOB:?$',
        r'^SSN:?$',
        r'^(Patient\s+)?Address:?$',
        r'^(Phone|Tel|Mobile):?$',
        r'^(Emergency\s+)?Contact:?$',
    ],
    
    "medical_labels": [
        r'^(Test\s+)?Results?:?$',
        r'^(Normal\s+)?Range:?$',
        r'^Diagnosis:?$',
        r'^Condition:?$',
        r'^Medication:?$',
        r'^Allergies:?$',
        r'^Symptoms:?$',
    ],
    
    "provider_labels": [
        r'^(Ordering\s+)?Physician:?$',
        r'^Doctor:?$',
        r'^Provider:?$',
        r'^Facility:?$',
        r'^Laboratory:?$',
    ],
    
    "insurance_labels": [
        r'^Insurance\s+(Provider|Company):?$',
        r'^Policy\s+(Number|#):?$',
        r'^Group\s+(Number|#):?$',
        r'^Member\s+ID:?$',
    ],
    
    "patient_values": [
        r'^[A-Z][a-z]+\s+[A-Z][a-z]+$',  # Names
        r'^\d{2}/\d{2}/\d{4}$',          # Dates
        r'^\d{3}-\d{2}-\d{4}$',          # SSN
        r'^\(\d{3}\)\s*\d{3}-\d{4}$',    # Phone numbers
    ],
    
    "medical_values": [
        r'^\d+\.?\d*\s*(mg/dL|mmol/L|units?)$',  # Lab values
        r'^(Normal|Abnormal|High|Low)$',          # Result flags
        r'^\d+\.?\d*\s*-\s*\d+\.?\d*$',          # Ranges
    ],
    
    "identifier_values": [
        r'^[A-Z]{2,3}-?\d{6,}$',         # Medical record numbers
        r'^LAB-\d+$',                    # Lab reference numbers
        r'^\d{10,}$',                    # Policy numbers
    ],
    
    "date_values": [
        r'^\d{1,2}/\d{1,2}/\d{4}$',      # MM/DD/YYYY
        r'^\d{4}-\d{2}-\d{2}$',          # YYYY-MM-DD
        r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}$',  # Month DD, YYYY
    ],
}
```

### Step 3: Update Entity Recognition Logic

Modify `scripts/preprocessing.py` to use your custom patterns:

```python
# scripts/preprocessing.py

def assign_entity_labels_custom(words: List[str], boxes: List[List[int]], 
                                image_size: Tuple[int, int], 
                                domain: str = "medical") -> List[int]:
    """
    Assign entity labels based on domain-specific patterns.
    
    Args:
        words: List of OCR words
        boxes: List of bounding boxes
        image_size: Image dimensions
        domain: Domain type ("medical", "legal", "financial")
        
    Returns:
        List of entity labels for each word
    """
    import re
    from patterns import get_patterns_for_domain
    
    labels = [0] * len(words)  # Default to 'O'
    width, height = image_size
    patterns = get_patterns_for_domain(domain)
    
    for i, (word, box) in enumerate(zip(words, boxes)):
        x1, y1, x2, y2 = box
        y_position = y1 / height
        
        # Check for headers (top 20% of document)
        is_header = False
        if y_position < 0.2:
            for pattern in patterns["headers"]:
                if re.match(pattern, word, re.IGNORECASE):
                    is_header = True
                    break
        
        # Check for patient information
        is_patient_label = any(re.match(p, word, re.IGNORECASE) 
                              for p in patterns["patient_labels"])
        is_patient_value = any(re.match(p, word) 
                              for p in patterns["patient_values"])
        
        # Check for medical information
        is_medical_label = any(re.match(p, word, re.IGNORECASE) 
                              for p in patterns["medical_labels"])
        is_medical_value = any(re.match(p, word) 
                              for p in patterns["medical_values"])
        
        # ... Continue for other entity types
        
        # Assign labels with BIO tagging
        if is_header:
            if i > 0 and labels[i-1] in [13, 14]:  # Previous was B-HEADER or I-HEADER
                labels[i] = 14  # I-HEADER
            else:
                labels[i] = 13  # B-HEADER
                
        elif is_patient_label or is_patient_value:
            if i > 0 and labels[i-1] in [1, 2]:  # Previous was B-PATIENT or I-PATIENT
                labels[i] = 2  # I-PATIENT
            else:
                labels[i] = 1  # B-PATIENT
                
        # ... Continue for other entity types
        
    return labels

def get_patterns_for_domain(domain: str) -> dict:
    """Load domain-specific patterns."""
    if domain == "medical":
        from patterns.medical_patterns import MEDICAL_PATTERNS
        return MEDICAL_PATTERNS
    elif domain == "legal":
        from patterns.legal_patterns import LEGAL_PATTERNS
        return LEGAL_PATTERNS
    elif domain == "financial":
        from patterns.financial_patterns import FINANCIAL_PATTERNS
        return FINANCIAL_PATTERNS
    else:
        # Default to general patterns
        from patterns.general_patterns import GENERAL_PATTERNS
        return GENERAL_PATTERNS
```

### Step 4: Update Model Configuration

Ensure the model can handle your label count:

```python
# scripts/layoutlm_model.py

class LayoutLMTrainer:
    def __init__(self, model_name: str, num_labels: int, max_seq_length: int = 512):
        self.model_name = model_name
        self.num_labels = num_labels  # Update this to match your schema
        self.max_seq_length = max_seq_length
        # ... rest of initialization
```

## Pattern Recognition Customization

### Writing Effective Patterns

#### 1. Header Patterns
```python
# Good header patterns
header_patterns = [
    r'^(MEDICAL|PATIENT)\s+(RECORD|INFORMATION|FORM)$',
    r'^LAB(ORATORY)?\s+(REPORT|RESULTS?)$',
    r'^INSURANCE\s+(CLAIM|FORM)$',
    
    # Case-insensitive variants
    r'^(?i)(prescription|rx)\s+(form|order)$',
]
```

#### 2. Field Label Patterns
```python
# Flexible field label patterns
field_patterns = [
    r'^(Patient\s+)?Name:?$',           # "Name:" or "Patient Name:"
    r'^(Date\s+of\s+)?Birth:?$',        # "Birth:" or "Date of Birth:"
    r'^(SSN|Social\s+Security):?$',     # Various SSN formats
    r'^(Phone|Tel|Mobile)(\s+#)?:?$',   # Phone variations
]
```

#### 3. Value Patterns
```python
# Specific value patterns
value_patterns = [
    r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+$',  # Full names
    r'^\d{1,2}/\d{1,2}/\d{4}$',           # Dates MM/DD/YYYY
    r'^\d{3}-\d{2}-\d{4}$',               # SSN format
    r'^\(\d{3}\)\s*\d{3}-\d{4}$',         # Phone (xxx) xxx-xxxx
    r'^\d+\.?\d*\s*(mg/dL|mmol/L)$',      # Lab values with units
]
```

### Advanced Pattern Techniques

#### 1. Context-Aware Labeling
```python
def assign_context_aware_labels(words, boxes, image_size):
    """Use context to improve labeling accuracy."""
    labels = [0] * len(words)
    
    for i, word in enumerate(words):
        # Look at surrounding words for context
        prev_word = words[i-1] if i > 0 else ""
        next_word = words[i+1] if i < len(words)-1 else ""
        
        # If previous word was a question, this might be an answer
        if is_question_pattern(prev_word) and is_value_like(word):
            labels[i] = get_answer_label_for_question(prev_word)
            
        # If this word and next word form a compound entity
        if is_compound_entity(word + " " + next_word):
            labels[i] = get_compound_start_label(word + " " + next_word)
            labels[i+1] = get_compound_continue_label(word + " " + next_word)
    
    return labels
```

#### 2. Position-Based Rules
```python
def apply_position_rules(words, boxes, labels, image_size):
    """Apply position-based labeling rules."""
    width, height = image_size
    
    for i, (word, box) in enumerate(zip(words, boxes)):
        x1, y1, x2, y2 = box
        
        # Top-left corner often contains headers
        if y1 < height * 0.15 and x1 < width * 0.3:
            if looks_like_header(word):
                labels[i] = get_header_label()
        
        # Right-aligned text often contains values
        if x2 > width * 0.7:
            if looks_like_value(word):
                labels[i] = get_value_label()
                
        # Bottom of document often contains signatures/dates
        if y1 > height * 0.85:
            if looks_like_date(word):
                labels[i] = get_date_label()
    
    return labels
```

## Configuration Updates

### Domain-Specific Configuration

Create domain-specific configuration files:

```yaml
# config/medical_config.yaml
model:
  num_labels: 15

labels:
  domain: "medical"
  mapping:
    0: "O"
    1: "B-PATIENT"
    2: "I-PATIENT"
    # ... rest of medical schema

patterns:
  domain_specific: true
  pattern_file: "patterns/medical_patterns.py"

data:
  document_types: ["lab_reports", "patient_forms", "prescriptions", "insurance_claims"]
```

### Environment-Specific Settings

```yaml
# config/production_config.yaml
inherit_from: "medical_config.yaml"

data:
  raw_data_dir: "${PRODUCTION_DATA_DIR}/medical_docs"
  processed_data_dir: "${PRODUCTION_DATA_DIR}/processed"
  
training:
  num_epochs: 10
  batch_size: 8
  learning_rate: 3.0e-5

production:
  confidence_threshold: 0.85
  validate_predictions: true
```

## Testing and Validation

### Step 1: Pattern Testing

Create test scripts to validate your patterns:

```python
# tests/test_patterns.py

import unittest
from scripts.patterns.medical_patterns import MEDICAL_PATTERNS

class TestMedicalPatterns(unittest.TestCase):
    
    def test_header_patterns(self):
        """Test header pattern recognition."""
        test_cases = [
            ("LABORATORY REPORT", True),
            ("PATIENT INFORMATION", True),
            ("random text", False),
        ]
        
        for text, expected in test_cases:
            with self.subTest(text=text):
                matches = any(re.match(pattern, text, re.IGNORECASE) 
                            for pattern in MEDICAL_PATTERNS["headers"])
                self.assertEqual(matches, expected)
    
    def test_patient_value_patterns(self):
        """Test patient value pattern recognition."""
        test_cases = [
            ("John Smith", True),
            ("01/15/1985", True),
            ("123-45-6789", True),
            ("random123", False),
        ]
        
        for text, expected in test_cases:
            with self.subTest(text=text):
                matches = any(re.match(pattern, text) 
                            for pattern in MEDICAL_PATTERNS["patient_values"])
                self.assertEqual(matches, expected)

if __name__ == "__main__":
    unittest.main()
```

### Step 2: Label Distribution Analysis

```python
# scripts/analyze_labels.py

def analyze_label_distribution(processed_data_dir: str, config_path: str):
    """Analyze the distribution of labels in processed data."""
    from collections import Counter
    import json
    from pathlib import Path
    
    label_counts = Counter()
    total_tokens = 0
    
    # Load label mapping
    config = load_config(config_path)
    label_mapping = config.get_label_mapping()
    
    # Process all annotation files
    for annotation_file in Path(processed_data_dir).glob("*_annotation.json"):
        with open(annotation_file, 'r') as f:
            data = json.load(f)
            
        labels = data.get('labels', [])
        for label in labels:
            label_name = label_mapping.get(label, f"UNKNOWN_{label}")
            label_counts[label_name] += 1
            total_tokens += 1
    
    # Print distribution
    print(f"Total tokens: {total_tokens}")
    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        percentage = (count / total_tokens) * 100
        print(f"  {label:15}: {count:6} ({percentage:5.1f}%)")
    
    # Check for potential issues
    if label_counts.get("O", 0) / total_tokens > 0.7:
        print("\n⚠️  WARNING: High percentage of 'O' labels. Consider:")
        print("   - More specific patterns")
        print("   - Additional entity types")
        print("   - Review pattern matching logic")
    
    return label_counts

# Usage
if __name__ == "__main__":
    analyze_label_distribution(
        "/path/to/processed_data",
        "config/medical_config.yaml"
    )
```

### Step 3: Validation on Sample Data

```python
# scripts/validate_schema.py

def validate_schema_on_samples(image_dir: str, config_path: str, num_samples: int = 10):
    """Validate the schema on a small sample of documents."""
    import random
    from pathlib import Path
    from preprocessing import extract_text_and_boxes, assign_entity_labels_custom
    
    # Load config
    config = load_config(config_path)
    label_mapping = config.get_label_mapping()
    domain = config.get("labels.domain", "general")
    
    # Get sample images
    image_files = list(Path(image_dir).glob("*.png"))
    sample_files = random.sample(image_files, min(num_samples, len(image_files)))
    
    print(f"Validating schema on {len(sample_files)} sample documents...\n")
    
    for image_file in sample_files:
        print(f"Processing: {image_file.name}")
        
        # Extract OCR data
        ocr_result = extract_text_and_boxes(str(image_file))
        
        # Apply custom labeling
        labels = assign_entity_labels_custom(
            ocr_result['words'], 
            ocr_result['boxes'], 
            ocr_result['image_size'],
            domain
        )
        
        # Show results
        print("  Words and labels:")
        for word, label in zip(ocr_result['words'], labels):
            label_name = label_mapping.get(label, f"UNKNOWN_{label}")
            if label != 0:  # Only show non-O labels
                print(f"    {word:20} -> {label_name}")
        
        # Count label distribution for this document
        from collections import Counter
        doc_labels = Counter(labels)
        print("  Label distribution:")
        for label_id, count in sorted(doc_labels.items()):
            label_name = label_mapping.get(label_id, f"UNKNOWN_{label_id}")
            print(f"    {label_name:15}: {count}")
        print()

# Usage
if __name__ == "__main__":
    validate_schema_on_samples(
        "/path/to/sample/images",
        "config/medical_config.yaml",
        5
    )
```

## Advanced Customization

### Multi-Domain Support

Support multiple document domains in a single pipeline:

```python
# scripts/multi_domain_processor.py

class MultiDomainProcessor:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.domain_configs = self.load_domain_configs()
    
    def detect_document_domain(self, words: List[str]) -> str:
        """Automatically detect document domain based on content."""
        # Medical keywords
        medical_keywords = {"patient", "laboratory", "prescription", "diagnosis", "medical"}
        
        # Legal keywords  
        legal_keywords = {"plaintiff", "defendant", "court", "attorney", "case"}
        
        # Financial keywords
        financial_keywords = {"account", "balance", "transaction", "payment", "bank"}
        
        word_text = " ".join(words).lower()
        
        medical_score = sum(1 for kw in medical_keywords if kw in word_text)
        legal_score = sum(1 for kw in legal_keywords if kw in word_text)
        financial_score = sum(1 for kw in financial_keywords if kw in word_text)
        
        scores = {
            "medical": medical_score,
            "legal": legal_score, 
            "financial": financial_score
        }
        
        return max(scores, key=scores.get)
    
    def process_document(self, image_path: str) -> Dict[str, Any]:
        """Process document with domain-specific labeling."""
        # Extract OCR data
        ocr_result = extract_text_and_boxes(image_path)
        
        # Detect domain
        domain = self.detect_document_domain(ocr_result['words'])
        
        # Apply domain-specific labeling
        labels = assign_entity_labels_custom(
            ocr_result['words'],
            ocr_result['boxes'], 
            ocr_result['image_size'],
            domain
        )
        
        return {
            "domain": domain,
            "words": ocr_result['words'],
            "boxes": ocr_result['boxes'],
            "labels": labels,
            "confidence": self.calculate_domain_confidence(ocr_result['words'], domain)
        }
```

### Custom Entity Hierarchies

Create hierarchical entity structures:

```python
# scripts/hierarchical_entities.py

ENTITY_HIERARCHY = {
    "PATIENT_INFO": {
        "parent": None,
        "children": ["PERSONAL_DETAILS", "CONTACT_INFO", "MEDICAL_ID"],
        "label_mapping": {1: "B-PATIENT", 2: "I-PATIENT"}
    },
    "PERSONAL_DETAILS": {
        "parent": "PATIENT_INFO", 
        "children": ["NAME", "DOB", "SSN"],
        "label_mapping": {3: "B-PERSONAL", 4: "I-PERSONAL"}
    },
    "CONTACT_INFO": {
        "parent": "PATIENT_INFO",
        "children": ["ADDRESS", "PHONE", "EMAIL"], 
        "label_mapping": {5: "B-CONTACT", 6: "I-CONTACT"}
    },
    # ... continue hierarchy
}

def assign_hierarchical_labels(words, boxes, image_size):
    """Assign labels using hierarchical entity structure."""
    labels = [0] * len(words)
    
    # First pass: identify top-level entities
    for i, word in enumerate(words):
        for entity_type, config in ENTITY_HIERARCHY.items():
            if config["parent"] is None:  # Top-level entity
                if matches_entity_pattern(word, entity_type):
                    labels[i] = config["label_mapping"][list(config["label_mapping"].keys())[0]]
    
    # Second pass: refine with sub-entities
    for i, word in enumerate(words):
        if labels[i] != 0:  # Already has a label
            continue
            
        # Check if this word belongs to a sub-entity
        for entity_type, config in ENTITY_HIERARCHY.items():
            if config["parent"] is not None:  # Sub-entity
                if matches_entity_pattern(word, entity_type):
                    # Check if parent entity is nearby
                    if has_parent_entity_nearby(i, words, labels, config["parent"]):
                        labels[i] = config["label_mapping"][list(config["label_mapping"].keys())[0]]
    
    return labels
```

### Machine Learning-Enhanced Patterns

Combine rule-based patterns with ML for better accuracy:

```python
# scripts/ml_enhanced_patterns.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

class MLEnhancedLabeler:
    def __init__(self, model_path: str = None):
        self.tfidf = TfidfVectorizer(max_features=1000)
        self.classifier = LogisticRegression()
        self.is_trained = False
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def train_entity_classifier(self, training_data: List[Tuple[str, str]]):
        """Train ML classifier to enhance pattern matching."""
        words, labels = zip(*training_data)
        
        # Vectorize words
        X = self.tfidf.fit_transform(words)
        y = labels
        
        # Train classifier
        self.classifier.fit(X, y)
        self.is_trained = True
    
    def predict_entity_type(self, word: str) -> Tuple[str, float]:
        """Predict entity type for a word with confidence."""
        if not self.is_trained:
            return "O", 0.0
            
        X = self.tfidf.transform([word])
        prediction = self.classifier.predict(X)[0]
        confidence = max(self.classifier.predict_proba(X)[0])
        
        return prediction, confidence
    
    def assign_enhanced_labels(self, words: List[str], boxes: List[List[int]], 
                              image_size: Tuple[int, int]) -> List[int]:
        """Combine rule-based and ML approaches."""
        # Start with rule-based labels
        rule_labels = assign_entity_labels_custom(words, boxes, image_size)
        
        # Enhance with ML predictions
        enhanced_labels = rule_labels.copy()
        
        for i, word in enumerate(words):
            if rule_labels[i] == 0:  # Only enhance 'O' labels
                ml_prediction, confidence = self.predict_entity_type(word)
                
                if confidence > 0.7:  # High confidence threshold
                    enhanced_labels[i] = self.convert_label_to_id(ml_prediction)
        
        return enhanced_labels
    
    def save_model(self, model_path: str):
        """Save trained model."""
        joblib.dump({
            'tfidf': self.tfidf,
            'classifier': self.classifier,
            'is_trained': self.is_trained
        }, model_path)
    
    def load_model(self, model_path: str):
        """Load trained model."""
        model_data = joblib.load(model_path)
        self.tfidf = model_data['tfidf']
        self.classifier = model_data['classifier'] 
        self.is_trained = model_data['is_trained']
```

## Troubleshooting

### Common Issues and Solutions

#### 1. High "O" Label Percentage

**Problem**: Most tokens are labeled as "O" (outside)

**Solutions**:
```python
# Add more inclusive patterns
question_patterns = [
    r'^[A-Z][a-z]+:?$',  # Any capitalized word with optional colon
    r'^\w+\s*[:#]$',     # Any word followed by colon or hash
    r'^[A-Z][A-Z\s]+:?$', # All caps words
]

# Lower confidence thresholds
if confidence > 0.5:  # Instead of 0.7
    assign_entity_label(word)

# Add position-based heuristics  
if y_position < 0.3:  # Top 30% instead of 20%
    consider_as_header(word)
```

#### 2. Label Imbalance

**Problem**: Some entity types are rarely detected

**Solutions**:
```python
# Add synthetic data generation for rare entities
def generate_synthetic_examples(entity_type: str, count: int):
    """Generate synthetic examples for rare entity types."""
    templates = {
        "B-INSURANCE": ["Policy #{}", "Group #{}", "Member ID: {}"],
        "B-MEDICAL": ["Test: {}", "Diagnosis: {}", "Condition: {}"],
    }
    
    examples = []
    for template in templates.get(entity_type, []):
        for i in range(count):
            examples.append(template.format(f"EXAMPLE_{i:03d}"))
    
    return examples

# Weighted loss function for imbalanced data
def create_weighted_loss(label_counts: Dict[int, int]):
    """Create weighted loss to handle label imbalance."""
    total = sum(label_counts.values())
    weights = {label: total / count for label, count in label_counts.items()}
    return weights
```

#### 3. Pattern Conflicts

**Problem**: Multiple patterns match the same text

**Solutions**:
```python
# Priority-based pattern matching
PATTERN_PRIORITY = {
    "specific_patterns": 1,    # Highest priority
    "general_patterns": 2, 
    "fallback_patterns": 3,   # Lowest priority
}

def resolve_pattern_conflicts(word: str, matches: List[Tuple[str, str]]):
    """Resolve conflicts when multiple patterns match."""
    if len(matches) <= 1:
        return matches[0] if matches else None
    
    # Sort by priority
    sorted_matches = sorted(matches, key=lambda x: PATTERN_PRIORITY.get(x[1], 999))
    return sorted_matches[0]

# Context-aware disambiguation
def disambiguate_with_context(word: str, context: List[str], matches: List[str]):
    """Use surrounding words to disambiguate pattern matches."""
    context_text = " ".join(context).lower()
    
    if "patient" in context_text and "B-PATIENT" in matches:
        return "B-PATIENT"
    elif "test" in context_text and "B-MEDICAL" in matches:
        return "B-MEDICAL"
    
    return matches[0]  # Default to first match
```

#### 4. Performance Issues

**Problem**: Processing is too slow

**Solutions**:
```python
# Compiled regex patterns
import re

class CompiledPatterns:
    def __init__(self, patterns: Dict[str, List[str]]):
        self.compiled = {}
        for category, pattern_list in patterns.items():
            self.compiled[category] = [
                re.compile(pattern, re.IGNORECASE) 
                for pattern in pattern_list
            ]
    
    def match(self, text: str, category: str) -> bool:
        return any(pattern.match(text) for pattern in self.compiled[category])

# Batch processing
def process_documents_batch(image_paths: List[str], batch_size: int = 10):
    """Process documents in batches for better performance."""
    results = []
    
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        batch_results = []
        
        for image_path in batch:
            result = process_single_document(image_path)
            batch_results.append(result)
        
        results.extend(batch_results)
        
        # Optional: garbage collection between batches
        import gc
        gc.collect()
    
    return results
```

### Validation Checklist

Before deploying your custom schema:

- [ ] **Pattern Testing**: All patterns tested with sample data
- [ ] **Label Distribution**: Balanced distribution across entity types  
- [ ] **Schema Validation**: BIO tagging consistency verified
- [ ] **Performance Testing**: Processing speed acceptable for production
- [ ] **Accuracy Testing**: Labels match human annotations
- [ ] **Edge Case Handling**: Unusual document formats handled
- [ ] **Configuration Testing**: All config files valid
- [ ] **Integration Testing**: Full pipeline works end-to-end

### Next Steps

1. **Start Small**: Begin with 3-5 entity types
2. **Iterate**: Add complexity gradually based on results
3. **Monitor**: Track label distribution and accuracy metrics
4. **Refine**: Continuously improve patterns based on real data
5. **Scale**: Expand to additional document types once stable

---

## Conclusion

Modifying the label schema requires careful analysis of your production data, thoughtful pattern design, and thorough testing. Start with a simple schema and gradually add complexity as you validate the approach with your specific documents.

The intelligent entity labeling system provides a solid foundation that can be adapted to virtually any document understanding task. Focus on understanding your data patterns first, then systematically implement and test your custom schema.

For additional support, refer to the main README.md and the example notebooks that demonstrate the complete pipeline with the default schema.