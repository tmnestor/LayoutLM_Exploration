# LayoutLM Production Deployment Guide

This guide covers deploying the LayoutLM document understanding system in production environments, particularly for KFP (Kubeflow Pipelines) pods without internet access.

## Overview

The application has been designed for production deployment with the following features:
- **Offline-first design** - defaults to offline operation for production
- **No internet access required** in production pods
- **External storage mounts** for data and models
- **YAML-based configuration** with environment variable substitution
- **Pre-downloaded Hugging Face models** for offline usage

## Configuration System

### YAML Configuration with Environment Variables

The application uses a YAML configuration system that supports environment variable substitution using `${VAR:default}` syntax:

```yaml
# Example from config/config.yaml
environment:
  data_dir: "${DATADIR:./data}"           # Uses DATADIR env var or ./data default
  model_dir: "${MODELDIR:./models}"       # Uses MODELDIR env var or ./models default
  hf_home: "${HF_HOME:./cache/huggingface}"  # Uses HF_HOME env var or cache default
```

### Environment Variables

Copy and customize the environment template:

```bash
cp .env.template .env
# Edit .env with your specific values
source .env
```

#### Required Production Variables

| Variable | Description | Example Value |
|----------|-------------|---------------|
| `DATADIR` | Data storage mount point (app uses DATADIR/layout_lm) | `/data` |
| `MODELDIR` | Model storage mount point | `/models/layoutlm` |
| `HF_HOME` | Hugging Face cache directory | `/cache/huggingface` |

#### Optional Override Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_NAME` | Hugging Face model identifier | `microsoft/layoutlm-base-uncased` |
| `USE_LOCAL_MODEL` | Use local model files | `true` |
| `OFFLINE_MODE` | Enable offline operation | `true` |
| `TRAIN_EPOCHS` | Training epochs | `3` |
| `TRAIN_BATCH_SIZE` | Training batch size | `8` |
| `LOG_LEVEL` | Logging level | `INFO` |

## Offline Model Setup

### 1. Pre-download Models

On a machine with internet access, download the required models:

```bash
# Set up environment
export HF_HOME=/path/to/cache
export MODELDIR=/path/to/models

# Download LayoutLM model
python -c "
from transformers import LayoutLMForTokenClassification, LayoutLMTokenizer
model = LayoutLMForTokenClassification.from_pretrained('microsoft/layoutlm-base-uncased')
tokenizer = LayoutLMTokenizer.from_pretrained('microsoft/layoutlm-base-uncased')
model.save_pretrained('${MODELDIR}/layoutlm-base-uncased', local_files_only=True)
tokenizer.save_pretrained('${MODELDIR}/layoutlm-base-uncased', local_files_only=True)
print('Model downloaded and saved locally')
"
```

### 2. Package Models for Deployment

Create a model package:

```bash
# Create model archive
tar -czf layoutlm-models.tar.gz -C ${MODELDIR} .

# Or create container image with models
cat > Dockerfile.models <<EOF
FROM scratch
COPY models/ /models/
EOF

docker build -f Dockerfile.models -t layoutlm-models:latest .
```

## KFP/Kubernetes Deployment

### 1. Create ConfigMap for Environment

```yaml
# layoutlm-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: layoutlm-config
data:
  DATADIR: "/data"
  MODELDIR: "/models/layoutlm"
  HF_HOME: "/cache/huggingface"
  OFFLINE_MODE: "true"
  USE_LOCAL_MODEL: "true"
  LOG_LEVEL: "INFO"
```

Apply the ConfigMap:
```bash
kubectl apply -f layoutlm-config.yaml
```

### 2. Create Persistent Volumes

```yaml
# layoutlm-storage.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: layoutlm-data
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: layoutlm-models
spec:
  accessModes:
    - ReadOnlyMany
  resources:
    requests:
      storage: 50Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: layoutlm-cache
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
```

### 3. KFP Pipeline Component

```python
# layoutlm-component.py
from kfp import dsl

@dsl.component(
    base_image="your-registry/layoutlm:latest",
    packages_to_install=[]  # All packages should be in base image
)
def layoutlm_training(
    config_path: str = "/config/config.yaml",
    data_mount: str = "/data/layoutlm",
    model_mount: str = "/models/layoutlm",
    output_mount: str = "/output/layoutlm"
):
    import subprocess
    import os
    
    # Set environment variables
    os.environ['DATADIR'] = data_mount  # App will use data_mount/layout_lm
    os.environ['MODELDIR'] = model_mount
    os.environ['HF_HOME'] = f"{model_mount}/cache"
    os.environ['OFFLINE_MODE'] = 'true'
    
    # Run training
    cmd = [
        "python", "/app/scripts/train.py",
        "--config", config_path,
        "--offline"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Training failed: {result.stderr}")
    
    print("Training completed successfully")
    print(result.stdout)

# Pipeline definition
@dsl.pipeline(name="layoutlm-training-pipeline")
def layoutlm_pipeline():
    training_task = layoutlm_training()
    
    # Mount volumes
    training_task.add_volume(
        k8s.V1Volume(
            name="data-volume",
            persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
                claim_name="layoutlm-data"
            )
        )
    ).add_volume_mount(
        k8s.V1VolumeMount(
            name="data-volume",
            mount_path="/data"
        )
    )
    
    training_task.add_volume(
        k8s.V1Volume(
            name="model-volume",
            persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
                claim_name="layoutlm-models"
            )
        )
    ).add_volume_mount(
        k8s.V1VolumeMount(
            name="model-volume",
            mount_path="/models/layoutlm",
            read_only=True
        )
    )
    
    # Add environment from ConfigMap
    training_task.add_env_from(
        k8s.V1EnvFromSource(
            config_map_ref=k8s.V1ConfigMapEnvSource(
                name="layoutlm-config"
            )
        )
    )
```

## Container Image

### Dockerfile for Production

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY scripts/ ./scripts/
COPY config/ ./config/
COPY notebooks/ ./notebooks/

# Create directories for mounts
RUN mkdir -p /data/layoutlm /models/layoutlm /cache/huggingface /output/layoutlm

# Set environment variables
ENV PYTHONPATH=/app
ENV DATADIR=/data/layoutlm
ENV MODELDIR=/models/layoutlm
ENV HF_HOME=/cache/huggingface

# Default command
CMD ["python", "scripts/train.py", "--config", "config/config.yaml"]
```

### Build and Push Image

```bash
# Build image
docker build -t your-registry/layoutlm:latest .

# Push to registry
docker push your-registry/layoutlm:latest
```

## Usage Examples

### Training with Custom Configuration

```bash
# Local development with online models  
DATADIR=/local MODELDIR=/local/models python scripts/train.py --online

# Production with offline mode (default)
DATADIR=/data python scripts/train.py --config config/config.yaml

# Override specific settings with offline mode (default)
DATADIR=/data python scripts/train.py \
    --config config/config.yaml \
    --num_epochs 5 \
    --batch_size 16
```

### Inference in Production

```bash
# Run inference on documents
python scripts/inference.py \
    --config config/config.yaml \
    --input_path /data/layoutlm/documents \
    --batch_process \
    --visualize
```

### Configuration Validation

```bash
# Test configuration
python scripts/yaml_config_manager.py

# Validate environment setup
python -c "
from scripts.yaml_config_manager import load_config
config = load_config()
config.print_summary()
warnings = config.validate_environment_variables()
for w in warnings: print(f'WARNING: {w}')
"
```

## Monitoring and Logging

### Log Configuration

Set logging levels via environment variables:

```bash
# Debug logging for development
export LOG_LEVEL=DEBUG

# Production logging
export LOG_LEVEL=INFO
```

### Health Checks

Add health check endpoints for monitoring:

```python
# health_check.py
def health_check():
    try:
        from scripts.yaml_config_manager import load_config
        config = load_config()
        
        # Check directories exist
        import os
        data_dir = config.get('environment.data_dir')
        model_dir = config.get('environment.model_dir')
        
        if not os.path.exists(data_dir):
            return False, f"Data directory missing: {data_dir}"
        
        if not os.path.exists(model_dir):
            return False, f"Model directory missing: {model_dir}"
        
        return True, "All systems operational"
    
    except Exception as e:
        return False, f"Health check failed: {e}"

if __name__ == "__main__":
    healthy, message = health_check()
    print(f"Status: {'HEALTHY' if healthy else 'UNHEALTHY'}")
    print(f"Message: {message}")
    exit(0 if healthy else 1)
```

## Troubleshooting

### Common Issues

1. **Model Download Errors**
   ```bash
   # Check HF_HOME is writable
   ls -la $HF_HOME
   
   # Clear cache if corrupted
   rm -rf $HF_HOME/*
   ```

2. **Permission Issues**
   ```bash
   # Fix directory permissions
   chown -R app:app /data/layoutlm /models/layoutlm
   chmod -R 755 /data/layoutlm /models/layoutlm
   ```

3. **Configuration Validation Errors**
   ```bash
   # Validate YAML syntax
   python -c "import yaml; yaml.safe_load(open('config/config.yaml'))"
   
   # Test environment variable substitution
   python scripts/yaml_config_manager.py
   ```

4. **Memory Issues**
   ```bash
   # Reduce batch size in config
   export TRAIN_BATCH_SIZE=4
   export INFERENCE_BATCH_SIZE=8
   ```

### Debug Mode

Enable debug logging for troubleshooting:

```bash
export LOG_LEVEL=DEBUG
python scripts/train.py --config config/config.yaml
```

## Security Considerations

1. **Secrets Management**: Use Kubernetes secrets for sensitive configuration
2. **Network Policies**: Restrict network access in production pods
3. **RBAC**: Use minimal required permissions for service accounts
4. **Image Security**: Scan container images for vulnerabilities
5. **Data Encryption**: Encrypt data at rest and in transit

## Performance Optimization

1. **Resource Limits**: Set appropriate CPU/memory limits
2. **Storage**: Use high-performance storage for model/data access
3. **Batch Processing**: Optimize batch sizes for available memory
4. **Model Quantization**: Consider model optimization for inference
5. **Caching**: Use persistent volumes for model caching

This deployment guide ensures your LayoutLM system runs reliably in production environments with proper configuration management and offline capabilities.