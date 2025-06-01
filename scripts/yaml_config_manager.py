"""
LayoutLM YAML Configuration Manager
Handles YAML configuration files with environment variable substitution and validation
for LayoutLM document understanding project.
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class ConfigValidationError(Exception):
    """Exception raised for configuration validation errors."""

    message: str
    config_path: Optional[str] = None
    key_path: Optional[str] = None


class YAMLConfigManager:
    """
    LayoutLM Configuration Manager that supports YAML with comments
    and environment variable substitution for production deployment.

    Supports syntax: ${ENV_VAR:default_value} in YAML files
    Handles configurable data/model directories and offline model usage.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self._config = None
        self._env_pattern = re.compile(r"\$\{([^}:]+)(?::([^}]*))?\}")

        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

    def load_config(self, validate: bool = True) -> Dict[str, Any]:
        """
        Load configuration with environment variable substitution and validation.

        Args:
            validate: Whether to validate the configuration after loading

        Returns:
            Loaded and processed configuration dictionary
        """
        if self._config is None:
            self._config = self._load_and_substitute()

            if validate:
                self._validate_config()

        return self._config

    def _load_and_substitute(self) -> Dict[str, Any]:
        """Load YAML and substitute environment variables"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        # Load raw YAML
        with self.config_path.open("r") as f:
            config_text = f.read()

        # Substitute environment variables
        substituted_text = self._substitute_env_vars(config_text)

        # Parse YAML
        try:
            config = yaml.safe_load(substituted_text)
            return self._convert_types(config)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}") from e

    def _substitute_env_vars(self, text: str) -> str:
        """Replace ${ENV_VAR:default} patterns with environment values"""

        def replace_env_var(match):
            env_var = match.group(1)
            default_value = match.group(2) if match.group(2) is not None else ""

            # Get environment variable or use default
            value = os.getenv(env_var, default_value)

            # Handle missing required variables (no default provided)
            if not value and match.group(2) is None:
                raise ValueError(f"Required environment variable {env_var} not set")

            return value

        return self._env_pattern.sub(replace_env_var, text)

    def _validate_config(self) -> None:
        """Validate the LayoutLM configuration structure and required fields."""
        required_sections = [
            "environment",
            "data",
            "model",
            "training",
            "inference",
            "labels",
        ]

        for section in required_sections:
            if section not in self._config:
                raise ConfigValidationError(
                    f"Required configuration section missing: {section}",
                    config_path=str(self.config_path),
                    key_path=section,
                )

        # Validate specific configuration values
        self._validate_paths()
        self._validate_model_config()
        self._validate_training_config()
        self._validate_data_config()
        self._validate_labels_config()

    def _validate_paths(self) -> None:
        """Validate that required directories exist or can be created."""
        paths_to_check = [
            "environment.data_dir",
            "environment.model_dir",
            "environment.hf_home",
        ]

        for path_key in paths_to_check:
            path_value = self.get(path_key)
            if path_value:
                path = Path(path_value)
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    logger.debug(f"Validated path: {path}")
                except Exception as e:
                    logger.warning(f"Cannot create directory {path}: {e}")

    def _validate_model_config(self) -> None:
        """Validate LayoutLM model configuration."""
        model_config = self._config.get("model", {})

        # Check num_labels
        num_labels = model_config.get("num_labels", 0)
        if not isinstance(num_labels, int) or num_labels <= 0:
            raise ConfigValidationError(
                "model.num_labels must be a positive integer",
                key_path="model.num_labels",
            )

        # Check max_seq_length
        max_seq_length = model_config.get("max_seq_length", 0)
        if not isinstance(max_seq_length, int) or max_seq_length <= 0:
            raise ConfigValidationError(
                "model.max_seq_length must be a positive integer",
                key_path="model.max_seq_length",
            )

        # Validate local model path if using local model
        if model_config.get("use_local_model", False):
            local_path = model_config.get("local_model_path")
            if not local_path or not Path(local_path).exists():
                logger.warning(f"Local model path does not exist: {local_path}")

    def _validate_training_config(self) -> None:
        """Validate training configuration."""
        training_config = self._config.get("training", {})

        # Validate numeric parameters
        numeric_params = {
            "num_epochs": (1, 1000),
            "batch_size": (1, 1024),
            "learning_rate": (1e-6, 1.0),
            "warmup_steps": (0, 10000),
        }

        for param, (min_val, max_val) in numeric_params.items():
            value = training_config.get(param)
            if value is not None:
                if not isinstance(value, (int, float)) or not (
                    min_val <= value <= max_val
                ):
                    raise ConfigValidationError(
                        f"training.{param} must be between {min_val} and {max_val}",
                        key_path=f"training.{param}",
                    )

    def _validate_data_config(self) -> None:
        """Validate data configuration."""
        data_config = self._config.get("data", {})

        # Validate train_split
        train_split = data_config.get("train_split", 0.8)
        if not isinstance(train_split, (int, float)) or not (0.0 < train_split < 1.0):
            raise ConfigValidationError(
                "data.train_split must be between 0.0 and 1.0",
                key_path="data.train_split",
            )

    def _validate_labels_config(self) -> None:
        """Validate labels configuration."""
        labels_config = self._config.get("labels", {})
        mapping = labels_config.get("mapping", {})

        if not mapping:
            raise ConfigValidationError(
                "labels.mapping cannot be empty", key_path="labels.mapping"
            )

        # Validate that label IDs are consecutive integers starting from 0
        label_ids = sorted([int(k) for k in mapping.keys()])
        expected_ids = list(range(len(label_ids)))

        if label_ids != expected_ids:
            raise ConfigValidationError(
                "Label IDs must be consecutive integers starting from 0",
                key_path="labels.mapping",
            )

    def _convert_types(self, obj: Any) -> Any:
        """Convert string values to appropriate types"""
        if isinstance(obj, dict):
            return {key: self._convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_types(item) for item in obj]
        elif isinstance(obj, str):
            # Convert boolean strings
            if obj.lower() in ("true", "false"):
                return obj.lower() == "true"

            # Convert None/null strings
            if obj.lower() in ("none", "null", ""):
                return None

            # Convert numeric strings
            try:
                # Try integer first
                if "." not in obj and "e" not in obj.lower():
                    return int(obj)
                else:
                    return float(obj)
            except ValueError:
                return obj

        return obj

    def _apply_environment_overrides(
        self, config: Dict[str, Any], environment: str
    ) -> Dict[str, Any]:
        """Apply environment-specific configuration overrides"""
        if environment not in config.get("environments", {}):
            return config

        # Create a deep copy of the base config
        import copy

        merged_config = copy.deepcopy(config)

        # Remove the environments section from the final config
        if "environments" in merged_config:
            env_overrides = merged_config.pop("environments")[environment]
            merged_config = self._deep_merge(merged_config, env_overrides)

        return merged_config

    def _deep_merge(
        self, base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key_path: Dot-separated path to the configuration key (e.g., 'model.num_labels')
            default: Default value if key is not found

        Returns:
            Configuration value or default
        """
        config = self.load_config(validate=False)

        try:
            keys = key_path.split(".")
            value = config
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key_path: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.

        Args:
            key_path: Dot-separated path to the configuration key
            value: Value to set
        """
        if self._config is None:
            self._config = self._load_and_substitute()

        keys = key_path.split(".")
        current = self._config

        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set the final key
        current[keys[-1]] = value

    def get_section(self, section_name: str) -> Dict[str, Any]:
        """
        Get an entire configuration section.

        Args:
            section_name: Name of the configuration section

        Returns:
            Configuration section dictionary
        """
        return self.get(section_name, {})

    def get_model_config(self) -> Dict[str, Any]:
        """Get LayoutLM model configuration."""
        return self.get_section("model")

    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.get_section("training")

    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.get_section("data")

    def get_inference_config(self) -> Dict[str, Any]:
        """Get inference configuration."""
        return self.get_section("inference")

    def get_label_mapping(self) -> Dict[int, str]:
        """Get label mapping for LayoutLM."""
        mapping = self.get("labels.mapping", {})
        return {int(k): v for k, v in mapping.items()}

    def get_label_colors(self) -> Dict[str, str]:
        """Get label colors for visualization."""
        return self.get("labels.colors", {})

    def validate_environment_variables(self) -> List[str]:
        """
        Validate that required environment variables are set.

        Returns:
            List of missing environment variable warnings
        """
        warnings_list = []

        # Check for critical environment variables
        critical_env_vars = {
            "DATADIR": "environment.data_dir",
            "MODELDIR": "environment.model_dir",
            "HF_HOME": "environment.hf_home",
        }

        # DATADIR is now required for production
        if not os.getenv("DATADIR"):
            warnings_list.append(
                "DATADIR environment variable is required for production deployment"
            )

        for env_var, _config_path in critical_env_vars.items():
            if not os.getenv(env_var):
                warnings_list.append(
                    f"Environment variable {env_var} not set, using default from config"
                )

        return warnings_list

    def create_directories(self) -> None:
        """Create necessary directories based on configuration."""
        directories_to_create = [
            self.get("environment.data_dir"),
            self.get("environment.model_dir"),
            self.get("environment.hf_home"),
            self.get("environment.output_dir"),
            self.get("environment.log_dir"),
            self.get("data.raw_data_dir"),
            self.get("data.processed_data_dir"),
            self.get("model.checkpoint_dir"),
            self.get("model.final_model_dir"),
        ]

        for directory in directories_to_create:
            if directory:
                path = Path(directory)
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    logger.debug(f"Created directory: {path}")
                except Exception as e:
                    logger.warning(f"Could not create directory {path}: {e}")

    def is_offline_mode(self) -> bool:
        """Check if the system is configured for offline operation."""
        return self.get("production.offline_mode")

    def should_use_local_model(self) -> bool:
        """Check if local model should be used instead of downloading."""
        return self.get("model.use_local_model") or self.is_offline_mode()

    def get_hf_cache_dir(self) -> str:
        """Get the Hugging Face cache directory."""
        return self.get("environment.hf_home")

    def export_to_json(self, output_path: str) -> str:
        """Export processed configuration to JSON format."""
        config = self.load_config()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with output_file.open("w") as f:
            json.dump(config, f, indent=2)

        return str(output_file)

    def save_processed_config(self, output_path: str) -> None:
        """Save the processed configuration (with environment variables substituted) to a file."""
        config = self.load_config()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

        logger.info(f"Processed configuration saved to {output_path}")

    def reload(self) -> Dict[str, Any]:
        """Reload configuration from file."""
        self._config = None
        return self.load_config()

    def print_summary(self) -> None:
        """Print a summary of the current configuration."""
        print("=== LayoutLM Configuration Summary ===")
        print(f"Config file: {self.config_path}")
        print(f"Data directory: {self.get('environment.data_dir')}")
        print(f"Model directory: {self.get('environment.model_dir')}")
        print(f"HF Home: {self.get('environment.hf_home')}")
        print(f"Model: {self.get('model.name')}")
        print(f"Use local model: {self.should_use_local_model()}")
        print(f"Offline mode: {self.is_offline_mode()}")
        print(f"Batch size: {self.get('training.batch_size')}")
        print(f"Learning rate: {self.get('training.learning_rate')}")
        print(f"Number of labels: {self.get('model.num_labels')}")
        print("=" * 40)


def load_config(config_path: Union[str, Path] = None) -> YAMLConfigManager:
    """
    Convenience function to load LayoutLM configuration.

    Args:
        config_path: Path to configuration file. If None, uses default path.

    Returns:
        YAMLConfigManager instance
    """
    if config_path is None:
        # Default config path relative to this script
        script_dir = Path(__file__).parent
        config_path = script_dir.parent / "config" / "config.yaml"

    return YAMLConfigManager(config_path)


def setup_environment_for_production():
    """Setup environment variables for production deployment."""
    required_env_vars = {
        "DATADIR": "/data/layoutlm",
        "MODELDIR": "/models/layoutlm",
        "HF_HOME": "/cache/huggingface",
    }

    print("Production Environment Setup")
    print("=" * 40)
    print("Required environment variables:")

    for env_var, example_path in required_env_vars.items():
        current_value = os.getenv(env_var)
        if current_value:
            print(f"✅ {env_var}: {current_value}")
        else:
            print(f"❌ {env_var}: Not set (example: {example_path})")

    print("\nTo set environment variables:")
    print("export DATADIR=/path/to/data")
    print("export MODELDIR=/path/to/models")
    print("export HF_HOME=/path/to/hf/cache")


if __name__ == "__main__":
    print("LayoutLM YAML Configuration Manager\n")

    try:
        # Load configuration
        config = load_config()
        config.print_summary()

        # Validate environment variables
        warnings_list = config.validate_environment_variables()
        if warnings_list:
            print("\nEnvironment Warnings:")
            for warning in warnings_list:
                print(f"  - {warning}")

        # Create directories
        config.create_directories()

        # Test specific configurations
        print("\nConfiguration Tests:")
        print(f"Model config: {config.get_model_config()}")
        print(f"Data config: {config.get_data_config()}")
        print(f"Label mapping: {config.get_label_mapping()}")
        print(f"Use local model: {config.should_use_local_model()}")
        print(f"Offline mode: {config.is_offline_mode()}")
        print(f"HF cache dir: {config.get_hf_cache_dir()}")

        print("\n✅ Configuration loaded and validated successfully!")

    except FileNotFoundError:
        print("❌ Configuration file not found")
        print("💡 Make sure config/config.yaml exists")
        setup_environment_for_production()
    except ConfigValidationError as e:
        print(f"❌ Configuration validation error: {e.message}")
        if e.key_path:
            print(f"   Key path: {e.key_path}")
    except Exception as e:
        print(f"❌ Error: {e}")
        setup_environment_for_production()
