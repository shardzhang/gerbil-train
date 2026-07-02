"""Custom exception hierarchy for gerbil-train."""


class GerbilTrainError(Exception):
    """Base exception for all gerbil-train errors."""


class ConfigError(GerbilTrainError):
    """Invalid model or training configuration."""


class FieldNotFoundError(GerbilTrainError):
    """A required field is missing from config or data."""


class FieldValidationError(ConfigError):
    """Field configuration is invalid (e.g. mismatched emb_size)."""


class DataError(GerbilTrainError):
    """Invalid or corrupt data (file not found, missing features)."""


class FeatureParsingError(DataError):
    """Failed to parse a feature from a TFRecord example."""


class OptimizerConfigError(ConfigError):
    """Invalid optimizer parameters."""


class ModelInitError(GerbilTrainError):
    """Model initialization failed (e.g. field validation)."""
