"""Lightweight feature output validation."""


STANDARD_FEATURE_KEYS = {"feature_name", "value", "confidence", "source_model", "metadata"}


class FeatureValidator:
    def validate_node_output(self, output: dict) -> bool:
        return STANDARD_FEATURE_KEYS.issubset(output.keys())
