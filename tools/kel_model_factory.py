#!/usr/bin/env python3
"""
KEL Model Factory
=================
Specialized factory for Knowledge Distillation-based Explainability Learning (KEL).
Handles Keyness-aware model variants that output temporal keyness vectors for interpretability.

Supports:
- Teacher models: Standard architectures (from model_factory)
- Student models: KeynessV2 variants with is_only_time parameter for keyness extraction
"""

import importlib
from typing import Dict, Tuple, Type


class KELStudentFactory:
    """
    Factory for KeynessV2 student models.
    These models return both predictions and temporal keyness vectors for distillation.
    """

    # Registry: Model Name -> (Module Path, Class Name)
    _REGISTRY: Dict[str, Tuple[str, str]] = {
        # KeynessV2 ConvTok variants
        "ConvTokMHSA": ("Models.kel_models.KEL_ConvTokMHSA", "KEL_ConvTokMHSA"),
        "TimeConvTokMHSA": ("Models.kel_models.KEL_ConvTokMHSA", "TimeKEL_ConvTokMHSA"),

        # KeynessV2 MMK Net
        "MMK_Net": ("Models.kel_models.KEL_MMK_Net", "KEL_MMK_Net"),
        "TimeMMK_Net": ("Models.kel_models.KEL_MMK_Net", "TimeKEL_MMK_Net"),

    }

    def __init__(self):
        self._cache: Dict[str, Type] = {}

    def _import_model(self, model_name: str, is_only_time: bool = True) -> Type:
        """Dynamically import student model class."""
        if model_name not in self._REGISTRY:
            available = ", ".join(self.available_models())
            raise ValueError(
                f"Student model '{model_name}' not registered. "
                f"Available: [{available}]"
            )

        module_path, class_name = self._REGISTRY[model_name]

        # Handle Time- prefixed models (is_only_time=True)
        if model_name.startswith("Time") and not is_only_time:
            # Map TimeXxx back to Xxx for non-time version
            base_name = model_name[4:]  # Remove "Time" prefix
            if base_name in self._REGISTRY:
                _, class_name = self._REGISTRY[base_name]

        try:
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            return model_class
        except ImportError as e:
            raise ImportError(f"Failed to import {module_path}: {e}")
        except AttributeError as e:
            raise AttributeError(f"Class {class_name} not found in {module_path}: {e}")

    def get_model_class(self, model_name: str, is_only_time: bool = True) -> Type:
        """
        Retrieve student model class.

        Args:
            model_name: Model identifier (e.g., "ConvTokMHSA" or "TimeConvTokMHSA")
            is_only_time: If True, returns TimeKeynessXxx variant (temporal-only keyness)

        Returns:
            Model class constructor
        """
        cache_key = f"{model_name}_{is_only_time}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        model_class = self._import_model(model_name, is_only_time)
        self._cache[cache_key] = model_class
        return model_class

    def available_models(self) -> list:
        """Return list of registered student models."""
        return list(self._REGISTRY.keys())


# Import standard factory for teacher models
from tools.model_factory import ModelFactory as KELTeacherFactory


def get_student_model(model_name: str, is_only_time: bool = True):
    """Convenience function for student model retrieval."""
    factory = KELStudentFactory()
    return factory.get_model_class(model_name, is_only_time)


def get_teacher_model(model_name: str):
    """Convenience function for teacher model retrieval (standard models)."""
    factory = KELTeacherFactory()
    return factory.get_model_class(model_name)