#!/usr/bin/env python3
"""
Model Factory Module
====================

Centralized model registry for dynamic instantiation.
Provides clean interface to instantiate models by string identifier.

Usage:
    from model_factory import ModelFactory

    factory = ModelFactory()
    model_class = factory.get_model_class("ConvTokMHSA")
    model = model_class(configs)

Supported Models:
    - Bi-LSTM: Bidirectional LSTM baseline
    - CNN: Convolutional Neural Network baseline
    - MLP: Multi-Layer Perceptron baseline
    - ConvTokMHSA: Convolutional Tokenizer + Multi-Head Self Attention
    - ConvTokSWLA: Convolutional Tokenizer + Sliding Window Local Attention
    - ConvTokMWLA: Convolutional Tokenizer + Multi-scale Window Local Attention
    - ConvTokLPLA: Convolutional Tokenizer + Local-Perception Linear Attention
    - InceptionTime: Inception-based time series classifier
    - MMK_Net: Multi-scale Multi-kernel Network
    - LMSD: Large-Mini Scale Diagnostician (proposed method)
"""

import importlib
from typing import Dict, Callable, Type


class ModelFactory:
    """
    Factory class for model instantiation.

    Maps model names to their module paths and class names for lazy loading.
    All models are expected to be in the 'Models' package.
    """

    # Registry: Model Name -> (Module Path, Class Name)
    _REGISTRY: Dict[str, tuple] = {
        # Baseline Models
        "Bi-LSTM": ("Models.Bi_LSTM", "BiLSTM"),
        "CNN": ("Models.CNN", "CNN"),
        "MLP": ("Models.MLP", "MLP"),

        # Inception-based Models
        "InceptionTime": ("Models.InceptionTime", "InceptionTime"),
        "MMK_Net": ("Models.MMK_Net", "MMK_Net"),

        # ConvTok Family (Tokenizer-based Transformers)
        "ConvTokMHSA": ("Models.ConvTokMHSA", "ConvTokMHSA"),
        "ConvTokSWLA": ("Models.ConvTokSWLA", "ConvTokSWLA"),
        "ConvTokMWLA": ("Models.ConvTokMWLA", "ConvTokMWLA"),
        "ConvTokLPLA": ("Models.ConvTokLPLA", "ConvTokLPLA"),

        # Proposed Architecture
        "LMSD": ("Models.LMSD", "LMSD"),
    }

    def __init__(self):
        """Initialize factory with empty cache."""
        self._cache: Dict[str, Type] = {}

    def _import_model(self, model_name: str) -> Type:
        """
        Dynamically import model class.

        Args:
            model_name: Registered model identifier

        Returns:
            Model class (not instance)

        Raises:
            ValueError: If model_name not registered or import fails
        """
        if model_name not in self._REGISTRY:
            available = ", ".join(self.available_models())
            raise ValueError(
                f"Model '{model_name}' not registered. "
                f"Available: [{available}]"
            )

        module_path, class_name = self._REGISTRY[model_name]

        try:
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            return model_class
        except ImportError as e:
            raise ImportError(
                f"Failed to import module '{module_path}' for model '{model_name}': {e}"
            )
        except AttributeError as e:
            raise AttributeError(
                f"Class '{class_name}' not found in '{module_path}': {e}"
            )

    def get_model_class(self, model_name: str) -> Type:
        """
        Retrieve model class by name (with caching).

        Args:
            model_name: Model identifier string

        Returns:
            Model class (constructor)
        """
        # Return cached version if available
        if model_name in self._cache:
            return self._cache[model_name]

        # Import and cache
        model_class = self._import_model(model_name)
        self._cache[model_name] = model_class
        return model_class

    def available_models(self) -> list:
        """Return list of registered model names."""
        return list(self._REGISTRY.keys())

    def register_model(self, name: str, module_path: str, class_name: str):
        """
        Register new model at runtime (for extension).

        Args:
            name: Model identifier
            module_path: Python module path (e.g., 'Models.NewModel')
            class_name: Class name within module
        """
        self._REGISTRY[name] = (module_path, class_name)
        # Invalidate cache if re-registering existing model
        self._cache.pop(name, None)


# Singleton instance for convenience
_factory = ModelFactory()


def get_model(model_name: str):
    """
    Convenience function for direct model retrieval.

    Usage:
        model_class = get_model("ConvTokMHSA")
    """
    return _factory.get_model_class(model_name)