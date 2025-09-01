# File: src/lmpro/modules/vision/__init__.py

"""
Vision modules for image classification and segmentation
"""

from .classifier import VisionClassifier
from .segmenter import VisionSegmenter

__all__ = [
    "VisionClassifier",
    "VisionSegmenter",
]