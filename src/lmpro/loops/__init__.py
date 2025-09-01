# File: src/lmpro/loops/__init__.py

"""
Custom training loops for advanced training strategies
"""

from .kfold_loop import KFoldLoop
from .curriculum_loop import CurriculumLoop

__all__ = [
    "KFoldLoop",
    "CurriculumLoop",
]