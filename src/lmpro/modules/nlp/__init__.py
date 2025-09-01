# File: src/lmpro/modules/nlp/__init__.py

"""
NLP modules for text classification and language modeling
"""

from .char_lm import CharacterLanguageModel
from .sentiment import SentimentClassifier

__all__ = [
    "CharacterLanguageModel",
    "SentimentClassifier",
]