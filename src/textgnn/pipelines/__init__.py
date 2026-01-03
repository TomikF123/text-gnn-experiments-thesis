"""
Training pipeline runners for different learning paradigms.

Each pipeline handles the complete flow from data loading to evaluation
for a specific type of learning (inductive, transductive, etc.).
"""

from .inductive import run_inductive_pipeline
from .transductive import run_transductive_pipeline

__all__ = [
    "run_inductive_pipeline",
    "run_transductive_pipeline",
]
