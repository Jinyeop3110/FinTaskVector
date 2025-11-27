"""Dataset loaders for Financial QA evaluation."""

from .finqa import FinQADataset, download_finqa

__all__ = [
    "FinQADataset",
    "download_finqa",
]
