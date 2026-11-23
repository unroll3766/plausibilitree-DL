"""plausdl: deep-learning companion library for plausibilitree.

Turns a folder produced by plausibilitree's run_pipeline.py into
per-plot feature/label file pairs (one pair per plot) that any deep
learning framework can load. See notes.md for the design document.
"""

from .config import PrepConfig, DEFAULT_LABELS

__all__ = ["PrepConfig", "DEFAULT_LABELS"]
__version__ = "0.1.0"
