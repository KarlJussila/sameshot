"""Find copies of the same photo across crops, edits, and watermarks."""

__version__ = '0.1.0'

from .align import compare  # noqa: E402
from .scan import scan  # noqa: E402

__all__ = ['compare', 'scan']
