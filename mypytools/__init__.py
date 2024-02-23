"""
My python tools
"""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s [%(name)s] %(message)s",
)
_log = logging.getLogger(__name__)
