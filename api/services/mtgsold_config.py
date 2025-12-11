"""
MTGSold feature flag utility.
Determines enablement based on the MTGSOLD_API_TOKEN variable from manapool_export.
"""

from api.services.manapool_export import MTGSOLD_API_TOKEN


def is_mtgsold_enabled() -> bool:
    """Return True when MTGSOLD_API_TOKEN is set."""
    return MTGSOLD_API_TOKEN is not None

