"""
Rate limiting configuration for the API.

Provides a shared Limiter instance that can be used across all route modules.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

# Shared rate limiter instance
# Using remote address (IP) as the key for rate limiting
limiter = Limiter(key_func=get_remote_address)
