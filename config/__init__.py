"""
Configuration Module
====================
Centralized configuration management for the bot detection pipeline.
"""

from .config import Config, load_config

__all__ = [
    'Config',
    'load_config',
]
