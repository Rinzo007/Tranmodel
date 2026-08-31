"""Compatibility entry point for the canonical route economics implementation."""
from .economics_core import calculate_annual_route_economics, REQUIRED_COST_KEYS

__all__ = ["calculate_annual_route_economics", "REQUIRED_COST_KEYS"]
