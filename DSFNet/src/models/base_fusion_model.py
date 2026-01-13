"""Compatibility shim after renaming base_fusion_model.py -> b2_fusion_model.py.
Do NOT remove this file; older code still imports models.base_fusion_model.
"""
from .b2_fusion_model import BaseFusionModel  # noqa: F401
