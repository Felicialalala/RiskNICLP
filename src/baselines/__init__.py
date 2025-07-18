"""
Baseline models for RiskNICLP-5G comparison
"""

from .graphsaint_baseline import GraphSAINTBaseline
from .tgat_baseline import TGATBaseline

__all__ = [
    'GraphSAINTBaseline',
    'TGATBaseline'
]