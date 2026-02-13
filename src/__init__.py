"""
Graph Fusion Experiment

A toolkit for experimenting with graph-level optimizations and operator fusion
in deep learning computation graphs.
"""

from .fusion_patterns import FusionPattern, PatternMatcher
from .graph_analyzer import GraphAnalyzer, compute_graph_stats
from .manual_fuser import ManualFuser, FusionRule, apply_fusion_rules
from .benchmark import FusionBenchmark, run_benchmark, compare_fused_vs_unfused

__version__ = "1.0.0"
__all__ = [
    # Fusion patterns
    "FusionPattern",
    "PatternMatcher",
    
    # Graph analysis
    "GraphAnalyzer",
    "compute_graph_stats",
    
    # Manual fusion
    "ManualFuser",
    "FusionRule",
    "apply_fusion_rules",
    
    # Benchmarking
    "FusionBenchmark",
    "run_benchmark",
    "compare_fused_vs_unfused",
]
