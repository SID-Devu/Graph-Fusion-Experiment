"""
Benchmark Module

Provides benchmarking utilities for comparing fused vs unfused graph execution.
"""

import time
import statistics
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple
from datetime import datetime
import logging

from .manual_fuser import ComputationGraph, ManualFuser, FusionRule, get_default_fusion_rules

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run"""
    name: str
    iterations: int
    warmup_iterations: int
    
    # Timing statistics (in milliseconds)
    mean_ms: float
    median_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    
    # Additional metrics
    total_time_ms: float
    throughput_ops: float  # Operations per second
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'iterations': self.iterations,
            'warmup_iterations': self.warmup_iterations,
            'timing': {
                'mean_ms': self.mean_ms,
                'median_ms': self.median_ms,
                'std_ms': self.std_ms,
                'min_ms': self.min_ms,
                'max_ms': self.max_ms,
                'p90_ms': self.p90_ms,
                'p95_ms': self.p95_ms,
                'p99_ms': self.p99_ms,
            },
            'total_time_ms': self.total_time_ms,
            'throughput_ops': self.throughput_ops,
            'timestamp': self.timestamp,
            'metadata': self.metadata,
        }
    
    def __str__(self) -> str:
        return (
            f"{self.name}:\n"
            f"  Mean: {self.mean_ms:.3f}ms, Median: {self.median_ms:.3f}ms\n"
            f"  Std: {self.std_ms:.3f}ms, Min: {self.min_ms:.3f}ms, Max: {self.max_ms:.3f}ms\n"
            f"  P90: {self.p90_ms:.3f}ms, P95: {self.p95_ms:.3f}ms, P99: {self.p99_ms:.3f}ms\n"
            f"  Throughput: {self.throughput_ops:.2f} ops/s"
        )


@dataclass
class ComparisonResult:
    """Results from comparing fused vs unfused execution"""
    unfused: BenchmarkResult
    fused: BenchmarkResult
    speedup: float  # fused_time / unfused_time ratio
    latency_reduction_ms: float
    latency_reduction_percent: float
    
    # Graph statistics
    original_op_count: int
    fused_op_count: int
    fusion_ratio: float  # ops eliminated / original ops
    
    # Fusion details
    fusion_stats: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'unfused': self.unfused.to_dict(),
            'fused': self.fused.to_dict(),
            'comparison': {
                'speedup': self.speedup,
                'latency_reduction_ms': self.latency_reduction_ms,
                'latency_reduction_percent': self.latency_reduction_percent,
            },
            'graph_stats': {
                'original_op_count': self.original_op_count,
                'fused_op_count': self.fused_op_count,
                'fusion_ratio': self.fusion_ratio,
            },
            'fusion_stats': self.fusion_stats,
        }
    
    def __str__(self) -> str:
        return (
            f"Fusion Benchmark Comparison\n"
            f"===========================\n"
            f"\nUnfused Graph ({self.original_op_count} ops):\n"
            f"  Mean latency: {self.unfused.mean_ms:.3f}ms\n"
            f"\nFused Graph ({self.fused_op_count} ops):\n"
            f"  Mean latency: {self.fused.mean_ms:.3f}ms\n"
            f"\nImprovement:\n"
            f"  Speedup: {self.speedup:.2f}x\n"
            f"  Latency reduction: {self.latency_reduction_ms:.3f}ms ({self.latency_reduction_percent:.1f}%)\n"
            f"  Op count reduction: {self.fusion_ratio*100:.1f}%"
        )


class GraphExecutor:
    """
    Simulated graph executor for benchmarking.
    
    In a real implementation, this would interface with actual
    inference frameworks like ONNX Runtime, TensorRT, etc.
    """
    
    def __init__(self, 
                 graph: ComputationGraph,
                 op_latencies: Optional[Dict[str, float]] = None):
        """
        Initialize executor.
        
        Args:
            graph: Computation graph to execute
            op_latencies: Optional dict mapping op_type to latency in ms
        """
        self.graph = graph
        self.op_latencies = op_latencies or self._default_latencies()
        self._execution_order = graph.topological_sort()
    
    def _default_latencies(self) -> Dict[str, float]:
        """Default operator latencies (in milliseconds)"""
        return {
            # Compute-heavy ops
            'Conv': 2.0,
            'MatMul': 1.5,
            'Gemm': 1.8,
            'ConvTranspose': 2.5,
            
            # Normalization
            'BatchNormalization': 0.3,
            'LayerNormalization': 0.4,
            'InstanceNorm': 0.35,
            
            # Activations
            'Relu': 0.05,
            'Sigmoid': 0.08,
            'Tanh': 0.08,
            'Gelu': 0.1,
            'Softmax': 0.15,
            
            # Element-wise
            'Add': 0.03,
            'Sub': 0.03,
            'Mul': 0.03,
            'Div': 0.04,
            'Pow': 0.06,
            'Sqrt': 0.05,
            
            # Reduction
            'ReduceMean': 0.1,
            'ReduceSum': 0.08,
            'ReduceMax': 0.08,
            
            # Fused ops (faster than sum of components)
            'FusedConvBn': 2.1,  # vs 2.3 separate
            'FusedConvBnRelu': 2.15,  # vs 2.35 separate
            
            # Default
            'default': 0.1,
        }
    
    def execute(self) -> float:
        """
        Execute graph and return total latency.
        
        Returns:
            Total execution time in milliseconds
        """
        total_latency = 0.0
        
        for node in self._execution_order:
            op_latency = self.op_latencies.get(
                node.op_type, 
                self.op_latencies.get('default', 0.1)
            )
            
            # Add some variance
            import random
            variance = random.gauss(0, op_latency * 0.05)
            actual_latency = max(0.001, op_latency + variance)
            
            total_latency += actual_latency
        
        return total_latency
    
    def profile(self) -> Dict[str, float]:
        """Profile execution time per operator"""
        profile = {}
        
        for node in self._execution_order:
            op_latency = self.op_latencies.get(
                node.op_type,
                self.op_latencies.get('default', 0.1)
            )
            profile[node.id] = op_latency
        
        return profile


class FusionBenchmark:
    """
    Benchmarks graph execution with and without fusion.
    """
    
    def __init__(self,
                 iterations: int = 100,
                 warmup: int = 10,
                 op_latencies: Optional[Dict[str, float]] = None):
        """
        Initialize benchmark.
        
        Args:
            iterations: Number of benchmark iterations
            warmup: Number of warmup iterations
            op_latencies: Custom operator latencies
        """
        self.iterations = iterations
        self.warmup = warmup
        self.op_latencies = op_latencies
    
    def benchmark_graph(self,
                        graph: ComputationGraph,
                        name: str = "benchmark") -> BenchmarkResult:
        """
        Benchmark a single graph.
        
        Args:
            graph: Graph to benchmark
            name: Benchmark name
            
        Returns:
            BenchmarkResult with timing statistics
        """
        executor = GraphExecutor(graph, self.op_latencies)
        
        # Warmup
        for _ in range(self.warmup):
            executor.execute()
        
        # Benchmark
        latencies = []
        start_total = time.perf_counter()
        
        for _ in range(self.iterations):
            latency = executor.execute()
            latencies.append(latency)
        
        end_total = time.perf_counter()
        total_time = (end_total - start_total) * 1000  # Convert to ms
        
        # Compute statistics
        latencies_sorted = sorted(latencies)
        
        return BenchmarkResult(
            name=name,
            iterations=self.iterations,
            warmup_iterations=self.warmup,
            mean_ms=statistics.mean(latencies),
            median_ms=statistics.median(latencies),
            std_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0,
            min_ms=min(latencies),
            max_ms=max(latencies),
            p90_ms=latencies_sorted[int(len(latencies) * 0.9)],
            p95_ms=latencies_sorted[int(len(latencies) * 0.95)],
            p99_ms=latencies_sorted[int(len(latencies) * 0.99)],
            total_time_ms=total_time,
            throughput_ops=self.iterations / (total_time / 1000),
            metadata={
                'op_count': len(graph.nodes),
            }
        )
    
    def compare(self,
                original_graph: ComputationGraph,
                rules: Optional[List[FusionRule]] = None) -> ComparisonResult:
        """
        Compare fused vs unfused graph execution.
        
        Args:
            original_graph: Original unfused graph
            rules: Fusion rules to apply (uses defaults if None)
            
        Returns:
            ComparisonResult with comparison metrics
        """
        # Benchmark original graph
        unfused_result = self.benchmark_graph(original_graph, "unfused")
        
        # Apply fusion
        fuser = ManualFuser()
        if rules is None:
            rules = get_default_fusion_rules()
        
        for rule in rules:
            fuser.add_rule(rule)
        
        fused_graph = fuser.fuse(original_graph)
        fusion_stats = fuser.get_stats()
        
        # Benchmark fused graph
        fused_result = self.benchmark_graph(fused_graph, "fused")
        
        # Compute comparison metrics
        speedup = unfused_result.mean_ms / fused_result.mean_ms if fused_result.mean_ms > 0 else 1.0
        latency_reduction = unfused_result.mean_ms - fused_result.mean_ms
        reduction_percent = (latency_reduction / unfused_result.mean_ms * 100) if unfused_result.mean_ms > 0 else 0
        
        original_ops = len(original_graph.nodes)
        fused_ops = len(fused_graph.nodes)
        fusion_ratio = (original_ops - fused_ops) / original_ops if original_ops > 0 else 0
        
        return ComparisonResult(
            unfused=unfused_result,
            fused=fused_result,
            speedup=speedup,
            latency_reduction_ms=latency_reduction,
            latency_reduction_percent=reduction_percent,
            original_op_count=original_ops,
            fused_op_count=fused_ops,
            fusion_ratio=fusion_ratio,
            fusion_stats=fusion_stats.get('by_rule', {}),
        )


# ============ Convenience Functions ============

def run_benchmark(graph: ComputationGraph,
                  iterations: int = 100,
                  warmup: int = 10,
                  name: str = "benchmark") -> BenchmarkResult:
    """
    Run a simple benchmark on a graph.
    
    Args:
        graph: Graph to benchmark
        iterations: Number of iterations
        warmup: Warmup iterations
        name: Benchmark name
        
    Returns:
        BenchmarkResult
    """
    benchmark = FusionBenchmark(iterations=iterations, warmup=warmup)
    return benchmark.benchmark_graph(graph, name)


def compare_fused_vs_unfused(graph: ComputationGraph,
                              rules: Optional[List[FusionRule]] = None,
                              iterations: int = 100) -> ComparisonResult:
    """
    Compare fused vs unfused execution.
    
    Args:
        graph: Original graph
        rules: Fusion rules (uses defaults if None)
        iterations: Benchmark iterations
        
    Returns:
        ComparisonResult
    """
    benchmark = FusionBenchmark(iterations=iterations)
    return benchmark.compare(graph, rules)


def create_sample_resnet_block() -> ComputationGraph:
    """Create a sample ResNet block for benchmarking"""
    from .manual_fuser import OperatorNode
    
    nodes = [
        OperatorNode(id="conv1", op_type="Conv", inputs=["input"], outputs=["conv1_out"]),
        OperatorNode(id="bn1", op_type="BatchNormalization", inputs=["conv1_out"], outputs=["bn1_out"]),
        OperatorNode(id="relu1", op_type="Relu", inputs=["bn1_out"], outputs=["relu1_out"]),
        OperatorNode(id="conv2", op_type="Conv", inputs=["relu1_out"], outputs=["conv2_out"]),
        OperatorNode(id="bn2", op_type="BatchNormalization", inputs=["conv2_out"], outputs=["bn2_out"]),
        OperatorNode(id="add", op_type="Add", inputs=["bn2_out", "input"], outputs=["add_out"]),
        OperatorNode(id="relu2", op_type="Relu", inputs=["add_out"], outputs=["output"]),
    ]
    
    return ComputationGraph(nodes)


def create_sample_transformer_block() -> ComputationGraph:
    """Create a sample Transformer block for benchmarking"""
    from .manual_fuser import OperatorNode
    
    nodes = [
        # Self-attention
        OperatorNode(id="q_matmul", op_type="MatMul", inputs=["input"], outputs=["q"]),
        OperatorNode(id="k_matmul", op_type="MatMul", inputs=["input"], outputs=["k"]),
        OperatorNode(id="v_matmul", op_type="MatMul", inputs=["input"], outputs=["v"]),
        OperatorNode(id="qk_matmul", op_type="MatMul", inputs=["q", "k"], outputs=["qk"]),
        OperatorNode(id="softmax", op_type="Softmax", inputs=["qk"], outputs=["attn"]),
        OperatorNode(id="attn_v", op_type="MatMul", inputs=["attn", "v"], outputs=["attn_out"]),
        OperatorNode(id="proj", op_type="MatMul", inputs=["attn_out"], outputs=["proj_out"]),
        OperatorNode(id="proj_add", op_type="Add", inputs=["proj_out"], outputs=["proj_bias"]),
        
        # Residual + LayerNorm
        OperatorNode(id="residual1", op_type="Add", inputs=["proj_bias", "input"], outputs=["res1"]),
        OperatorNode(id="ln1_mean", op_type="ReduceMean", inputs=["res1"], outputs=["ln1_mean_out"]),
        OperatorNode(id="ln1_sub", op_type="Sub", inputs=["res1", "ln1_mean_out"], outputs=["ln1_sub_out"]),
        
        # FFN
        OperatorNode(id="ffn1", op_type="MatMul", inputs=["ln1_sub_out"], outputs=["ffn1_out"]),
        OperatorNode(id="ffn1_add", op_type="Add", inputs=["ffn1_out"], outputs=["ffn1_bias"]),
        OperatorNode(id="gelu", op_type="Gelu", inputs=["ffn1_bias"], outputs=["gelu_out"]),
        OperatorNode(id="ffn2", op_type="MatMul", inputs=["gelu_out"], outputs=["ffn2_out"]),
        OperatorNode(id="ffn2_add", op_type="Add", inputs=["ffn2_out"], outputs=["ffn2_bias"]),
        
        # Final residual
        OperatorNode(id="residual2", op_type="Add", inputs=["ffn2_bias", "ln1_sub_out"], outputs=["output"]),
    ]
    
    return ComputationGraph(nodes)


def save_benchmark_results(results: List[Dict[str, Any]], 
                           filepath: str) -> None:
    """Save benchmark results to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)


def load_benchmark_results(filepath: str) -> List[Dict[str, Any]]:
    """Load benchmark results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)
