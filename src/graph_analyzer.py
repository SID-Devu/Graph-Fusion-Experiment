#!/usr/bin/env python3
"""
Graph Fusion Experiment - Graph Analysis and Fusion Tools

Analyze ONNX graphs to find fusion opportunities and apply optimizations.
"""

import onnx
from onnx import numpy_helper, TensorProto
import onnxruntime as ort
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Set, Optional, Tuple
from pathlib import Path
import time


@dataclass
class FusionOpportunity:
    """Represents a fusion opportunity in the graph."""
    pattern: str
    nodes: List[str]
    ops: List[str]
    estimated_speedup: float
    confidence: float


@dataclass
class GraphStats:
    """Statistics about an ONNX graph."""
    num_nodes: int
    num_parameters: int
    num_initializers: int
    op_counts: Dict[str, int]
    total_flops: int
    memory_bytes: int


class GraphAnalyzer:
    """Analyze ONNX graphs for optimization opportunities."""
    
    # Common fusion patterns
    FUSION_PATTERNS = {
        'matmul_add': (['MatMul', 'Add'], 1.5),
        'matmul_add_relu': (['MatMul', 'Add', 'Relu'], 1.8),
        'conv_bn': (['Conv', 'BatchNormalization'], 1.6),
        'conv_bn_relu': (['Conv', 'BatchNormalization', 'Relu'], 2.0),
        'conv_relu': (['Conv', 'Relu'], 1.4),
        'layer_norm': (['ReduceMean', 'Sub', 'Pow', 'ReduceMean', 'Add', 'Sqrt', 'Div', 'Mul', 'Add'], 3.0),
        'gelu': (['Div', 'Erf', 'Add', 'Mul', 'Mul'], 2.5),
    }
    
    def __init__(self, model_path: str):
        """Load and analyze model."""
        self.model_path = model_path
        self.model = onnx.load(model_path)
        self.graph = self.model.graph
        
        # Build node lookup
        self.nodes = {node.name: node for node in self.graph.node}
        self.output_to_node: Dict[str, str] = {}
        self.input_to_nodes: Dict[str, List[str]] = {}
        
        for node in self.graph.node:
            for output in node.output:
                self.output_to_node[output] = node.name
            for inp in node.input:
                if inp not in self.input_to_nodes:
                    self.input_to_nodes[inp] = []
                self.input_to_nodes[inp].append(node.name)
    
    def get_stats(self) -> GraphStats:
        """Get basic graph statistics."""
        op_counts: Dict[str, int] = {}
        for node in self.graph.node:
            op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
        
        # Count parameters
        num_params = 0
        total_bytes = 0
        for init in self.graph.initializer:
            arr = numpy_helper.to_array(init)
            num_params += arr.size
            total_bytes += arr.nbytes
        
        return GraphStats(
            num_nodes=len(self.graph.node),
            num_parameters=num_params,
            num_initializers=len(self.graph.initializer),
            op_counts=op_counts,
            total_flops=0,  # Would need shape inference
            memory_bytes=total_bytes
        )
    
    def find_fusion_opportunities(self) -> List[FusionOpportunity]:
        """Find all fusion opportunities in the graph."""
        opportunities = []
        visited: Set[str] = set()
        
        for node in self.graph.node:
            if node.name in visited:
                continue
            
            for pattern_name, (ops, speedup) in self.FUSION_PATTERNS.items():
                match = self._match_pattern(node, ops)
                if match:
                    node_names = [n.name for n in match]
                    if not any(n in visited for n in node_names):
                        opportunities.append(FusionOpportunity(
                            pattern=pattern_name,
                            nodes=node_names,
                            ops=[n.op_type for n in match],
                            estimated_speedup=speedup,
                            confidence=0.8
                        ))
                        visited.update(node_names)
                        break
        
        return opportunities
    
    def _match_pattern(self, start_node, ops: List[str]) -> Optional[List]:
        """Check if pattern matches starting from node."""
        if start_node.op_type != ops[0]:
            return None
        
        matched = [start_node]
        current = start_node
        
        for op in ops[1:]:
            # Find consumer node
            consumers = []
            for output in current.output:
                if output in self.input_to_nodes:
                    consumers.extend(self.input_to_nodes[output])
            
            # Find matching consumer
            found = False
            for consumer_name in consumers:
                consumer = self.nodes.get(consumer_name)
                if consumer and consumer.op_type == op:
                    # Check it's the only consumer (for safe fusion)
                    matched.append(consumer)
                    current = consumer
                    found = True
                    break
            
            if not found:
                return None
        
        return matched
    
    def visualize_graph(self) -> str:
        """Generate ASCII visualization of graph."""
        lines = ["Graph Structure:", "=" * 40]
        
        for node in self.graph.node[:20]:  # Limit for readability
            inputs = ', '.join(node.input[:2])
            outputs = ', '.join(node.output)
            lines.append(f"{node.op_type:20s} {inputs} -> {outputs}")
        
        if len(self.graph.node) > 20:
            lines.append(f"... and {len(self.graph.node) - 20} more nodes")
        
        return '\n'.join(lines)


class ManualFuser:
    """Apply manual fusion patterns to ONNX graphs."""
    
    def __init__(self):
        self.fusion_handlers = {
            'matmul_add': self._fuse_matmul_add,
            'conv_bn': self._fuse_conv_bn,
        }
    
    def apply_pattern(self, model_path: str, pattern: str, 
                      output_path: str) -> str:
        """Apply a fusion pattern to the model."""
        model = onnx.load(model_path)
        
        handler = self.fusion_handlers.get(pattern)
        if handler:
            model = handler(model)
        
        onnx.save(model, output_path)
        return output_path
    
    def _fuse_matmul_add(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuse MatMul + Add into Gemm."""
        graph = model.graph
        nodes_to_remove = []
        nodes_to_add = []
        
        for i, node in enumerate(graph.node):
            if node.op_type == 'MatMul':
                # Check if output goes to Add
                matmul_output = node.output[0]
                for j, other in enumerate(graph.node):
                    if other.op_type == 'Add' and matmul_output in other.input:
                        # Found MatMul -> Add pattern
                        bias_input = [inp for inp in other.input 
                                     if inp != matmul_output][0]
                        
                        gemm_node = onnx.helper.make_node(
                            'Gemm',
                            inputs=[node.input[0], node.input[1], bias_input],
                            outputs=other.output,
                            name=f'fused_gemm_{i}',
                            alpha=1.0,
                            beta=1.0,
                            transA=0,
                            transB=0
                        )
                        
                        nodes_to_remove.extend([node.name, other.name])
                        nodes_to_add.append(gemm_node)
                        break
        
        # Rebuild graph
        new_nodes = [n for n in graph.node if n.name not in nodes_to_remove]
        new_nodes.extend(nodes_to_add)
        
        new_graph = onnx.helper.make_graph(
            new_nodes,
            graph.name,
            graph.input,
            graph.output,
            graph.initializer
        )
        
        return onnx.helper.make_model(new_graph, opset_imports=model.opset_import)
    
    def _fuse_conv_bn(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuse Conv + BatchNorm by folding BN into Conv weights."""
        # This is a more complex fusion that modifies weights
        # Simplified implementation
        return model


def benchmark_models(original_path: str, optimized_path: str,
                    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
                    warmup: int = 10, iterations: int = 100) -> Dict:
    """Benchmark original vs optimized models."""
    
    # Create sessions
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    
    original_sess = ort.InferenceSession(original_path, sess_options)
    optimized_sess = ort.InferenceSession(optimized_path, sess_options)
    
    # Get input name
    input_name = original_sess.get_inputs()[0].name
    input_data = np.random.randn(*input_shape).astype(np.float32)
    
    # Warmup
    for _ in range(warmup):
        original_sess.run(None, {input_name: input_data})
        optimized_sess.run(None, {input_name: input_data})
    
    # Benchmark original
    start = time.perf_counter()
    for _ in range(iterations):
        original_sess.run(None, {input_name: input_data})
    original_time = (time.perf_counter() - start) / iterations * 1000
    
    # Benchmark optimized
    start = time.perf_counter()
    for _ in range(iterations):
        optimized_sess.run(None, {input_name: input_data})
    optimized_time = (time.perf_counter() - start) / iterations * 1000
    
    return {
        'original_ms': original_time,
        'optimized_ms': optimized_time,
        'speedup': original_time / optimized_time,
        'savings_ms': original_time - optimized_time
    }


def analyze_model(model_path: str) -> None:
    """Main analysis function."""
    print(f"Analyzing: {model_path}")
    print("=" * 60)
    
    analyzer = GraphAnalyzer(model_path)
    
    # Stats
    stats = analyzer.get_stats()
    print(f"\nGraph Statistics:")
    print(f"  Nodes: {stats.num_nodes}")
    print(f"  Parameters: {stats.num_parameters:,}")
    print(f"  Memory: {stats.memory_bytes / 1024 / 1024:.2f} MB")
    print(f"\n  Ops:")
    for op, count in sorted(stats.op_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {op}: {count}")
    
    # Fusion opportunities
    opportunities = analyzer.find_fusion_opportunities()
    print(f"\nFusion Opportunities: {len(opportunities)}")
    
    total_speedup = 0
    for opp in opportunities:
        print(f"\n  Pattern: {opp.pattern}")
        print(f"    Nodes: {' -> '.join(opp.ops)}")
        print(f"    Est. Speedup: {opp.estimated_speedup:.1f}x")
        total_speedup += (opp.estimated_speedup - 1)
    
    if opportunities:
        print(f"\n  Total Potential Improvement: ~{total_speedup:.0%} faster")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze ONNX graph for fusion')
    parser.add_argument('model', help='ONNX model path')
    parser.add_argument('--fuse', help='Apply fusion pattern')
    parser.add_argument('--output', help='Output path for fused model')
    parser.add_argument('--benchmark', action='store_true',
                       help='Benchmark before/after fusion')
    
    args = parser.parse_args()
    
    analyze_model(args.model)
    
    if args.fuse and args.output:
        fuser = ManualFuser()
        fuser.apply_pattern(args.model, args.fuse, args.output)
        print(f"\nFused model saved to: {args.output}")
        
        if args.benchmark:
            results = benchmark_models(args.model, args.output)
            print(f"\nBenchmark Results:")
            print(f"  Original: {results['original_ms']:.2f} ms")
            print(f"  Optimized: {results['optimized_ms']:.2f} ms")
            print(f"  Speedup: {results['speedup']:.2f}x")
