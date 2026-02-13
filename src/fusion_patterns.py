"""
Fusion Pattern Library for Graph Optimization

This module contains common fusion patterns for ONNX graph optimization.
"""

import onnx
from onnx import helper, TensorProto
from typing import List, Dict, Tuple, Optional


class FusionPattern:
    """Base class for fusion patterns"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def match(self, graph: onnx.GraphProto, start_node: onnx.NodeProto) -> bool:
        """Check if pattern matches starting from given node"""
        raise NotImplementedError
    
    def apply(self, graph: onnx.GraphProto, matched_nodes: List[onnx.NodeProto]) -> onnx.NodeProto:
        """Apply fusion and return new fused node"""
        raise NotImplementedError


class ConvBNFusion(FusionPattern):
    """Fuse Conv + BatchNormalization into single Conv"""
    
    def __init__(self):
        super().__init__(
            "ConvBNFusion",
            "Fuses Conv2D followed by BatchNormalization into optimized Conv2D"
        )
    
    def match(self, graph: onnx.GraphProto, node: onnx.NodeProto) -> Optional[Tuple]:
        if node.op_type != "Conv":
            return None
        
        # Find consumer of Conv output
        conv_output = node.output[0]
        for next_node in graph.node:
            if next_node.op_type == "BatchNormalization":
                if conv_output in next_node.input:
                    return (node, next_node)
        return None
    
    def apply(self, conv_node: onnx.NodeProto, bn_node: onnx.NodeProto,
              weights: Dict[str, onnx.TensorProto]) -> Tuple[onnx.NodeProto, Dict]:
        """
        Fuse Conv and BN: y = gamma * (conv(x) - mean) / sqrt(var + eps) + beta
        
        Fused weights:
        w_fused = gamma * w / sqrt(var + eps)
        b_fused = gamma * (b - mean) / sqrt(var + eps) + beta
        """
        import numpy as np
        
        # Get Conv weights
        w = numpy_helper.to_array(weights[conv_node.input[1]])
        b = numpy_helper.to_array(weights[conv_node.input[2]]) if len(conv_node.input) > 2 else np.zeros(w.shape[0])
        
        # Get BN parameters
        gamma = numpy_helper.to_array(weights[bn_node.input[1]])
        beta = numpy_helper.to_array(weights[bn_node.input[2]])
        mean = numpy_helper.to_array(weights[bn_node.input[3]])
        var = numpy_helper.to_array(weights[bn_node.input[4]])
        
        eps = 1e-5  # Default epsilon
        for attr in bn_node.attribute:
            if attr.name == "epsilon":
                eps = attr.f
        
        # Compute fused parameters
        scale = gamma / np.sqrt(var + eps)
        w_fused = w * scale.reshape(-1, 1, 1, 1)
        b_fused = (b - mean) * scale + beta
        
        # Create fused Conv node
        fused_node = helper.make_node(
            "Conv",
            inputs=[conv_node.input[0], f"{conv_node.name}_fused_weight", f"{conv_node.name}_fused_bias"],
            outputs=bn_node.output,
            name=f"{conv_node.name}_bn_fused",
            **{attr.name: attr for attr in conv_node.attribute}
        )
        
        return fused_node, {
            f"{conv_node.name}_fused_weight": w_fused,
            f"{conv_node.name}_fused_bias": b_fused
        }


class MatMulAddFusion(FusionPattern):
    """Fuse MatMul + Add into Gemm"""
    
    def __init__(self):
        super().__init__(
            "MatMulAddFusion",
            "Fuses MatMul followed by bias Add into Gemm operation"
        )
    
    def match(self, graph: onnx.GraphProto, node: onnx.NodeProto) -> Optional[Tuple]:
        if node.op_type != "MatMul":
            return None
        
        matmul_output = node.output[0]
        for next_node in graph.node:
            if next_node.op_type == "Add":
                if matmul_output in next_node.input:
                    return (node, next_node)
        return None
    
    def apply(self, matmul_node: onnx.NodeProto, add_node: onnx.NodeProto) -> onnx.NodeProto:
        # Determine which Add input is the bias
        bias_input = add_node.input[1] if add_node.input[0] == matmul_node.output[0] else add_node.input[0]
        
        fused_node = helper.make_node(
            "Gemm",
            inputs=[matmul_node.input[0], matmul_node.input[1], bias_input],
            outputs=add_node.output,
            name=f"{matmul_node.name}_gemm_fused",
            alpha=1.0,
            beta=1.0,
            transA=0,
            transB=0
        )
        return fused_node


class GeLUFusion(FusionPattern):
    """Fuse GELU approximation pattern"""
    
    def __init__(self):
        super().__init__(
            "GeLUFusion",
            "Fuses GELU approximation (x * 0.5 * (1 + tanh(...))) into single op"
        )
        # GELU pattern: Mul -> Add -> Mul -> Tanh -> Add -> Mul -> Mul
    
    def match(self, graph: onnx.GraphProto, node: onnx.NodeProto) -> Optional[List]:
        # Complex pattern matching for GELU
        # Pattern: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        if node.op_type != "Pow":
            return None
        
        # Simplified: Look for characteristic constants
        # This is a heuristic - real implementation needs full pattern matching
        return None  # Placeholder
    
    def apply(self, matched_nodes: List[onnx.NodeProto]) -> onnx.NodeProto:
        # Create fused GELU node
        input_name = matched_nodes[0].input[0]
        output_name = matched_nodes[-1].output[0]
        
        return helper.make_node(
            "Gelu",  # Custom op or use domain
            inputs=[input_name],
            outputs=[output_name],
            name="fused_gelu"
        )


class LayerNormFusion(FusionPattern):
    """Fuse LayerNorm pattern (ReduceMean, Sub, Pow, ReduceMean, Add, Sqrt, Div, Mul, Add)"""
    
    def __init__(self):
        super().__init__(
            "LayerNormFusion",
            "Fuses LayerNormalization pattern into single operation"
        )


class AttentionFusion(FusionPattern):
    """Fuse Multi-Head Attention pattern"""
    
    def __init__(self):
        super().__init__(
            "AttentionFusion",
            "Fuses Q, K, V projections with attention computation"
        )


# Pattern registry
FUSION_PATTERNS = [
    ConvBNFusion(),
    MatMulAddFusion(),
    GeLUFusion(),
    LayerNormFusion(),
    AttentionFusion(),
]


def find_fusable_patterns(graph: onnx.GraphProto) -> List[Tuple[FusionPattern, List]]:
    """Find all fusable patterns in a graph"""
    found = []
    
    for node in graph.node:
        for pattern in FUSION_PATTERNS:
            match = pattern.match(graph, node)
            if match:
                found.append((pattern, match))
    
    return found


def apply_fusions(model: onnx.ModelProto, patterns: List[Tuple]) -> onnx.ModelProto:
    """Apply all matched fusion patterns to model"""
    # Implementation would modify graph in-place or create new graph
    pass
