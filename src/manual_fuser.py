"""
Manual Fuser Module

Provides utilities for manually specifying and applying operator fusion rules
to computation graphs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Set, Tuple
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


class FusionStrategy(Enum):
    """Strategies for applying fusion rules"""
    GREEDY = "greedy"  # Apply first matching rule
    MAXIMIZE_FUSION = "maximize_fusion"  # Maximize number of fused ops
    MINIMIZE_MEMORY = "minimize_memory"  # Minimize intermediate memory
    BALANCE = "balance"  # Balance between fusion and memory


@dataclass
class OperatorNode:
    """Represents a node in the computation graph"""
    id: str
    op_type: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    shape: Optional[Tuple[int, ...]] = None
    dtype: str = "float32"
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if isinstance(other, OperatorNode):
            return self.id == other.id
        return False


@dataclass
class FusionRule:
    """Defines a fusion rule for combining operators"""
    name: str
    pattern: List[str]  # List of op_types to match in sequence
    fused_op: str  # Name of the fused operator
    constraints: Optional[Callable[[List[OperatorNode]], bool]] = None
    transform: Optional[Callable[[List[OperatorNode]], Dict[str, Any]]] = None
    priority: int = 0  # Higher = applied first
    description: str = ""
    
    def matches(self, nodes: List[OperatorNode]) -> bool:
        """Check if nodes match this fusion pattern"""
        if len(nodes) != len(self.pattern):
            return False
        
        for node, expected_op in zip(nodes, self.pattern):
            if node.op_type != expected_op:
                return False
        
        # Check additional constraints
        if self.constraints is not None:
            try:
                if not self.constraints(nodes):
                    return False
            except Exception as e:
                logger.warning(f"Constraint check failed for rule {self.name}: {e}")
                return False
        
        return True
    
    def apply(self, nodes: List[OperatorNode]) -> OperatorNode:
        """Apply fusion to create a new fused node"""
        # Get attributes from transform or merge
        if self.transform is not None:
            attrs = self.transform(nodes)
        else:
            attrs = {}
            for node in nodes:
                attrs.update(node.attributes)
        
        # Get inputs (first node's inputs minus internal connections)
        internal_outputs = set()
        for node in nodes[:-1]:
            internal_outputs.update(node.outputs)
        
        inputs = []
        for node in nodes:
            for inp in node.inputs:
                if inp not in internal_outputs and inp not in inputs:
                    inputs.append(inp)
        
        # Get outputs (last node's outputs)
        outputs = nodes[-1].outputs.copy()
        
        return OperatorNode(
            id=f"fused_{nodes[0].id}_{nodes[-1].id}",
            op_type=self.fused_op,
            inputs=inputs,
            outputs=outputs,
            attributes={
                **attrs,
                '_fused_from': [n.id for n in nodes],
                '_fusion_rule': self.name,
            },
            shape=nodes[-1].shape,
            dtype=nodes[-1].dtype,
        )


class ComputationGraph:
    """Represents a computation graph"""
    
    def __init__(self, nodes: Optional[List[OperatorNode]] = None):
        self.nodes: Dict[str, OperatorNode] = {}
        self.edges: Dict[str, List[str]] = {}  # output -> consumers
        
        if nodes:
            for node in nodes:
                self.add_node(node)
    
    def add_node(self, node: OperatorNode) -> None:
        """Add a node to the graph"""
        self.nodes[node.id] = node
        
        # Update edges
        for output in node.outputs:
            if output not in self.edges:
                self.edges[output] = []
    
    def remove_node(self, node_id: str) -> Optional[OperatorNode]:
        """Remove a node from the graph"""
        if node_id not in self.nodes:
            return None
        
        node = self.nodes.pop(node_id)
        
        # Remove edges
        for output in node.outputs:
            if output in self.edges:
                del self.edges[output]
        
        return node
    
    def get_consumers(self, output_name: str) -> List[OperatorNode]:
        """Get all nodes that consume a given output"""
        consumers = []
        for node in self.nodes.values():
            if output_name in node.inputs:
                consumers.append(node)
        return consumers
    
    def get_producers(self, input_name: str) -> List[OperatorNode]:
        """Get all nodes that produce a given input"""
        producers = []
        for node in self.nodes.values():
            if input_name in node.outputs:
                producers.append(node)
        return producers
    
    def topological_sort(self) -> List[OperatorNode]:
        """Return nodes in topological order"""
        visited = set()
        order = []
        
        def visit(node_id: str):
            if node_id in visited:
                return
            visited.add(node_id)
            
            node = self.nodes.get(node_id)
            if node is None:
                return
            
            for inp in node.inputs:
                for producer in self.get_producers(inp):
                    visit(producer.id)
            
            order.append(node)
        
        for node_id in self.nodes:
            visit(node_id)
        
        return order
    
    def find_chains(self, max_length: int = 10) -> List[List[OperatorNode]]:
        """Find linear chains of operators"""
        chains = []
        visited = set()
        
        sorted_nodes = self.topological_sort()
        
        for start_node in sorted_nodes:
            if start_node.id in visited:
                continue
            
            # Start a new chain
            chain = [start_node]
            current = start_node
            
            while len(chain) < max_length:
                # Check if current node has exactly one output consumer
                if len(current.outputs) != 1:
                    break
                
                consumers = self.get_consumers(current.outputs[0])
                if len(consumers) != 1:
                    break
                
                next_node = consumers[0]
                
                # Check if next node has exactly one input from current
                producer_count = sum(1 for inp in next_node.inputs 
                                   if inp in current.outputs)
                if producer_count != 1:
                    break
                
                # Also check next node doesn't have other inputs from visited
                other_inputs = [inp for inp in next_node.inputs 
                               if inp not in current.outputs]
                if any(self.get_producers(inp) for inp in other_inputs 
                       if any(p.id in visited for p in self.get_producers(inp))):
                    break
                
                chain.append(next_node)
                current = next_node
            
            if len(chain) > 1:
                chains.append(chain)
            
            for node in chain:
                visited.add(node.id)
        
        return chains
    
    def clone(self) -> 'ComputationGraph':
        """Create a deep copy of the graph"""
        new_graph = ComputationGraph()
        for node in self.nodes.values():
            new_node = OperatorNode(
                id=node.id,
                op_type=node.op_type,
                inputs=node.inputs.copy(),
                outputs=node.outputs.copy(),
                attributes=node.attributes.copy(),
                shape=node.shape,
                dtype=node.dtype,
            )
            new_graph.add_node(new_node)
        return new_graph
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary representation"""
        return {
            'nodes': [
                {
                    'id': n.id,
                    'op_type': n.op_type,
                    'inputs': n.inputs,
                    'outputs': n.outputs,
                    'attributes': n.attributes,
                    'shape': n.shape,
                    'dtype': n.dtype,
                }
                for n in self.nodes.values()
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComputationGraph':
        """Create graph from dictionary representation"""
        nodes = [
            OperatorNode(
                id=n['id'],
                op_type=n['op_type'],
                inputs=n.get('inputs', []),
                outputs=n.get('outputs', []),
                attributes=n.get('attributes', {}),
                shape=tuple(n['shape']) if n.get('shape') else None,
                dtype=n.get('dtype', 'float32'),
            )
            for n in data.get('nodes', [])
        ]
        return cls(nodes)


class ManualFuser:
    """
    Applies manual fusion rules to computation graphs.
    
    Example usage:
        fuser = ManualFuser()
        fuser.add_rule(FusionRule(
            name="conv_bn_relu",
            pattern=["Conv", "BatchNormalization", "Relu"],
            fused_op="FusedConvBnRelu",
        ))
        fused_graph = fuser.fuse(graph)
    """
    
    def __init__(self, strategy: FusionStrategy = FusionStrategy.GREEDY):
        self.rules: List[FusionRule] = []
        self.strategy = strategy
        self.fusion_stats: Dict[str, int] = {}
    
    def add_rule(self, rule: FusionRule) -> None:
        """Add a fusion rule"""
        self.rules.append(rule)
        # Sort by priority (descending)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
        logger.info(f"Added fusion rule: {rule.name}")
    
    def remove_rule(self, name: str) -> bool:
        """Remove a fusion rule by name"""
        for i, rule in enumerate(self.rules):
            if rule.name == name:
                self.rules.pop(i)
                return True
        return False
    
    def find_fusible_sequences(self, 
                                graph: ComputationGraph) -> List[Tuple[FusionRule, List[OperatorNode]]]:
        """Find all sequences that can be fused"""
        fusible = []
        chains = graph.find_chains()
        
        for chain in chains:
            for rule in self.rules:
                pattern_len = len(rule.pattern)
                
                # Slide window over chain
                for i in range(len(chain) - pattern_len + 1):
                    subchain = chain[i:i + pattern_len]
                    if rule.matches(subchain):
                        fusible.append((rule, subchain))
        
        return fusible
    
    def fuse(self, 
             graph: ComputationGraph,
             max_iterations: int = 100) -> ComputationGraph:
        """
        Apply fusion rules to the graph.
        
        Args:
            graph: Input computation graph
            max_iterations: Maximum fusion iterations
            
        Returns:
            New graph with fused operators
        """
        result = graph.clone()
        self.fusion_stats = {}
        
        for iteration in range(max_iterations):
            fusible = self.find_fusible_sequences(result)
            
            if not fusible:
                logger.info(f"Fusion complete after {iteration} iterations")
                break
            
            if self.strategy == FusionStrategy.GREEDY:
                # Apply first match
                rule, nodes = fusible[0]
            elif self.strategy == FusionStrategy.MAXIMIZE_FUSION:
                # Apply rule that fuses most operators
                rule, nodes = max(fusible, key=lambda x: len(x[1]))
            else:
                # Default to greedy
                rule, nodes = fusible[0]
            
            # Apply the fusion
            fused_node = rule.apply(nodes)
            
            # Update graph
            for node in nodes:
                result.remove_node(node.id)
            result.add_node(fused_node)
            
            # Update connections
            self._update_connections(result, nodes, fused_node)
            
            # Update stats
            self.fusion_stats[rule.name] = self.fusion_stats.get(rule.name, 0) + 1
            
            logger.debug(f"Applied fusion rule '{rule.name}' to create '{fused_node.id}'")
        
        return result
    
    def _update_connections(self,
                            graph: ComputationGraph,
                            old_nodes: List[OperatorNode],
                            new_node: OperatorNode) -> None:
        """Update graph connections after fusion"""
        # Map old outputs to new outputs
        old_final_outputs = set(old_nodes[-1].outputs)
        
        # Update consumers to use new node's outputs
        for other_node in graph.nodes.values():
            if other_node.id == new_node.id:
                continue
            
            new_inputs = []
            for inp in other_node.inputs:
                if inp in old_final_outputs:
                    # Map to corresponding new output
                    idx = old_nodes[-1].outputs.index(inp)
                    if idx < len(new_node.outputs):
                        new_inputs.append(new_node.outputs[idx])
                    else:
                        new_inputs.append(inp)
                else:
                    new_inputs.append(inp)
            other_node.inputs = new_inputs
    
    def get_stats(self) -> Dict[str, Any]:
        """Get fusion statistics"""
        return {
            'total_fusions': sum(self.fusion_stats.values()),
            'by_rule': self.fusion_stats.copy(),
        }


# ============ Predefined Fusion Rules ============

def create_conv_bn_fusion() -> FusionRule:
    """Create Conv + BatchNorm fusion rule"""
    def transform(nodes: List[OperatorNode]) -> Dict[str, Any]:
        conv_attrs = nodes[0].attributes.copy()
        bn_attrs = nodes[1].attributes
        return {
            **conv_attrs,
            'fused_bn_epsilon': bn_attrs.get('epsilon', 1e-5),
            'fused_bn_momentum': bn_attrs.get('momentum', 0.9),
        }
    
    return FusionRule(
        name="conv_bn",
        pattern=["Conv", "BatchNormalization"],
        fused_op="FusedConvBn",
        transform=transform,
        priority=10,
        description="Fuse Conv2D with BatchNormalization",
    )


def create_conv_bn_relu_fusion() -> FusionRule:
    """Create Conv + BatchNorm + ReLU fusion rule"""
    return FusionRule(
        name="conv_bn_relu",
        pattern=["Conv", "BatchNormalization", "Relu"],
        fused_op="FusedConvBnRelu",
        priority=20,
        description="Fuse Conv2D + BatchNormalization + ReLU",
    )


def create_matmul_add_fusion() -> FusionRule:
    """Create MatMul + Add (bias) fusion rule"""
    return FusionRule(
        name="matmul_add",
        pattern=["MatMul", "Add"],
        fused_op="Gemm",
        priority=10,
        description="Fuse MatMul with bias Add into Gemm",
    )


def create_layer_norm_fusion() -> FusionRule:
    """Create LayerNorm component fusion rule"""
    return FusionRule(
        name="layer_norm_components",
        pattern=["ReduceMean", "Sub", "Pow", "ReduceMean", "Add", "Sqrt", "Div"],
        fused_op="LayerNormalization",
        priority=30,
        description="Fuse LayerNorm components into single op",
    )


def create_gelu_fusion() -> FusionRule:
    """Create GELU approximation fusion rule"""
    return FusionRule(
        name="gelu_approx",
        pattern=["Mul", "Pow", "Mul", "Add", "Mul", "Tanh", "Add", "Mul"],
        fused_op="Gelu",
        priority=25,
        description="Fuse GELU approximation into single op",
    )


def get_default_fusion_rules() -> List[FusionRule]:
    """Get list of commonly used fusion rules"""
    return [
        create_conv_bn_fusion(),
        create_conv_bn_relu_fusion(),
        create_matmul_add_fusion(),
        create_layer_norm_fusion(),
        create_gelu_fusion(),
    ]


# ============ Convenience Functions ============

def apply_fusion_rules(graph: ComputationGraph,
                       rules: Optional[List[FusionRule]] = None,
                       strategy: FusionStrategy = FusionStrategy.GREEDY) -> Tuple[ComputationGraph, Dict[str, Any]]:
    """
    Apply fusion rules to a graph.
    
    Args:
        graph: Input computation graph
        rules: List of fusion rules (uses defaults if None)
        strategy: Fusion strategy to use
        
    Returns:
        Tuple of (fused_graph, fusion_stats)
    """
    fuser = ManualFuser(strategy=strategy)
    
    if rules is None:
        rules = get_default_fusion_rules()
    
    for rule in rules:
        fuser.add_rule(rule)
    
    fused_graph = fuser.fuse(graph)
    return fused_graph, fuser.get_stats()
