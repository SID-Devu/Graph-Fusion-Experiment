# Graph Fusion Tutorial

## Introduction

Graph fusion combines multiple operations into a single optimized kernel, reducing memory bandwidth and improving performance.

## Prerequisites

- Python 3.8+
- ONNX (for model loading)
- NetworkX (for graph analysis)

## Quick Start

### 1. Load a Model

```python
import onnx
from src.graph_loader import load_graph

model = onnx.load("model.onnx")
graph = load_graph(model)
```

### 2. Detect Fusion Patterns

```python
from src.pattern_detector import PatternDetector

detector = PatternDetector("patterns/fusion_patterns.yaml")
candidates = detector.find_patterns(graph)

print(f"Found {len(candidates)} fusion candidates")
```

### 3. Apply Fusion

```python
from src.fusion_engine import FusionEngine

engine = FusionEngine()
optimized_graph = engine.apply_fusions(graph, candidates)
```

### 4. Export Optimized Model

```python
from src.export import export_onnx

export_onnx(optimized_graph, "model_fused.onnx")
```

## Common Patterns

### Conv-BN-ReLU Fusion

Most impactful in CNN models:

```python
# Before: 3 kernels, 3 memory round-trips
conv_out = Conv2D(input)
bn_out = BatchNorm(conv_out)
relu_out = ReLU(bn_out)

# After: 1 fused kernel
fused_out = FusedConvBNRelu(input)
```

### Verifying Results

```python
import numpy as np

# Run both versions
original_out = run_original(model, test_input)
fused_out = run_fused(optimized_model, test_input)

# Verify numerical accuracy
np.testing.assert_allclose(original_out, fused_out, rtol=1e-5)
```

## Troubleshooting

- **Fusion rejected**: Check constraint violations in logs
- **Accuracy loss**: Some fusions may need higher precision
- **No patterns found**: Verify model format compatibility
