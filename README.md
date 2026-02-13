# Graph Fusion Experiment

> Explore operator fusion techniques in ONNX graphs to understand how graph optimizations affect inference performance.

## 🎯 Purpose

Graph fusion is one of the most impactful optimizations for AI inference. This project demonstrates:
- **What fusion is**: Combining multiple operators into single optimized kernels
- **Why it matters**: Reduced memory bandwidth, fewer kernel launches
- **How to experiment**: Tools to manually fuse and measure impact

## 📊 Fusion Benefits

```
Before Fusion:              After Fusion:
┌────────┐                  ┌────────────────┐
│ MatMul │                  │                │
└───┬────┘                  │   Fused        │
    │        Memory         │   MatMul+      │
┌───▼────┐   Traffic        │   Bias+        │
│  Add   │   ────────▶      │   ReLU         │
└───┬────┘                  │                │
    │                       └────────────────┘
┌───▼────┐
│  ReLU  │                  Result:
└────────┘                  - 1 kernel vs 3
                            - 2x less memory traffic
                            - 30-50% faster
```

## 📁 Project Structure

```
Graph-Fusion-Experiment/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── graph_analyzer.py      # Analyze ONNX graphs
│   ├── fusion_patterns.py     # Common fusion patterns
│   ├── manual_fuser.py        # Apply manual fusions
│   └── benchmark.py           # Measure fusion impact
├── patterns/
│   ├── matmul_add.py          # MatMul + Add fusion
│   ├── conv_bn_relu.py        # Conv + BatchNorm + ReLU
│   └── attention.py           # Multi-head attention fusion
├── notebooks/
│   └── fusion_analysis.ipynb
└── models/
    └── README.md              # Where to place test models
```

## 🚀 Quick Start

```python
from graph_fusion import GraphAnalyzer, ManualFuser

# Analyze fusion opportunities
analyzer = GraphAnalyzer("model.onnx")
opportunities = analyzer.find_fusion_opportunities()

print(f"Found {len(opportunities)} fusion opportunities:")
for opp in opportunities:
    print(f"  - {opp.pattern}: {opp.nodes}")

# Apply fusion
fuser = ManualFuser()
optimized = fuser.apply_pattern(
    "model.onnx",
    pattern="matmul_add",
    output="model_fused.onnx"
)

# Benchmark
from graph_fusion import benchmark_models
results = benchmark_models("model.onnx", "model_fused.onnx")
print(f"Speedup: {results['speedup']:.2f}x")
```

## 🔧 Common Fusion Patterns

### 1. MatMul + Add (Bias)
```
MatMul(A, B) + C  →  Gemm(A, B, C)
```

### 2. Conv + BatchNorm + ReLU
```
Conv → BatchNorm → ReLU  →  ConvBnRelu (fused)
```

### 3. Attention Pattern
```
QKV Projection → Reshape → Attention → Output
  → FusedMultiHeadAttention
```

### 4. LayerNorm
```
ReduceMean → Sub → Pow → ReduceMean → Add → Sqrt → Div → Mul → Add
  → LayerNormalization
```

## 📈 Expected Results

| Pattern | Unfused (ms) | Fused (ms) | Speedup |
|---------|--------------|------------|---------|
| MatMul+Add | 1.2 | 0.8 | 1.5x |
| Conv+BN+ReLU | 2.5 | 1.4 | 1.8x |
| Full Attention | 5.0 | 2.8 | 1.8x |
| Transformer Block | 15.0 | 8.0 | 1.9x |

## 📚 Learning Resources

- [ONNX Graph Optimization](https://onnxruntime.ai/docs/performance/graph-optimizations.html)
- [Operator Fusion Paper (TVM)](https://arxiv.org/abs/1802.04799)
- [cuDNN Fusion Guide](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/)

## License

MIT
