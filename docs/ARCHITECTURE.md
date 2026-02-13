# Graph Fusion Architecture

## Overview

This document explains the architecture of the graph fusion detection and optimization framework.

## Components

### 1. Pattern Detector (`src/pattern_detector.py`)

Identifies fusible operations in computational graphs:
- **Pointwise chains**: Element-wise ops (add, mul, relu) that can be fused
- **Conv-BN-Relu patterns**: Classic CNN fusion opportunity
- **MatMul-Add patterns**: Dense layer optimizations

### 2. Fusion Engine

Applies detected patterns to create optimized graphs:
- Validates fusion safety (no side effects)
- Generates fused kernel specifications
- Updates graph topology

## Pattern Categories

| Pattern | Operations | Speedup |
|---------|-----------|---------|
| Conv-BN-Relu | Conv2D + BatchNorm + ReLU | 1.5-2x |
| MatMul-Add | MatMul + BiasAdd | 1.2-1.4x |
| Pointwise Chain | Multiple elementwise ops | 1.3-2x |

## Data Flow

```
Input Graph → Pattern Detection → Candidate Selection → Fusion → Output Graph
```

## Configuration

Patterns are defined in `patterns/fusion_patterns.yaml`:

```yaml
patterns:
  conv_bn_relu:
    ops: [Conv2D, BatchNorm, Relu]
    constraints:
      - same_dtype
      - contiguous_memory
```

## Extending Patterns

1. Add pattern definition to YAML
2. Implement validator in `src/validators/`
3. Add fusion rule to engine
