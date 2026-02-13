# Common Fusion Patterns

This directory contains example fusion pattern definitions for various
deep learning architectures.

## Pattern Files

- `conv_patterns.yaml` - Convolutional neural network fusion patterns
- `transformer_patterns.yaml` - Transformer architecture fusion patterns  
- `normalization_patterns.yaml` - Normalization layer fusion patterns

## Pattern Format

Each pattern file defines fusion rules in YAML format:

```yaml
patterns:
  - name: pattern_name
    ops: [Op1, Op2, Op3]
    fused_op: FusedOp
    priority: 10
    description: "Description of the fusion"
    constraints:
      - same_dtype
      - contiguous
```

## Usage

```python
from src.manual_fuser import ManualFuser, load_patterns_from_yaml

fuser = ManualFuser()
rules = load_patterns_from_yaml("patterns/conv_patterns.yaml")
for rule in rules:
    fuser.add_rule(rule)
```
