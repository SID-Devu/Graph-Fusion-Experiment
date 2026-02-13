# Contributing to Graph Fusion Experiment

Thank you for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/sudheerdevu/Graph-Fusion-Experiment.git
cd Graph-Fusion-Experiment

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install pytest black isort
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

```bash
black src/ tests/
isort src/ tests/
```

## Adding Fusion Patterns

1. Add pattern to `patterns/*.yaml`
2. Update pattern detection in `src/fusion_patterns.py`
3. Add benchmark case
4. Document in README

## Pull Request Process

1. Fork repository
2. Create feature branch
3. Add tests
4. Submit PR

## License

Contributions are licensed under MIT License.
