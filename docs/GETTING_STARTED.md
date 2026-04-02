# Getting Started with Post Graduation Utils

## What's Included?

This library provides utilities for:

1. **Machine Learning (ML)**
   - Data preprocessing
   - Model building and evaluation
   - Feature engineering

2. **Deep Learning**
   - Neural network architectures
   - Training utilities
   - Transfer learning helpers

3. **Generative AI (GenAI)**
   - Large Language Model (LLM) utilities
   - Retrieval-Augmented Generation (RAG)
   - Prompt engineering tools

4. **Institution-Specific Extensions**
   - PUC-tailored configurations and examples
   - IBMEC-tailored configurations and examples

---

## Basic Usage Examples

### Example 1: Using Core Utilities

```python
from post_graduation_utils import core

# Access core utilities
result = core.utils.placeholder()
print(result)
```

### Example 2: Machine Learning Preprocessing

```python
from post_graduation_utils.ml import preprocessing

# Use preprocessing tools
# Example implementation coming soon
```

### Example 3: Deep Learning

```python
from post_graduation_utils.deep_learning import draw_neural_network

# Use Deep Learning tools
# Example: Draw neural network architecture
# draw_neural_network(model)
```

### Example 4: Generative AI

```python
from post_graduation_utils.genai import llm, rag

# Use GenAI tools
# Example implementation coming soon
```

### Example 5: Institution-Specific (PUC)

```python
from post_graduation_utils.puc import config

# Get PUC-specific configuration
info = config.PUCConfig.get_info()
print(info)  # Output: PUC Utils v1.0.0
```

### Example 6: Institution-Specific (IBMEC)

```python
from post_graduation_utils.ibmec import config

# Get IBMEC-specific configuration
info = config.IBMECConfig.get_info()
print(info)  # Output: IBMEC Utils v1.0.0
```

---

## Project Structure for Reference

```
post_graduation_utils/
├── post_graduation_utils/          # Main package
│   ├── core/                       # Shared utilities
│   ├── ml/                         # Machine Learning
│   ├── dl/                         # Deep Learning
│   ├── genai/                      # Generative AI
│   ├── puc/                        # PUC-specific
│   └── ibmec/                      # IBMEC-specific
├── examples/                       # Example scripts
├── docs/                           # Documentation
└── tests/                          # Unit tests
```

---

## Common Tasks

### Running Examples

```bash
python examples/example_basic.py
```

### Running Tests

```bash
pytest tests/
```

### Installing Development Tools

```bash
pip install "post-graduation-utils[dev]"
```

---

## Next Steps

1. Read the [Installation Guide](INSTALLATION_GUIDE.md)
2. Check the [Architecture Documentation](ARCHITECTURE.md)
3. Explore the `examples/` directory
4. Read the [README.md](../README.md) for more details

---

## Support

- 📚 Check existing examples in `examples/`
- 📖 Read the documentation in `docs/`
- 🐛 Report issues on GitHub
- 💬 Ask questions in class or office hours
