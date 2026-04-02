# PGL Utils

A comprehensive library for Machine Learning, Deep Learning, and Generative AI utilities, designed for PUC and IBMEC post-graduation students.

## Features

- **Machine Learning (ML)**: Preprocessing, models, and utilities for classical ML
- **Deep Learning**: Architectures, training utilities, and pre-trained models
- **Generative AI (GenAI)**: LLM utilities, RAG implementations, and prompt engineering tools
- **Institution-specific extensions**: Customized tools for PUC and IBMEC students

## Installation

### Basic Installation

```bash
pip install pgl-utils
```

### Installation with specific features

```bash
# Machine Learning only
pip install pgl-utils[ml]

# Deep Learning only
pip install pgl-utils[deep_learning]

# Generative AI only
pip install pgl-utils[genai]

# All features
pip install pgl-utils[all]

# Development
pip install pgl-utils[dev]
```

### Installation from source

```bash
git clone https://github.com/renansantosmendes/pgl_utils.git
cd pgl_utils
pip install -e .
```

## Quick Start

### Using Core Utilities

```python
from pgl_utils import core

# Your code here
```

### Using Machine Learning Tools

```python
from pgl_utils.ml import preprocessing, models

# Your code here
```

### Using Deep Learning Tools

```python
from pgl_utils.deep_learning import draw_neural_network

# Your code here
```

### Using Generative AI Tools

```python
from pgl_utils.genai import llm, rag

# Your code here
```

### Institution-Specific Tools

#### For PUC Students

```python
from pgl_utils.puc import config

puc_info = config.PUCConfig.get_info()
```

#### For IBMEC Students

```python
from pgl_utils.ibmec import config

ibmec_info = config.IBMECConfig.get_info()
```

## Project Structure

```
pgl_utils/
├── pgl_utils/          # Main package
│   ├── __init__.py
│   ├── core/                       # Shared utilities
│   │   ├── __init__.py
│   │   └── utils.py
│   ├── ml/                         # Machine Learning module
│   │   ├── __init__.py
│   │   ├── preprocessing.py
│   │   └── models.py
│   ├── deep_learning/              # Deep Learning module
│   │   ├── __init__.py
│   │   ├── architectures.py
│   │   └── training.py
│   ├── genai/                      # Generative AI module
│   │   ├── __init__.py
│   │   ├── llm.py
│   │   └── rag.py
│   ├── puc/                        # PUC-specific extensions
│   │   ├── __init__.py
│   │   └── config.py
│   └── ibmec/                      # IBMEC-specific extensions
│       ├── __init__.py
│       └── config.py
├── tests/                          # Unit tests
├── examples/                       # Example notebooks and scripts
├── docs/                           # Documentation
├── setup.py                        # Package configuration
├── requirements.txt                # Dependencies
├── README.md                       # This file
└── .gitignore                      # Git ignore rules
```

## Requirements

- Python >= 3.8
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0

## Dependencies by Module

### Machine Learning (ML)
- scikit-learn
- xgboost
- lightgbm

### Deep Learning
- torch
- tensorflow
- keras

### Generative AI (GenAI)
- openai
- langchain
- huggingface-hub

## Examples

See the `examples/` directory for jupyter notebooks and scripts demonstrating library usage.

## Testing

Run tests with pytest:

```bash
pytest tests/
```

With coverage:

```bash
pytest tests/ --cov=pgl_utils
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues, questions, or suggestions, please open an issue on [GitHub](https://github.com/renansantosmendes/pgl_utils/issues).

## Changelog

### Version 0.1.0
- Initial release
- Core functionality for ML, Deep Learning, and GenAI
- Institution-specific extensions for PUC and IBMEC
