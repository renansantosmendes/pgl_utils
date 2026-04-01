# Project Architecture

## Overview

Post Graduation Utils is organized into **functional modules** with **institution-specific extensions**.

### Design Principles

1. **Shared Core**: Common functionality in `core/` used by all students
2. **Feature Modules**: Organized by domain (ML, DL, GenAI)
3. **Institution Extensions**: Specific configurations for PUC and IBMEC

## Module Structure

### `core/`
Contains shared utilities used across the library:
- General utility functions
- Common configurations
- Helper functions

### `ml/` (Machine Learning)
Machine Learning utilities:
- `preprocessing.py`: Data preprocessing, cleaning, feature engineering
- `models.py`: Model training, evaluation, and selection utilities

### `dl/` (Deep Learning)
Deep Learning utilities:
- `architectures.py`: Pre-built neural network architectures
- `training.py`: Training loops, callbacks, and utilities

### `genai/` (Generative AI)
Generative AI utilities:
- `llm.py`: Large Language Model wrappers and utilities
- `rag.py`: Retrieval-Augmented Generation implementations

### `puc/` (PUC-Specific)
Customizations for PUC students:
- Institution-specific configurations
- PUC-preferred frameworks and approaches
- Curated examples and tutorials

### `ibmec/` (IBMEC-Specific)
Customizations for IBMEC students:
- Institution-specific configurations
- IBMEC-preferred frameworks and approaches
- Curated examples and tutorials

## Installation Patterns

```bash
# Just core (lightweight)
pip install post-graduation-utils

# With ML tools
pip install post-graduation-utils[ml]

# With DL tools
pip install post-graduation-utils[dl]

# With GenAI tools
pip install post-graduation-utils[genai]

# Everything
pip install post-graduation-utils[all]
```

## Development Workflow

### Adding New Features

1. Identify the appropriate module (ml, dl, genai, core)
2. Add implementation to corresponding `.py` file
3. Update `__init__.py` to export new features
4. Add tests in `tests/`
5. Document in `docs/` or add examples in `examples/`

### Adding Institution-Specific Features

1. Add to `puc/` or `ibmec/` module
2. Import from core modules as needed
3. Document institutional differences in module docstrings

## Dependency Management

Dependencies are organized by feature:
- **Base**: numpy, pandas, scikit-learn
- **ML**: xgboost, lightgbm
- **DL**: torch, tensorflow, keras
- **GenAI**: openai, langchain, huggingface-hub

Students install only what they need!
