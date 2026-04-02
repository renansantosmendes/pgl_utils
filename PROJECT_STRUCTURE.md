# 📁 Project Structure Overview

## Complete Directory Tree

```
post_graduation_utils/
│
├── 📦 post_graduation_utils/          # Main package (what gets installed)
│   ├── __init__.py                    # Package initialization
│   │
│   ├── 📁 core/                       # ⭐ CORE - Shared Utilities
│   │   ├── __init__.py
│   │   └── utils.py                   # Common utility functions
│   │
│   ├── 📁 ml/                         # 🤖 MACHINE LEARNING
│   │   ├── __init__.py
│   │   ├── preprocessing.py           # Data prep, cleaning, feature engineering
│   │   └── models.py                  # Model building, evaluation, utilities
│   │
│   ├── 📁 deep_learning/              # 🧠 DEEP LEARNING
│   │   ├── __init__.py
│   │   ├── architectures.py           # Neural network architectures
│   │   └── training.py                # Training loops, callbacks, utilities
│   │
│   ├── 📁 genai/                      # ✨ GENERATIVE AI
│   │   ├── __init__.py
│   │   ├── llm.py                     # Large Language Model utilities
│   │   └── rag.py                     # RAG implementations
│   │
│   ├── 📁 puc/                        # 🏫 PUC-SPECIFIC
│   │   ├── __init__.py
│   │   └── config.py                  # PUC-specific configurations
│   │
│   └── 📁 ibmec/                      # 🏫 IBMEC-SPECIFIC
│       ├── __init__.py
│       └── config.py                  # IBMEC-specific configurations
│
├── 📁 tests/                          # ✅ UNIT TESTS
│   ├── __init__.py
│   └── test_core.py                   # Tests for core module
│
├── 📁 examples/                       # 📚 EXAMPLE SCRIPTS
│   └── example_basic.py               # Basic usage example
│
├── 📁 docs/                           # 📖 DOCUMENTATION
│   ├── ARCHITECTURE.md                # Project architecture details
│   ├── INSTALLATION_GUIDE.md          # How to install (for students)
│   └── GETTING_STARTED.md             # Quick start guide
│
├── 📄 setup.py                        # Package installation config
├── 📄 pyproject.toml                  # Modern Python project config
├── 📄 requirements.txt                # Dependencies list
├── 📄 pytest.ini                      # Pytest configuration
├── 📄 .flake8                         # Code style configuration
├── 📄 .gitignore                      # Git ignore rules
├── 📄 README.md                       # Main documentation
├── 📄 STUDENT_README.md               # Student-friendly guide
├── 📄 CONTRIBUTING.md                 # How to contribute
├── 📄 LICENSE                         # MIT License
├── 📄 MANIFEST.in                     # Package manifest
└── 📄 PROJECT_STRUCTURE.md            # This file!

```

---

## Module Purposes

### 🔹 `core/`
**Shared utilities used by all modules**
- Common helper functions
- Utility classes
- Base implementations

### 🔹 `ml/`
**Machine Learning Tools**
- Data preprocessing (scaling, normalization, encoding)
- Feature engineering utilities
- Model selection and evaluation
- Training and prediction helpers

### 🔹 `deep_learning/`
**Deep Learning Tools**
- Pre-built neural network architectures
- Training utilities (loss functions, optimizers)
- Callbacks and monitoring
- Transfer learning helpers

### 🔹 `genai/`
**Generative AI Tools**
- Large Language Model (LLM) wrappers
- Prompt engineering utilities
- Retrieval-Augmented Generation (RAG)
- Vector databases and embeddings

### 🔹 `puc/`
**PUC-Specific Extensions**
- PUC institutional configurations
- Preferred tools and frameworks for PUC
- Custom examples for PUC curriculum
- PUC student resources

### 🔹 `ibmec/`
**IBMEC-Specific Extensions**
- IBMEC institutional configurations
- Preferred tools and frameworks for IBMEC
- Custom examples for IBMEC curriculum
- IBMEC student resources

---

## Key Files

| File | Purpose |
|------|---------|
| `setup.py` | Installation configuration (pip) |
| `pyproject.toml` | Modern Python project config |
| `requirements.txt` | Dependency list |
| `README.md` | Technical documentation |
| `STUDENT_README.md` | Student-friendly guide |
| `CONTRIBUTING.md` | How to contribute |
| `.gitignore` | Git exclusion rules |
| `pytest.ini` | Test configuration |
| `.flake8` | Code style configuration |

---

## Installation Patterns

### For Students

```bash
# Basic installation
pip install post-graduation-utils

# With ML tools
pip install "post-graduation-utils[ml]"

# With Deep Learning tools
pip install "post-graduation-utils[deep_learning]"

# With GenAI tools
pip install "post-graduation-utils[genai]"

# Everything
pip install "post-graduation-utils[all]"
```

### For Developers

```bash
# Clone and develop
git clone https://github.com/renansantosmendes/post_graduation_utils.git
cd post_graduation_utils

# Install in editable mode with dev tools
pip install -e ".[dev]"
```

---

## Usage by Module

### Importing from Core

```python
from post_graduation_utils import core
from post_graduation_utils.core import utils
```

### Importing from ML

```python
from post_graduation_utils import ml
from post_graduation_utils.ml import preprocessing, models
```

### Importing from Deep Learning

```python
from post_graduation_utils import deep_learning
from post_graduation_utils.deep_learning import architectures, training
```

### Importing from GenAI

```python
from post_graduation_utils import genai
from post_graduation_utils.genai import llm, rag
```

### Importing Institution-Specific

```python
# PUC Students
from post_graduation_utils.puc import config
puc_info = config.PUCConfig.get_info()

# IBMEC Students
from post_graduation_utils.ibmec import config
ibmec_info = config.IBMECConfig.get_info()
```

---

## File Naming Conventions

- `__init__.py` - Package initialization
- `{feature}.py` - Implementation files
- `test_{module}.py` - Unit tests for a module
- `example_{feature}.py` - Example scripts

---

## Documentation Links

- 📖 [Main README](README.md)
- 🎓 [Student Guide](STUDENT_README.md)
- 🚀 [Installation Guide](docs/INSTALLATION_GUIDE.md)
- 🎯 [Getting Started](docs/GETTING_STARTED.md)
- 🏗️ [Architecture](docs/ARCHITECTURE.md)
- 🤝 [Contributing Guide](CONTRIBUTING.md)

---

## Development Workflow

1. **Clone the repo**
   ```bash
   git clone https://github.com/renansantosmendes/post_graduation_utils.git
   cd post_graduation_utils
   ```

2. **Install with dev tools**
   ```bash
   pip install -e ".[dev]"
   ```

3. **Make changes** in appropriate modules

4. **Write tests**
   ```bash
   pytest tests/
   ```

5. **Format code**
   ```bash
   black post_graduation_utils/
   flake8 post_graduation_utils/
   ```

6. **Submit PR**

---

## Quick Reference

| Task | Command |
|------|---------|
| Install | `pip install post-graduation-utils` |
| Install with ML | `pip install "post-graduation-utils[ml]"` |
| Run tests | `pytest tests/` |
| Format code | `black post_graduation_utils/` |
| Check style | `flake8 post_graduation_utils/` |
| Build package | `python setup.py sdist bdist_wheel` |
| Upload to PyPI | `twine upload dist/*` |

---

## Next Steps

1. ✅ Project structure created
2. 📦 Ready for students to install
3. 📝 Ready to add your content to modules
4. 🚀 Ready to push to GitHub and distribute

---

**Happy coding!** 🎉
