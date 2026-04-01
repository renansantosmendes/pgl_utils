# Installation Guide for Students

## For PUC Students

### Step 1: Install the library

```bash
pip install post-graduation-utils
```

### Step 2: Verify PUC-specific tools

```python
from post_graduation_utils.puc import config

print(config.PUCConfig.get_info())
# Output: PUC Utils v1.0.0
```

### Step 3: Start using the library

```python
from post_graduation_utils import ml, dl, genai
# Your code here
```

---

## For IBMEC Students

### Step 1: Install the library

```bash
pip install post-graduation-utils
```

### Step 2: Verify IBMEC-specific tools

```python
from post_graduation_utils.ibmec import config

print(config.IBMECConfig.get_info())
# Output: IBMEC Utils v1.0.0
```

### Step 3: Start using the library

```python
from post_graduation_utils import ml, dl, genai
# Your code here
```

---

## Installation with Specific Features

### Machine Learning Focus

```bash
pip install "post-graduation-utils[ml]"
```

Includes: scikit-learn, xgboost, lightgbm

### Deep Learning Focus

```bash
pip install "post-graduation-utils[dl]"
```

Includes: torch, tensorflow, keras

### Generative AI Focus

```bash
pip install "post-graduation-utils[genai]"
```

Includes: openai, langchain, huggingface-hub

### All Features

```bash
pip install "post-graduation-utils[all]"
```

Includes everything above

---

## Development Installation (For instructors)

```bash
git clone https://github.com/renansantosmendes/post_graduation_utils.git
cd post_graduation_utils
pip install -e ".[dev]"
```

---

## Troubleshooting

### Import errors

If you get `ModuleNotFoundError`, ensure the library is installed:

```bash
pip list | grep post-graduation-utils
```

If not installed, run:

```bash
pip install post-graduation-utils
```

### Missing dependencies for specific features

If you get errors about missing modules, install the feature extras:

```bash
# For ML features
pip install "post-graduation-utils[ml]"

# For DL features
pip install "post-graduation-utils[dl]"

# For GenAI features
pip install "post-graduation-utils[genai]"
```

### Upgrade to latest version

```bash
pip install --upgrade post-graduation-utils
```

---

## Uninstall

```bash
pip uninstall post-graduation-utils
```
