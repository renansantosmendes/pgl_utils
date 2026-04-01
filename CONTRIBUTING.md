# Contributing to Post Graduation Utils

We welcome contributions from students and instructors! Here's how you can help improve this library.

## Getting Started

### 1. Set up Development Environment

```bash
git clone https://github.com/renansantosmendes/post_graduation_utils.git
cd post_graduation_utils
pip install -e ".[dev]"
```

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 3. Make Your Changes

Follow these guidelines:
- Write clear, documented code
- Add tests for new features
- Follow Python best practices (PEP 8)
- Use meaningful commit messages

### 4. Test Your Changes

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=post_graduation_utils
```

### 5. Code Style

Use black for formatting:

```bash
black post_graduation_utils/
```

Check with flake8:

```bash
flake8 post_graduation_utils/
```

### 6. Submit a Pull Request

Push to your branch and create a PR on GitHub.

---

## Adding New Features

### Adding a Feature to an Existing Module

1. Edit the appropriate file in `post_graduation_utils/{module}/`
2. Update the `__init__.py` to export your function/class
3. Write tests in `tests/test_{module}.py`
4. Add documentation in docstrings
5. Update `README.md` if necessary

Example:

```python
# In post_graduation_utils/ml/preprocessing.py
def new_preprocessing_function(data):
    """
    Description of what this function does.
    
    Parameters:
    -----------
    data : array-like
        Input data
        
    Returns:
    --------
    array-like
        Processed data
    """
    # Implementation
    return processed_data
```

### Adding a New Module

1. Create directory: `post_graduation_utils/new_module/`
2. Create `__init__.py` in the new directory
3. Add feature files: `feature1.py`, `feature2.py`, etc.
4. Update main `post_graduation_utils/__init__.py`
5. Create tests in `tests/test_new_module.py`
6. Document in `docs/`

### Adding Institution-Specific Features

For PUC:
- Add to `post_graduation_utils/puc/`
- Follow naming conventions and document well

For IBMEC:
- Add to `post_graduation_utils/ibmec/`
- Follow naming conventions and document well

---

## Writing Tests

Tests are essential! Here's the structure:

```python
# In tests/test_module.py
import pytest
from post_graduation_utils.module import function

class TestMyFunction:
    def test_basic_functionality(self):
        result = function(input_data)
        assert result == expected_output
    
    def test_edge_case(self):
        with pytest.raises(ValueError):
            function(invalid_input)
```

Run tests:

```bash
pytest tests/
pytest tests/test_module.py::TestMyFunction::test_basic_functionality
```

---

## Documentation

Good documentation is crucial:

### Docstrings

```python
def my_function(param1, param2):
    """
    Brief description of function.
    
    Longer description explaining what the function does,
    any important details, and usage patterns.
    
    Parameters:
    -----------
    param1 : type
        Description of param1
    param2 : type
        Description of param2
        
    Returns:
    --------
    type
        Description of return value
        
    Examples:
    ---------
    >>> result = my_function(1, 2)
    >>> print(result)
    3
    
    Notes:
    ------
    Any additional notes or warnings.
    """
    # Implementation
    return result
```

### Module Documentation

Update relevant files in `docs/`:
- `ARCHITECTURE.md` - if changing overall structure
- Create new `.md` files for major features

---

## Commit Message Guidelines

Write clear commit messages:

```
[TYPE] Brief description (50 chars max)

Longer description explaining what changed and why.
Mention issue numbers if relevant: #123

- Bullet point 1
- Bullet point 2
```

Types:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `test:` - Tests
- `refactor:` - Code refactoring
- `style:` - Code style (formatting)

Example:

```
feat: Add data normalization to ml.preprocessing

- Implements MinMaxScaler wrapper
- Adds unit tests
- Updates documentation
```

---

## Code Review Process

1. **Create PR**: Push to your branch and open a PR
2. **Automated Checks**: Tests run automatically
3. **Review**: Instructors review your code
4. **Feedback**: Address any feedback
5. **Merge**: Your code is merged!

---

## Questions?

- Open an issue on GitHub
- Ask during office hours
- Check existing documentation
- Review similar existing code

---

## Code of Conduct

Be respectful and constructive. We value:
- Clear communication
- Helpful feedback
- Learning from mistakes
- Collaborative improvement

---

Thank you for contributing! 🎉
