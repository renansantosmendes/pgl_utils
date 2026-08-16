# CI/CD Setup Guide

This guide explains how to set up Continuous Integration/Continuous Deployment (CI/CD) for automatic testing and publishing to PyPI.

## Overview

The project uses **GitHub Actions** to:

1. **Test** - Run tests on every push and pull request
2. **Publish** - Automatically publish to PyPI when code is merged to `main`

## Workflows Included

### 1. `tests.yml` - Continuous Testing
- Runs on every **push** and **pull request**
- Tests on multiple OS: Ubuntu, Windows, macOS
- Tests on Python versions: 3.10, 3.11
- Checks code style with black
<!-- - flake8 linting foi desabilitado -->
- Runs type checking with mypy
- Uploads coverage reports to Codecov

### 2. `release.yml` - Automated Versioning
- Runs on every push to `main`
- Analyzes Conventional Commit messages since the last release
- Bumps the version in `pyproject.toml`/`setup.py`, updates `CHANGELOG.md`, tags the commit (`vX.Y.Z`), and creates the GitHub Release
- Skips itself when there's nothing releasable, or when the triggering commit is its own `chore(release):` commit (avoids loops)

### 3. `publish-to-pypi.yml` - Automated Publishing
- Triggers on the `vX.Y.Z` tag created by `release.yml`
- Runs tests first
- Builds and publishes the package to PyPI

## Setup Instructions

### Step 1: Create PyPI Account

1. Go to [https://pypi.org/](https://pypi.org/)
2. Click "Register" or login if you already have an account
3. Create your account

### Step 2: Create PyPI API Token

1. Login to PyPI
2. Go to Account Settings → API Tokens → [Add API token](https://pypi.org/manage/account/tokens/create/)
3. Name it: `GitHub Actions`
4. Scope: `Entire account` (or limit to specific project)
5. Copy the token (format: `pypi-AgEIcHlwaS5vcmc...`)

### Step 3: Add Secret to GitHub

1. Go to your GitHub repository
2. Settings → Secrets and variables → Actions
3. Click "New repository secret"
4. Name: `PYPI_API_TOKEN`
5. Value: Paste your PyPI token
6. Click "Add secret"

### Step 4: Update Project Name in PyPI

⚠️ **Important**: The package name in your GitHub workflow must match what's in PyPI!

In your terminal:

```bash
# Reserve the package name on PyPI
python -m pip install twine
python -m build
twine upload dist/* --username __token__ --password pypi-YOUR_TOKEN_HERE
```

Or manually upload a test version:

```bash
# Build the package
python -m build

# Upload to PyPI (first time only)
twine upload dist/*
```

## Workflow Details

### Testing Workflow (`tests.yml`)

Runs on:
- Every push to `main` or `develop`
- Every pull request to `main` or `develop`

Steps:
1. Checkout code
2. Set up Python
3. Cache pip packages
4. Install dev dependencies
5. Format check with black
<!-- 5. Lint with flake8 (desabilitado) -->
6. Type check with mypy
7. Run tests with pytest
8. Upload coverage to Codecov

### Publishing Workflow (`publish-to-pypi.yml`)

Runs on:
- Push of a `vX.Y.Z` tag (created automatically by `release.yml`)

Steps:
1. **Run all tests** (matrix test job) - must pass!
2. **Build distribution**
   - Builds source distribution (`.tar.gz`)
   - Builds wheel (`.whl`)
3. **Check distribution** with twine
4. **Publish to PyPI** using `PYPI_API_TOKEN`
5. **Create GitHub Release** with version tag

## Versioning

Versioning is fully automated with **[python-semantic-release](https://python-semantic-release.readthedocs.io/)**, driven by
[Conventional Commits](https://www.conventionalcommits.org/) on `main`. There is nothing to bump by hand.

The commit prefix on your PR determines the bump:

| Commit prefix | Bump | Example |
|---|---|---|
| `fix:` | patch (`0.1.12` → `0.1.13`) | `fix: correct off-by-one in sliding window` |
| `feat:` | minor (`0.1.12` → `0.2.0`) | `feat: add plot_full_sliding_progress` |
| `BREAKING CHANGE:` in body/footer, or `feat!:`/`fix!:` | major* | `feat!: rename fetch_price_series signature` |
| `chore:`, `docs:`, `refactor:`, `test:`, `ci:` | no release | `docs: fix README typo` |

\* While the package is `0.x`, `major_on_zero = false` keeps breaking changes bumping the **minor** version instead of jumping to `1.0.0`
(see `[tool.semantic_release]` in `pyproject.toml`).

### What happens when a PR is merged to `main`

1. **`release.yml`** runs, inspects commits since the last tag, and — if there's a releasable change:
   - bumps `version` in `pyproject.toml` and `setup.py`
   - updates `CHANGELOG.md`
   - commits as `chore(release): X.Y.Z [skip ci]` and pushes a `vX.Y.Z` tag
   - creates the GitHub Release
2. The new tag triggers **`publish-to-pypi.yml`**, which runs the tests, builds the package, and publishes to PyPI.

If a merge doesn't contain any `feat:`/`fix:`/breaking-change commits (e.g. only `docs:`/`chore:`), no release is created and nothing is published — as intended.

### Manually triggering a release

You normally never need to do this — just merge a PR with a Conventional Commit message and the pipeline handles the rest.

## Testing Locally

Before pushing to GitHub, test locally:

### Run tests
```bash
pytest tests/
```

### Check code style
```bash
black --check post_graduation_utils/ tests/
# flake8 post_graduation_utils/ tests/  # Desabilitado
```

### Fix style issues
```bash
black post_graduation_utils/ tests/
```

### Build package
```bash
python -m build
```

### Check distribution
```bash
twine check dist/*
```

## Troubleshooting

### Tests fail in CI but pass locally

- Check Python version differences
- Check OS differences (Windows vs Linux)
- Look at CI logs for specific errors

### PyPI upload fails

1. **Token expired/wrong**: Update `PYPI_API_TOKEN` secret
2. **Package name taken**: Change name in `setup.py` and `pyproject.toml`
3. **Version already exists**: Bump version number
4. **Twine check fails**: Fix distribution issues locally first

### How to check logs

1. Go to GitHub repo → Actions tab
2. Click on failed workflow
3. Expand job to see detailed logs

## Security Best Practices

1. ✅ **Use API tokens, not passwords**
2. ✅ **Limit token scope** if possible
3. ✅ **Never commit tokens** to code
4. ✅ **Rotate tokens** periodically
5. ✅ **Use GitHub Secrets** (never hardcode)
6. ✅ **Review** workflow files for security

## Advanced Configuration

### Custom Test Matrix

Edit `.github/workflows/tests.yml`:

```yaml
strategy:
  matrix:
    python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']  # Add 3.12
    os: [ubuntu-latest, windows-latest, macos-latest]
```

### Skip Publishing for Certain Commits

Add `[skip ci]` to commit message:

```bash
git commit -m "Update README [skip ci]"
```

### Conditional Publishing

To publish only on specific conditions, edit `publish-to-pypi.yml`:

```yaml
if: github.event_name == 'push' && github.ref == 'refs/heads/main' && contains(github.event.head_commit.message, 'Release')
```

## Useful Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [PyPI Help](https://pypi.org/help/)
- [twine Documentation](https://twine.readthedocs.io/)
- [Python Packaging Guide](https://packaging.python.org/)

## Support

For issues:
1. Check GitHub Actions logs
2. Review workflow files
3. Check PyPI documentation
4. Ask for help in office hours

---

**Your CI/CD is now set up!** 🚀

Next steps:
1. Get PyPI token
2. Add `PYPI_API_TOKEN` secret to GitHub
3. Make changes and commit to `main`
4. Watch automation happen! ✨
