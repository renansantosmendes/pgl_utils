# CHANGELOG


## v0.3.0 (2026-08-20)

### Bug Fixes

- Publish to PyPI in the same run as the version release
  ([`9175236`](https://github.com/renansantosmendes/pgl_utils/commit/9175236ce1f8b554343df43349229bdc424e997a))

GITHUB_TOKEN-authored tag pushes from python-semantic-release don't trigger other workflows (GitHub
  Actions loop-prevention), so publish-to-pypi.yml never fired after release.yml tagged v0.2.0.
  Publishing now happens in a `publish` job inside release.yml, gated on release.outputs.released,
  so it runs in the same workflow run as the bump. publish-to-pypi.yml is kept as a manual
  workflow_dispatch fallback to catch up any release that was tagged but never published.

### Documentation

- Document curated ticker list loaders
  ([`3e1beea`](https://github.com/renansantosmendes/pgl_utils/commit/3e1beeadce49868ef60762800a7d276aa510d7b2))

Adds README coverage for load_brazil_tickers()/load_us_tickers(), including the JSON resource paths
  and the snake_case sector keys.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

### Features

- Add curated Brazil and US stock ticker lists
  ([`41b473c`](https://github.com/renansantosmendes/pgl_utils/commit/41b473cd24ff19dca93fe1f6bd840be3013a491a))

Adds JSON resources with B3 and US tickers grouped by sector, plus
  load_brazil_tickers()/load_us_tickers() helpers exposed from pgl_utils.deep_learning so they can
  be imported once the package is installed (resources are loaded via importlib.resources).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>

### Refactoring

- Use snake_case keys in ticker JSON files
  ([`9341403`](https://github.com/renansantosmendes/pgl_utils/commit/934140352e98cc371b54b095d4942a1776b8827a))

Sector names were plain Portuguese labels with accents and spaces, inconvenient as dict keys.
  Renamed to snake_case identifiers, e.g. "Alimentos / bebidas" -> "alimentos_bebidas".

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>


## v0.2.0 (2026-08-16)

### Bug Fixes

- Change pyproject
  ([`aa082ab`](https://github.com/renansantosmendes/pgl_utils/commit/aa082ab4c0afb575db8b80bb26e1e649450e251b))

### Features

- Add stock price plots
  ([`358de8b`](https://github.com/renansantosmendes/pgl_utils/commit/358de8b16858cc30959ea90970fd93d92ed5dcc1))


## v0.1.16 (2026-08-03)


## v0.1.14 (2026-07-01)


## v0.1.13 (2026-05-04)


## v0.1.12 (2026-05-03)


## v0.1.10 (2026-04-30)


## v0.1.9 (2026-04-30)


## v0.1.8 (2026-04-27)


## v0.1.7 (2026-04-26)


## v0.1.5 (2026-04-26)


## v0.1.4 (2026-04-21)


## v0.1.3 (2026-04-19)


## v0.1.2 (2026-04-16)


## v0.1.1 (2026-04-13)

### Bug Fixes

- Correct import path for utils in test_core.py
  ([`e85511a`](https://github.com/renansantosmendes/pgl_utils/commit/e85511a2c1fe820735dc1abef9ca25e23d6271ec))

### Code Style

- Remove unnecessary blank lines in IBMEC and PUC configuration files
  ([`0f868fb`](https://github.com/renansantosmendes/pgl_utils/commit/0f868fb0fdca7883a7e56293df78676b428feee1))

### Features

- Initial commit of Post Graduation Utils library
  ([`6393949`](https://github.com/renansantosmendes/pgl_utils/commit/6393949a65ead60f2bd8eca888c166aa94f2430b))

- Added README.md for project overview and installation instructions. - Created STUDENT_README.md
  for student-specific guidance. - Established project architecture documentation in
  ARCHITECTURE.md. - Implemented CI/CD setup guide in CI_CD_SETUP.md. - Developed Getting Started
  guide in GETTING_STARTED.md. - Provided detailed installation instructions in
  INSTALLATION_GUIDE.md. - Created example script example_basic.py to demonstrate library usage. -
  Initialized core library structure with __init__.py and utility functions. - Developed Machine
  Learning, Deep Learning, and Generative AI modules with placeholder functions. - Added
  institution-specific configurations for PUC and IBMEC. - Set up package configuration in setup.py
  and pyproject.toml. - Configured testing framework with pytest and initial test cases.
