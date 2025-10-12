# Contributing

Many hands make light work! Here are some ways you can contribute to this project:

### Open an issue

- You can [open an issue](https://github.com/nflverse/nflreadpy/issues/new/choose) if you'd like to request a specific function or report a bug/error.

### Fixing typos

*   You can fix typos, spelling mistakes, or grammatical errors in the documentation directly using the GitHub web interface, as long as the changes are made in the _source_ file.

### Bigger changes

*   If you want to make a bigger change, it's a good idea to first file an issue and make sure someone from the team agrees that it’s needed. If you’ve found a bug, please file an issue that illustrates the bug with a minimal
[reprex](https://www.tidyverse.org/help/#reprex) (this will also help you write a unit test, if needed).

*   You can also bring up changes in the [nflverse discord](https://discord.com/invite/5Er2FBnnQa) and our team will be happy to discuss!

### Tooling and Code Practices

This project uses the following tooling:

- uv or pip for dependency management
- ruff and flake8 for linting the betting toolkit
- mypy for type checking (with strict settings on the betting toolkit)
- pytest for testing
- mkdocs for documentation site

You can choose between uv or pip for managing development dependencies.

```bash
# Install development dependencies with uv
uv sync --dev

# Alternatively, install development dependencies with pip
pip install -e .[dev]

# Run tests
uv run pytest

# Lint betting modules
uv run ruff check src/nflreadpy/betting
uv run flake8 src/nflreadpy/betting

# Format code
uv run ruff format

# Type check
uv run mypy src/nflreadpy/betting

# Serve docs site locally
uv run mkdocs serve

# Build docs site
uv run mkdocs build
```

If you prefer pip, you can run the tools directly from your environment, for example
`pytest`, `ruff`, `mypy`, or `mkdocs build --strict`.

### Continuous Integration expectations

Every pull request runs the following GitHub Actions workflows:

- **`ci-test.yaml` / lint job** – executes Ruff, Flake8, and MyPy against `src/nflreadpy/betting/`. Failing lint or type checks will block the PR.
- **`ci-test.yaml` / test matrix** – validates betting configurations, runs ingestion smoke tests, and executes the full pytest suite with coverage across the supported operating systems and Python versions.
- **`ci-docs.yaml`** – ensures the documentation builds cleanly.

Release automation (`ci-publish.yaml`) is gated on the same linting, typing, and testing checks so the published artifacts always reflect a green CI state.

## Code of Conduct

Please note that the nflverse project is released with a
[Contributor Code of Conduct](https://github.com/nflverse/.github/blob/main/.github/CODE_OF_CONDUCT.md). By contributing to this
project you agree to abide by its terms.
