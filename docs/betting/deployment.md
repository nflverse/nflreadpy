# Betting Deployment Guide

The Bloomberg-style betting stack ships with a layered configuration system and
container images that make it straightforward to deploy the analytics suite to
local workstations or production infrastructure. This guide walks through the
recommended environment setup, secrets management strategy, and container
workflows.

## Environment setup

1. **Install Python and uv** – The betting utilities target Python 3.10+.
   Installing [uv](https://docs.astral.sh/uv/) keeps the local environment in
   sync with the repository lock file:

   ```bash
   pip install --upgrade uv
   uv sync --locked --all-extras --dev
   ```

2. **Review the configuration files** – The default betting configuration lives
   at `config/betting.yaml`. Copy this file to a safe location before editing or
   create an environment-specific overlay such as
   `config/betting.production.yaml`.

3. **Verify the stack** – Validate the active configuration and run the
   regression tests before rolling out changes:

   ```bash
   uv run nflreadpy-betting validate-config --config config/betting.yaml \
     --environment production --warnings-as-errors
   uv run pytest tests/betting/test_configuration.py
   ```

## Layered configuration

Configuration is resolved in multiple layers:

1. `config/betting.yaml` provides the baseline definitions for scrapers,
   ingestion defaults, and analytics thresholds.
2. If the environment variable `NFLREADPY_BETTING_ENV` is set (for example to
   `production`), the loader merges `config/betting.<env>.yaml` on top of the
   base file.
3. Additional overrides can be supplied through `NFLREADPY_BETTING_CONFIG`,
   which accepts one or more YAML files separated by the OS path separator.
4. Finally, any environment variable matching the pattern
   `NFLREADPY_BETTING__SECTION__KEY=value` updates individual fields. Values are
   parsed as JSON when possible, so numbers and booleans work naturally.

Strings in the YAML files support shell-style placeholders. A value such as
`"${STORAGE_ROOT}/odds.sqlite3"` will expand using `STORAGE_ROOT` from the
process environment, allowing secrets or machine-specific paths to be supplied
at deploy time without committing them to source control.

The resulting configuration object is used to build sportsbook scrapers,
initialise the `OddsIngestionService`, and seed the analytics layer, ensuring
that defaults remain consistent across the CLI, ingestion scheduler, and
visualisations.

## Secrets management

* **Environment variables first** – Use `${VAR_NAME}` placeholders in the YAML
  files to pull API keys or credentials from the runtime environment. For
  example, the DraftKings scraper headers in
  `config/betting.production.yaml` reference `${DRAFTKINGS_API_TOKEN}`.
* **Per-environment overrides** – Store production-only secrets in a separate
  YAML file that is referenced via `NFLREADPY_BETTING_CONFIG`. This keeps the
  base configuration usable for local development while allowing a deployment
  system (GitHub Actions, Kubernetes, etc.) to inject sensitive values.
* **Secret stores** – When running in managed environments, populate the
  environment variables from your platform’s secret manager (e.g. GitHub
  Actions secrets, AWS Secrets Manager, HashiCorp Vault). The layered loader will
  pick them up automatically.

## Container images

Two Dockerfiles are provided at the repository root:

* `Dockerfile.cpu` – A slim Python 3.11 image optimised for general-purpose
  deployments.
* `Dockerfile.gpu` – An NVIDIA CUDA 12.2 runtime image suitable for GPU-backed
  analytics or accelerated simulations.

Build the images locally with:

```bash
docker build -f Dockerfile.cpu -t nflreadpy:cpu .
docker build -f Dockerfile.gpu -t nflreadpy:gpu .
```

Run the CPU image while mounting a configuration directory and exposing the
SQLite output:

```bash
docker run --rm \
  -e NFLREADPY_BETTING_ENV=production \
  -v "$PWD/config":/app/config \
  -v "$PWD/data":/app/data \
  nflreadpy:cpu nflreadpy-betting ingest --interval 60
```

## Continuous delivery

The release workflow (`.github/workflows/ci-publish.yaml`) now builds and pushes
both container variants to the GitHub Container Registry (`ghcr.io/<owner>/<repo>`)
whenever a GitHub release is published. Tags follow the pattern
`<version>-cpu` / `<version>-gpu` alongside rolling `latest-*` tags, so staging
clusters can track stable versions while developers have an easy latest tag for
experimentation.

By combining layered configuration, environment-driven secrets, and automated
container builds, you can tailor the betting stack to multiple environments and
deployments without forking the codebase.
