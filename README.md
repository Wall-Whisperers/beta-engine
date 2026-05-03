# Beta Engine – Python + Docker Starter

This repository is a minimal Python starter that is ready for local development, Docker builds, and GitHub Actions CI.

## What's included

- `app.py`: tiny runtime script that uses one dependency (`requests`) so we can validate dependency install.
- `test_everything_working.py`: smoke test for import + simple HTTP behavior.
- `requirements.txt`: locked top-level dependency list.
- `Dockerfile`: production-lean container build.
- `.dockerignore`: keeps Docker build context small.
- `.github/workflows/ci.yml`: CI for tests + Docker image build on push/PR.

## Quick start (local)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
pytest -q
```

## Docker quick start

```bash
docker build -t beta-engine:dev .
docker run --rm beta-engine:dev
```

## 2026 setup notes (Docker + GitHub)

These project defaults follow current official guidance as of **April 2026**:

1. Use modern Dockerfile syntax header and keep Dockerfiles readable and explicit.
2. Prefer small, trusted base images and pin to stable version tags.
3. Use `.dockerignore` to reduce build context size and improve build speed.
4. Keep containers non-root where practical.
5. Build and test images in CI on every PR/push.
6. In GitHub Actions for Python, use `actions/setup-python` and dependency caching.

References:
- Docker build best practices: https://docs.docker.com/build/building/best-practices/
- Dockerfile concepts: https://docs.docker.com/build/concepts/dockerfile/
- GitHub Actions workflow syntax: https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax
- GitHub guide for building/testing Python: https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python

