# Beta Engine – Docker Demo (Image to Black-and-White)

This repository is designed as a **clear Docker demo** for collaborators using different operating systems.

The app is intentionally simple but dependency-heavy enough to demonstrate Docker value:
- Web server: Flask
- Image processing: Pillow

You upload an image, and the app returns a black-and-white PNG.

---

## Why this is useful for teams

When everyone runs the app in Docker:
- macOS, Windows, and Linux users run the same environment.
- No “it works on my machine” package/version mismatch.
- No need to manually troubleshoot local Python/Pillow setup differences.

---

## What this app is (and is not)

- ✅ A **demo/test app** to show Docker workflow and portability.
- ✅ Good for onboarding teammates who are new to containers.
- ❌ Not production-hardened (no auth, no persistent storage, minimal validation).

The explanation above is also documented in `app.py` itself.

---

## Project files

- `app.py` — demo Flask server with image upload + black-and-white conversion.
- `requirements.txt` — Python dependencies.
- `Dockerfile` — image build instructions.

---

## Prerequisite: install Docker Desktop / Docker Engine

- **macOS**: install Docker Desktop, open it once, wait until Docker is running.
- **Windows**: install Docker Desktop, enable WSL2 integration when prompted, then start Docker Desktop.
- **Linux**: install Docker Engine + Docker CLI for your distro and ensure the Docker daemon is running.

To verify Docker is available, run:

```bash
docker --version
```

---

## Quick start (same commands on macOS, Windows PowerShell, and Linux)

From the repository root:

```bash
docker build -t beta-engine:demo .
docker run --rm -p 8000:8000 beta-engine:demo
```

Now open:

- http://localhost:8000

Upload an image and download the converted black-and-white output.

---

## OS-specific notes

### macOS

- If port `8000` is busy, change to another local port:
  ```bash
  docker run --rm -p 8080:8000 beta-engine:demo
  ```
  Then open `http://localhost:8080`.

### Windows (PowerShell)

- Use the same commands as above in PowerShell.
- If you see a file-sharing warning in Docker Desktop, allow access to your project folder.
- If port `8000` is busy:
  ```powershell
  docker run --rm -p 8080:8000 beta-engine:demo
  ```

### Linux

- If `docker` needs sudo on your machine:
  ```bash
  sudo docker build -t beta-engine:demo .
  sudo docker run --rm -p 8000:8000 beta-engine:demo
  ```
- Optional: configure your user for non-sudo Docker usage.

---

## Stop the app

In the terminal running the container, press `Ctrl + C`.

---

## Optional local (non-Docker) run

If you want to compare Docker vs local Python setup:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Then visit `http://localhost:8000`.
