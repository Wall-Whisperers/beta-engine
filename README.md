# Beta Engine – Docker Demo (Image to Black-and-White)

This repository is designed as a **clear Docker demo** for collaborators across macOS, Windows, and Linux.

The demo app:
- runs a small Flask web server,
- accepts image uploads,
- converts images to black-and-white using Pillow,
- returns a downloadable PNG.

---

## Why Docker helps this team

Docker gives everyone the same runtime and dependency environment regardless of machine:
- same Python version,
- same system libraries,
- same package versions,
- fewer setup issues between collaborators.

This is a **demo/test app** intended to teach workflow and portability, not production hardening.

---

## Prerequisites

Install Docker first:
- **macOS**: Docker Desktop
- **Windows**: Docker Desktop (enable WSL2 integration)
- **Linux**: Docker Engine + Docker CLI

Verify install:

```bash
docker --version
```

---

## Full Docker workflow (recommended for first-time users)

### 1) Initial setup using Docker

From the repository root, build the image:

```bash
docker build -t beta-engine:demo .
```

What this does:
- reads the `Dockerfile`,
- installs dependencies from `requirements.txt`,
- creates a reusable image named `beta-engine:demo`.

### 2) Running the app

Start a container from the image:

```bash
docker run --rm -p 8000:8000 --name beta-engine-demo beta-engine:demo
```

Then open:
- http://localhost:8000

Upload an image and the app returns a black-and-white PNG.

### 3) Closing the app

In the terminal running the container, press `Ctrl + C`.

Because we used `--rm`, the stopped container is removed automatically.

### 4) Cleaning

Optional cleanup commands:

```bash
docker ps -a
docker images
```

Remove the demo image if you want a completely clean state:

```bash
docker rmi beta-engine:demo
```

Optional system cleanup (removes unused Docker data):

```bash
docker system prune
```

### 5) Re-running the app

If the image still exists:

```bash
docker run --rm -p 8000:8000 --name beta-engine-demo beta-engine:demo
```

If you removed the image in cleanup, rebuild first:

```bash
docker build -t beta-engine:demo .
docker run --rm -p 8000:8000 --name beta-engine-demo beta-engine:demo
```

### 6) Closing again

Press `Ctrl + C` in the running container terminal.

---

## Helpful commands

View running containers:

```bash
docker ps
```

View all containers (including stopped):

```bash
docker ps -a
```

View images:

```bash
docker images
```

Follow container logs (if running detached):

```bash
docker logs -f beta-engine-demo
```

Run in background (detached mode):

```bash
docker run -d --rm -p 8000:8000 --name beta-engine-demo beta-engine:demo
```

Stop detached container:

```bash
docker stop beta-engine-demo
```

---

## OS-specific notes

### macOS
- If port 8000 is busy, use another local port:
  ```bash
  docker run --rm -p 8080:8000 --name beta-engine-demo beta-engine:demo
  ```
  Then open `http://localhost:8080`.

### Windows (PowerShell)
- Use the same commands in PowerShell.
- If Docker asks for file-sharing permissions, allow access to the project directory.
- Alternate port example:
  ```powershell
  docker run --rm -p 8080:8000 --name beta-engine-demo beta-engine:demo
  ```

### Linux
- If your Docker install requires sudo:
  ```bash
  sudo docker build -t beta-engine:demo .
  sudo docker run --rm -p 8000:8000 --name beta-engine-demo beta-engine:demo
  ```

---

## Optional local run (without Docker)

If you want to compare local setup vs Docker setup:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:8000`.
