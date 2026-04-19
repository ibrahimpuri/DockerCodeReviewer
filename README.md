# DockerCodeReviewer

An AI-powered code review tool with a FastAPI backend and Streamlit frontend. It detects code defects using a fine-tunable transformer model (UniXcoder), runs GPT-4 or Claude for natural-language feedback, and lints Python/JavaScript files automatically.

---

## Features

- **AI Defect Detection** — Uses `microsoft/unixcoder-base` to classify code as defective or clean
- **Natural-Language Review** — GPT-4 or Claude AI provides detailed, actionable feedback
- **Linting** — Pylint (Python) and ESLint (JavaScript) results displayed inline
- **SQLite Database** — Every analysis is auto-saved; browse submission history
- **Fine-Tuning Pipeline** — Confirm/correct predictions to build a training set, then retrain the model in one click
- **Multi-Vendor GPU Acceleration** — NVIDIA CUDA, AMD ROCm, Intel Arc/Xe, Apple Silicon MPS, CPU fallback
- **Professional Dark UI** — Custom-designed Streamlit interface
- **Cross-Platform Docker** — Works on Linux, macOS, and Windows (WSL2)

---

## Quick Start

### 1. Clone the repository

```sh
git clone https://github.com/ibrahimpuri/DockerCodeReviewer.git
cd DockerCodeReviewer
```

### 2. Configure environment variables

```sh
cp .env.example .env
# Edit .env and fill in your API keys
```

`.env` keys:

| Key | Required | Description |
|-----|----------|-------------|
| `OPENAI_API_KEY` | When using GPT-4 | OpenAI API key |
| `CLAUDE_API_KEY` | When using Claude | Anthropic API key |
| `DB_PATH` | No (has default) | SQLite file path (default: `./data/codelens.db`) |
| `MODEL_WEIGHTS_DIR` | No (has default) | Fine-tuned weights directory (default: `./data/model_weights`) |

### 3. Run with Docker

**CPU (works on all platforms):**
```sh
docker compose up --build
```

**NVIDIA GPU** (requires driver ≥ 525 + nvidia-container-toolkit):
```sh
docker compose -f docker-compose.yml -f docker-compose.nvidia.yml up --build
```

**AMD GPU** (requires ROCm 6.2+ on Linux):
```sh
docker compose -f docker-compose.yml -f docker-compose.amd.yml up --build
```

**Intel Arc / Iris Xe** (requires Intel compute runtime on Linux):
```sh
docker compose -f docker-compose.yml -f docker-compose.intel.yml up --build
```

**Apple Silicon** — MPS is not available inside Docker containers (Linux VM). For GPU acceleration on Apple Silicon, run natively:
```sh
pip install -r requirements.txt
uvicorn fastapi_backend:app --reload &
streamlit run streamlit_app.py
```

Then open **http://localhost:8501** in your browser.

---

## GPU Support Matrix

| Vendor | Hardware | Backend | Docker override |
|--------|----------|---------|-----------------|
| NVIDIA | RTX / Tesla / Data Center | CUDA 12.6 | `docker-compose.nvidia.yml` |
| AMD | RX / Instinct | ROCm 6.2 (presents as CUDA) | `docker-compose.amd.yml` |
| Intel | Arc / Iris Xe / Flex / Max | Intel XPU (IPEX) | `docker-compose.intel.yml` |
| Apple | M1 / M2 / M3 / M4 | MPS (native only) | — |
| Any | — | CPU fallback | (default) |

The application auto-detects the best available device at startup. No manual configuration needed.

---

## Fine-Tuning Pipeline

1. Upload and review code files — each analysis is automatically saved to the database
2. After viewing results, click **"👍 Prediction was correct"** or **"👎 Prediction was wrong"**
3. Once 20+ labeled samples are collected, the **"🔁 Retrain Model"** button activates in the sidebar
4. Click it — the model fine-tunes in the background and hot-swaps without restarting the container
5. Fine-tuned weights are saved to `data/model_weights/` and loaded automatically on the next restart

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/review` | `POST` | Analyze code; returns defect prediction, AI feedback, and lint results |
| `/feedback` | `POST` | Submit a label correction for a saved submission |
| `/retrain` | `POST` | Fine-tune the model on collected training samples |
| `/stats` | `GET` | Return submission and training sample counts |
| `/health` | `GET` | Health check (used by Docker Compose) |

### `/review` request

```json
{
  "code": "def foo():\n    pass",
  "language": "python",
  "ai_tool": "gpt-4"
}
```

### `/feedback` request

```json
{
  "submission_id": 42,
  "correct_label": 0
}
```

`correct_label`: `0` = clean, `1` = defective

---

## Local Development

```sh
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# CPU
pip install -r requirements.txt

# NVIDIA CUDA 12.6
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# AMD ROCm 6.2
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/rocm6.2
pip install -r requirements.txt

# Run
uvicorn fastapi_backend:app --reload
streamlit run streamlit_app.py
```

---

## Project Structure

```
DockerCodeReviewer/
├── ai_code_reviewer_backend.py  # UniXcoder model, device detection, inference
├── fastapi_backend.py           # REST API (FastAPI)
├── streamlit_app.py             # Frontend UI (Streamlit)
├── database.py                  # SQLite persistence (submissions + training_samples)
├── trainer.py                   # AdamW fine-tuning loop
├── requirements.txt
├── Dockerfile_fastapi
├── Dockerfile_streamlit
├── docker-compose.yml           # Base (CPU)
├── docker-compose.nvidia.yml    # NVIDIA overlay
├── docker-compose.amd.yml       # AMD ROCm overlay
├── docker-compose.intel.yml     # Intel XPU overlay
└── .env.example
```

---

## Technologies

- **Python 3.12**, FastAPI, Streamlit
- **PyTorch 2.6** — CUDA 12.6, ROCm 6.2, Intel IPEX, MPS, CPU
- **Hugging Face Transformers** — `microsoft/unixcoder-base`
- **SQLite** (WAL mode, thread-safe)
- **OpenAI API** (GPT-4), **Anthropic API** (Claude)
- **Pylint**, **ESLint**
- **Docker** / **Docker Compose V2**

---

## Contributing

Pull requests are welcome. Please open an issue first to discuss significant changes.

## License

MIT
