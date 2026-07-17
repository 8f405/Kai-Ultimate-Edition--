# 🤖 AI-Pulse Analyst
### Real-time AI/ML News & Research Assistant

AI-Pulse is an AI research assistant specialized in the Artificial Intelligence industry—covering model releases, architectures, hardware, and breakthrough news from **2025-2026**. Unlike standard LLMs, AI-Pulse eliminates "knowledge cutoff" issues by integrating live web search.

🔗 **Live URL:** [https://ai-pulse-239750720157.us-central1.run.app/)

---

## 🌟 Key Features

* **Real-Time RAG:** Powered by the **Tavily Search API** to fetch breaking news and technical papers.
* **Two-Step Agentic Pipeline:**
*  1.  **Optimization:** Refines user queries into high-intent technical search terms.
    2.  **Synthesis:** Generates structured answers grounded strictly in verified live results.
* **Built-in Safety:** Sophisticated system prompts to filter out-of-scope requests (e.g., cooking, politics) and block adversarial prompt injection.
* **Cloud Native:** Fully containerized and deployed on **Google Cloud Run** for serverless scaling.

---

## 🛠️ Tech Stack

* **Backend:** Python 3.11 + FastAPI
* **AI Engine:** Google Vertex AI (Gemini 1.5) + Tavily AI Search
* **Environment Management:** `uv` (Astral) for lightning-fast builds
* **Infrastructure:** Docker + Google Cloud Build + Cloud Run
* **Evaluation:** Custom eval framework with a 70% pass-rate threshold for CI/CD readiness.

---

## 📂 Project Structure

```text
ai-pulse/
├── app/
│   ├── main.py            # FastAPI entrypoint & routing
│   ├── prompt_config.py   # System prompts & safety logic
│   └── static/            # Web UI (Frontend)
├── evals/
│   ├── dataset.json       # Test cases (In-domain, adversarial, etc.)
│   └── run_evals.py       # Evaluation engine
├── Dockerfile             # Production container config
├── pyproject.toml         # Dependency management (uv)
└── README.md



## How to run locally

### Prerequisites

- [uv](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- GCP project with **Vertex AI API** enabled
- [Tavily API key](https://tavily.com) (free tier available)

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env:
#   TAVILY_API_KEY=tvly-...
#   GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json
#   VERTEX_PROJECT=your-gcp-project-id
#   VERTEX_LOCATION=us-central1
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Start server

```bash
uv run uvicorn app.main:app --reload --port 8000
```

### 4. Open the UI

```
http://localhost:8000
```

### 5. Test manually

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is vibe coding?"}' | python3 -m json.tool
```

---

## Run evals

```bash
# Full evaluation: deterministic + Golden MaaJ + Rubric MaaJ 
uv run python -m evals.run_evals

# Deterministic only — no LLM judge, fast 
uv run python -m evals.run_evals --no-maaj

# Single category
uv run python -m evals.run_evals --category in-domain
uv run python -m evals.run_evals --category out-of-scope
uv run python -m evals.run_evals --category adversarial

# Save full results to JSON
uv run python -m evals.run_evals --save

#Run evals against live URL
CHATBOT_URL=https://ai-pulse-239750720157.us-central1.run.app/ \
  uv run python -m evals.run_evals --no-maaj
```

Exits with code `0` if overall pass rate ≥ 70%, `1` otherwise (CI-friendly).



