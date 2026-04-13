# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ai4ceo** — "CEO를 위한 AI coding school" by 장동인. A collection of Streamlit-based AI apps used for teaching. Python 3.13 managed with **uv** (not pip). The UI language is Korean.

The actual Python patch version is pinned in `.python-version` and `pyproject.toml` — treat those files as the source of truth, not this document.

## Setup & Environment

```bash
# Windows (PowerShell)
.\setup.ps1                # installs uv, Python 3.13, creates .venv, runs uv sync

# Linux/macOS
bash setup.sh

# Recreate venv from scratch
.\setup.ps1 -RecreateVenv          # Windows
RECREATE_VENV=1 bash setup.sh      # Linux/macOS
```

Dependencies are declared in `pyproject.toml`. **Always use `uv sync` and `uv add` — never `pip install`.**

## Running Apps

All apps are Streamlit apps. Use `uv run` for platform-neutral execution (no manual venv activation needed):

```bash
# Combined 4-in-1 hub app (recommended entry point)
uv run streamlit run one.py

# Individual apps
uv run streamlit run chatbot.py
uv run streamlit run internet.py
uv run streamlit run rag.py
uv run streamlit run time.py

# Run on a specific port
uv run streamlit run chatbot.py --server.port 8502
```

Windows shortcut (auto-detects `.venv` and runs setup if missing):

```bash
runst one.py
runst chatbot.py --server.port 8502
```

## Architecture

### Core Apps (root directory)

- **`one.py`** — Hub app. Sidebar radio selects between the four sub-apps. Each sub-app exposes a `render_app()` function that `one.py` calls.

  ⚠️ **`time.py` is loaded via `importlib` to avoid shadowing the stdlib `time` module. DO NOT refactor this to a normal `import time` statement — it will break the entire app. This is an intentional design decision, not technical debt.**

- **`chatbot.py`** — Simple OpenAI chatbot using LangChain `ChatOpenAI`. The exact model name is defined inside the file; do not hardcode model assumptions in this document.

- **`internet.py`** — Internet-search chatbot using OpenAI Responses API with the built-in `web_search` tool, enabled via `use_responses_api=True` on `ChatOpenAI`. The exact model name is defined inside the file.

- **`rag.py`** — PDF upload → text chunking → FAISS + BM25 ensemble retriever → RAG Q&A.

- **`time.py`** — Real-time digital clock widget with auto-rerun. Loaded by `one.py` via `importlib` (see warning above).

### `prompt/` — Teaching materials (READ-ONLY)

Contains numbered subdirectories (e.g., `3.app`, `4.ref.py`, `5.SQL-Agent`, `9.Agentic-AI`) with lesson-by-lesson teaching materials and example prompts. **These files are not imported by any app at runtime.** Read these only when explicitly asked, and do not modify them unless 장동인 교수 explicitly requests changes.

### Reference scripts (READ-ONLY)

`ref.py`, `ref-multi.py`, `ref1.py`, `ai4ceo-open.py`, `Internet-RAG-LLM.py` are standalone class demos and reference implementations (multi-page Streamlit apps with SQL agents, data analysis, multi-user sessions, etc.). They are **independent from the `one.py` hub** and should not be modified unless explicitly requested. Do not borrow code from these files into the core apps without explicit instruction.

## Key Conventions

### Streamlit sub-app contract

- Every Streamlit sub-app **must expose a `render_app()` function**.
- Do **NOT** call `st.set_page_config` inside `render_app()` — that is the caller's responsibility (`one.py` or `main()`).

### Caching strategy

- Use `@st.cache_resource` for **LLM clients, retrievers, vectorstores, and any object holding network connections or model weights** (e.g., `ChatOpenAI`, `FAISS`, `BM25Retriever`, `EnsembleRetriever`).
- Use `@st.cache_data` for **dataframes, parsed documents, computed results, and any serializable data**.
- Never use `@st.cache_data` on objects that hold open connections or model state — it will break.

### LangChain imports

The project uses both `langchain` and `langchain-classic` packages. Use them as follows:

- General LLM/chain code: import from `langchain` and its standard subpackages (`langchain_openai`, `langchain_community`, etc.).
- **`EnsembleRetriever`**: must be imported from `langchain_classic.retrievers` — this is the **only** component in this project that requires `langchain-classic`.

### Environment & secrets

- API keys are loaded from `.env` via `python-dotenv`. See `.env-sample` for the required keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, etc.).
- Never commit `.env` or hardcode keys in source files.

### Project metadata

- `[tool.uv] package = false` in `pyproject.toml` — this is a non-packaged **app project**, not a library. Do not add `[project.scripts]` entry points or restructure as a package without explicit instruction.

## Communication

- The UI language is **Korean**. User-facing strings (titles, labels, error messages, prompts) should be written in Korean.
- Code comments and docstrings can be in English or Korean; follow the existing style of the file being edited.
- When responding to 장동인 교수 in chat, use Korean with 존댓말 (formal speech).