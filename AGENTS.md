# AGENTS.md

## Cursor Cloud specific instructions

### Project Overview

SERA (Soft-Verified Efficient Repository Agents) is a Python 3.12 research framework for generating synthetic training data to fine-tune coding agents. It consists of three installable packages:

- **sera** (`/workspace/`) — core pipeline (generate, distill, eval, postprocess)
- **sweagent** (`/workspace/modules/SWE-agent/`) — LLM-based coding agent
- **code2flow** (`/workspace/modules/code2flow/`) — call graph generator

### Dev Install

Per README: `pip install -e . -e modules/code2flow -e modules/SWE-agent`

### Required Setup Gotchas

- SWE-agent requires `config/` and `trajectories/` directories under `modules/SWE-agent/` to exist (asserted on import in `sweagent/__init__.py`). They are not tracked by git in this repo. The update script creates them automatically.
- System dependencies: `graphviz` and `libgraphviz-dev` are needed for `pygraphviz` (used by code2flow tests). `acorn` (npm) is needed for JS parsing tests.
- `$HOME/.local/bin` must be on `PATH` for `ruff`, `pytest`, `sweagent`, and `code2flow` CLI tools.

### Linting

- `ruff check sera/` — lint the sera package
- `ruff check modules/SWE-agent/sweagent/` — lint SWE-agent (ruff config in its `pyproject.toml`)
- `ruff check modules/code2flow/code2flow/` — lint code2flow
- Pre-existing lint errors exist in all three packages; these are upstream issues.

### Testing

- **code2flow**: `cd modules/code2flow && pytest` — Python tests pass; JS/PHP/Ruby tests require external parsers and some have pre-existing failures.
- **SWE-agent**: `cd modules/SWE-agent && pytest tests/test_models.py tests/test_parsing.py tests/test_history_processors.py tests/test_utils.py tests/test_packaging.py` — unit tests that don't require Docker.
- **sera**: No automated tests exist. The main entry point is `python sera/main.py --config-name=<config>` which requires Docker and an LLM inference endpoint.

### Running the Pipeline

The full SERA pipeline (`python sera/main.py --config-name=specialization_django ...`) requires:
1. Docker Engine (for swesmith container builds)
2. An LLM inference endpoint (OpenAI API, Anthropic API, or self-hosted via SGLang/vLLM)

These are external dependencies not available in the default cloud VM. See `README.md` for full usage.
