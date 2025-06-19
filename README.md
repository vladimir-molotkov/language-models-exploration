# Language models exploration

Explore and train language models

## Installation guide

1. Clone this repository to local using `git clone`
2. Install `uv`
   You can install uv on macOS using `curl`:
   ```
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   Aslo you can use `homebrew`:
   ```
   brew install uv
   ```
3. Create python virtural environment
   ```
   uv venv
   ```
4. Install depencies
   ```
   uv pip install -r pyproject.toml --all-extras
   ```
5. Setup pre-commit
   ```
   pre-commit install
   ```
6. Now you can train models and make experiments
