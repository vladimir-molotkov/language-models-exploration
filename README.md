# Language models exploration

Explore and train language models

## Installation guide

1. Clone this repository to local using `git clone`
2. Navigate to `language-models-exploration`:
   ```
   cd language-models-exploration
   ```
4. Install `uv`
   You can install uv on macOS using `curl`:
   ```
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   Aslo you can use `homebrew`:
   ```
   brew install uv
   ```
5. Create and activate python virtural environment
   ```
   uv venv
   source .venv/bin/activate
   ```
6. Install depencies
   ```
   uv pip install -r pyproject.toml --all-extras
   ```
7. Setup pre-commit
   ```
   pre-commit install
   ```
8. Now you can train models and make experiments. For example:
   ```
   uv run models/bert_sst_classification.py
   ```
Note. By default it will not work without existing MLFlow instance. To disable this behavior change hydra config or run with `ml_flow.logging_enable=False`. For example
```
uv run models/bert_sst_classification.py ml_flow.logging_enable=False
