# Aviation Safety NLP (Advanced Data Mining)

A data mining and natural language processing project for the **Advanced Data Mining** course. The project focuses on text mining, exploratory data analysis, and multi-class classification of aviation incident narratives from the NASA ASRS (*Aviation Safety Reporting System*) database.

---

## 🛠️ Prerequisites

- [**uv**](https://docs.astral.sh/uv/) (extremely fast Python package and project manager)
- **Python 3.14** (if not already installed on your system, `uv` will download and manage it automatically)

If you don't have `uv` installed yet, you can install it via PowerShell:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
or on Linux / macOS / WSL:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## 🚀 Quick Start (Automated Setup)

Ready-to-use setup scripts are included in the repository. They automatically set up Python 3.14, sync the virtual environment (`.venv`), install all required dependencies, register the Jupyter kernel, and fetch essential NLP corpora.

### Windows (PowerShell)
```powershell
.\setup_env.ps1
```

### Linux / macOS / WSL
```bash
chmod +x setup_env.sh
./setup_env.sh
```

---

## 💻 Daily Workflow

### 1. Direct Script Execution with `uv run` (Recommended)
You do not need to activate the virtual environment manually. `uv run` automatically executes commands inside the project's virtual environment:
```powershell
# Run feature extraction pipeline
uv run python temp/attributes-extraction.py

# Run baseline TF-IDF + LogisticRegression classification
uv run python temp/pre-classification.py
```

### 2. Interactive Notebooks & JupyterLab
The setup script registers a dedicated Jupyter kernel: `Python 3.14 (aviation-safety-nlp)`.
To launch JupyterLab:
```powershell
uv run jupyter lab
```
When creating or opening a notebook, select the **Python 3.14 (aviation-safety-nlp)** kernel from the kernel dropdown.

### 3. Traditional Virtual Environment Activation
If you prefer activating the shell environment directly:
```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows Command Prompt (CMD)
.\.venv\Scripts\activate.bat

# Linux / macOS
source .venv/bin/activate
```

### 4. Dependency Management
- **Add a package:**
  ```powershell
  uv add <package-name>
  # Example: uv add polars
  ```
- **Add a development/testing package:**
  ```powershell
  uv add --dev pytest ruff
  ```
- **Sync dependencies after git pull:**
  ```powershell
  uv sync
  ```

---

## 📁 Project Structure

```text
aviation-safety-nlp/
├── .python-version          # Pinned Python version (3.14)
├── pyproject.toml           # Project metadata and dependency definitions
├── uv.lock                  # Deterministic dependency lockfile
├── setup_env.ps1            # Setup script for Windows PowerShell
├── setup_env.sh             # Setup script for Linux / macOS / WSL
├── .gitignore               # Ignored artifacts (.venv, caches, checkpoints)
├── data/                    # Datasets directory
│   ├── raw/                 # Raw source data (e.g., ASRS_DBOnline.csv)
│   └── processed/           # Cleaned and engineered datasets (e.g., asrs_selected_features.pkl)
├── notebooks/               # Jupyter notebooks for EDA, modeling, and experiments
├── src/                     # Reusable Python package source code
│   └── aviation_safety_nlp/
│       └── __init__.py
└── temp/                    # Initial research scripts and baseline datasets
    ├── ASRS_DBOnline.csv
    ├── asrs_selected_features.csv
    ├── asrs_selected_features.pkl
    ├── attributes-extraction.py
    ├── parser-test.py
    └── pre-classification.py
```

---

## 📦 Core Installed Packages

- **Data Processing & ML:** `pandas`, `numpy`, `scipy`, `scikit-learn`
- **NLP & Text Mining:** `nltk`, `wordcloud`
- **Visualization:** `matplotlib`, `seaborn`, `plotly`
- **Interactive Environment:** `jupyterlab`, `notebook`, `ipykernel`
- **Utilities:** `tqdm`, `openpyxl`
