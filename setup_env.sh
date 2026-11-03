#!/usr/bin/env bash
# Environment setup script for aviation-safety-nlp (Advanced Data Mining)
set -e

echo "=========================================================="
echo "   Environment Setup: aviation-safety-nlp (Python 3.14)   "
echo "=========================================================="

# 1. Check for uv installation
if ! command -v uv &> /dev/null; then
    echo "[!] 'uv' was not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "[+] Detected uv: $(uv --version)"

# 2. Ensure Python 3.14 is installed
echo ""
echo "[*] [1/3] Ensuring Python 3.14 is available..."
uv python install 3.14

# 3. Synchronize virtual environment (.venv) and dependencies
echo ""
echo "[*] [2/3] Syncing virtual environment and installing packages..."
uv sync

# 4. Register Jupyter kernel
echo ""
echo "[*] [3/3] Registering Jupyter kernel..."
uv run python -m ipykernel install --user --name "aviation-safety-nlp" --display-name "Python 3.14 (aviation-safety-nlp)"

# 5. Optional NLTK resource setup
echo ""
echo "[*] Pre-downloading common NLTK resources..."
uv run python -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('wordnet', quiet=True); print('[+] NLTK resources ready.')" || true

echo ""
echo "=========================================================="
echo "   Environment setup completed successfully!              "
echo "=========================================================="
echo "Usage options:"
echo "  1. Direct execution via uv (recommended):"
echo "       uv run jupyter lab"
echo "       uv run python <script.py>"
echo "  2. Traditional environment activation:"
echo "       source .venv/bin/activate"
echo "  3. Adding dependencies:"
echo "       uv add <package-name>"
echo "=========================================================="
