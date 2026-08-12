"""Launch the Agentic Finance dashboard.

Convenience wrapper so the UI starts with a single command:

    python run_ui.py

(equivalent to ``streamlit run ui/app.py``). Requires the ``ui`` extra:
``pip install -e ".[ui]"`` (or ``.[dev]``).
"""
import os
import sys

from streamlit.web import cli as stcli

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    app = os.path.join(HERE, "ui", "app.py")
    sys.argv = ["streamlit", "run", app, "--server.headless=true"]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
