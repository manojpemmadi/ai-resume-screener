"""Launch the Streamlit app."""
import os
import subprocess
import sys

if __name__ == "__main__":
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", app_path]))
