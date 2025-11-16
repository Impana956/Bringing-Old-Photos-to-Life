"""
Convenience launcher for the Streamlit app.
Run with: python run_app.py
This will invoke the same Python interpreter to run "streamlit run app.py".
"""
import sys
import subprocess
import os

HERE = os.path.dirname(__file__)
APP = os.path.join(HERE, "app.py")

cmd = [sys.executable, "-m", "streamlit", "run", APP]

print("Starting Image Processing Suite: ", " ".join(cmd))
subprocess.run(cmd)
