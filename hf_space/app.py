import os
import sys
from pathlib import Path

# Add the space root directory to the python path
sys.path.append(str(Path(__file__).resolve().parent))

# Import the demo interface from the demo package
from demo.app import demo

if __name__ == "__main__":
    demo.launch()
