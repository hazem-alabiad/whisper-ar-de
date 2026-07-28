#!/usr/bin/env python
"""
Root execution wrapper for whisper-tools.
Forward execution to src/whisper_tools/main.py.
"""
import sys
from pathlib import Path

# Ensure src/ directory is in sys.path
src_dir = Path(__file__).parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from whisper_tools.main import main

if __name__ == "__main__":
    main()
