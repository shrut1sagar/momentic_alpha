"""
Pytest configuration file.

Ensures the repo root is on sys.path so that 'import src...' works.
"""
import sys
from pathlib import Path

# Add the repo root to sys.path
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
