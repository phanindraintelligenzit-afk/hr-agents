"""Fix sys.path to prefer project venv over Hermes system site-packages."""
import sys

# Remove Hermes site-packages from path to avoid version conflicts
sys.path[:] = [p for p in sys.path if 'hermes' not in p.lower()]