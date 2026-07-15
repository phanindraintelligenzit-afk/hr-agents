#!/usr/bin/env python3
"""Path-sanitized runner for HR Agents demo.
Removes Hermes paths from sys.path and PYTHONPATH to avoid version conflicts.
"""
import os
import sys

# Strip Hermes paths from PYTHONPATH
orig_pythonpath = os.environ.get("PYTHONPATH", "")
cleaned = ";".join(
    p for p in orig_pythonpath.split(";")
    if p.strip() and "hermes" not in p.lower()
)
os.environ["PYTHONPATH"] = cleaned

# Fix sys.path
sys.path[:] = [p for p in sys.path if "hermes" not in p.lower()]

# Add project src
project_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if project_src not in sys.path:
    sys.path.insert(0, project_src)

# Now run the demo
os.chdir(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    import demo
    demo.main()