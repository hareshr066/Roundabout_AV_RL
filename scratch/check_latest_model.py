import os
import glob

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
search_dir = os.path.join(_ROOT, "results", "models")
candidates = [
    c for c in glob.glob(os.path.join(search_dir, "*.zip"))
    if "val_test" not in os.path.basename(c)
]
if candidates:
    latest = max(candidates, key=os.path.getmtime)
    print("Latest Model:", latest)
    print("All Candidates and modification times:")
    for c in sorted(candidates, key=os.path.getmtime):
        print(f"  {os.path.basename(c)}: {os.path.getmtime(c)}")
else:
    print("No models found!")
