# Repo export instructions

Run `assemble_repo.py` from the project root to create a ready-to-upload folder named `repo_export/`.

Usage:

Python 3 required. From the project root:

```
python assemble_repo.py         # copies code, data/cicids2017, and results summary
python assemble_repo.py --include-models   # also include results/saved_models (may be large)
```

The script will copy:
- core files: `config.py`, `main.py`, `README.md`, `requirements.txt`, `demo_synthetic.py`
- directories: `data/`, `models/`, `utils/`, `explainability/`
- always: `data/cicids2017/` (full CSV dataset if present)
- results/detection_results.json (if present)

After running, `repo_export/` will contain a clean directory you can initialize as a git repo and push to GitHub.
