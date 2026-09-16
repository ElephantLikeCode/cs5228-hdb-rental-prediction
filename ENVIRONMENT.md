# Reproducible development environment

The verified reference environment uses Python 3.12. Each teammate must create
a new local virtual environment; never copy or commit another machine's
`.venv` directory.

## Windows PowerShell

From the repository root:

```powershell
py -3.12 --version
py -3.12 -m venv .venv
& '.\.venv\Scripts\Activate.ps1'
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python src\check_setup.py --skip-data
```

If PowerShell blocks `Activate.ps1`, activation is optional. Run commands
directly with `.\.venv\Scripts\python.exe`, or allow locally created scripts
for the current user according to your institution's security policy.

## macOS/Linux

From the repository root:

```bash
python3.12 --version
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python src/check_setup.py --skip-data
```

If an exact locked package version is unavailable on a platform, install
`requirements.txt` instead and record the resulting versions in the pull
request or experiment log.

## Add the private course data

The repository deliberately excludes all course/Kaggle data. Follow
`data/README.md`, then run the complete check:

```bash
python src/check_setup.py
```

Do not continue until it prints `SETUP CHECK PASSED`.

## VS Code and Jupyter

Open the cloned repository folder in VS Code and select the Python interpreter
inside `.venv`. To register an explicit Jupyter kernel:

```bash
python -m ipykernel install --user --name cs5228-project --display-name "CS5228 Project (Python 3.12)"
jupyter lab
```

Choose `CS5228 Project (Python 3.12)` when opening a notebook.

## Reproduce the current checkpoint

```bash
python src/data_audit.py
python -m pytest -q
python src/train_prototype.py
```

Expected results are documented in `docs/TEAM_HANDOFF.md`. Deactivate the
environment with `deactivate` when finished.

## Dependency files

- `requirements-lock.txt`: exact top-level versions used for the verified run.
- `requirements.txt`: compatible version ranges for platforms where an exact
  locked version is unavailable.

When adding or upgrading a direct dependency, update both files and rerun the
setup check, tests, and reference pipeline before merging.
