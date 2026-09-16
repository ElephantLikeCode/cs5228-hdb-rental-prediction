# Team collaboration guide

## Before starting work

1. Read `README.md`, `docs/TEAM_HANDOFF.md`, and `docs/PROJECT_PLAN.md`.
2. Build a local Python 3.12 environment using `ENVIRONMENT.md`.
3. Obtain the supplied data privately and follow `data/README.md`.
4. Run `python src/check_setup.py` and `python -m pytest -q`.

Do not commit course/Kaggle data, supplied templates, credentials, trained
models, generated submissions, or local environments. The `.gitignore` covers
the expected locations, but always review `git status` before committing.

## Git workflow

Do not develop directly on `main`. For each task:

```bash
git switch main
git pull --ff-only
git switch -c feature/short-description
```

Make focused commits, push the branch, and open a pull request:

```bash
git add <specific-files>
git commit -m "Describe the change"
git push -u origin feature/short-description
```

At least one teammate should review changes that affect data preparation,
validation, target-derived features, model evaluation, or submission creation.

## Experiment contract

Every meaningful experiment should record one row in
`docs/EXPERIMENT_LOG.csv`, including:

- code/version identifier;
- data/features used;
- chronological validation window;
- metric and score;
- model and important hyperparameters;
- runtime, seed, and artifact path;
- conclusion or next action.

Use seed `42` unless an experiment explicitly studies seed sensitivity. Never
compare models using different validation windows without labeling that change.

## Before opening a pull request

Run from the repository root:

```bash
python src/check_setup.py
python src/data_audit.py
python -m pytest -q
```

If dependencies change, update both `requirements.txt` (compatible ranges) and
`requirements-lock.txt` (the exact verified versions). Update documentation and
tests whenever behavior, paths, schemas, or reproduction commands change.

