# CS5228 HDB Rental Prediction Project

This is a self-contained working directory for the CS5228 group project. It is intentionally stored outside OneDrive so it can be evaluated, shared as a ZIP file, or deleted without changing the original course folder.

## Confirmed project requirements

- Task: predict the monthly rent of Singapore HDB flats as a regression problem.
- Competition format: Kaggle InClass Competition.
- Project weight: 25% of the course grade.
- Progress report: PDF slide deck of approximately 10-15 slides. The last 1-2 slides should state the next steps.
- Progress report deadline: Thursday, 8 October 2026, 11:59 pm. Earlier submission is encouraged.
- Final report: scientific-paper PDF, at most 8 pages excluding references and appendix.
- Final report deadline: Friday, 6 November 2026, 11:59 pm (Week 12).
- Final submission: report PDF plus complete source code in a ZIP file, or report PDF plus a GitHub repository link.
- Marking: methodological quality 60%, report quality 30%, reproducibility 10%.
- Report: include every member's name and student ID, team name, and a workload breakdown in the appendix.

The Canvas Project page still shows placeholder dates, but the newer Canvas Overview page provides the dates above. The Kaggle competition has now been confirmed; its metric is RMSE and the current limit is five submissions per day.

## Current status

The reproducible skeleton and first prototype are implemented:

- isolated project structure created;
- supplied data and report template copied into this workspace;
- requirements and project plan recorded;
- reproducible data-audit program added;
- automated data-contract checks added;
- chronological validation covering January-March 2025 added;
- global median, town/flat-type median, and CatBoost prototype evaluated;
- a schema-valid 50,000-row Kaggle submission generated locally.

See `docs/STATUS.md` for the latest checkpoint and `docs/PROJECT_PLAN.md` for the full plan.

## Quick start

Python 3.12 is the reference version. After cloning the repository, obtain the
course dataset separately and follow `data/README.md` to place it under
`data/raw/`. The dataset, trained models, generated submissions, local virtual
environments, and supplied report templates are intentionally not stored on
GitHub.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src\data_audit.py
python -m pytest -q
python src\train_prototype.py
```

Generated audit outputs are written to:

- `artifacts/data_audit.json`
- `reports/data_audit.md`
- `reports/prototype_metrics.json`
- `submissions/prototype_catboost_v1.csv`

See `ENVIRONMENT.md` for Windows, macOS/Linux, VS Code, and Jupyter setup notes.

## Directory structure

```text
CS5228_Project_Workspace/
|-- data/
|   |-- raw/               Supplied, immutable input data
|   `-- processed/         Reproducibly generated feature tables
|-- docs/                  Requirements, plan, decisions, and progress
|-- notebooks/             Exploration only; final logic belongs in src/
|-- reports/               Report drafts and figures
|-- submissions/           Kaggle submission CSV files
|-- src/                   Reusable project code
|-- tests/                 Data and pipeline checks
|-- artifacts/             Machine-generated audit and experiment outputs
`-- templates/             Supplied Word and LaTeX report templates
```

## Collaboration rules

1. Never edit files under `data/raw/`.
2. Put reusable processing and modeling logic in `src/`, not only in notebooks.
3. Log every meaningful experiment in `docs/EXPERIMENT_LOG.csv`.
4. Use time-based validation as the primary evaluation method.
5. Do not use post-release public rental records to reconstruct hidden Kaggle targets.
6. Do not publish supplied course data or templates publicly unless the teaching team permits it.

See `docs/COURSE_ALIGNMENT.md` for the mapping between our methods, lecture topics, and project guidance.
For a Chinese team-facing overview, see `docs/项目总方案_中文.md`.

## Source material

The plan is based on the supplied CS5228 lecture PDFs, project data, report templates, and the Canvas Project page for course 93866. Canvas was reviewed read-only on 16 September 2026; no Canvas content was edited or submitted.
