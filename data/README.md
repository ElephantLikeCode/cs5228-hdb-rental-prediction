# Local data directory

Course and competition datasets are intentionally excluded from GitHub.
Obtain them through the official course/Kaggle channel and place them under
`data/raw/` with this structure:

```text
data/raw/
|-- train.csv
|-- test.csv
|-- example-submission.csv
`-- auxiliary/
    |-- sg-coe-prices.csv
    |-- sg-hdb-block.csv
    |-- sg-mrt-stations.csv
    |-- sg-schools.csv
    |-- sg-shopping-malls.csv
    `-- sg-stock-prices.csv
```

Do not commit these files. The repository's `.gitignore` keeps the entire data
directory private except for this instruction file.

Expected row counts (excluding headers) are 150,000 for `train.csv`, 50,000
for `test.csv`, and 50,000 for `example-submission.csv`. After copying the
files, run the read-only version/schema/checksum validation from the repository
root:

```bash
python src/check_setup.py
```

Only proceed when the command prints `SETUP CHECK PASSED`. A checksum failure
means the member has a different or modified data release and should obtain the
official files again rather than editing the local copy.
