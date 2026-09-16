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

