# Kaggle Competition Audit

Checked on 18 September 2026 using the signed-in competition pages for
`cs-5228-2610-project`. No submission or discussion action was performed.

## Data provenance

- The official download contains `train.csv`, `test.csv`, and six files under
  `auxiliary/` (8 CSV files, 23.6 MB uncompressed as reported by Kaggle).
- Downloaded archive: `data/source/cs-5228-2610-project.zip`.
- Archive SHA-256:
  `BB7FF2EE4AE7EABE73DAF929D352B27F98592CD9AEA89EAA99804011612F3BC6`.
- Every downloaded CSV has the same SHA-256 digest as the file already stored
  under `data/raw/`; the existing raw data was therefore not overwritten.
- `example-submission.csv` is described by the overview but is not included in
  the ZIP download. The existing course-supplied copy remains checksum-locked by
  `src/check_setup.py`.

## Requirements that affect the project

- Evaluation metric: root mean squared error (RMSE).
- Required submission columns: `Id,Predicted`.
- The public leaderboard covers approximately 50% of test data; final ranking
  uses the remaining private portion.
- Official auxiliary data includes HDB blocks, MRT stations (open and planned),
  malls, schools, COE prices, and Singapore stock prices.
- Additional external data collection is explicitly permitted and encouraged.
  The page gives traffic/travel-time data from LTA as an example.
- Competition-specific rules are only: do not cheat, apply yourself, and have
  fun. Kaggle foundational rules prohibit hand-labeling hidden targets and
  private code/data sharing outside the registered team.
- The competition discussion page contained no topics at the time of review.

## Planning implications

1. Continue using chronological rolling validation; public leaderboard scores
   are supporting evidence only.
2. Continue stress-testing temporal extrapolation; past-only location statistics
   are now implemented, but the hidden test range extends well beyond training.
3. Test COE/stock and any external source in separate, leakage-safe ablations.
4. Record source, retrieval date, license, transformation, and prediction-time
   availability for every external feature used in the report.
5. Keep all code and private data sharing within the registered Kaggle team.

## Closeout verification — 21 September 2026

Read-only browser checks confirmed:
- `submission_history_diversity.csv` scored 432.47187, rank 3 of 16 at the time
  checked; previous best was 436.05684.
- The Submissions page allows up to two final submissions. If fewer than two
  are chosen, Kaggle automatically fills from best-scoring submissions.
  No selection or upload was performed by the agent.
- The displayed selection notice does not explicitly state how the two scores
  determine the final result; do not claim averaging from this notice.
- Public/private test proportions remain approximately 50/50. Their exact
  temporal/random allocation was not verified.

Canvas project page was also reread:
https://canvas.nus.edu.sg/courses/93866/pages/project
Progress slides: approximately 10-15 pages as PDF, target Sunday 4 October
(not a hard deadline), with next steps in the last 1-2 slides.
Final report: at most eight main-text pages, excluding references/appendix,
plus code or GitHub link and member workload attribution.
The final deadline is still `Nov XX, 11.59 pm`, not a confirmed 6 November.
Grading: methodology 60%, report quality 30%, reproducibility 10%.

Auxiliary timing caveat: current spatial features use supplied facility snapshots
and MRT status `open`, not opening dates as of each historical rental month.
Past-only target histories are tested separately; they do not eliminate this
snapshot-availability limitation.

## Score update — 24 September 2026

The user reported public RMSE **432.14881** for
`submission_local_trend_diversity.csv`, improving the previous 432.47187
mainline by 0.32306. The exact submitted CSV has SHA-256
`a88da3be518a4793486765f5d9ff44c9b27b3248bb195e6f2d0264ae7dabb57a`.
No rank was supplied or inferred, and no final-submission selection was changed.
