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

## External-information boundary check — 6 October 2026

Read-only Kaggle checks reconfirmed that the Overview welcomes collecting
additional data. The Rules page contains the short competition-specific rules
and Kaggle's foundational rules. This does not explicitly settle whether
external rental records overlapping the hidden test interval may be used.
No post, message, upload, rule acceptance or final-selection change was made.

Official HDB open rental records cover January 2021 through September 2026,
overlapping the competition test period. Their six fields include rental month,
town, block, street, flat type and monthly rent. The site describes rents as
owner-declared and not verified by HDB. That supports treating extreme values
as uncertain observations, not automatically as known errors or room rentals.
Source (read 6 October):
https://data.gov.sg/datasets/d_c9f57187485a850908655db0e8cfe651/view

No transaction-level external rent data was downloaded, matched or used for
training. Direct test-period rent lookup would change the information problem
and requires explicit instructor clarification before consideration. This
availability finding does not explain, or allege anything about, other teams'
scores. The current pipeline uses only supplied training rental labels.

Non-transaction macro candidates are available from official sources:
- Monthly CPI, SingStat/data.gov.sg:
  https://data.gov.sg/datasets/d_bdaff844e3ef89d39fceb962ff8f0791/view
- Quarterly private-residential rental indices, URA:
  https://www.ura.gov.sg/property-data/
  https://eservice.ura.gov.sg/property-market-information/pmiResidentialTimeseries

These sources were investigated, not imported or trained this round. Release
lags, historical revisions, licences and HDB/private-market applicability still
need checking before use. Provided COE and stock tables are being tested
separately with a previous-month join and an origin-only availability control;
their snapshot timing is not equivalent to verified real-time availability.

The 2023 competition is a related HDB rent/RMSE task, not a verified identical
dataset. Its public winner scored 477.52195 and final private winner 478.81397;
the older public/private scores cannot establish a noise floor for this edition.
Source: https://www.kaggle.com/competitions/cs5228-2310-final-project/leaderboard

### Project decision — 6 October 2026

The user declines the test-period public transaction-label matching route.
Do not download or match such rental labels for this project. The unresolved
permission question is therefore no longer an action item. This decision does
not establish how any other team obtained its score. Aggregate CPI/URA inputs
are a separate optional proposal, subject to availability checks before use.

### Aggregate indices and operating rail calendar — 6 October 2026

The user's subsequent request authorized importing aggregate price/rental indices
and testing MRT service timing. Official aggregate snapshots are now cached under
ignored `data/external/official_macro/`; no individual external rental records
were retrieved. Sources:

- CPI All Items, 2024 base, monthly: SingStat table M213751,
  https://tablebuilder.singstat.gov.sg/table/TS/M213751 and its official tabledata
  API. The downloaded All Items row has 788 months, Jan 1961-Aug 2026.
  Original JSON SHA-256:
  `90d24f5623c3134349466b3accbb5c8718cbdc07427dcb556a8cd15e7dd0d372`.
- URA private residential rental index, official data.gov.sg dataset
  https://data.gov.sg/datasets/d_8e4c50283fb7052a391dfb746a05c853/view .
  The All Residential series in this download ends at 2021-Q2. Use the continuous
  Non-Landed / Whole Island series instead: 90 quarters, 2004-Q1 through 2026-Q2;
  do not splice property types. The index covers private properties, not HDB.
  The dataset is available under the Singapore Open Data Licence.
  Original JSON SHA-256:
  `f7efb8468f39624ac7a63db059197dc7cf2bbfda060495bd54453c135a509b85`.

CPI is normally published on the 23rd of the next month or next working day:
https://www.singstat.gov.sg/find-data/explore-data-themes/economy-prices/consumer-price-index/our-data-explained .
The conservative month-start join activates CPI at observation month +2 and
URA at quarter-end month +2, after the usual following-month release. URA's
2024-Q4 release was 24 January 2025:
https://www.ura.gov.sg/news/media/pr25-05/ .
These are latest-vintage series, not reconstructed historical vintages; CPI
rebasing and index revisions remain limitations. Single-origin forecasts freeze
indices at the origin. Separately labelled monthly controls use only covariates
released by each rental month; this changes information availability and must
not be reported as the same single-origin forecast.

`configs/mrt_openings.json` records verified TEL2 (28 Aug 2021), TEL3 (13 Nov
2022), TEL4 (23 Jun 2024), Punggol Coast (10 Dec 2024) and Hume (28 Feb 2025)
service dates and dated official announcement sources. Only the first full
month after service begins is treated as open. The origin-available variant
excludes opening dates announced after the origin. Monthly controls reconstruct
service at each query month and are labelled separately. Legacy flags mean
operational before 2021, not fabricated exact opening dates. Unknown/future codes
stay excluded; interchanges are deduplicated by physical station name. This is
an audit of the supplied catalog, not a complete reconstruction of every LRT stop.

Of 243 supplied station-line rows, 150 are verified legacy rows and 26 belong
to the dated opening events. Sixty-seven rows are excluded; 55 of these were
supplied as `open`. The January 2025 verified catalog has 143 physical station
names. Compared with the original supplied-snapshot distances, nearest distance
changes by over 10 metres for 1,691 of 9,464 HDB blocks (17.87%). This comparison
combines operating-service filtering and interchange deduplication; it does not
attribute the change to a single repair or establish a causal rental premium.

## Canvas final-report verification -- 23 October 2026

Read the project page, Policies, 8 October feedback announcement and Assignments
through Chrome with no edits/posts/submissions. The official scientific-paper
template ZIP (Canvas file 9565679) was downloaded; SHA-256 is
`ab2974691f767969dd6b2b1de18217ccaa7829f4439c5232a3fbbaa03bbfda54`.
It matches the previously cached ZIP. Word and LaTeX templates are supplied;
the draft uses the unchanged IEEEtran conference class.

The PDF body limit is eight pages including figures/tables, excluding references
and a sparse appendix. Names/IDs and team name are required; workload attribution
may be in the appendix. Submit report plus code to Canvas as a ZIP, or include
a repository link in place of code. Methodology/report/reproducibility weights
are 60/30/10. The 8 October announcement confirms no presentation and that Kaggle
score is not the grading criterion. The Policies page explicitly permits AI for
the group project with appropriate documentation. These are verified course
requirements, not a policy invented by the assistant or imported from a skill.

The project page still gives `Xxx, Nov XX, 11.59 pm`. No Final Report assignment
is visible yet; the deadline remains unconfirmed. Latest announcements inspected
do not supply a replacement date. Sources:
https://canvas.nus.edu.sg/courses/93866/pages/project ,
https://canvas.nus.edu.sg/courses/93866/pages/policies , and
https://canvas.nus.edu.sg/courses/93866/discussion_topics/501350 .
