# Confirmed Requirements

Sources: CS5228 Canvas Project and Overview pages, course 93866, reviewed read-only on 16 September 2026. The Project page reports a last update of 4 August 2026; the newer Overview page reports a last update of 25 August 2026.

## Task

- Predict expected monthly rent for HDB flats from historical rental data and flat information.
- Regression task implemented as a Kaggle InClass Competition.
- The emphasis is on understanding the data, task, results, alternatives, design decisions, and limitations rather than final leaderboard rank alone.
- Additional data may be collected if the team believes it will help, but it is explicitly not mandatory.
- Teams are not required to use every supplied attribute.

## Supplied data described by Canvas

- Approved HDB rental applications beginning January 2021.
- Core fields include approval month, town, block, street, and monthly rent.
- Teaching staff extended the data with flat type, size, and lease commencement information.
- Auxiliary data includes block details, geolocation, open/planned MRT stations, malls, primary schools, stock prices, and COE prices.

Canvas states that the complete data covers January 2021 through June 2026. The local `test.csv` includes July 2026, so the page or local package may have been updated independently. This is tracked as an open question.

## Kaggle competition

- Competition: `CS5228-2610 Project` (`cs-5228-2610-project`).
- Private invitation link: `https://www.kaggle.com/t/f00c64ff5d67488793caaa5caa5fa4ae`.
- Evaluation metric: Root Mean Square Error (RMSE); lower is better.
- Submission file: exactly 50,000 rows plus a header, with columns `Id,Predicted`.
- Current platform allowance: five submissions per day, with a daily reset shown in the submission dialog.
- Kaggle displays a public leaderboard during the competition and uses a private test split for the final leaderboard.
- The instructor describes the leaderboard as a friendly competition with no direct impact on project marks.
- Each member must join individually before accepting a team invitation. The Kaggle team name must contain the Canvas team number.
- Team 44 currently uses the name `elephant(44)`; capitalization/spacing may be refined later without changing the required number.

## Progress report

- PDF slide deck.
- Approximately 10-15 slides.
- No prescribed layout.
- Cover goals/questions, understanding of task/data, early decisions/results, and issues/challenges.
- Last 1-2 slides: next steps through project completion.
- Not explicitly graded, but failure to submit negatively affects the final grade.
- Earlier submission is encouraged.
- Deadline: Thursday, 8 October 2026, 11:59 pm. Earlier submission is encouraged.

## Final report

- Scientific-paper PDF.
- Maximum 8 pages including tables, plots, and figures.
- References and appendix are excluded from the limit.
- Appendix should be used sparingly; the report must be understandable without it.
- Use the supplied Word or LaTeX template.
- Include team name, names and student IDs of all members.
- Include a workload breakdown, such as a table or Gantt chart, in the appendix.

Expected content:

- Motivation.
- Exploratory Data Analysis and preprocessing.
- Data mining methods and hyperparameter choices.
- Evaluation and interpretation.
- Alternatives, failure cases, limitations, and possible future improvements.

## Final submission

- Report PDF plus source code uploaded to Canvas as a ZIP file; or report PDF plus a GitHub repository link.
- Reproducibility, organization, documentation, and readability are assessed.
- Deadline: Friday, 6 November 2026, 11:59 pm (Week 12).

## Marking

- The group project is worth 25% of the course grade.
- Methodological quality: 60%.
- Report quality: 30%.
- Reproducibility: 10%.

Methodological quality includes data preparation, visualization, method choice and implementation, evaluation setup and metrics, common errors, limitations, and justified design decisions.

## Team assessment

- Members normally receive the same mark.
- Marks may differ when individual contribution warrants it.
- TEAMMATES peer review will be used.
- Teaching staff recommend starting early, meeting once or twice per week, and keeping buffer time.

## Course environment and relevant syllabus

- Course notebooks are tested with Python 3.12.
- Common course packages include NumPy, pandas, scikit-learn and NetworkX.
- Week 1 covers EDA and preprocessing.
- Weeks 4-6 cover regression/classification evaluation, kNN, decision/regression trees, linear regression, bagging and boosting.
- Week 8 covers dimensionality reduction.
- Week 11 covers classical time-series analysis.

Our core model comparison is therefore built around descriptive baselines, kNN regression, linear regression, regression trees and course-covered ensemble methods. More specialized libraries are optional extensions.

## Course integrity notes

- Course slides state zero tolerance for plagiarism.
- AI tools are generally allowed for ideation, self-learning, and writing improvement.
- Take-home assignments require AI use to be acknowledged.
- Confirm whether the project has additional AI-use rules when the Kaggle/assignment instructions are published.
