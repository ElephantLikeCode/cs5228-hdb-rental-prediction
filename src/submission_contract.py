"""Submission row layout without requiring an optional course template."""
from pathlib import Path

import numpy as np
import pandas as pd


def submission_frame(raw_dir: Path, rows: int) -> pd.DataFrame:
    """Validate a supplied template, or construct positional IDs in memory only.

    No prediction, training label or raw file is changed by this helper.
    """
    path = raw_dir / "example-submission.csv"
    if path.is_file():
        frame = pd.read_csv(path)
        if (frame.columns.tolist() != ["Id", "Predicted"] or len(frame) != rows
                or frame.Id.tolist() != list(range(rows))):
            raise ValueError("Invalid example-submission.csv: expected Id,Predicted and ordered zero-based IDs matching test rows")
        return frame
    return pd.DataFrame({"Id": np.arange(rows), "Predicted": np.zeros(rows)})
