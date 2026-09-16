from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def test_core_schema() -> None:
    train = pd.read_csv(RAW / "train.csv", nrows=5)
    test = pd.read_csv(RAW / "test.csv", nrows=5)
    assert "MONTHLY_RENT" in train.columns
    assert "MONTHLY_RENT" not in test.columns
    assert train.drop(columns=["MONTHLY_RENT"]).columns.tolist() == test.columns.tolist()


def test_submission_contract() -> None:
    test = pd.read_csv(RAW / "test.csv", usecols=["RENT_APPROVAL_DATE"])
    sample = pd.read_csv(RAW / "example-submission.csv")
    assert sample.columns.tolist() == ["Id", "Predicted"]
    assert len(sample) == len(test)
    assert sample["Id"].tolist() == list(range(len(test)))
    assert sample["Predicted"].notna().all()


def test_auxiliary_files_exist() -> None:
    expected = {
        "sg-coe-prices.csv",
        "sg-hdb-block.csv",
        "sg-mrt-stations.csv",
        "sg-schools.csv",
        "sg-shopping-malls.csv",
        "sg-stock-prices.csv",
    }
    actual = {path.name for path in (RAW / "auxiliary").glob("*.csv")}
    assert actual == expected


def main() -> None:
    checks = [test_core_schema, test_submission_contract, test_auxiliary_files_exist]
    for check in checks:
        check()
        print(f"PASS {check.__name__}")


if __name__ == "__main__":
    main()
