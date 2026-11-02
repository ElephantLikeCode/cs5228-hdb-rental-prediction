"""Download HDB resale/building attributes, never external rental labels."""

import hashlib
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/external/attribute_sources_20261102"
SOURCES = {
    "resale": "d_8b84c4ee58e3cfc0ece0d773c8ca6abc",
    "property": "d_17f5382f26140b1fdae0ba2ef6239d2f",
}


def collect(name, dataset):
    destination = OUTPUT / f"{name}.csv"
    if destination.exists():
        print(f"Preserving cached {destination}", flush=True)
        return
    rows, offset, expected_total = [], 0, None
    while True:
        params = {"resource_id": dataset, "limit": 5000, "offset": offset}
        params["sort"] = "_id asc"
        url = "https://data.gov.sg/api/action/datastore_search?" + urllib.parse.urlencode(params)
        page_file = OUTPUT / f"{name}_page_{offset}.json"
        if page_file.exists():
            payload = page_file.read_bytes()
        else:
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url, timeout=30) as response:
                        payload = response.read()
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2)
            page_file.write_bytes(payload)
        result = json.loads(payload)
        if not result.get("success"):
            raise ValueError(result.get("error"))
        result = result["result"]
        total = result["total"]
        if expected_total is None:
            expected_total = total
        if total != expected_total:
            raise ValueError("Source changed during pagination")
        page = result["records"]
        if not page:
            break
        if any("monthly_rent" in record for record in page):
            raise ValueError("External rental labels are not an attribute source")
        rows.extend(page)
        offset += len(page)
        print(f"{name}: {offset}/{total}", flush=True)
        if offset >= total:
            break
        time.sleep(0.25)
    frame = pd.DataFrame(rows)
    if len(frame) != expected_total:
        raise ValueError("Incomplete source pagination")
    if frame._id.duplicated().any():
        raise ValueError("Duplicate source IDs in pagination")
    if name == "resale":
        frame = frame.loc[frame.month.le("2024-12")]
    frame.to_csv(destination, index=False)
    receipt = dict(
        dataset=dataset, url=f"https://data.gov.sg/datasets/{dataset}/view",
        retrieved_utc=datetime.now(timezone.utc).isoformat(), rows=len(frame),
        sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        licence="Singapore Open Data Licence",
        note="Latest-vintage snapshot; not a reconstructed historical vintage. Resale rows restricted to 2017-01..2024-12. No external rental labels.",
    )
    (OUTPUT / f"{name}_manifest.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, dataset in SOURCES.items():
        collect(name, dataset)
