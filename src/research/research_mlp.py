"""One fixed-budget embedding-MLP screen, with train-only preprocessing."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from src.research.cache_inputs import require_files

import numpy as np
import pandas as pd
import torch
from torch import nn


class RentalMLP(nn.Module):
    def __init__(self, cardinalities, numeric_count):
        super().__init__()
        dimensions = [min(32, max(2, int(np.sqrt(c)))) for c in cardinalities]
        self.embeddings = nn.ModuleList([
            nn.Embedding(c, d, padding_idx=0)
            for c, d in zip(cardinalities, dimensions)
        ])
        size = numeric_count + sum(dimensions)
        self.layers = nn.Sequential(
            nn.Linear(size, 256), nn.SiLU(), nn.Dropout(0.15),
            nn.Linear(256, 128), nn.SiLU(), nn.Dropout(0.10),
            nn.Linear(128, 1),
        )

    def forward(self, numeric, categorical):
        pieces = [numeric]
        pieces.extend(embedding(categorical[:, j])
                      for j, embedding in enumerate(self.embeddings))
        return self.layers(torch.cat(pieces, dim=1)).squeeze(1)


def prepare(train, query):
    categorical = train.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric = train.columns.difference(categorical, sort=False).tolist()
    a = train[numeric].replace([np.inf, -np.inf], np.nan).astype(float)
    b = query[numeric].replace([np.inf, -np.inf], np.nan).astype(float)
    median = a.median().fillna(0)
    missing_a, missing_b = a.isna().to_numpy(float), b.isna().to_numpy(float)
    a, b = a.fillna(median), b.fillna(median)
    mean, scale = a.mean(), a.std().replace(0, 1).fillna(1)
    numeric_a = np.concatenate([((a-mean)/scale).to_numpy(), missing_a], axis=1).astype("float32")
    numeric_b = np.concatenate([((b-mean)/scale).to_numpy(), missing_b], axis=1).astype("float32")
    vocabularies, left, right, cardinalities = {}, [], [], []
    for column in categorical:
        vocabulary = sorted(train[column].fillna("missing").astype(str).unique().tolist())
        lookup = {value: index+1 for index, value in enumerate(vocabulary)}
        vocabularies[column] = vocabulary
        left.append(train[column].fillna("missing").astype(str).map(lookup).fillna(0).to_numpy("int64"))
        right.append(query[column].fillna("missing").astype(str).map(lookup).fillna(0).to_numpy("int64"))
        cardinalities.append(len(vocabulary)+1)
    metadata = dict(categorical=categorical, numeric=numeric, vocabulary=vocabularies,
                    median=median.to_dict(), mean=mean.to_dict(), scale=scale.to_dict())
    return (numeric_a, np.column_stack(left), numeric_b, np.column_stack(right),
            cardinalities, metadata)


def predict(model, numeric, categorical):
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(numeric), 4096):
            result.append(model(numeric[start:start+4096], categorical[start:start+4096]).numpy())
    return np.concatenate(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_folder", type=Path)
    args = parser.parse_args()
    folder = require_files(args.run_folder, ['short_recent_local_trend_0.pkl',
        'short_recent_scoring_rows.csv','short_recent_reference_ensemble.npy'], 'run_folder')
    output = folder / "embedding_mlp_followup"
    output.mkdir(exist_ok=True)
    train, query, target, weights, _ = pd.read_pickle(folder / "short_recent_local_trend_0.pkl")
    protocol = dict(
        rationale="Three LightGBM configurations did not improve the CPU CatBoost reference. Test one different function family and its fixed 25-percent blend.",
        epochs=32, batch_size=2048, seed=42, learning_rate=0.001,
        weight_decay=0.0001, device="cpu", threads=2,
        structure="categorical embeddings + numeric/missing flags; 256/128 SiLU; dropout .15/.10; residual rent target /1000",
        preprocessing="numeric imputation/scaling and category vocabularies fit on training only",
        validation="2025-01 through 2025-03; not used for early stopping or training",
        stopping="One preset, no hyperparameter search. Same advancement rule as other candidates.",
        torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__, platform=platform.platform(),
        code_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        parent_protocol=json.loads((folder / "protocol.json").read_text()),
    )
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.use_deterministic_algorithms(True)
    a, ca, b, cb, cardinalities, metadata = prepare(train, query)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    numeric, categorical = torch.from_numpy(a), torch.from_numpy(ca)
    query_numeric, query_categorical = torch.from_numpy(b), torch.from_numpy(cb)
    offset = train.ADAPT_ADDRESS_FLAT_AREA_LEVEL.to_numpy(float)
    query_offset = query.ADAPT_ADDRESS_FLAT_AREA_LEVEL.to_numpy(float)
    assert np.isfinite(offset).all() and np.isfinite(query_offset).all()
    response = torch.from_numpy(((np.asarray(target)-offset)/1000).astype("float32"))
    mass = torch.from_numpy(np.asarray(weights, dtype="float32"))
    model = RentalMLP(cardinalities, a.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=32, eta_min=0.00005)
    history, started = [], time.perf_counter()
    for epoch in range(32):
        model.train()
        order = torch.randperm(len(response))
        total_loss, total_weight = 0., 0.
        for start in range(0, len(order), 2048):
            ids = order[start:start+2048]
            optimizer.zero_grad(set_to_none=True)
            error = model(numeric[ids], categorical[ids])-response[ids]
            loss = (mass[ids]*error.square()).sum()/mass[ids].sum()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total_loss += float((mass[ids]*error.detach().square()).sum())
            total_weight += float(mass[ids].sum())
        scheduler.step()
        record = dict(epoch=epoch+1, training_batch_rmse=1000*np.sqrt(total_loss/total_weight))
        history.append(record)
        if (epoch+1) % 4 == 0:
            print(json.dumps(record), flush=True)
    estimate = predict(model, query_numeric, query_categorical)*1000 + query_offset
    assert np.isfinite(estimate).all()
    np.save(output / "ensemble.npy", estimate)
    torch.save(dict(state=model.state_dict(), cardinalities=cardinalities, numeric_count=a.shape[1]), output / "model.pt")
    (output / "preprocessing.json").write_text(json.dumps(metadata))
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    restored = RentalMLP(cardinalities, a.shape[1])
    restored.load_state_dict(torch.load(output / "model.pt", weights_only=True)["state"])
    replay = predict(restored, query_numeric, query_categorical)*1000 + query_offset
    assert np.array_equal(replay, estimate)
    # Read scoring labels only after fixed-budget fitting has finished.
    truth = pd.read_csv(folder / "short_recent_scoring_rows.csv").target.to_numpy()
    reference = np.load(folder / "short_recent_reference_ensemble.npy")
    metrics = dict(candidate="embedding_mlp", seconds=time.perf_counter()-started,
                   rmse=float(np.sqrt(np.mean((estimate-truth)**2))),
                   blend25_rmse=float(np.sqrt(np.mean((0.25*estimate+0.75*reference-truth)**2))),
                   reload_max_difference=float(np.max(np.abs(replay-estimate))))
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
