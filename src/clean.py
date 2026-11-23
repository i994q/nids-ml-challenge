"""Clean the CSE-CIC-IDS2018 day files into one training table.

Three real problems in these CSVs, all confirmed on the Thursday-01-03 day:
  1. repeated header rows inside the file (25 of them in that day alone)
  2. +/-inf in the rate columns (Flow Byts/s, Flow Pkts/s) from zero-duration flows
  3. each day holds Benign plus only the attacks staged that day, so training
     needs several days or the model never sees most classes

Label strings are used exactly as they appear in the files -- 'DDOS attack-HOIC'
in capitals, 'Infilteration' with its misspelling. Correcting either invents a
class that does not exist in the data.

    python src/clean.py                 # clean every day found in data/
    python src/clean.py --cap 200000    # cap Benign rows per day
"""
import argparse
import glob
import os

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "models")
CHUNK = 200_000           # the biggest day is ~4 GB, so it is read in pieces
DROP_COLS = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Timestamp"]
LABEL = "Label"


def clean_day(path, benign_cap=None, seed=42):
    """Return a cleaned DataFrame for one day file."""
    kept, stats = [], {"raw": 0, "header_rows": 0, "inf_rows": 0, "na_rows": 0}
    rng = np.random.default_rng(seed)

    for chunk in pd.read_csv(path, chunksize=CHUNK, low_memory=False, dtype=str):
        stats["raw"] += len(chunk)
        chunk.columns = [c.strip() for c in chunk.columns]

        # (1) a repeated header row shows up as a row whose Label literally reads 'Label'
        bad_header = chunk[LABEL].astype(str).str.strip() == LABEL
        stats["header_rows"] += int(bad_header.sum())
        chunk = chunk[~bad_header]

        # columns that carry no signal or leak ordering; absent in some days, hence errors="ignore"
        chunk = chunk.drop(columns=[c for c in DROP_COLS if c in chunk.columns], errors="ignore")

        y = chunk[LABEL].astype(str).str.strip()
        X = chunk.drop(columns=[LABEL]).apply(pd.to_numeric, errors="coerce")

        # (2) inf -> NaN, then drop: a zero-duration flow has no meaningful rate
        n_inf = int(np.isinf(X.to_numpy(dtype="float64", na_value=np.nan)).sum())
        stats["inf_rows"] += n_inf
        X = X.replace([np.inf, -np.inf], np.nan)

        keep = X.notna().all(axis=1)
        stats["na_rows"] += int((~keep).sum())
        X, y = X[keep], y[keep]

        df = X.astype("float32")
        df[LABEL] = y.values
        kept.append(df)

    day = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()
    del kept

    # (3) Benign dominates every day; capping it keeps the table in memory without
    # touching any attack row. The cap is reported so the paper can state it.
    if benign_cap is not None and len(day):
        benign = day[day[LABEL] == "Benign"]
        if len(benign) > benign_cap:
            drop_idx = rng.choice(benign.index.to_numpy(), size=len(benign) - benign_cap,
                                  replace=False)
            day = day.drop(index=drop_idx).reset_index(drop=True)
            stats["benign_capped"] = len(benign) - benign_cap

    return day, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=None,
                    help="maximum Benign rows kept per day (attacks are never dropped)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(DATA, "*TrafficForML_CICFlowMeter.csv")))
    if not files:
        raise SystemExit(f"no day files in {DATA}; run src/download.sh first")

    parts, per_day = [], {}
    for path in files:
        name = os.path.basename(path)
        day, stats = clean_day(path, benign_cap=args.cap)
        counts = day[LABEL].value_counts()
        per_day[name] = counts
        parts.append(day)
        print(f"\n=== {name} ===")
        print(f"raw {stats['raw']:,}  header rows {stats['header_rows']}  "
              f"inf cells {stats['inf_rows']:,}  rows dropped for NaN/inf {stats['na_rows']:,}"
              + (f"  benign capped {stats['benign_capped']:,}" if "benign_capped" in stats else ""))
        print(counts.to_string())

    table = pd.concat(parts, ignore_index=True)
    del parts

    print("\n=== combined table ===")
    print(f"rows {len(table):,}   features {table.shape[1] - 1}   classes {table[LABEL].nunique()}")
    counts = table[LABEL].value_counts()
    print(counts.to_string())
    attack = int((table[LABEL] != "Benign").sum())
    print(f"\nattack share: {attack / len(table):.2%}  ({attack:,} of {len(table):,})")
    print("rarest three classes:")
    print(counts.tail(3).to_string())

    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, "clean.joblib")
    joblib.dump({"table": table, "label_col": LABEL, "per_day": per_day,
                 "benign_cap": args.cap}, out, compress=3)
    print(f"\nwrote {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
