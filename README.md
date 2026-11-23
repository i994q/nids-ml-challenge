# Network Intrusion Detection Challenge — MSCS 640 Final Project

Multi-class network intrusion detection on **CSE-CIC-IDS2018**: 79 CICFlowMeter flow
features per record, classified as `Benign` or one of 14 attack types. Optimised for
**macro F1**, not accuracy — attacks are about 1% of real traffic, so a model that
answers "Benign" every time scores 99% and is useless.

## Team

| Member | Owns |
|---|---|
| A | data pipeline, feature selection, class imbalance, model training |
| B | dashboard, deployment, reproducibility, paper |

Both members commit to this repository, as the project rules require.

## Quick start

```bash
git clone <repo> && cd <repo>
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

bash src/download.sh                    # ten day files, smallest first (~6 GB total)
.venv/bin/python src/clean.py --cap 200000
.venv/bin/python src/train.py           # (next step)
.venv/bin/streamlit run main.py         # dashboard
```

## Layout

```
data/      day CSVs from the public S3 bucket (not committed)
src/
  download.sh   fetches the ten days over https, resumable
  clean.py      cleaning + per-day label report -> models/clean.joblib
  train.py      baseline + advanced model, macro F1 comparison
models/    clean.joblib, model.joblib (not committed)
docs/
  VIDEO_SCRIPT.md   demo-day script with roles and timings
main.py    Streamlit dashboard: CSV upload -> threat report
```

## Data notes — three real problems, all measured

Figures below are from the smallest day, `Thursday-01-03-2018` (103 MB, 331,125 rows):

| Problem | What we found | How we handle it |
|---|---|---|
| Repeated header rows | **25 rows** whose `Label` literally reads `Label` | dropped by value, not by line number |
| Infinite rates | **4,004** `inf` cells in `Flow Byts/s` / `Flow Pkts/s` from zero-duration flows | `inf` → NaN, then drop the row (**2,919 rows**) |
| One day ≠ all attacks | this day holds only `Benign` + `Infilteration` | train across days; `download.sh` lists which attacks each day staged |

**Label strings are used verbatim.** `DDOS attack-HOIC` is capitalised differently from
`DDoS attacks-LOIC-HTTP`, and `Infilteration` is misspelled in the source. Correcting
either invents a class that does not exist in the CSVs.

`Flow ID`, `Src IP`, `Src Port`, `Dst IP` and `Timestamp` are dropped: in a staged
capture the attacker addresses and the clock separate the classes perfectly, so keeping
them produces a model that has memorised the lab setup rather than attack behaviour.

Benign rows are capped per day (`--cap`) to keep the table in memory. **No attack row is
ever dropped**, and the cap is reported so the paper can state it.

## Honest limits

- Scores are measured on held-out data from the **same capture**, so they read higher than
  the model would manage on another network.
- `SQL Injection`, `Brute Force -Web` and `Brute Force -XSS` have only dozens to hundreds
  of rows across all ten days. Per-class recall for them is the number that matters, and a
  high macro F1 can hide a weak one.

## AI use disclosure

Claude Code (Anthropic) was used for parts of this repository: scaffolding the data
pipeline, the dashboard, and drafting documentation. All design decisions, metrics and
claims were reviewed and verified by the team against the actual data output. Every
reference in the paper was checked against its original source.

## Dataset citation

I. Sharafaldin, A. Habibi Lashkari and A. A. Ghorbani, "Toward generating a new intrusion
detection dataset and intrusion traffic characterization," *ICISSP*, pp. 108–116, 2018.
<https://www.unb.ca/cic/datasets/ids-2018.html>
