# System documentation

`NIDS-System-Documentation.pdf` is the print-ready technical dossier for this repository: system and repository topology, data pipeline, model contract, measured results, dashboard walkthrough, deployment, and limits.

## What is here

| Path | Purpose |
|---|---|
| `NIDS-System-Documentation.pdf` | Final 27-page A4 document (committed so it is readable without tooling). |
| `src/documentation.html` | Editable document source. |
| `src/documentation.css` | Print stylesheet, including the cover and landscape spread. |
| `assets/system-topology.svg` | End-to-end build/run topology (Figure 1). |
| `assets/confusion-matrix.svg` | Row-normalised 15 × 15 test matrix. |
| `assets/class-distribution.svg` | Log-scale test support per class. |
| `assets/dashboard-*.png` | Live captures of `main.py` running locally. |
| `assets/logo.svg` | Project shield mark used on the cover and section badges. |
| `build.py` | Renders the HTML/CSS source to the PDF with WeasyPrint. |

## Rebuild the PDF

WeasyPrint is a documentation dependency, not a runtime dependency of the dashboard:

```bash
python3 -m venv .venv-docs
.venv-docs/bin/pip install weasyprint
.venv-docs/bin/python docs/documentation/build.py
```

The script writes `docs/documentation/NIDS-System-Documentation.pdf` relative to the repository. The PDF embeds all fonts and images; no network access is needed to render it.

## Refreshing the evidence

The numbers in the document come from `models/metrics.json` and `models/confusion_matrix.csv`. If the model is retrained:

1. Update the figures in `src/documentation.html` from the new metrics.
2. Re-capture the dashboard screenshots if the UI changed.
3. Re-run `build.py` and check the page count and figures.

The dashboard screenshots were captured from `streamlit run main.py` with a 4,200-row sample of `Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv`. The sample deliberately retained 25 repeated header rows and rows with infinite values so the cleaning notes are real; the dashboard scored 4,115 usable flows.

## Editing notes

- Keep labels verbatim, including `Infilteration` and the mixed-case DDoS spellings.
- The topology is a hand-authored SVG; keep its text sizes at 11–12 units because it is printed at ~165 mm height on a landscape page.
- `@page landscape` is used only for the topology spread. The rest of the document is A4 portrait.
- If page numbers shift, the table of contents updates automatically through `target-counter`; no manual page numbers exist in the HTML.
- Do not copy metrics from prose. Re-read the committed JSON/CSV first.
