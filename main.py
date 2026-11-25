"""Threat detection dashboard: upload a CSV of network flows, get a threat report.

    .venv/bin/streamlit run main.py      # or: source .venv/bin/activate first

Loads models/model.joblib once per session and applies the same cleaning rules as
src/clean.py, so a raw day file from the dataset can be dropped in unchanged.
"""
import io
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "model.joblib")
METRICS_PATH = os.path.join(HERE, "models", "metrics.json")
BENIGN = "Benign"
DROP_COLS = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Timestamp"]

st.set_page_config(page_title="NIDS Threat Report", page_icon="🛡", layout="wide")

CSS = """
<style>
:root{
  --bg:#060914; --bg2:#0b1124; --card:rgba(17,24,46,.72);
  --text:#e6eaf5; --muted:#8b95b2; --border:rgba(148,163,207,.14);
  --accent:#818cf8; --accent2:#22d3ee;
  --danger:#f87171; --ok:#34d399; --warn:#fbbf24;
}
html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"]{
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Oxygen,Ubuntu,Cantarell,"Helvetica Neue",Arial,sans-serif;
}
[data-testid="stAppViewContainer"]{
  background-color:var(--bg);
  background-image:
    radial-gradient(1100px 620px at 10% -10%, rgba(129,140,248,.17), transparent 62%),
    radial-gradient(900px 520px at 92% 2%, rgba(34,211,238,.12), transparent 60%),
    linear-gradient(to right, rgba(148,163,207,.08) 1px, transparent 1px),
    linear-gradient(to bottom, rgba(148,163,207,.08) 1px, transparent 1px),
    linear-gradient(180deg, var(--bg) 0%, var(--bg2) 100%);
  background-size:auto, auto, 48px 48px, 48px 48px, auto;
  background-attachment:fixed;
  color:var(--text);
}
[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:2.2rem; max-width:1280px;}

/* ---------- hero ---------- */
.nids-eyebrow{
  font-size:.72rem; font-weight:700; letter-spacing:.18em; text-transform:uppercase;
  color:var(--muted); margin-bottom:.55rem;
}
h1.nids-h1, .nids-h1{
  font-size:clamp(1.9rem,3.4vw,3rem) !important; line-height:1.08; letter-spacing:-.025em;
  font-weight:700; color:var(--text); margin:0 0 .6rem 0 !important; padding:0 !important;
}
.nids-grad{
  background:linear-gradient(90deg, var(--accent), var(--accent2));
  -webkit-background-clip:text; background-clip:text;
  -webkit-text-fill-color:transparent; color:transparent;
}
.nids-sub{color:var(--muted); font-size:1rem; max-width:62ch; margin-bottom:.9rem;}
.nids-pill{
  display:inline-flex; align-items:center; gap:.5rem;
  padding:.3rem .7rem; border:1px solid var(--border); border-radius:999px;
  background:var(--card); color:var(--muted);
  font-size:.74rem; font-weight:600; letter-spacing:.08em; text-transform:uppercase;
  margin-bottom:1.1rem;
}
.nids-dot{
  width:.5rem; height:.5rem; border-radius:999px; background:var(--ok);
  box-shadow:0 0 0 0 rgba(52,211,153,.55); animation:nids-pulse 2s infinite;
}
.nids-dot.off{background:var(--danger); animation:none; box-shadow:none;}
@keyframes nids-pulse{
  0%{box-shadow:0 0 0 0 rgba(52,211,153,.55);}
  70%{box-shadow:0 0 0 .5rem rgba(52,211,153,0);}
  100%{box-shadow:0 0 0 0 rgba(52,211,153,0);}
}
@media (prefers-reduced-motion: reduce){
  .nids-dot{animation:none !important; box-shadow:0 0 0 .18rem rgba(52,211,153,.22);}
}
.nids-rule{height:1px; background:var(--border); border:0; margin:1.6rem 0 1.3rem;}
.nids-sec{
  font-size:1.18rem; font-weight:650; letter-spacing:-.01em; color:var(--text); margin:0 0 .7rem 0;
}

/* ---------- metric cards ---------- */
[data-testid="stMetric"]{
  background:var(--card); border:1px solid var(--border); border-radius:16px;
  padding:1rem 1.1rem;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.04), 0 18px 34px -26px rgba(0,0,0,.9);
  backdrop-filter:blur(8px);
}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p{
  font-size:.7rem !important; font-weight:700; letter-spacing:.14em; text-transform:uppercase;
  color:var(--muted) !important;
}
[data-testid="stMetricValue"]{
  font-size:1.9rem; font-weight:700; letter-spacing:-.02em;
  font-variant-numeric:tabular-nums; font-feature-settings:"tnum" 1;
}
[data-testid="stMetricDelta"]{font-variant-numeric:tabular-nums;}
.st-key-nids-metric-malicious [data-testid="stMetricValue"]{color:var(--danger);}
.st-key-nids-metric-benign [data-testid="stMetricValue"]{color:var(--ok);}
.st-key-nids-metric-types [data-testid="stMetricValue"]{color:var(--accent2);}

/* ---------- data display ---------- */
[data-testid="stDataFrame"], [data-testid="stDataFrameResizable"]{
  border:1px solid var(--border); border-radius:14px; overflow:hidden;
}
[data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"]{
  background:var(--card); border:1px solid var(--border); border-radius:16px;
  padding:.6rem; overflow:hidden;
}
[data-testid="stAlert"]{border:1px solid var(--border); border-radius:14px;}

/* ---------- uploader ---------- */
[data-testid="stFileUploaderDropzone"]{
  background:var(--card); border:1px dashed rgba(148,163,207,.32); border-radius:16px;
  transition:border-color .18s ease, background .18s ease;
}
[data-testid="stFileUploaderDropzone"]:hover{
  border-color:var(--accent); background:rgba(129,140,248,.07);
}
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small{color:var(--muted);}

/* ---------- buttons ---------- */
.stButton > button,
[data-testid="stBaseButton-secondary"],
[data-testid="stDownloadButton"] > button,
[data-testid="stFileUploaderDropzone"] button{
  background-image:linear-gradient(90deg, var(--accent), var(--accent2));
  background-color:transparent; color:#060914 !important;
  border:0; border-radius:11px; font-weight:700; letter-spacing:.01em;
  padding:.5rem 1.05rem;
  box-shadow:0 10px 22px -14px rgba(34,211,238,.75);
  transition:transform .15s ease, box-shadow .15s ease, filter .15s ease;
}
.stButton > button:hover,
[data-testid="stDownloadButton"] > button:hover,
[data-testid="stFileUploaderDropzone"] button:hover{
  transform:translateY(-1px); filter:brightness(1.06);
  box-shadow:0 16px 28px -14px rgba(34,211,238,.85);
}
.stButton > button:focus-visible,
[data-testid="stDownloadButton"] > button:focus-visible{
  outline:2px solid var(--accent2); outline-offset:2px;
}
@media (prefers-reduced-motion: reduce){
  .stButton > button, [data-testid="stDownloadButton"] > button{transition:none;}
  .stButton > button:hover, [data-testid="stDownloadButton"] > button:hover{transform:none;}
}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"]{
  background-image:linear-gradient(180deg, rgba(11,17,36,.96), rgba(6,9,20,.96));
  border-right:1px solid var(--border);
}
[data-testid="stSidebar"] [data-testid="stMetric"]{padding:.7rem .85rem; border-radius:14px;}
[data-testid="stSidebar"] [data-testid="stMetricValue"]{font-size:1.25rem; color:var(--accent);}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3{
  font-size:.74rem !important; font-weight:700; letter-spacing:.18em; text-transform:uppercase;
  color:var(--muted);
}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p{color:var(--muted);}
</style>
"""

HERO = """<div class="nids-eyebrow">Network intrusion detection</div><h1 class="nids-h1">Is this traffic <span class="nids-grad">under attack?</span></h1><div class="nids-sub">Upload a CSV of CICFlowMeter network flows. Every flow is classified as benign traffic or a specific attack type.</div>"""
PILL_ON = '<div class="nids-pill"><span class="nids-dot"></span>Model online</div>'
PILL_OFF = '<div class="nids-pill"><span class="nids-dot off"></span>Model offline</div>'


def eyebrow(label, title):
    """Small uppercase kicker above a section heading (static text only)."""
    st.markdown(f'<div class="nids-rule"></div>'
                f'<div class="nids-eyebrow">{label}</div>'
                f'<div class="nids-sec">{title}</div>', unsafe_allow_html=True)


def verdict_style(value):
    """Fixed CSS per cell: green for benign, red for anything else. No interpolation."""
    return ("color:#34d399;font-weight:600" if str(value).strip() == BENIGN
            else "color:#f87171;font-weight:600")


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    import json
    with open(METRICS_PATH) as fh:
        return json.load(fh)


def prepare(df, features):
    """Apply the training-time cleaning rules and report what was dropped."""
    notes = []
    df.columns = [c.strip() for c in df.columns]

    truth = None
    if "Label" in df.columns:
        lab = df["Label"].astype(str).str.strip()
        bad_header = lab == "Label"          # repeated header rows inside the file
        if bad_header.any():
            notes.append(f"dropped {int(bad_header.sum())} repeated header row(s)")
            df, lab = df[~bad_header], lab[~bad_header]
        truth = lab
        df = df.drop(columns=["Label"])

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing {len(missing)} feature column(s), "
                         f"e.g. {', '.join(missing[:4])}")

    X = df[features].apply(pd.to_numeric, errors="coerce")
    n_inf = int(np.isinf(X.to_numpy(dtype="float64", na_value=np.nan)).sum())
    if n_inf:
        notes.append(f"{n_inf:,} infinite value(s) from zero-duration flows")
    X = X.replace([np.inf, -np.inf], np.nan)

    keep = X.notna().all(axis=1)
    if (~keep).any():
        notes.append(f"skipped {int((~keep).sum()):,} row(s) with missing or non-numeric values")
    X = X[keep].astype("float32")
    if truth is not None:
        truth = truth[keep]
    return X, truth, notes


bundle = load_model()
metrics = load_metrics()

st.markdown(CSS, unsafe_allow_html=True)
st.markdown(HERO + (PILL_ON if bundle is not None else PILL_OFF),
            unsafe_allow_html=True)

if bundle is None:
    st.error("No trained model found. Run `python src/train.py` first "
             "(it writes models/model.joblib).")
    st.stop()

model, features = bundle["model"], bundle["features"]
# the MLP is trained on integer labels, so the bundle carries the encoder that
# maps them back to 'Benign' / 'DDOS attack-HOIC'. The forest needs no decoding.
label_encoder = bundle.get("label_encoder")

with st.sidebar:
    st.header("Model")
    st.write(f"**{bundle.get('model_name', type(model).__name__)}**")
    st.write(f"{len(features)} flow features &middot; {len(bundle['classes'])} classes")
    if metrics:
        r = metrics["results"][metrics["best"]]
        st.metric("macro F1 (test split)", f"{r['macro_f1']:.3f}")
        st.metric("attack recall", f"{r['attack_recall']:.3f}")
        st.metric("false-positive rate", f"{r['fpr']:.3f}")
        st.caption(f"trained on {metrics['n_train']:,} flows, "
                   f"tested on {metrics['n_test']:,}")
    st.divider()
    st.caption("Scores come from held-out data of the same capture, so they read higher "
               "than this model would manage on another network.")

upload = st.file_uploader("Flow CSV", type="csv")
if upload is None:
    st.info("Expecting the CICFlowMeter columns. A `Label` column is optional and is "
            "used only to score the predictions, never to make them.")
    st.stop()

try:
    raw = pd.read_csv(upload, low_memory=False, dtype=str)
except Exception as exc:
    st.error(f"Could not read that CSV: {exc}")
    st.stop()

try:
    X, truth, notes = prepare(raw, features)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

if X.empty:
    st.error("No usable rows in that CSV.")
    st.stop()

with st.spinner(f"Scoring {len(X):,} flows..."):
    pred = model.predict(X)
    if label_encoder is not None:
        pred = label_encoder.inverse_transform(pred)

attacks = pred != BENIGN
n_attack, n_total = int(attacks.sum()), len(pred)

eyebrow("Summary", "Threat report")
c1, c2, c3, c4 = st.columns(4)
c1.container(key="nids-metric-total").metric("Flows scanned", f"{n_total:,}")
c2.container(key="nids-metric-malicious").metric(
    "Malicious", f"{n_attack:,}", f"{n_attack / n_total:.1%}",
    delta_color="inverse")
c3.container(key="nids-metric-benign").metric("Benign", f"{n_total - n_attack:,}")
c4.container(key="nids-metric-types").metric(
    "Attack types seen", f"{len(set(pred[attacks]))}")

if notes:
    st.caption("Cleaning: " + "; ".join(notes))

if n_attack:
    eyebrow("Breakdown", "Attack types detected")
    breakdown = (pd.Series(pred[attacks]).value_counts()
                 .rename_axis("attack type").reset_index(name="flows"))
    breakdown["share of malicious"] = (breakdown["flows"] / n_attack).map("{:.1%}".format)
    left, right = st.columns([1, 1])
    left.dataframe(breakdown, hide_index=True, use_container_width=True)
    right.bar_chart(breakdown.set_index("attack type")["flows"], color="#f87171")
else:
    st.success("No malicious flows found in this file.")

eyebrow("Detail", "Suspicious flows")
out = pd.DataFrame({"row": X.index + 2, "prediction": pred})   # +2: 1-based plus header
if truth is not None:
    out["true label"] = truth.values
    correct = (pred == truth.values).mean()
    st.caption(f"A `Label` column was present: {correct:.1%} of rows match "
               f"({int((pred == truth.values).sum()):,} of {n_total:,}).")
show = out[out["prediction"] != BENIGN] if n_attack else out
head = show.head(200)
verdict_cols = [c for c in ("prediction", "true label") if c in head.columns]
st.dataframe(head.style.map(verdict_style, subset=verdict_cols),
             hide_index=True, use_container_width=True)
if len(show) > 200:
    st.caption(f"Showing the first 200 of {len(show):,}.")

buf = io.StringIO()
out.to_csv(buf, index=False)
st.download_button("Download full report (CSV)", buf.getvalue(),
                   file_name="threat_report.csv", mime="text/csv")
