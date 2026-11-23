#!/bin/bash
# Fetch the ten CSE-CIC-IDS2018 "Processed Traffic Data for ML Algorithms" days.
#
# The bucket is public and anonymous, so no aws CLI and no credentials are needed:
#   aws s3 sync --no-sign-request s3://cse-cic-ids2018/ .    # the lab's way
# curl over https works the same and skips the aws dependency.
#
# Smallest day first (103 MB) so the pipeline can be built before the 4 GB day lands.
set -u

BASE="https://cse-cic-ids2018.s3.amazonaws.com/Processed%20Traffic%20Data%20for%20ML%20Algorithms"
DEST="$(dirname "$0")/../data"
mkdir -p "$DEST"

DAYS=(
  "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv"   # Infilteration      103 MB
  "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv"  # DoS GoldenEye/Slowloris
  "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv"   # DoS Hulk/SlowHTTPTest
  "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv"     # DoS Hulk
  "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv"   # Brute Force Web/XSS, SQL Injection
  "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv"     # Brute Force Web/XSS, SQL Injection
  "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv"  # DDOS LOIC-UDP / HOIC
  # NOTE: the key in the bucket is misspelled "Thuesday", not "Tuesday".
  # Using the correct spelling returns HTTP 404, so the typo is kept deliberately.
  "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"   # DDoS LOIC-HTTP     ~4 GB
  "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"     # Bot
  "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv"  # Infilteration
)

for f in "${DAYS[@]}"; do
  out="$DEST/$f"
  if [ -s "$out" ]; then
    echo "have    $f  ($(du -h "$out" | cut -f1))"
    continue
  fi
  echo "fetching $f ..."
  # -C - resumes a partial file, so a dropped connection does not restart the 4 GB day
  if curl -fL -C - --retry 3 --retry-delay 5 -o "$out" "$BASE/${f// /%20}"; then
    echo "   got  $f  ($(du -h "$out" | cut -f1))"
  else
    echo "   FAILED $f" >&2
  fi
done

echo
echo "--- data directory ---"
ls -lh "$DEST"
