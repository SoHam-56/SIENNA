#!/bin/bash
# Downloads ECG5000 (UCR time-series archive, Chen et al.; from timeseriesclassification.com) into models/ecg5000 (git-ignored) for the LSTM model; sha256-checked.
set -e
cd "$(dirname "$0")" && mkdir -p ecg5000 && cd ecg5000
[ -f ECG5000_TRAIN.txt ] && [ -f ECG5000_TEST.txt ] && { echo "fetch_ecg5000: ECG5000_TRAIN.txt and ECG5000_TEST.txt are already there"; exit 0; }
curl -fL --retry 3 -o ECG5000.zip https://www.timeseriesclassification.com/aeon-toolkit/ECG5000.zip
echo "41f6de20ac895e9ce31753860995518951f1ed42a405d0e51c909d27e3b3c5a4  ECG5000.zip" | sha256sum -c --quiet - || { rm -f ECG5000.zip; echo "fetch_ecg5000: ECG5000.zip does not match its sha256"; exit 1; }
unzip -o -q ECG5000.zip ECG5000_TRAIN.txt ECG5000_TEST.txt && rm ECG5000.zip
echo "fetch_ecg5000: ready (cite the UCR archive and Chen et al. if you use it)"
