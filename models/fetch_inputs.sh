#!/bin/bash
# Downloads the kws and ad01 models' input files from MLCommons Tiny (Apache 2.0, the commit below) into models/ (git-ignored); each checked by sha256 first.
set -e
cd "$(dirname "$0")"
base=https://raw.githubusercontent.com/mlcommons/tiny/4addd0fa08d216e20637637874e084895f289da4/benchmark
fetch() {  # NAME SHA256 PATH under $base
  if [ -f "$1" ] && echo "$2  $1" | sha256sum -c --quiet - 2>/dev/null; then echo "fetch_inputs: $1 is already there"; return; fi
  curl -fL --retry 3 -o "$1.part" "$base/$3"
  echo "$2  $1.part" | sha256sum -c --quiet - || { rm -f "$1.part"; echo "fetch_inputs: $1 does not match its sha256"; exit 1; }
  mv "$1.part" "$1"; echo "fetch_inputs: $1 ready"
}
fetch marvin_617de221_0.wav d08ee300c6a9fb2ce78b1f0505de80b2af19a4c456b48d24bb29949271bf6986 runner/sd_card/marvin_617de221_0.wav
fetch normal_id_01_00000000_hist_librosa.bin 31bc130d27e3732e1c09db946ccc7bfa130f98739bcd90cfa39d590f61f4d6fa \
  reference_submissions/anomaly_detection/datasets/dcase01/normal_id_01_00000000_hist_librosa.bin
