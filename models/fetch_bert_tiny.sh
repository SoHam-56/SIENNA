#!/bin/bash
# Downloads BERT-Tiny (google/bert_uncased_L-2_H-128_A-2, Apache 2.0) from Hugging Face into models/bert_tiny (git-ignored); each file checked by sha256.
set -e
cd "$(dirname "$0")" && mkdir -p bert_tiny && cd bert_tiny
base=https://huggingface.co/google/bert_uncased_L-2_H-128_A-2/resolve/main
fetch() {  # NAME SHA256
  if [ -f "$1" ] && echo "$2  $1" | sha256sum -c --quiet - 2>/dev/null; then echo "fetch_bert_tiny: $1 is already there"; return; fi
  curl -fL --retry 3 -o "$1.part" "$base/$1"
  echo "$2  $1.part" | sha256sum -c --quiet - || { rm -f "$1.part"; echo "fetch_bert_tiny: $1 does not match its sha256"; exit 1; }
  mv "$1.part" "$1"; echo "fetch_bert_tiny: $1 ready"
}
fetch config.json 508e1f01aae55d73355cbdd82609be2f43ba5a0d3428837adbe56cf8391f8b39
fetch vocab.txt 07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3
fetch model.safetensors 7fb69ad9f6866d8983183c930e33828f326470bf6ad8bbb2ad4ed957a92e9414
