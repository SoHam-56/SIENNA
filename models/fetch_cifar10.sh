#!/bin/bash
# Downloads CIFAR-10 (python version) from its official page into models/data/ (git-ignored) for the resnet8 model; checks its md5 first.
# With the argument train it also extracts the five training batches (regression.py --action selu-model trains on them).
set -e
cd "$(dirname "$0")"
url=https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
md5=c58f30108f718f92721af3b95e74349a  # as published on https://www.cs.toronto.edu/~kriz/cifar.html
mkdir -p data
want="cifar-10-batches-py/test_batch cifar-10-batches-py/batches.meta cifar-10-batches-py/readme.html"
[ "$1" = train ] && want="$want $(for i in 1 2 3 4 5; do echo -n "cifar-10-batches-py/data_batch_$i "; done)"
have=1; for f in $want; do [ -f "data/$f" ] || have=0; done
[ $have = 1 ] && { echo "fetch_cifar10: $want already there"; exit 0; }
curl -fL --retry 3 -o data/cifar-10-python.tar.gz "$url"
echo "$md5  data/cifar-10-python.tar.gz" | md5sum -c -
tar -xzf data/cifar-10-python.tar.gz -C data $want
rm data/cifar-10-python.tar.gz
echo "fetch_cifar10: data/cifar-10-batches-py/test_batch ready (cite Krizhevsky 2009 if you use it)"
