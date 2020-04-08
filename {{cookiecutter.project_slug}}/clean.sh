#!/bin/bash
#
# 対応するpyファイルの無いmodels配下のディレクトリがあれば _trash 配下へ移動するスクリプト。
# 逆の復元もする。
#
# 前提とする命名規則:
#    ./{xxx}.py  <=>  ./models/{xxx}/
#
set -eu

mkdir -p _trash/models/

for d in models/* ; do
    if [ ! -f "$(basename $d).py" ] ; then
        mv $d _trash/models/
        echo "'$d' removed."
    fi
done

for f in *.py ; do
    if [ ! -d "models/$(basename $f .py)" -a -d "_trash/models/$(basename $f .py)" ] ; then
        mv "_trash/models/$(basename $f .py)" models/
        echo "'models/$(basename $f .py)' restored."
    fi
done
