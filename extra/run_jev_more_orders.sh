#!/usr/bin/env bash
cd "$(dirname "$0")/.."
PY=.venv/bin/python; P=extra/predictions; L=extra/logs
for s in 3 4 5 6; do $PY atis/predict_jev_atis.py --variant raw --shuffle $s --out $P/atis__jev__raw__s$s.jsonl > $L/atis_s$s.log 2>&1; done
for s in 3 4; do $PY jev/predict.py --variant raw --shuffle $s --out $P/banking__jev__raw__s$s.jsonl > $L/banking_s$s.log 2>&1; done
echo DONE > $L/MORE_ORDERS_DONE
