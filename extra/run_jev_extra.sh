#!/usr/bin/env bash
cd "$(dirname "$0")/.."
PY=.venv/bin/python; P=extra/predictions; L=extra/logs
$PY extra/predict_jev_atis_extra.py --mode desc > $L/atis_desc.log 2>&1
$PY extra/predict_jev_atis_extra.py --mode noul > $L/atis_noul.log 2>&1
for s in 1 2; do $PY atis/predict_jev_atis.py --variant raw --shuffle $s --out $P/atis__jev__raw__s$s.jsonl > $L/atis_s$s.log 2>&1; done
for s in 1 2; do $PY jev/predict.py --variant raw --shuffle $s --out $P/banking__jev__raw__s$s.jsonl > $L/banking_s$s.log 2>&1; done
for i in 1 2 3; do
  $PY atis/predict_jev_atis.py --variant raw --limit 500 --out $P/atis__jev__det$i.jsonl > $L/atis_det$i.log 2>&1
  $PY jev/predict.py --variant raw --limit 500 --out $P/banking__jev__det$i.jsonl > $L/banking_det$i.log 2>&1
done
echo ALL_DONE > $L/ALL_DONE
