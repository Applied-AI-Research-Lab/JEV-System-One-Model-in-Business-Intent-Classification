# Datasets (not redistributed as separate files)

The prediction files in this repository contain the utterance text next to each prediction, so that every number can be
re-checked without downloading anything. To **re-run** the models you need the original corpora:

## Banking77 (77 intents, 13,083 utterances = official train 10,003 + test 3,080)
* Source: https://huggingface.co/datasets/PolyAI/banking77 (original files: `train.csv`, `test.csv` in
  https://github.com/PolyAI-LDN/task-specific-datasets, `banking_data/`). Licence: CC BY 4.0.
* Put `train.csv` and `test.csv` in `Datasets/banking77/`, then run `python prepare_data.py` (merges both files, normalises
  the single label `reverted_card_payment?`). Output: `Datasets/banking77/banking77_all.csv`.

## ATIS (8 intents, 5,634 utterances = train + test of the intent-labelled release)
* Source: https://www.kaggle.com/datasets/hassanamin/atis-airlinetravelinformationsystem (files `atis_intents_train.csv`,
  `atis_intents_test.csv`; the third file `atis_intents.csv` is not the union of the two and is not used).
  The original ATIS corpus (Hemphill et al., 1990) is distributed by the LDC; check the licence of the release you use.
* Put the two files in `Datasets/ATIS Airline Travel Information System/`, then run `python atis/prepare_atis.py`.

For the small open models copy the two merged files to `JEVvsLLMs/data/` (`banking77_all.csv`, `atis_all.csv`).
