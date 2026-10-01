# Jev vs small LLMs vs GLM-5.3 — comparison


## Banking77 (77 intents) — raw labels — 13083 rows in common (majority baseline 0.017)

| system | acc [95% CI] | macro-F1 | invalid | ECE | AURC | acc@50% | acc@80% | in tok | out tok | s/utt (see note) | McNemar p vs Jev |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Jev | 0.779 [0.772, 0.786] | 0.777 | 0.000 | 0.097 | 0.074 | 0.955 | 0.869 | 1028 | 827 | 0.33 | - |
| Qwen3.5-4B | 0.639 [0.630, 0.647] | 0.635 | 0.000 | 0.227 | 0.163 | 0.845 | 0.727 | 529 | 0 | 0.30 | 5.3e-305 |
| Gemma-3-4B | 0.558 [0.550, 0.566] | 0.541 | 0.000 | 0.414 | 0.318 | 0.712 | 0.633 | 671 | 0 | 0.77 | 0 |
| GLM-5.3 (cloud) | 0.801 [0.794, 0.807] | 0.797 | 0.002 | n/a | n/a | n/a | n/a | 530 | 151 | 3.62 | 1.8e-18 |

Note: s/utt = mean latency per utterance; Jev = hosted API call, SLMs = batched (8) on one H100, GLM = Ollama cloud with reasoning. Not like-for-like: compare tokens and cost, not seconds.

Paired Jev vs GLM: both right 9789, only GLM right 686, only Jev right 398, both wrong 2210.

Cascade Jev -> GLM (least confident Jev answers go to GLM):

| share sent to GLM | accuracy | GLM calls |
|---|---|---|
| 0% | 0.779 | 0 |
| 5% | 0.788 | 654 |
| 10% | 0.793 | 1308 |
| 20% | 0.800 | 2617 |
| 30% | 0.804 | 3925 |
| 50% | 0.801 | 6542 |
| 100% | 0.801 | 13083 |

## Banking77 (77 intents) — human labels — 13083 rows in common (majority baseline 0.017)

| system | acc [95% CI] | macro-F1 | invalid | ECE | AURC | acc@50% | acc@80% | in tok | out tok | s/utt (see note) | McNemar p vs Jev |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Jev | 0.777 [0.771, 0.785] | 0.775 | 0.000 | 0.100 | 0.075 | 0.956 | 0.868 | 956 | 754 | 0.33 | - |
| Qwen3.5-4B | 0.643 [0.635, 0.651] | 0.638 | 0.000 | 0.230 | 0.161 | 0.849 | 0.736 | 490 | 0 | 0.27 | 1.2e-287 |
| Gemma-3-4B | 0.559 [0.550, 0.567] | 0.539 | 0.000 | 0.417 | 0.322 | 0.712 | 0.629 | 483 | 0 | 0.49 | 0 |

Note: s/utt = mean latency per utterance; Jev = hosted API call, SLMs = batched (8) on one H100, GLM = Ollama cloud with reasoning. Not like-for-like: compare tokens and cost, not seconds.

## ATIS (8 intents) — raw labels — 5634 rows in common (majority baseline 0.763)

| system | acc [95% CI] | macro-F1 | invalid | ECE | AURC | acc@50% | acc@80% | in tok | out tok | s/utt (see note) | McNemar p vs Jev |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Jev | 0.865 [0.857, 0.874] | 0.738 | 0.000 | 0.023 | 0.046 | 0.962 | 0.922 | 364 | 94 | 0.32 | - |
| Qwen3.5-4B | 0.887 [0.878, 0.895] | 0.686 | 0.000 | 0.049 | 0.062 | 0.938 | 0.926 | 123 | 0 | 0.20 | 1.7e-07 |
| Gemma-3-4B | 0.925 [0.918, 0.932] | 0.728 | 0.000 | 0.071 | 0.077 | 0.940 | 0.949 | 123 | 0 | 0.46 | 2.2e-39 |
| GLM-5.3 (cloud) | 0.934 [0.928, 0.941] | 0.859 | 0.001 | n/a | n/a | n/a | n/a | 124 | 100 | 3.27 | 2.1e-67 |

Note: s/utt = mean latency per utterance; Jev = hosted API call, SLMs = batched (8) on one H100, GLM = Ollama cloud with reasoning. Not like-for-like: compare tokens and cost, not seconds.

Paired Jev vs GLM: both right 4795, only GLM right 469, only Jev right 81, both wrong 289.

Cascade Jev -> GLM (least confident Jev answers go to GLM):

| share sent to GLM | accuracy | GLM calls |
|---|---|---|
| 0% | 0.865 | 0 |
| 5% | 0.887 | 282 |
| 10% | 0.903 | 563 |
| 20% | 0.920 | 1127 |
| 30% | 0.924 | 1690 |
| 50% | 0.928 | 2817 |
| 100% | 0.934 | 5634 |

## ATIS (8 intents) — human labels — 5634 rows in common (majority baseline 0.763)

| system | acc [95% CI] | macro-F1 | invalid | ECE | AURC | acc@50% | acc@80% | in tok | out tok | s/utt (see note) | McNemar p vs Jev |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Jev | 0.869 [0.861, 0.878] | 0.766 | 0.000 | 0.018 | 0.032 | 0.986 | 0.942 | 354 | 82 | 0.32 | - |
| Qwen3.5-4B | 0.916 [0.909, 0.923] | 0.810 | 0.000 | 0.030 | 0.020 | 0.985 | 0.973 | 104 | 0 | 0.17 | 1.1e-30 |
| Gemma-3-4B | 0.920 [0.913, 0.927] | 0.766 | 0.000 | 0.077 | 0.031 | 0.969 | 0.966 | 96 | 0 | 0.28 | 1.9e-26 |

Note: s/utt = mean latency per utterance; Jev = hosted API call, SLMs = batched (8) on one H100, GLM = Ollama cloud with reasoning. Not like-for-like: compare tokens and cost, not seconds.
