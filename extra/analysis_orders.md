# Jev across option orders


## Banking77: 13083 rows, 5 orders (s0 = alphabetical)

| order | accuracy | macro-F1 | ECE | acc@50% coverage |
|---|---|---|---|---|
| s0 | 0.7786 | 0.7770 | 0.097 | 0.955 |
| s1 | 0.7864 | 0.7820 | 0.092 | 0.952 |
| s2 | 0.7889 | 0.7870 | 0.089 | 0.956 |
| s3 | 0.7822 | 0.7800 | 0.097 | 0.953 |
| s4 | 0.7873 | 0.7838 | 0.096 | 0.958 |
| **mean ± sd** | **0.7847 ± 0.0042** | **0.7820 ± 0.0038** | **0.094 ± 0.004** | |
| min / max | 0.7786 / 0.7889 | 0.7770 / 0.7870 | | |

Majority vote over 5 orders: accuracy 0.7897, macro-F1 0.7867.
Rows where all orders agree: 87.8% (vote accuracy there 0.841); the rest: vote accuracy 0.423.

Single-order comparison systems (alphabetical order, same rows):

| system | accuracy | macro-F1 | Jev orders above this accuracy |
|---|---|---|---|
| Qwen3.5-4B | 0.6387 | 0.6348 | 5 of 5 |
| Gemma-3-4B | 0.5582 | 0.5408 | 5 of 5 |
| GLM-5.3 | 0.8007 | 0.7969 | 0 of 5 |

## ATIS: 5634 rows, 7 orders (s0 = alphabetical)

| order | accuracy | macro-F1 | ECE | acc@50% coverage |
|---|---|---|---|---|
| s0 | 0.8655 | 0.7376 | 0.023 | 0.962 |
| s1 | 0.9018 | 0.7631 | 0.030 | 0.971 |
| s2 | 0.9246 | 0.8149 | 0.023 | 0.978 |
| s3 | 0.9102 | 0.7701 | 0.036 | 0.990 |
| s4 | 0.9231 | 0.8460 | 0.040 | 0.947 |
| s5 | 0.9088 | 0.7857 | 0.024 | 0.995 |
| s6 | 0.9187 | 0.7919 | 0.027 | 0.977 |
| **mean ± sd** | **0.9075 ± 0.0203** | **0.7870 ± 0.0355** | **0.029 ± 0.007** | |
| min / max | 0.8655 / 0.9246 | 0.7376 / 0.8460 | | |

Majority vote over 7 orders: accuracy 0.9173, macro-F1 0.8244.
Rows where all orders agree: 87.3% (vote accuracy there 0.944); the rest: vote accuracy 0.736.

Single-order comparison systems (alphabetical order, same rows):

| system | accuracy | macro-F1 | Jev orders above this accuracy |
|---|---|---|---|
| Qwen3.5-4B | 0.8871 | 0.6858 | 6 of 7 |
| Gemma-3-4B | 0.9251 | 0.7282 | 0 of 7 |
| GLM-5.3 | 0.9343 | 0.8590 | 0 of 7 |
