# Option-order sensitivity: Jev vs Qwen3.5-4B vs Gemma-3-4B

Seed 0 = alphabetical order; seeds >= 1 = the same random orders for every system.


## Banking77: 3000 rows, seeds used: [0, 1, 2, 3]

| system | order | accuracy | macro-F1 | ECE |
|---|---|---|---|---|
| Jev | s0 | 0.7773 | 0.7735 | 0.099 |
| Jev | s1 | 0.7860 | 0.7808 | 0.092 |
| Jev | s2 | 0.7883 | 0.7852 | 0.088 |
| Jev | s3 | 0.7770 | 0.7719 | 0.099 |
| Qwen3.5-4B | s0 | 0.6327 | 0.6274 | 0.229 |
| Qwen3.5-4B | s1 | 0.6697 | 0.6548 | 0.206 |
| Qwen3.5-4B | s2 | 0.6517 | 0.6461 | 0.213 |
| Qwen3.5-4B | s3 | 0.6507 | 0.6352 | 0.222 |
| Gemma-3-4B | s0 | 0.5420 | 0.5244 | 0.429 |
| Gemma-3-4B | s1 | 0.5307 | 0.5124 | 0.445 |
| Gemma-3-4B | s2 | 0.6037 | 0.5880 | 0.366 |
| Gemma-3-4B | s3 | 0.5417 | 0.5156 | 0.431 |

### Summary over orders

| system | accuracy mean ± sd | range (max-min) | macro-F1 mean ± sd | ECE mean | orders agreeing on a row (all) | accuracy: agree / disagree | vote accuracy |
|---|---|---|---|---|---|---|---|
| Jev | 0.7822 ± 0.0059 | 0.0113 | 0.7779 ± 0.0062 | 0.095 | 88.7% | 0.836 / 0.317 | 0.7787 |
| Qwen3.5-4B | 0.6512 ± 0.0151 | 0.0370 | 0.6409 ± 0.0120 | 0.217 | 65.2% | 0.800 / 0.318 | 0.6583 |
| Gemma-3-4B | 0.5545 ± 0.0332 | 0.0730 | 0.5351 ± 0.0357 | 0.418 | 44.3% | 0.776 / 0.356 | 0.5860 |

### Does Jev beat each small model under the SAME option order?

| comparison | seeds where Jev accuracy is higher | seeds where Jev macro-F1 is higher |
|---|---|---|
| Jev vs Qwen3.5-4B | 4 of 4 | 4 of 4 |
| Jev vs Gemma-3-4B | 4 of 4 | 4 of 4 |

## ATIS: 5634 rows, seeds used: [0, 1, 2, 3]

| system | order | accuracy | macro-F1 | ECE |
|---|---|---|---|---|
| Jev | s0 | 0.8655 | 0.7376 | 0.023 |
| Jev | s1 | 0.9018 | 0.7631 | 0.030 |
| Jev | s2 | 0.9246 | 0.8149 | 0.023 |
| Jev | s3 | 0.9102 | 0.7701 | 0.036 |
| Qwen3.5-4B | s0 | 0.8871 | 0.6858 | 0.049 |
| Qwen3.5-4B | s1 | 0.9201 | 0.8207 | 0.044 |
| Qwen3.5-4B | s2 | 0.8229 | 0.7626 | 0.057 |
| Qwen3.5-4B | s3 | 0.8976 | 0.7385 | 0.059 |
| Gemma-3-4B | s0 | 0.9251 | 0.7282 | 0.071 |
| Gemma-3-4B | s1 | 0.9310 | 0.7582 | 0.063 |
| Gemma-3-4B | s2 | 0.9228 | 0.8039 | 0.069 |
| Gemma-3-4B | s3 | 0.8953 | 0.6768 | 0.100 |

### Summary over orders

| system | accuracy mean ± sd | range (max-min) | macro-F1 mean ± sd | ECE mean | orders agreeing on a row (all) | accuracy: agree / disagree | vote accuracy |
|---|---|---|---|---|---|---|---|
| Jev | 0.9005 ± 0.0252 | 0.0591 | 0.7715 ± 0.0322 | 0.028 | 88.5% | 0.939 / 0.297 | 0.9132 |
| Qwen3.5-4B | 0.8819 ± 0.0417 | 0.0973 | 0.7519 ± 0.0559 | 0.052 | 83.3% | 0.932 / 0.662 | 0.8953 |
| Gemma-3-4B | 0.9185 ± 0.0159 | 0.0357 | 0.7418 ± 0.0534 | 0.076 | 90.2% | 0.963 / 0.578 | 0.9230 |

### Does Jev beat each small model under the SAME option order?

| comparison | seeds where Jev accuracy is higher | seeds where Jev macro-F1 is higher |
|---|---|---|
| Jev vs Qwen3.5-4B | 2 of 4 | 3 of 4 |
| Jev vs Gemma-3-4B | 2 of 4 | 4 of 4 |
