# Extra Jev-API analyses

## 1. Sensitivity to the order of the options (alphabetical s0 vs two random orders)

### Banking77 (13083 rows)

| order | accuracy | macro-F1 | ECE |
|---|---|---|---|
| s0 (alphabetical) | 0.7786 | 0.7770 | 0.097 |
| s1 | 0.7864 | 0.7820 | 0.092 |
| s2 | 0.7889 | 0.7870 | 0.089 |

Accuracy mean 0.7847, range 0.0102 (max-min).
- s0 (alphabetical) vs s1: identical prediction on 93.3% of rows; McNemar p = 2.2e-05
- s0 (alphabetical) vs s2: identical prediction on 93.7% of rows; McNemar p = 1.7e-08
- s1 vs s2: identical prediction on 94.0% of rows; McNemar p = 0.17
- All three orders agree on 90.7% of rows. Majority vote over the three orders: accuracy 0.7879 (single run 0.7786).
- Accuracy when the three orders agree: 0.829 (11870 rows); when they disagree: 0.284 (1213 rows) -> order-disagreement is itself an uncertainty signal.

### ATIS (5634 rows)

| order | accuracy | macro-F1 | ECE |
|---|---|---|---|
| s0 (alphabetical) | 0.8655 | 0.7376 | 0.023 |
| s1 | 0.9018 | 0.7631 | 0.030 |
| s2 | 0.9246 | 0.8149 | 0.023 |

Accuracy mean 0.8973, range 0.0591 (max-min).
- s0 (alphabetical) vs s1: identical prediction on 91.2% of rows; McNemar p = 7.5e-21
- s0 (alphabetical) vs s2: identical prediction on 91.3% of rows; McNemar p = 1.8e-56
- s1 vs s2: identical prediction on 95.8% of rows; McNemar p = 5e-18
- All three orders agree on 89.2% of rows. Majority vote over the three orders: accuracy 0.9139 (single run 0.8655).
- Accuracy when the three orders agree: 0.937 (5024 rows); when they disagree: 0.274 (610 rows) -> order-disagreement is itself an uncertainty signal.

## 2. Repeatability of the API (same 500 rows, 3 repetitions, temperature-free)

- Banking77: 500 rows x 3 runs -> identical prediction in 97.0% of rows; top-probability spread (max-min across runs): mean 0.0226, max 0.1900.
- ATIS: 500 rows x 3 runs -> identical prediction in 96.8% of rows; top-probability spread (max-min across runs): mean 0.0252, max 0.1700.

## 3. ATIS with option descriptions (post-hoc diagnostic, NOT pure zero-shot)

| condition | accuracy | macro-F1 | ECE | McNemar p vs names-only |
|---|---|---|---|---|
| names only (zero-shot) | 0.8655 | 0.7376 | 0.023 | - |
| + descriptions | 0.9272 | 0.8317 | 0.050 | 5.1e-51 |

Per-label F1 (precision / recall):

| label | names only | + descriptions |
|---|---|---|
| atis_flight | 0.92 (0.98 / 0.87) | 0.96 (0.98 / 0.93) |
| atis_airfare | 0.72 (0.59 / 0.94) | 0.79 (0.69 / 0.93) |
| atis_ground_service | 0.94 (0.89 / 1.00) | 0.98 (0.95 / 1.00) |
| atis_airline | 0.87 (0.85 / 0.88) | 0.79 (0.79 / 0.78) |
| atis_abbreviation | 0.50 (0.98 / 0.33) | 0.90 (0.82 / 0.99) |
| atis_aircraft | 0.88 (0.85 / 0.91) | 0.92 (0.90 / 0.94) |
| atis_flight_time | 0.20 (0.11 / 0.67) | 0.37 (0.44 / 0.33) |
| atis_quantity | 0.88 (0.84 / 0.91) | 0.95 (0.90 / 1.00) |

## 4. ATIS with eight Noul questions (one per intent, absolute probabilities)

- argmax of the eight Noul values: accuracy 0.9214, macro-F1 0.6884 (Choice names-only: 0.8655 / 0.7376); McNemar p = 8.9e-35.
- ECE of the winning Noul value: 0.038. Sum of the eight probabilities: mean 3.05 (a Choice sums to 1).
- Rows with more than one intent above 0.5: 92.7%; rows with no intent above 0.5: 0.2%.

| label | Noul F1 (precision / recall) | Choice F1 (precision / recall) |
|---|---|---|
| atis_flight | 0.96 (0.95 / 0.98) | 0.92 (0.98 / 0.87) |
| atis_airfare | 0.82 (0.75 / 0.90) | 0.72 (0.59 / 0.94) |
| atis_ground_service | 0.98 (0.96 / 0.99) | 0.94 (0.89 / 1.00) |
| atis_airline | 0.62 (0.76 / 0.53) | 0.87 (0.85 / 0.88) |
| atis_abbreviation | 0.63 (0.93 / 0.47) | 0.50 (0.98 / 0.33) |
| atis_aircraft | 0.76 (0.81 / 0.72) | 0.88 (0.85 / 0.91) |
| atis_flight_time | 0.03 (0.12 / 0.02) | 0.20 (0.11 / 0.67) |
| atis_quantity | 0.71 (0.75 / 0.67) | 0.88 (0.84 / 0.91) |
