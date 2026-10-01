# Offline analyses (raw labels)


## Banking77: 13083 rows, systems: Jev, Qwen, Gemma, GLM

### A. Consensus disagreement with the gold label (possible label noise / ambiguity)

- All 4 systems predict the SAME label and it differs from gold: **807** rows (6.2%).
- All systems wrong (labels may differ): 1935 rows (14.8%).
- Accuracy if the consensus-disagreement rows are excluded: Jev 0.830 (was 0.779), Qwen 0.681 (was 0.639), Gemma 0.595 (was 0.558), GLM 0.853 (was 0.801)
- Sample of consensus-disagreement rows for a manual audit (gold -> consensus prediction):

  - `top_up_reverted` -> `top_up_failed` : The app wouldn't accept my top up.
  - `top_up_reverted` -> `top_up_failed` : I believe my money did not go through with my top up, was there a problem on your end?
  - `top_up_reverted` -> `top_up_failed` : I put money into my account for the minimum balance but the application didn't accept.
  - `top_up_reverted` -> `top_up_failed` : I believed crypto top up with something you offered. This does not seem to be working. The money has been remo
  - `top_up_reverted` -> `top_up_failed` : On my last transaction it seem that my top-up was not successful.
  - `card_payment_not_recognised` -> `compromised_card` : Has someone accessed my card there are payments I did not make that are showing up on the app.
  - `card_arrival` -> `card_delivery_estimate` : How long does a card delivery take?
  - `card_payment_not_recognised` -> `compromised_card` : What should I do to get transactions off of my account if I didn't make them?  My card must have been compromi
  - `card_payment_not_recognised` -> `compromised_card` : I see a charge on my account that I don't recall making. I feel like my account may have been compromised.
  - `why_verify_identity` -> `verify_my_identity` : What other methods are there to verify my identity?
  - `unable_to_verify_identity` -> `verify_my_identity` : Help my verify my id.
  - `card_payment_wrong_exchange_rate` -> `exchange_charge` : I made a currency exchange and think I was charged more than I should of been.

### B. Calibration direction (bins of the top-label probability; + = overconfident)

| system | mean confidence | accuracy | gap | AUROC(conf -> correct) | [0.0,0.1) | [0.1,0.2) | [0.2,0.3) | [0.3,0.4) | [0.4,0.5) | [0.5,0.6) | [0.6,0.7) | [0.7,0.8) | [0.8,0.9) | [0.9,1.0) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Jev | 0.876 | 0.779 | +0.097 | 0.840 | - | 4: 0.00 | 50: 0.12 | 203: 0.22 | 488: 0.35 | 934: 0.42 | 846: 0.49 | 944: 0.56 | 1363: 0.68 | 8658: 0.92 |
| Qwen | 0.866 | 0.639 | +0.227 | 0.801 | - | 38: 0.08 | 133: 0.15 | 309: 0.20 | 561: 0.25 | 744: 0.29 | 711: 0.36 | 879: 0.38 | 1197: 0.44 | 8511: 0.80 |
| Gemma | 0.972 | 0.558 | +0.414 | 0.707 | - | - | 2: 0.00 | 11: 0.18 | 103: 0.18 | 103: 0.20 | 251: 0.24 | 248: 0.22 | 390: 0.27 | 11975: 0.59 |
| GLM | n/a | 0.801 | n/a | n/a | | | | | | | | | | |

(bin cells = rows in bin: accuracy in bin)

Jev's own `confidence` field vs top probability as an error detector: AUROC 0.843 vs 0.840.

### C. Paired bootstrap of the accuracy difference (Jev minus system), 95% CI

| system | difference | 95% CI |
|---|---|---|
| Qwen | +0.140 | [+0.133, +0.147] |
| Gemma | +0.220 | [+0.212, +0.229] |
| GLM | -0.022 | [-0.027, -0.017] |

### D. Row difficulty: how many of the systems get the row right

| systems correct | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| rows | 1935 (14.8%) | 778 (5.9%) | 1785 (13.6%) | 2367 (18.1%) | 6218 (47.5%) |

Oracle (at least one system right): 0.852. Majority vote of systems right (>= half): 0.793.

### E. Accuracy by utterance length (words, quartiles)

| length | rows | Jev | Qwen | Gemma | GLM |
|---|---|---|---|---|---|
| <=7 | 3548 | 0.786 | 0.660 | 0.595 | 0.797 |
| (7,10] | 4237 | 0.805 | 0.672 | 0.598 | 0.838 |
| (10,13] | 2411 | 0.811 | 0.640 | 0.562 | 0.837 |
| (13,max] | 2887 | 0.704 | 0.561 | 0.452 | 0.721 |

### F. Jev top-k accuracy

| k | accuracy |
|---|---|
| 1 | 0.778 |
| 2 | 0.869 |
| 3 | 0.906 |
| 5 | 0.929 |

### G. Labels where Jev is weakest (per-label accuracy)

| label | rows | Jev | Qwen | Gemma | GLM |
|---|---|---|---|---|---|
| get_physical_card | 146 | 0.00 | 0.01 | 0.00 | 0.00 |
| order_physical_card | 160 | 0.19 | 0.29 | 0.04 | 0.16 |
| top_up_by_bank_transfer_charge | 151 | 0.21 | 0.20 | 0.20 | 0.22 |
| beneficiary_not_allowed | 196 | 0.25 | 0.42 | 0.28 | 0.43 |
| balance_not_updated_after_bank_transfer | 211 | 0.38 | 0.42 | 0.41 | 0.53 |
| topping_up_by_card | 143 | 0.39 | 0.03 | 0.02 | 0.38 |
| transfer_not_received_by_recipient | 211 | 0.49 | 0.56 | 0.64 | 0.60 |
| supported_cards_and_currencies | 169 | 0.54 | 0.31 | 0.36 | 0.30 |
| wrong_exchange_rate_for_cash_withdrawal | 203 | 0.56 | 0.55 | 0.15 | 0.61 |
| direct_debit_payment_not_recognised | 222 | 0.56 | 0.09 | 0.29 | 0.56 |

Data check: 0 identical texts carry more than one gold label; 11 rows repeat an earlier text.

## ATIS: 5634 rows, systems: Jev, Qwen, Gemma, GLM

### A. Consensus disagreement with the gold label (possible label noise / ambiguity)

- All 4 systems predict the SAME label and it differs from gold: **87** rows (1.5%).
- All systems wrong (labels may differ): 111 rows (2.0%).
- Accuracy if the consensus-disagreement rows are excluded: Jev 0.879 (was 0.865), Qwen 0.901 (was 0.887), Gemma 0.940 (was 0.925), GLM 0.949 (was 0.934)
- Sample of consensus-disagreement rows for a manual audit (gold -> consensus prediction):

  - `atis_airfare` -> `atis_flight` : on april first i need a ticket from tacoma to san jose departing before 7 am
  - `atis_flight` -> `atis_airfare` : show me the cheapest first class round trip from new york to miami
  - `atis_flight` -> `atis_airfare` : i want to fly from nashville to seattle and i want the cheapest fare round trip
  - `atis_flight` -> `atis_airfare` : show me the cheapest one way flights from montreal to orlando
  - `atis_flight` -> `atis_airfare` : what's the cheapest one way flight from oakland to boston
  - `atis_flight_time` -> `atis_flight` : show me the schedule for airlines leaving pittsburgh going to san francisco for next monday
  - `atis_flight` -> `atis_airline` : yes i live in washington and i want to make a trip to san francisco which airlines may i use for this trip
  - `atis_flight` -> `atis_airfare` : can you tell me the cheapest flight between boston and san francisco
  - `atis_airline` -> `atis_airfare` : which airline is the cheapest to fly from dallas to baltimore on december twenty fourth
  - `atis_flight` -> `atis_flight_time` : when is the first flight in the morning from boston to denver
  - `atis_aircraft` -> `atis_flight` : i want to go and take a plane in atlanta and fly to boston
  - `atis_flight` -> `atis_airfare` : what is the cheapest way to travel round trip from milwaukee to san francisco

### B. Calibration direction (bins of the top-label probability; + = overconfident)

| system | mean confidence | accuracy | gap | AUROC(conf -> correct) | [0.0,0.1) | [0.1,0.2) | [0.2,0.3) | [0.3,0.4) | [0.4,0.5) | [0.5,0.6) | [0.6,0.7) | [0.7,0.8) | [0.8,0.9) | [0.9,1.0) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Jev | 0.866 | 0.865 | +0.001 | 0.799 | - | - | 2: 0.50 | 4: 0.25 | 52: 0.48 | 507: 0.53 | 455: 0.73 | 507: 0.80 | 929: 0.84 | 3407: 0.95 |
| Qwen | 0.923 | 0.887 | +0.036 | 0.707 | - | 1: 0.00 | 6: 0.33 | 28: 0.25 | 92: 0.55 | 158: 0.53 | 186: 0.73 | 272: 0.76 | 526: 0.89 | 4365: 0.93 |
| Gemma | 0.995 | 0.925 | +0.070 | 0.588 | - | - | - | - | 9: 0.78 | 8: 0.62 | 15: 0.53 | 14: 0.36 | 37: 0.54 | 5551: 0.93 |
| GLM | n/a | 0.934 | n/a | n/a | | | | | | | | | | |

(bin cells = rows in bin: accuracy in bin)

Jev's own `confidence` field vs top probability as an error detector: AUROC 0.799 vs 0.799.

### C. Paired bootstrap of the accuracy difference (Jev minus system), 95% CI

| system | difference | 95% CI |
|---|---|---|
| Qwen | -0.022 | [-0.029, -0.014] |
| Gemma | -0.060 | [-0.068, -0.051] |
| GLM | -0.069 | [-0.077, -0.061] |

### D. Row difficulty: how many of the systems get the row right

| systems correct | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| rows | 111 (2.0%) | 257 (4.6%) | 252 (4.5%) | 467 (8.3%) | 4547 (80.7%) |

Oracle (at least one system right): 0.980. Majority vote of systems right (>= half): 0.935.

### E. Accuracy by utterance length (words, quartiles)

| length | rows | Jev | Qwen | Gemma | GLM |
|---|---|---|---|---|---|
| <=8 | 1510 | 0.885 | 0.858 | 0.892 | 0.944 |
| (8,11] | 1809 | 0.878 | 0.907 | 0.938 | 0.939 |
| (11,13] | 924 | 0.840 | 0.895 | 0.938 | 0.927 |
| (13,max] | 1391 | 0.844 | 0.888 | 0.935 | 0.923 |

### F. Jev top-k accuracy

| k | accuracy |
|---|---|
| 1 | 0.866 |
| 2 | 0.990 |
| 3 | 0.996 |
| 5 | 0.999 |

### G. Labels where Jev is weakest (per-label accuracy)

| label | rows | Jev | Qwen | Gemma | GLM |
|---|---|---|---|---|---|
| atis_abbreviation | 180 | 0.33 | 0.27 | 0.50 | 0.99 |
| atis_flight_time | 55 | 0.67 | 0.73 | 0.64 | 0.69 |
| atis_flight | 4298 | 0.87 | 0.90 | 0.96 | 0.94 |
| atis_airline | 195 | 0.88 | 0.94 | 0.77 | 0.72 |
| atis_quantity | 54 | 0.91 | 0.24 | 0.22 | 1.00 |
| atis_aircraft | 90 | 0.91 | 0.93 | 0.89 | 0.94 |
| atis_airfare | 471 | 0.94 | 1.00 | 0.94 | 0.96 |
| atis_ground_service | 291 | 1.00 | 0.98 | 0.93 | 1.00 |
