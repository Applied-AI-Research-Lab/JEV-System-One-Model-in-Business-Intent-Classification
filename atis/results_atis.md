# ATIS results

| system | variant | order | rows | n | acc [95% CI] | majority baseline | macro-F1 | weighted-F1 | top2 | top3 | ECE | AURC | acc@50% | acc@80% | invalid | p50 lat (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jev-1.13.0 | human | s0 | all | 5634 | 0.869 [0.861, 0.877] | 0.763 | 0.766 | 0.891 | 0.992 | 0.996 | 0.018 | 0.036 | 0.986 | 0.942 | 0.000 | 0.32 |
| jev-1.13.0 | human | s0 | unique | 5251 | 0.871 [0.862, 0.880] | 0.769 | 0.768 | 0.893 | 0.991 | 0.995 | 0.016 | 0.037 | 0.984 | 0.940 | 0.000 | 0.32 |
| jev-1.13.0 | raw | s0 | all | 5634 | 0.865 [0.857, 0.874] | 0.763 | 0.738 | 0.883 | 0.990 | 0.996 | 0.023 | 0.047 | 0.963 | 0.922 | 0.000 | 0.32 |
| jev-1.13.0 | raw | s0 | unique | 5251 | 0.867 [0.858, 0.876] | 0.769 | 0.740 | 0.886 | 0.990 | 0.995 | 0.021 | 0.046 | 0.965 | 0.925 | 0.000 | 0.32 |

### jev-1.13.0 / human / s0 (all rows)

| label | precision | recall | F1 | support |
|---|---|---|---|---|
| atis_flight | 0.985 | 0.867 | 0.923 | 4298 |
| atis_airfare | 0.614 | 0.941 | 0.743 | 471 |
| atis_ground_service | 0.945 | 1.000 | 0.972 | 291 |
| atis_airline | 0.899 | 0.821 | 0.858 | 195 |
| atis_abbreviation | 0.866 | 0.539 | 0.664 | 180 |
| atis_aircraft | 0.802 | 0.944 | 0.867 | 90 |
| atis_flight_time | 0.110 | 0.727 | 0.190 | 55 |
| atis_quantity | 0.867 | 0.963 | 0.912 | 54 |

Top confusions (gold → predicted): atis_flight → atis_flight_time (320); atis_flight → atis_airfare (195); atis_abbreviation → atis_airfare (79); atis_airfare → atis_flight (25); atis_flight → atis_aircraft (18); atis_flight → atis_airline (16)

### jev-1.13.0 / raw / s0 (all rows)

| label | precision | recall | F1 | support |
|---|---|---|---|---|
| atis_flight | 0.981 | 0.871 | 0.923 | 4298 |
| atis_airfare | 0.589 | 0.938 | 0.723 | 471 |
| atis_ground_service | 0.893 | 1.000 | 0.943 | 291 |
| atis_airline | 0.851 | 0.882 | 0.866 | 195 |
| atis_abbreviation | 0.984 | 0.333 | 0.498 | 180 |
| atis_aircraft | 0.845 | 0.911 | 0.877 | 90 |
| atis_flight_time | 0.114 | 0.673 | 0.195 | 55 |
| atis_quantity | 0.845 | 0.907 | 0.875 | 54 |

Top confusions (gold → predicted): atis_flight → atis_flight_time (283); atis_flight → atis_airfare (212); atis_abbreviation → atis_airfare (92); atis_airfare → atis_flight (26); atis_flight → atis_ground_service (24); atis_flight → atis_airline (17)
