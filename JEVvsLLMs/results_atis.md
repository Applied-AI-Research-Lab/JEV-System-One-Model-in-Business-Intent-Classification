# ATIS results

| system | variant | order | rows | n | acc [95% CI] | majority baseline | macro-F1 | weighted-F1 | top2 | top3 | ECE | AURC | acc@50% | acc@80% | invalid | p50 lat (s) | McNemar p vs Jev |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jev-1.13.0 | human | s0 | all | 5634 | 0.869 [0.861, 0.877] | 0.763 | 0.766 | 0.891 | 0.992 | 0.996 | 0.018 | 0.036 | 0.986 | 0.942 | 0.000 | 0.32 | - |
| jev-1.13.0 | human | s0 | unique | 5251 | 0.871 [0.862, 0.880] | 0.769 | 0.768 | 0.893 | 0.991 | 0.995 | 0.016 | 0.037 | 0.984 | 0.940 | 0.000 | 0.32 | - |
| jev-1.13.0 | raw | s0 | all | 5634 | 0.865 [0.857, 0.874] | 0.763 | 0.738 | 0.883 | 0.990 | 0.996 | 0.023 | 0.047 | 0.963 | 0.922 | 0.000 | 0.32 | 0.24 |
| jev-1.13.0 | raw | s0 | unique | 5251 | 0.867 [0.858, 0.876] | 0.769 | 0.740 | 0.886 | 0.990 | 0.995 | 0.021 | 0.046 | 0.965 | 0.925 | 0.000 | 0.32 | 0.33 |
| gemma-3-4b-it | human | s0 | all | 5634 | 0.920 [0.912, 0.927] | 0.763 | 0.766 | 0.922 | nan | nan | 0.077 | 0.040 | 0.965 | 0.965 | 0.000 | 0.28 | 1.9e-26 |
| gemma-3-4b-it | human | s0 | unique | 5251 | 0.917 [0.909, 0.925] | 0.769 | 0.758 | 0.919 | nan | nan | 0.080 | 0.042 | 0.963 | 0.964 | 0.000 | 0.28 | 1.6e-21 |
| gemma-3-4b-it | raw | s0 | all | 5634 | 0.925 [0.918, 0.931] | 0.763 | 0.728 | 0.924 | nan | nan | 0.071 | 0.078 | 0.940 | 0.949 | 0.000 | 0.46 | 4.7e-35 |
| gemma-3-4b-it | raw | s0 | unique | 5251 | 0.928 [0.920, 0.934] | 0.769 | 0.735 | 0.927 | nan | nan | 0.069 | 0.069 | 0.947 | 0.952 | 0.000 | 0.46 | 9e-34 |
| qwen3.5-4b | human | s0 | all | 5634 | 0.916 [0.909, 0.924] | 0.763 | 0.810 | 0.921 | nan | nan | 0.030 | 0.020 | 0.985 | 0.973 | 0.000 | 0.17 | 1.1e-30 |
| qwen3.5-4b | human | s0 | unique | 5251 | 0.916 [0.909, 0.923] | 0.769 | 0.810 | 0.921 | nan | nan | 0.032 | 0.021 | 0.985 | 0.972 | 0.000 | 0.17 | 3e-27 |
| qwen3.5-4b | raw | s0 | all | 5634 | 0.887 [0.879, 0.896] | 0.763 | 0.686 | 0.890 | nan | nan | 0.049 | 0.062 | 0.938 | 0.926 | 0.000 | 0.20 | 4e-05 |
| qwen3.5-4b | raw | s0 | unique | 5251 | 0.890 [0.881, 0.898] | 0.769 | 0.690 | 0.894 | nan | nan | 0.044 | 0.060 | 0.943 | 0.930 | 0.000 | 0.20 | 2.6e-05 |

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

### gemma-3-4b-it / human / s0 (all rows)

| label | precision | recall | F1 | support |
|---|---|---|---|---|
| atis_flight | 0.975 | 0.931 | 0.952 | 4298 |
| atis_airfare | 0.763 | 0.966 | 0.853 | 471 |
| atis_ground_service | 0.976 | 0.979 | 0.978 | 291 |
| atis_airline | 0.491 | 0.815 | 0.613 | 195 |
| atis_abbreviation | 0.887 | 0.917 | 0.902 | 180 |
| atis_aircraft | 0.875 | 0.933 | 0.903 | 90 |
| atis_flight_time | 0.947 | 0.327 | 0.486 | 55 |
| atis_quantity | 0.842 | 0.296 | 0.438 | 54 |

Top confusions (gold → predicted): atis_flight → atis_airline (151); atis_flight → atis_airfare (127); atis_flight_time → atis_flight (36); atis_quantity → atis_flight (29); atis_airline → atis_abbreviation (18); atis_airfare → atis_flight (16)

### gemma-3-4b-it / raw / s0 (all rows)

| label | precision | recall | F1 | support |
|---|---|---|---|---|
| atis_flight | 0.972 | 0.961 | 0.967 | 4298 |
| atis_airfare | 0.762 | 0.938 | 0.841 | 471 |
| atis_ground_service | 0.975 | 0.928 | 0.951 | 291 |
| atis_airline | 0.654 | 0.774 | 0.709 | 195 |
| atis_abbreviation | 0.918 | 0.500 | 0.647 | 180 |
| atis_aircraft | 0.870 | 0.889 | 0.879 | 90 |
| atis_flight_time | 0.493 | 0.636 | 0.556 | 55 |
| atis_quantity | 0.364 | 0.222 | 0.276 | 54 |

Top confusions (gold → predicted): atis_abbreviation → atis_airfare (79); atis_flight → atis_airline (70); atis_flight → atis_airfare (54); atis_airline → atis_flight (32); atis_airfare → atis_flight (29); atis_quantity → atis_flight (28)

### qwen3.5-4b / human / s0 (all rows)

| label | precision | recall | F1 | support |
|---|---|---|---|---|
| atis_flight | 0.990 | 0.920 | 0.954 | 4298 |
| atis_airfare | 0.680 | 0.968 | 0.799 | 471 |
| atis_ground_service | 0.949 | 0.955 | 0.952 | 291 |
| atis_airline | 0.624 | 0.944 | 0.751 | 195 |
| atis_abbreviation | 0.967 | 0.650 | 0.777 | 180 |
| atis_aircraft | 0.711 | 0.956 | 0.815 | 90 |
| atis_flight_time | 0.440 | 0.673 | 0.532 | 55 |
| atis_quantity | 0.891 | 0.907 | 0.899 | 54 |

Top confusions (gold → predicted): atis_flight → atis_airfare (176); atis_flight → atis_airline (85); atis_flight → atis_flight_time (46); atis_abbreviation → atis_airfare (34); atis_flight → atis_aircraft (17); atis_flight_time → atis_flight (16)

### qwen3.5-4b / raw / s0 (all rows)

| label | precision | recall | F1 | support |
|---|---|---|---|---|
| atis_flight | 0.988 | 0.902 | 0.943 | 4298 |
| atis_airfare | 0.597 | 0.996 | 0.746 | 471 |
| atis_ground_service | 0.938 | 0.979 | 0.958 | 291 |
| atis_airline | 0.526 | 0.938 | 0.674 | 195 |
| atis_abbreviation | 0.941 | 0.267 | 0.416 | 180 |
| atis_aircraft | 0.816 | 0.933 | 0.870 | 90 |
| atis_flight_time | 0.377 | 0.727 | 0.497 | 55 |
| atis_quantity | 0.929 | 0.241 | 0.382 | 54 |

Top confusions (gold → predicted): atis_flight → atis_airfare (238); atis_flight → atis_airline (87); atis_abbreviation → atis_airfare (71); atis_flight → atis_flight_time (64); atis_abbreviation → atis_airline (55); atis_quantity → atis_flight (20)
