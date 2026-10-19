# Multi-seed evaluation of the frozen FFE-v2

Each seed retrains the targets on a new stratified split and redraws the cohort (up to 300 flows detected by XGBoost; LightGBM / CatBoost are attacked on the part of that cohort they also detect). Config frozen (results/ffev2_config.json, score-only). No tuning on these runs. Values are mean +- std (sample, ddof=1) over seeds.

## UNSW-NB15

### xgboost (seeds [1, 2, 3, 4, 5], n per seed [300, 300, 300, 300, 300])

| Attack | Real ASR % | PFR % | Queries | Median op-cost | Real ASR per seed |
|---|---|---|---|---|---|
| FFE-v2 | 97.7 +- 2.5 | 100.0 +- 0.0 | 38.4 +- 5.1 | 39.8720 +- 9.0158 | 98.3, 100.0, 94.3, 100.0, 96.0 |
| FFE-v1 | 58.3 +- 4.3 | 100.0 +- 0.0 | 93.7 +- 8.6 | 6.0380 +- 0.9238 | 60.7, 52.0, 56.7, 63.3, 58.7 |
| White-box control | 86.0 +- 8.0 | 100.0 +- 0.0 | 2.0 +- 0.0 | 93.0660 +- 68.1745 | 88.7, 81.7, 74.3, 94.7, 90.7 |
| CAPGD | 31.5 +- 2.8 | 100.0 +- 0.0 | 14.5 +- 0.3 | 30.2382 +- 3.2298 | 29.0, 35.7, 29.7, 30.0, 33.0 |

| FFE-v2 vs | seeds win/tie/loss | per-seed v2-only/other-only | per-seed exact p | pooled v2-only / other-only | pooled exact p |
|---|---|---|---|---|---|
| FFE-v1 | 5/0/0 | 114/1; 144/0; 120/7; 110/0; 114/2 | 5.6e-33; 9e-44; 1.1e-27; 1.5e-33; 1.6e-31 | 602 / 10 | 2.26e-163 |
| White-box control | 5/0/0 | 29/0; 55/0; 68/8; 16/0; 20/4 | 3.7e-09; 5.6e-17; 5.6e-13; 3.1e-05; 0.0015 | 188 / 12 | 8.11e-42 |
| CAPGD | 5/0/0 | 211/3; 193/0; 194/0; 210/0; 190/1 | 1.2e-58; 1.6e-58; 8e-59; 1.2e-63; 1.2e-55 | 998 / 4 | 1.96e-291 |

### lightgbm (seeds [1, 2, 3], n per seed [297, 297, 298])

| Attack | Real ASR % | PFR % | Queries | Median op-cost | Real ASR per seed |
|---|---|---|---|---|---|
| FFE-v2 | 98.9 +- 1.0 | 100.0 +- 0.0 | 37.0 +- 1.3 | 40.6400 +- 8.1061 | 98.7, 100.0, 98.0 |
| FFE-v1 | 57.2 +- 5.5 | 100.0 +- 0.0 | 94.8 +- 10.1 | 5.5634 +- 1.4653 | 60.9, 50.8, 59.7 |
| White-box control | 90.9 +- 8.2 | 100.0 +- 0.0 | 2.0 +- 0.0 | 123.3800 +- 56.2673 | 96.6, 81.5, 94.6 |
| CAPGD | 30.4 +- 4.9 | 100.0 +- 0.0 | 14.8 +- 0.6 | 29.8400 +- 3.4814 | 30.3, 35.4, 25.5 |

| FFE-v2 vs | seeds win/tie/loss | per-seed v2-only/other-only | per-seed exact p | pooled v2-only / other-only | pooled exact p |
|---|---|---|---|---|---|
| FFE-v1 | 3/0/0 | 113/1; 146/0; 114/0 | 1.1e-32; 2.2e-44; 9.6e-35 | 373 / 1 | 1.95e-110 |
| White-box control | 3/0/0 | 7/1; 55/0; 14/4 | 0.07; 5.6e-17; 0.031 | 76 / 5 | 2.26e-17 |
| CAPGD | 3/0/0 | 204/1; 192/0; 216/0 | 8e-60; 3.2e-58; 1.9e-65 | 612 / 1 | 3.61e-182 |

### catboost (seeds [1, 2, 3], n per seed [294, 300, 288])

| Attack | Real ASR % | PFR % | Queries | Median op-cost | Real ASR per seed |
|---|---|---|---|---|---|
| FFE-v2 | 99.3 +- 0.3 | 100.0 +- 0.0 | 37.2 +- 0.7 | 47.7867 +- 13.6866 | 99.0, 99.3, 99.7 |
| FFE-v1 | 72.1 +- 0.9 | 100.0 +- 0.0 | 68.6 +- 1.1 | 18.9434 +- 15.9547 | 72.1, 73.0, 71.2 |
| White-box control | 74.4 +- 5.6 | 100.0 +- 0.0 | 2.0 +- 0.0 | 54.1933 +- 40.7116 | 78.6, 68.0, 76.7 |
| CAPGD | 30.1 +- 6.2 | 100.0 +- 0.0 | 14.7 +- 0.9 | 29.8879 +- 3.5643 | 30.6, 36.0, 23.6 |

| FFE-v2 vs | seeds win/tie/loss | per-seed v2-only/other-only | per-seed exact p | pooled v2-only / other-only | pooled exact p |
|---|---|---|---|---|---|
| FFE-v1 | 3/0/0 | 80/1; 81/2; 82/0 | 6.8e-23; 7.2e-22; 4.1e-25 | 243 / 3 | 4.39e-68 |
| White-box control | 3/0/0 | 61/1; 94/0; 66/0 | 2.7e-17; 1e-28; 2.7e-20 | 221 / 1 | 6.62e-65 |
| CAPGD | 3/0/0 | 202/1; 190/0; 219/0 | 3.2e-59; 1.3e-57; 2.4e-66 | 611 / 1 | 7.21e-182 |

## CIC-IDS2017

### xgboost (seeds [1, 2, 3, 4, 5], n per seed [300, 300, 300, 300, 300])

| Attack | Real ASR % | PFR % | Queries | Median op-cost | Real ASR per seed |
|---|---|---|---|---|---|
| FFE-v2 | 100.0 +- 0.0 | 100.0 +- 0.0 | 29.1 +- 1.0 | 0.0006 +- 0.0000 | 100.0, 100.0, 100.0, 100.0, 100.0 |
| FFE-v1 | 97.3 +- 1.5 | 100.0 +- 0.0 | 17.0 +- 3.3 | 0.0008 +- 0.0001 | 97.7, 98.7, 94.7, 98.0, 97.7 |
| White-box control | 100.0 +- 0.0 | 100.0 +- 0.0 | 2.0 +- 0.0 | 0.0003 +- 0.0001 | 100.0, 100.0, 100.0, 100.0, 100.0 |
| CAPGD | 82.4 +- 11.9 | 100.0 +- 0.0 | 4.4 +- 2.3 | 15.4730 +- 5.0819 | 64.7, 83.0, 80.0, 87.0, 97.3 |

| FFE-v2 vs | seeds win/tie/loss | per-seed v2-only/other-only | per-seed exact p | pooled v2-only / other-only | pooled exact p |
|---|---|---|---|---|---|
| FFE-v1 | 5/0/0 | 7/0; 4/0; 16/0; 6/0; 7/0 | 0.016; 0.12; 3.1e-05; 0.031; 0.016 | 40 / 0 | 1.82e-12 |
| White-box control | 0/5/0 | 0/0; 0/0; 0/0; 0/0; 0/0 | 1; 1; 1; 1; 1 | 0 / 0 | 1 |
| CAPGD | 5/0/0 | 106/0; 51/0; 60/0; 39/0; 8/0 | 2.5e-32; 8.9e-16; 1.7e-18; 3.6e-12; 0.0078 | 264 / 0 | 6.75e-80 |

### lightgbm (seeds [1, 2, 3], n per seed [300, 300, 300])

| Attack | Real ASR % | PFR % | Queries | Median op-cost | Real ASR per seed |
|---|---|---|---|---|---|
| FFE-v2 | 95.6 +- 0.2 | 100.0 +- 0.0 | 30.3 +- 1.0 | 0.0006 +- 0.0000 | 95.7, 95.3, 95.7 |
| FFE-v1 | 36.1 +- 4.2 | 100.0 +- 0.0 | 120.3 +- 4.3 | 0.3340 +- 0.2884 | 32.3, 40.7, 35.3 |
| White-box control | 77.4 +- 31.6 | 100.0 +- 0.0 | 2.0 +- 0.0 | 0.0004 +- 0.0003 | 95.7, 41.0, 95.7 |
| CAPGD | 50.0 +- 2.4 | 100.0 +- 0.0 | 10.6 +- 0.5 | 6.6670 +- 4.6193 | 52.7, 49.3, 48.0 |

| FFE-v2 vs | seeds win/tie/loss | per-seed v2-only/other-only | per-seed exact p | pooled v2-only / other-only | pooled exact p |
|---|---|---|---|---|---|
| FFE-v1 | 3/0/0 | 190/0; 164/0; 181/0 | 1.3e-57; 8.6e-50; 6.5e-55 | 535 / 0 | 1.78e-161 |
| White-box control | 1/2/0 | 0/0; 163/0; 0/0 | 1; 1.7e-49; 1 | 163 / 0 | 1.71e-49 |
| CAPGD | 3/0/0 | 129/0; 138/0; 143/0 | 2.9e-39; 5.7e-42; 1.8e-43 | 410 / 0 | 7.56e-124 |

### catboost (seeds [1, 2, 3], n per seed [300, 298, 297])

| Attack | Real ASR % | PFR % | Queries | Median op-cost | Real ASR per seed |
|---|---|---|---|---|---|
| FFE-v2 | 36.7 +- 3.5 | 100.0 +- 0.0 | 122.1 +- 12.4 | 0.0000 +- 0.0000 | 34.0, 40.6, 35.4 |
| FFE-v1 | 35.1 +- 3.1 | 100.0 +- 0.0 | 74.9 +- 7.0 | 0.5006 +- 0.0006 | 32.7, 38.6, 34.0 |
| White-box control | 36.5 +- 3.3 | 100.0 +- 0.0 | 2.0 +- 0.0 | 0.4397 +- 0.7537 | 34.0, 40.3, 35.4 |
| CAPGD | 23.0 +- 2.8 | 100.0 +- 0.0 | 15.7 +- 0.5 | 1.0267 +- 1.7094 | 21.3, 21.5, 26.3 |

| FFE-v2 vs | seeds win/tie/loss | per-seed v2-only/other-only | per-seed exact p | pooled v2-only / other-only | pooled exact p |
|---|---|---|---|---|---|
| FFE-v1 | 3/0/0 | 4/0; 6/0; 4/0 | 0.12; 0.031; 0.12 | 14 / 0 | 0.000122 |
| White-box control | 1/2/0 | 0/0; 1/0; 0/0 | 1; 1; 1 | 1 / 0 | 1 |
| CAPGD | 3/0/0 | 38/0; 57/0; 27/0 | 7.3e-12; 1.4e-17; 1.5e-08 | 122 / 0 | 3.76e-37 |

## CSE-CIC-IDS2018

### xgboost (seeds [1, 2, 3, 4, 5], n per seed [300, 300, 300, 300, 300])

| Attack | Real ASR % | PFR % | Queries | Median op-cost | Real ASR per seed |
|---|---|---|---|---|---|
| FFE-v2 | 99.7 +- 0.3 | 100.0 +- 0.0 | 31.1 +- 1.0 | 0.0001 +- 0.0001 | 99.7, 100.0, 99.3, 100.0, 99.7 |
| FFE-v1 | 99.7 +- 0.3 | 100.0 +- 0.0 | 18.3 +- 2.7 | 0.0010 +- 0.0000 | 99.7, 100.0, 99.3, 100.0, 99.7 |
| White-box control | 99.7 +- 0.3 | 100.0 +- 0.0 | 2.0 +- 0.0 | 0.0007 +- 0.0002 | 99.7, 100.0, 99.3, 100.0, 99.7 |
| CAPGD | 97.7 +- 2.4 | 100.0 +- 0.0 | 1.5 +- 0.4 | 32.0663 +- 15.4680 | 97.7, 99.7, 99.3, 93.7, 98.3 |

| FFE-v2 vs | seeds win/tie/loss | per-seed v2-only/other-only | per-seed exact p | pooled v2-only / other-only | pooled exact p |
|---|---|---|---|---|---|
| FFE-v1 | 0/5/0 | 0/0; 0/0; 0/0; 0/0; 0/0 | 1; 1; 1; 1; 1 | 0 / 0 | 1 |
| White-box control | 0/5/0 | 0/0; 0/0; 0/0; 0/0; 0/0 | 1; 1; 1; 1; 1 | 0 / 0 | 1 |
| CAPGD | 4/1/0 | 6/0; 1/0; 0/0; 19/0; 4/0 | 0.031; 1; 1; 3.8e-06; 0.12 | 30 / 0 | 1.86e-09 |

### lightgbm (seeds [1, 2, 3], n per seed [299, 300, 299])

| Attack | Real ASR % | PFR % | Queries | Median op-cost | Real ASR per seed |
|---|---|---|---|---|---|
| FFE-v2 | 99.7 +- 0.3 | 100.0 +- 0.0 | 31.2 +- 0.1 | 0.0000 +- 0.0000 | 99.7, 100.0, 99.3 |
| FFE-v1 | 99.7 +- 0.3 | 100.0 +- 0.0 | 22.2 +- 2.1 | 0.0010 +- 0.0000 | 99.7, 100.0, 99.3 |
| White-box control | 99.7 +- 0.3 | 100.0 +- 0.0 | 2.0 +- 0.0 | 0.0008 +- 0.0002 | 99.7, 100.0, 99.3 |
| CAPGD | 98.9 +- 1.1 | 100.0 +- 0.0 | 1.2 +- 0.2 | 29.1840 +- 13.0693 | 97.7, 99.7, 99.3 |

| FFE-v2 vs | seeds win/tie/loss | per-seed v2-only/other-only | per-seed exact p | pooled v2-only / other-only | pooled exact p |
|---|---|---|---|---|---|
| FFE-v1 | 0/3/0 | 0/0; 0/0; 0/0 | 1; 1; 1 | 0 / 0 | 1 |
| White-box control | 0/3/0 | 0/0; 0/0; 0/0 | 1; 1; 1 | 0 / 0 | 1 |
| CAPGD | 2/1/0 | 6/0; 1/0; 0/0 | 0.031; 1; 1 | 7 / 0 | 0.0156 |

### catboost (seeds [1, 2, 3], n per seed [299, 300, 298])

| Attack | Real ASR % | PFR % | Queries | Median op-cost | Real ASR per seed |
|---|---|---|---|---|---|
| FFE-v2 | 99.7 +- 0.3 | 100.0 +- 0.0 | 29.2 +- 0.1 | 0.0003 +- 0.0000 | 99.7, 100.0, 99.3 |
| FFE-v1 | 56.4 +- 4.3 | 100.0 +- 0.0 | 66.5 +- 8.2 | 0.1676 +- 0.2887 | 54.2, 53.7, 61.4 |
| White-box control | 99.7 +- 0.3 | 100.0 +- 0.0 | 2.0 +- 0.0 | 0.0004 +- 0.0000 | 99.7, 100.0, 99.3 |
| CAPGD | 94.4 +- 2.5 | 100.0 +- 0.0 | 2.2 +- 0.3 | 32.3884 +- 10.6314 | 97.0, 92.0, 94.3 |

| FFE-v2 vs | seeds win/tie/loss | per-seed v2-only/other-only | per-seed exact p | pooled v2-only / other-only | pooled exact p |
|---|---|---|---|---|---|
| FFE-v1 | 3/0/0 | 136/0; 139/0; 113/0 | 2.3e-41; 2.9e-42; 1.9e-34 | 388 / 0 | 3.17e-117 |
| White-box control | 0/3/0 | 0/0; 0/0; 0/0 | 1; 1; 1 | 0 / 0 | 1 |
| CAPGD | 3/0/0 | 8/0; 24/0; 15/0 | 0.0078; 1.2e-07; 6.1e-05 | 47 / 0 | 1.42e-14 |

