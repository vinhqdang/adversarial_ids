| Corpus | Target | N | FFE-v1 | Control | CAPGD | FFE-v2 score-only | FFE-v2 + snapping |
|---|---|---|---|---|---|---|---|
| CSE-CIC-IDS2018 | xgboost | 1000 | 99.5 | 99.5 | 88.5 | 99.5 | 99.5 |
| CSE-CIC-IDS2018 | lightgbm | 1000 | 99.5 | 99.5 | n/a | 99.5 | 99.5 |
| CSE-CIC-IDS2018 | catboost | 1000 | 59.3 | 99.5 | n/a | 99.5 | 99.5 |
| CIC-IDS2017 | xgboost | 1000 | 99.2 | 100.0 | 89.5 | 100.0 | 100.0 |
| CIC-IDS2017 | lightgbm | 1000 | 35.1 | 36.6 | n/a | 96.0 | 96.0 |
| CIC-IDS2017 | catboost | 997 | 34.8 | 36.4 | n/a | 36.4 | 36.4 |
| UNSW-NB15 | xgboost | 1000 | 57.6 | 90.2 | 21.1 | 98.3 | 98.5 |
| UNSW-NB15 | lightgbm | 1000 | 62.4 | 88.1 | n/a | 96.7 | 99.2 |
| UNSW-NB15 | catboost | 983 | 70.1 | 56.8 | n/a | 97.5 | 97.7 |

Queries (mean per flow) and median operational cost of the successful flows, and median cost on the flows both methods evade (paired):

| Corpus | Target | q v1 | q control | q v2 | q v2+snap | med cost v1 | med cost control | med cost v2 | paired med v2 / v1 (n) | paired med v2 / control (n) |
|---|---|---|---|---|---|---|---|---|---|---|
| CSE-CIC-IDS2018 | xgboost | 16.5 | 2.0 | 30.3 | 30.8 | 0.00057 | 0.000495 | 3.02e-05 | 3.02e-05 / 0.00057 (995) | 3.02e-05 / 0.000495 (995) |
| CSE-CIC-IDS2018 | lightgbm | 20.2 | 2.0 | 31.1 | 31.6 | 0.000977 | 0.000496 | 2.42e-05 | 2.42e-05 / 0.000977 (995) | 2.42e-05 / 0.000496 (995) |
| CSE-CIC-IDS2018 | catboost | 52.5 | 2.0 | 29.1 | 29.2 | 0.000977 | 0.000285 | 0.000285 | 0.000121 / 0.000977 (593) | 0.000285 / 0.000285 (995) |
| CIC-IDS2017 | xgboost | 14.6 | 2.0 | 28.3 | 28.3 | 0.000855 | 0.000376 | 0.000554 | 0.000554 / 0.000855 (992) | 0.000554 / 0.000376 (1000) |
| CIC-IDS2017 | lightgbm | 120.1 | 2.0 | 32.4 | 36.5 | 1.62 | 1.56e-05 | 0.000554 | 1.17e-05 / 1.62 (351) | 1.17e-05 / 1.56e-05 (366) |
| CIC-IDS2017 | catboost | 76.3 | 2.0 | 109.0 | 140.4 | 0.5 | 0.00671 | 3.22e-06 | 3.22e-06 / 0.5 (347) | 3.22e-06 / 0.00671 (363) |
| UNSW-NB15 | xgboost | 93.7 | 2.0 | 36.2 | 37.9 | 5.61 | 213 | 40 | 6.72 / 5.77 (571) | 32.5 / 213 (894) |
| UNSW-NB15 | lightgbm | 89.0 | 2.0 | 38.3 | 41.5 | 7.49 | 49 | 50.4 | 14.6 / 7.49 (618) | 35.8 / 49 (860) |
| UNSW-NB15 | catboost | 67.9 | 2.0 | 38.0 | 42.3 | 9.8 | 10.5 | 56 | 29.1 / 9.78 (688) | 22.1 / 10.5 (558) |

Exact McNemar (Real ASR, same flows), FFE-v2 score-only versus each method: v2-only / other-only / p.

| Corpus | Target | vs v1 | vs control | vs CAPGD | vs v2+snap |
|---|---|---|---|---|---|
| CSE-CIC-IDS2018 | xgboost | 0/0/1 | 0/0/1 | 110/0/1.5e-33 | 0/0/1 |
| CSE-CIC-IDS2018 | lightgbm | 0/0/1 | 0/0/1 | n/a | 0/0/1 |
| CSE-CIC-IDS2018 | catboost | 402/0/1.9e-121 | 0/0/1 | n/a | 0/0/1 |
| CIC-IDS2017 | xgboost | 8/0/0.0078 | 0/0/1 | 105/0/4.9e-32 | 0/0/1 |
| CIC-IDS2017 | lightgbm | 609/0/9.4e-184 | 594/0/3.1e-179 | n/a | 0/0/1 |
| CIC-IDS2017 | catboost | 16/0/3.1e-05 | 0/0/1 | n/a | 0/0/1 |
| UNSW-NB15 | xgboost | 412/5/6.1e-115 | 89/8/2e-18 | 772/0/8.1e-233 | 0/2/0.5 |
| UNSW-NB15 | lightgbm | 349/6/7.4e-95 | 107/21/4.5e-15 | n/a | 0/25/6e-08 |
| UNSW-NB15 | catboost | 270/1/1.4e-79 | 400/0/7.7e-121 | n/a | 0/2/0.5 |

Clean accuracy: CSE-CIC-IDS2018 97.30/97.28/97.26; CIC-IDS2017 99.96/99.96/99.85; UNSW-NB15 88.51/88.47/86.67 (xgboost/lightgbm/catboost).

Generator usage of FFE-v2 successful flows (share of flows using each; score-only):

| Corpus | Target | time | pad | chaff | seg | win |
|---|---|---|---|---|---|---|
| CSE-CIC-IDS2018 | xgboost | 31% | 0% | 0% | 0% | 69% |
| CSE-CIC-IDS2018 | lightgbm | 49% | 0% | 2% | 0% | 49% |
| CSE-CIC-IDS2018 | catboost | 2% | 0% | 0% | 0% | 98% |
| CIC-IDS2017 | xgboost | 12% | 0% | 0% | 0% | 87% |
| CIC-IDS2017 | lightgbm | 14% | 1% | 0% | 1% | 87% |
| CIC-IDS2017 | catboost | 79% | 0% | 6% | 0% | 15% |
| UNSW-NB15 | xgboost | 14% | 13% | 37% | 39% | 0% |
| UNSW-NB15 | lightgbm | 12% | 14% | 64% | 16% | 0% |
| UNSW-NB15 | catboost | 11% | 7% | 61% | 24% | 0% |

Per-family Real ASR of FFE-v2 (score-only):

* CSE-CIC-IDS2018 / xgboost: Botnet 100.0 (n=35), BruteForce 100.0 (n=200), DDoS 98.7 (n=375), DoS 100.0 (n=390)
* CSE-CIC-IDS2018 / lightgbm: Botnet 100.0 (n=35), BruteForce 100.0 (n=200), DDoS 98.7 (n=375), DoS 100.0 (n=390)
* CSE-CIC-IDS2018 / catboost: Botnet 100.0 (n=35), BruteForce 100.0 (n=200), DDoS 98.7 (n=375), DoS 100.0 (n=390)
* CIC-IDS2017 / xgboost: Botnet 100.0 (n=1), BruteForce 100.0 (n=20), DDoS 100.0 (n=235), DoS 100.0 (n=422), PortScan 100.0 (n=322)
* CIC-IDS2017 / lightgbm: Botnet 100.0 (n=1), BruteForce 100.0 (n=20), DDoS 100.0 (n=235), DoS 90.5 (n=422), PortScan 100.0 (n=322)
* CIC-IDS2017 / catboost: Botnet 100.0 (n=1), BruteForce 100.0 (n=20), DDoS 0.0 (n=235), DoS 5.0 (n=420), PortScan 100.0 (n=321)
* UNSW-NB15 / xgboost: Analysis 100.0 (n=1), Backdoor 100.0 (n=6), DoS 93.8 (n=80), Exploits 99.8 (n=504), Fuzzers 95.9 (n=219), Generic 90.9 (n=22), Reconnaissance 100.0 (n=143), Shellcode 100.0 (n=19), Worms 100.0 (n=6)
* UNSW-NB15 / lightgbm: Analysis 100.0 (n=1), Backdoor 100.0 (n=6), DoS 95.0 (n=80), Exploits 98.0 (n=504), Fuzzers 94.1 (n=219), Generic 100.0 (n=22), Reconnaissance 96.5 (n=143), Shellcode 100.0 (n=19), Worms 83.3 (n=6)
* UNSW-NB15 / catboost: Analysis 100.0 (n=1), Backdoor 100.0 (n=6), DoS 97.4 (n=77), Exploits 97.2 (n=497), Fuzzers 98.1 (n=213), Generic 100.0 (n=21), Reconnaissance 96.5 (n=143), Shellcode 100.0 (n=19), Worms 100.0 (n=6)