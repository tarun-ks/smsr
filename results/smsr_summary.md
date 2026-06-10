# SMSR Full Evaluation Results
- scenarios: 15  clean tasks: 20

## Attack Results
| Configuration | ASR | neither | utility |
|---|---|---|---|
| unsigned/direct/n=1/none | 93.3% | 6.7% | 90.0% |
| unsigned/direct/n=3/none | 93.3% | 6.7% | 0.0% |
| unsigned/flooding/n=1/none | 93.3% | 6.7% | 0.0% |
| unsigned/flooding/n=3/none | 100.0% | 0.0% | 0.0% |
| unsigned/subtle/n=1/none | 100.0% | 0.0% | 0.0% |
| unsigned/subtle/n=3/none | 100.0% | 0.0% | 0.0% |
| unsigned/direct/n=1/heuristic | 86.7% | 13.3% | 90.0% |
| unsigned/direct/n=3/heuristic | 86.7% | 13.3% | 0.0% |
| unsigned/flooding/n=1/heuristic | 86.7% | 13.3% | 0.0% |
| unsigned/flooding/n=3/heuristic | 100.0% | 0.0% | 0.0% |
| unsigned/subtle/n=1/heuristic | 100.0% | 0.0% | 0.0% |
| unsigned/subtle/n=3/heuristic | 100.0% | 0.0% | 0.0% |
| unsigned/direct/n=1/c1 | 0.0% | 100.0% | 90.0% |
| unsigned/direct/n=3/c1 | 0.0% | 100.0% | 0.0% |
| unsigned/flooding/n=1/c1 | 0.0% | 93.3% | 0.0% |
| unsigned/flooding/n=3/c1 | 0.0% | 100.0% | 0.0% |
| unsigned/subtle/n=1/c1 | 0.0% | 93.3% | 0.0% |
| unsigned/subtle/n=3/c1 | 0.0% | 100.0% | 0.0% |
| unsigned/direct/n=1/c1c2 | 0.0% | 100.0% | 85.0% |
| unsigned/direct/n=3/c1c2 | 0.0% | 100.0% | 0.0% |
| unsigned/flooding/n=1/c1c2 | 0.0% | 100.0% | 0.0% |
| unsigned/flooding/n=3/c1c2 | 0.0% | 100.0% | 0.0% |
| unsigned/subtle/n=1/c1c2 | 0.0% | 93.3% | 0.0% |
| unsigned/subtle/n=3/c1c2 | 0.0% | 100.0% | 0.0% |
| authenticated/direct/n=1/none | 93.3% | 6.7% | 0.0% |
| authenticated/direct/n=3/none | 93.3% | 6.7% | 0.0% |
| authenticated/flooding/n=1/none | 100.0% | 0.0% | 0.0% |
| authenticated/flooding/n=3/none | 100.0% | 0.0% | 0.0% |
| authenticated/direct/n=1/c1 | 100.0% | 0.0% | 0.0% |
| authenticated/direct/n=3/c1 | 93.3% | 6.7% | 0.0% |
| authenticated/flooding/n=1/c1 | 93.3% | 6.7% | 0.0% |
| authenticated/flooding/n=3/c1 | 100.0% | 0.0% | 0.0% |
| authenticated/direct/n=1/c1c2 | 93.3% | 6.7% | 0.0% |
| authenticated/direct/n=3/c1c2 | 93.3% | 6.7% | 0.0% |
| authenticated/flooding/n=1/c1c2 | 46.7% | 46.7% | 0.0% |
| authenticated/flooding/n=3/c1c2 | 93.3% | 6.7% | 0.0% |
| bypass/subtle/n=3/none | 100.0% | 0.0% | 0.0% |
| bypass/subtle/n=3/heuristic | 100.0% | 0.0% | 0.0% |
| bypass/subtle/n=3/c1 | 0.0% | 100.0% | 0.0% |

## Formal Certificates (c1c2, r=1)
| t_adv | k | m | n_runs | p_clean/run | delta |
|---|---|---|---|---|---|
| 1 | 5 | 20 | 7 | 0.7500 | 0.0706 |
| 2 | 5 | 20 | 7 | 0.5526 | 0.3861 |
| 3 | 5 | 20 | 7 | 0.3991 | 0.7119 |
| 5 | 5 | 20 | 7 | 0.1937 | 0.9701 |
| 8 | 5 | 20 | 7 | 0.0511 | 0.9998 |
| 10 | 5 | 20 | 7 | 0.0163 | 1.0000 |
| 15 | 5 | 20 | 7 | 0.0001 | 1.0000 |
