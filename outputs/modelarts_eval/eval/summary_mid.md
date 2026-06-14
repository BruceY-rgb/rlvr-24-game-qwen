# 24-game evaluation summary

Conditions: solver, base, trained

## pass@1

| split | solver | base | trained |
|---|---|---|---|
| dev | 1.000 | 0.000 | 0.022 |
| ood_test | 1.000 | 0.015 | 0.029 |
| hard_test | 1.000 | 0.000 | 0.030 |
| unsolvable | 0.000 | 0.000 | — |
| countdown_ood | 1.000 | — | — |
| countdown_unique | 1.000 | — | — |

## pass@k

| split | solver | base | trained |
|---|---|---|---|
| dev | 1.000 | 0.007 | 0.066 |
| ood_test | 1.000 | 0.037 | 0.074 |
| hard_test | 1.000 | 0.000 | 0.070 |
| unsolvable | 0.000 | 0.000 | — |
| countdown_ood | 1.000 | — | — |
| countdown_unique | 1.000 | — | — |

## legal_rate

| split | solver | base | trained |
|---|---|---|---|
| dev | 1.000 | 0.051 | 0.581 |
| ood_test | 1.000 | 0.059 | 0.581 |
| hard_test | 1.000 | 0.050 | 0.590 |
| unsolvable | 0.000 | 0.050 | — |
| countdown_ood | 1.000 | — | — |
| countdown_unique | 1.000 | — | — |

## format_rate

| split | solver | base | trained |
|---|---|---|---|
| dev | 1.000 | 0.110 | 0.551 |
| ood_test | 1.000 | 0.051 | 0.610 |
| hard_test | 1.000 | 0.050 | 0.650 |
| unsolvable | 1.000 | 0.120 | — |
| countdown_ood | 1.000 | — | — |
| countdown_unique | 1.000 | — | — |

## hallucination_rate

| split | solver | base | trained |
|---|---|---|---|
| dev | 0.000 | 0.000 | 0.000 |
| ood_test | 0.000 | 0.000 | 0.000 |
| hard_test | 0.000 | 0.000 | 0.000 |
| unsolvable | 0.000 | 1.000 | — |
| countdown_ood | 0.000 | — | — |
| countdown_unique | 0.000 | — | — |
