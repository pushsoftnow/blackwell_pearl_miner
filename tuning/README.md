# Tuning results

Reports of [`tools/tune-wizard.py`](../tools/tune-wizard.py), one JSON file per GPU, CUDA version,
power limit and date (`results/<gpu>-cuda<version>-<watts>w-<date>.json`). Each report lists the
baselines (variants of [`kernels/variants.txt`](../kernels/variants.txt) and ptxas optimisation
levels), the five best candidates of the search after a 60-second confirmation and the best one,
all relative to the default variant (positive = faster).

Submitted results are collected here. A placement that is measurably faster on some GPU is
added to `kernels/variants.txt` with its origin, so the miner's own tuning tries it on every GPU
and keeps it where it is fastest.

To submit a result, use the link the wizard prints, or open an issue with the
"Tuning result" template and paste the JSON report.
