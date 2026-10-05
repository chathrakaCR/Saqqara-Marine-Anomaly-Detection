# Stage 2 results: SegFormer-B0 with expanded training data

Mrigank Dube, 29 Sept 2026

## What was tested

Belasith's Stage 2 notebook was retrained with the new CVAT annotations added to the training set. Everything else was kept the same: 30 epochs, the same LR, 437 negatives, crop augmentation, and thresholds.

- Training data: 38 prawn + 63 tuna (baseline: 15 prawn + 16 tuna), plus 437 negatives
- Test set: unchanged. It is the 10 prawn + 10 tuna images that two people annotated (the t2 JSONs).
- 10 new images come from the same scenes as test images. They were removed from training so the test images stay unseen.
- To get a like-for-like comparison, the unchanged baseline was also re-run on the same machine (laptop GPU).

Notebook: `Stage_02/Stage_02_Pipeline_EXPANDED_clean.ipynb`. Set `USE_NEW_DATA = True` for the expanded run and `False` for the baseline.

## Results

| | Original baseline (Belasith) | Baseline re-run (same machine) | Expanded data |
|---|---|---|---|
| Prawn IoU (10 test images) | 47.8% | 51.8% | 52.4% |
| Tuna IoU (10 test images) | 78.4% | 79.4% | 74.8% |
| Unseen prawn masked | 39/57 | 41/57 | 38/47 |
| Unseen tuna masked (of flagged) | 154/196 | 167/191 | 140/187 |
| Plain sea images wrongly masked | 2/50 | 10/50 | 5/50 |

Human agreement ceiling: prawn 58.7%, tuna 72.0%.

Raw outputs are in `results/expanded_run/` and `results/baseline_rerun/`.

## What it shows

- Prawn: no clear change against the same-machine baseline (+0.6).
- Tuna: about 4 points lower with the new data, but still above the human ceiling. Possible reasons are differences in how the tuna boils were outlined between annotators, and many of the added tuna images being augmented copies (flipped/noisy/transformed).
- The same baseline code run twice varied by about 4 IoU points on prawn and 2 vs 10 wrongly masked sea images. With only 10 test images per species, one run per setup can't show small improvements.
- In the expanded run, the two prawn test images the original baseline missed completely (WA0002 and 20260307-WA0005, both 0% IoU) scored 24% and 52%. The same-machine re-run wasn't checked per image, so this could also be run-to-run variation.
- The unseen-pool numbers come from different random samples in each run, so they aren't a direct comparison.

## Known issues

- The negative controls in the connected-pipeline test are sampled from the same 437 negatives used in training, so the false alarm rate is likely optimistic in every run.
- The best checkpoint is picked using the test set, which gives a small optimistic bias. It applies to all runs equally.

## Next steps

1. Run each setup with 3 random seeds and compare the averages.
2. Grow the test set (more double-annotated images) so the comparison is less noisy.
3. Check tuna annotation consistency against the baseline annotations.


## Update, 5 Oct 2026: re-annotated run (v2)

The same 80 images were annotated again by hand in a new CVAT project (`Stage2_anomaly_v2`) and the notebook was re-run. `prawn_aug_012` was left out because its source photo (WA0014) is a byte-identical copy of test image WA0047. The same 10 images were excluded by the leakage guard. Training: 37 prawn + 63 tuna + 437 negatives.

| | Baseline re-run | Expanded v1 | Expanded v2 |
|---|---|---|---|
| Prawn IoU | 51.8% | 52.4% | 46.1% |
| Tuna IoU | 79.4% | 74.8% | 76.2% |
| Best checkpoint | epoch 20 | epoch 30 | epoch 10 |
| Unseen prawn masked | 41/57 | 38/47 | 39/47 |
| Unseen tuna masked (of flagged) | 167/191 | 140/187 | 159/185 |
| Plain sea wrongly masked | 10/50 | 5/50 | 3/50 |

- Prawn: the clear images improved (WA0007 84.8%, WA0076 83.1%), but four faint ones dropped (WA0041 22%, WA0042 20%, WA0005 34%, WA0002 0%), which pulls the mean down.
- Removing prawn_aug_012 did not change WA0047 (46.4% vs 46.5% in v1).
- Checkpoints swung again: prawn 46.1 / 37.0 / 44.3% at epochs 10 / 20 / 30.
- Conclusion: no clear improvement from the new data yet. Run-to-run variation (4–9 points) is as large as the effect. Next: 3 seeds per setup and a larger prawn test set.

