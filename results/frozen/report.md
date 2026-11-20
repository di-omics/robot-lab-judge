# A linear probe on frozen features, on the scene set

Question: is the pass or fail answer in the top-view pixels at all, for a probe fitted on the train split?

A frozen vision tower turns pictures into numbers. A small logistic regression per chore, trained on train only, turns the numbers into PASS or FAIL. Every choice was made on train. Dev was predicted at the end.

## Features

- Vision tower: the SigLIP 2 encoder inside `LiquidAI/LFM2.5-VL-450M-MLX-bf16` (snapshot `ed71acdae0799c98cb450787d049d1343b65cd52`), 12 layers, hidden size 768, weights mlx.core.bfloat16, mlx 0.32.4, mlx-vlm 0.8.0.
- Tokens: vision tower last hidden state after its post layer norm, before the multimodal projector. Pooled in float32.
- Patch grid, "top" picture (640 x 360 resized to 512 x 288): 18 rows x 32 columns = 576 patches.
- Patch grid, each of the 8 "fine" crops (256 x 256 resized to 512 x 512): 32 x 32 = 1024 patches, 8192 per frame.
- Resize filter: bilinear, as the checkpoint's own image processor. Our patches match that processor's to 1.2e-07 on both picture sizes (`smoke.log`).
- Feature cache: `data/features/`, 49 MB in all (pooled numbers only, no per-patch features are kept).
- Train: 1768 scenes in 300 layouts. Dev: 475 scenes in 80 layouts.

## Head

- 1 logistic regression per chore (10 per row), class-balanced loss (weight n / (2 n_class)), L2 penalty on the weights (bias not penalized), loss = sum of weighted log losses + L2 / 2 * |w|^2.
- Inputs standardized with the mean and standard deviation of the rows the head is fitted on.
- L2 picked from [0.01, 0.1, 1.0, 10.0, 100.0] by 5-fold cross-validation on train, folds grouped by layout seed (the same folds for every chore and row), scaler fitted inside each fold. Score: mean over chores of the balanced accuracy of the pooled out-of-fold verdicts. A tie goes to the larger L2. Then refit on all of train.
- Verdict PASS when p >= 0.5. The head sees only the pixel features and the chore name.

## Results on dev

Balanced accuracy is pooled over all dev scenes, as on the board. Ranges are exact 95% (Clopper-Pearson).

| row | condition | L2 | train CV balanced accuracy | dev pass recall | dev fail recall | dev false-pass rate | dev balanced accuracy | mean of per-chore dev balanced accuracy | mean s per scene | dev predictions made |
|---|---|---|---|---|---|---|---|---|---|---|
| `top-logreg` | after | 1.0 | 0.944 | 274 of 283 (94-99%) | 178 of 192 (88-96%) | 14 of 192 (4-12%) | 0.948 | 0.950 | 0.042 | 1 |
| `fine-logreg` | after | 0.01 | 0.960 | 270 of 283 (92-98%) | 182 of 192 (91-97%) | 10 of 192 (3-9%) | 0.951 | 0.948 | 0.461 | 1 |
| `fine-change-logreg` | before_after | 10.0 | 0.889 | 247 of 283 (83-91%) | 175 of 192 (86-95%) | 17 of 192 (5-14%) | 0.892 | 0.889 | 0.935 | 1 |

Dev confusion counts:

| row | true pass judged PASS | true pass judged FAIL | true fail judged FAIL | true fail judged PASS |
|---|---|---|---|---|
| `top-logreg` | 274 | 9 | 178 | 14 |
| `fine-logreg` | 270 | 13 | 182 | 10 |
| `fine-change-logreg` | 247 | 36 | 175 | 17 |

For reference, the best zero-shot judge on the board scored 0.65 on dev (`results/simset_dev/board.md`).

## Train cross-validation by L2

Mean over chores of the out-of-fold balanced accuracy.

| row | L2 0.01 | L2 0.1 | L2 1.0 | L2 10.0 | L2 100.0 | chosen |
|---|---|---|---|---|---|---|
| `top-logreg` | 0.943 | 0.943 | 0.944 | 0.943 | 0.942 | 1.0 |
| `fine-logreg` | 0.960 | 0.959 | 0.958 | 0.957 | 0.953 | 0.01 |
| `fine-change-logreg` | 0.884 | 0.884 | 0.885 | 0.889 | 0.886 | 10.0 |

## Per chore: `top-logreg` (after)

Feature: top picture of the last frame (512 x 288), mean and max over patches, 1536 numbers.

| chore | n pass | n fail | true pass judged PASS | true fail judged FAIL | dev balanced accuracy | dev AUROC | train CV balanced accuracy |
|---|---|---|---|---|---|---|---|
| bottle_to_shelf | 38 | 10 | 37 of 38 (86-100%) | 10 of 10 (69-100%) | 0.987 | 1.000 | 0.895 |
| plate_to_hotel_bottom (plate) | 26 | 18 | 25 of 26 (80-100%) | 17 of 18 (73-100%) | 0.953 | 0.994 | 0.975 |
| plate_to_hotel_top (plate) | 27 | 21 | 27 of 27 (87-100%) | 21 of 21 (84-100%) | 1.000 | 1.000 | 0.968 |
| plate_to_tray (plate) | 20 | 28 | 20 of 20 (83-100%) | 27 of 28 (82-100%) | 0.982 | 0.995 | 0.896 |
| sort_by_cap | 37 | 11 | 37 of 37 (91-100%) | 11 of 11 (72-100%) | 1.000 | 1.000 | 0.985 |
| stack_plates (plate) | 23 | 25 | 20 of 23 (66-97%) | 21 of 25 (64-95%) | 0.855 | 0.948 | 0.916 |
| stack_tip_boxes (plate) | 23 | 25 | 22 of 23 (78-100%) | 22 of 25 (69-97%) | 0.918 | 0.977 | 0.922 |
| tubes_to_rack | 31 | 16 | 31 of 31 (89-100%) | 14 of 16 (62-98%) | 0.938 | 1.000 | 0.993 |
| tubes_to_waste | 39 | 9 | 39 of 39 (91-100%) | 9 of 9 (66-100%) | 1.000 | 1.000 | 0.985 |
| unstack_plate (plate) | 19 | 29 | 16 of 19 (60-97%) | 26 of 29 (73-98%) | 0.869 | 0.969 | 0.907 |
| mean of the 6 plate chores | | | | | 0.930 | 0.980 | 0.931 |
| mean of the 4 other chores | | | | | 0.981 | 1.000 | 0.964 |

## Per chore: `fine-logreg` (after)

Feature: 8 fine crops of the last frame (512 x 512 each), mean and max over all patches, 1536 numbers.

| chore | n pass | n fail | true pass judged PASS | true fail judged FAIL | dev balanced accuracy | dev AUROC | train CV balanced accuracy |
|---|---|---|---|---|---|---|---|
| bottle_to_shelf | 38 | 10 | 35 of 38 (79-98%) | 9 of 10 (55-100%) | 0.911 | 0.982 | 0.927 |
| plate_to_hotel_bottom (plate) | 26 | 18 | 25 of 26 (80-100%) | 17 of 18 (73-100%) | 0.953 | 0.989 | 0.975 |
| plate_to_hotel_top (plate) | 27 | 21 | 27 of 27 (87-100%) | 21 of 21 (84-100%) | 1.000 | 1.000 | 0.980 |
| plate_to_tray (plate) | 20 | 28 | 20 of 20 (83-100%) | 28 of 28 (88-100%) | 1.000 | 1.000 | 0.939 |
| sort_by_cap | 37 | 11 | 37 of 37 (91-100%) | 11 of 11 (72-100%) | 1.000 | 1.000 | 0.985 |
| stack_plates (plate) | 23 | 25 | 21 of 23 (72-99%) | 21 of 25 (64-95%) | 0.877 | 0.962 | 0.916 |
| stack_tip_boxes (plate) | 23 | 25 | 21 of 23 (72-99%) | 23 of 25 (74-99%) | 0.917 | 0.958 | 0.917 |
| tubes_to_rack | 31 | 16 | 31 of 31 (89-100%) | 15 of 16 (70-100%) | 0.969 | 0.938 | 1.000 |
| tubes_to_waste | 39 | 9 | 39 of 39 (91-100%) | 9 of 9 (66-100%) | 1.000 | 1.000 | 1.000 |
| unstack_plate (plate) | 19 | 29 | 14 of 19 (49-91%) | 28 of 29 (82-100%) | 0.851 | 0.967 | 0.961 |
| mean of the 6 plate chores | | | | | 0.933 | 0.979 | 0.948 |
| mean of the 4 other chores | | | | | 0.970 | 0.980 | 0.978 |

## Per chore: `fine-change-logreg` (before_after)

Feature: 8 fine crops of the first and last frame, the 32 patches with the largest change, [mean f_last, mean f_first, mean d], 2304 numbers.

| chore | n pass | n fail | true pass judged PASS | true fail judged FAIL | dev balanced accuracy | dev AUROC | train CV balanced accuracy |
|---|---|---|---|---|---|---|---|
| bottle_to_shelf | 38 | 10 | 35 of 38 (79-98%) | 10 of 10 (69-100%) | 0.961 | 0.997 | 0.873 |
| plate_to_hotel_bottom (plate) | 26 | 18 | 25 of 26 (80-100%) | 16 of 18 (65-99%) | 0.925 | 0.979 | 0.945 |
| plate_to_hotel_top (plate) | 27 | 21 | 25 of 27 (76-99%) | 21 of 21 (84-100%) | 0.963 | 0.998 | 0.973 |
| plate_to_tray (plate) | 20 | 28 | 10 of 20 (27-73%) | 26 of 28 (76-99%) | 0.714 | 0.904 | 0.807 |
| sort_by_cap | 37 | 11 | 36 of 37 (86-100%) | 11 of 11 (72-100%) | 0.986 | 1.000 | 0.876 |
| stack_plates (plate) | 23 | 25 | 18 of 23 (56-93%) | 19 of 25 (55-91%) | 0.771 | 0.901 | 0.844 |
| stack_tip_boxes (plate) | 23 | 25 | 19 of 23 (61-95%) | 24 of 25 (80-100%) | 0.893 | 0.970 | 0.844 |
| tubes_to_rack | 31 | 16 | 28 of 31 (74-98%) | 15 of 16 (70-100%) | 0.920 | 0.940 | 0.931 |
| tubes_to_waste | 39 | 9 | 37 of 39 (83-99%) | 9 of 9 (66-100%) | 0.974 | 1.000 | 0.951 |
| unstack_plate (plate) | 19 | 29 | 14 of 19 (49-91%) | 24 of 29 (64-94%) | 0.782 | 0.882 | 0.840 |
| mean of the 6 plate chores | | | | | 0.842 | 0.939 | 0.876 |
| mean of the 4 other chores | | | | | 0.960 | 0.984 | 0.908 |

## Control: row B with shuffled train labels

Row B's features with the train labels shuffled within each chore (seeded), then the same cross-validation. It must land near 0.50. It is not a verdict file.

| | L2 0.01 | L2 0.1 | L2 1.0 | L2 10.0 | L2 100.0 |
|---|---|---|---|---|---|
| mean over chores | 0.468 | 0.466 | 0.460 | 0.458 | 0.462 |

Single chores range from 0.357 to 0.544 over all L2 values (about 180 scenes each, so single chores swing more).


## Leak check: arm pixels, last top view against the first top view of the same layout

Arm mask: a pixel is dark when its brightest channel is below 50 (of 255). The mask keeps the pixels that are dark in the first top view of all 300 train layouts (14386 pixels; a single layout has 14386 to 14473 dark pixels), then shrinks it by 3 pixels (8892 pixels left). The same mask is used for dev.

### train

| kind | truth | n | scenes with any differing arm pixel | scenes with an arm pixel off by more than 16 | largest arm pixel difference (0-255) | most differing arm pixels in 1 scene |
|---|---|---|---|---|---|---|
| clean | pass | 300 | 0 of 300 | 0 of 300 | 0 | 0 |
| offset | pass | 493 | 0 of 493 | 0 of 493 | 0 | 0 |
| tilt | pass | 253 | 0 of 253 | 0 of 253 | 0 | 0 |
| offset | fail | 381 | 0 of 381 | 0 of 381 | 0 | 0 |
| partial | fail | 45 | 0 of 45 | 0 of 45 | 0 | 0 |
| tilt | fail | 46 | 0 of 46 | 0 of 46 | 0 | 0 |
| trap | fail | 145 | 3 of 145 | 3 of 145 | 222 | 10 |
| untouched | fail | 105 | 0 of 105 | 0 of 105 | 0 | 0 |

3 of 1768 train scenes have any differing pixel inside the arm mask. Outside the mask, the median scene has 2533 differing pixels (the labware moved).

- `train-unstack_plate-s10000-01-trap` (trap, fail): 4 arm pixels differ, largest difference 85.
- `train-unstack_plate-s10008-01-trap` (trap, fail): 10 arm pixels differ, largest difference 72.
- `train-unstack_plate-s10028-01-trap` (trap, fail): 3 arm pixels differ, largest difference 222.

### dev

| kind | truth | n | scenes with any differing arm pixel | scenes with an arm pixel off by more than 16 | largest arm pixel difference (0-255) | most differing arm pixels in 1 scene |
|---|---|---|---|---|---|---|
| clean | pass | 80 | 0 of 80 | 0 of 80 | 0 | 0 |
| offset | pass | 134 | 0 of 134 | 0 of 134 | 0 | 0 |
| tilt | pass | 69 | 0 of 69 | 0 of 69 | 0 | 0 |
| offset | fail | 102 | 0 of 102 | 0 of 102 | 0 | 0 |
| partial | fail | 12 | 0 of 12 | 0 of 12 | 0 | 0 |
| tilt | fail | 11 | 0 of 11 | 0 of 11 | 0 | 0 |
| trap | fail | 39 | 2 of 39 | 1 of 39 | 222 | 15 |
| untouched | fail | 28 | 0 of 28 | 0 of 28 | 0 | 0 |

2 of 475 dev scenes have any differing pixel inside the arm mask. Outside the mask, the median scene has 2502 differing pixels (the labware moved).

- `dev-unstack_plate-s20000-01-trap` (trap, fail): 15 arm pixels differ, largest difference 222.
- `dev-unstack_plate-s20004-01-trap` (trap, fail): 13 arm pixels differ, largest difference 16.


All 5 scenes with a differing arm pixel are `unstack_plate` traps where the knocked-away plate or its shadow ends up on the edge of the left arm's black base plate (each one was looked at). The arms themselves are pixel-identical between the first and last top view in every train and dev scene, so nothing about the arms goes with kind or truth, and the arms-flat control row B2 was not added.

## Seconds per scene

Measured during dev extraction on 1 machine, wall time with the GPU possibly shared. Encoding includes reading the PNG, cropping and resizing. Row C counts the first frame's 8 crops in full for every scene (a judge on its own would encode it every time).

| row | mean s per scene (encode + head) | median encode s | head s per scene |
|---|---|---|---|
| `top-logreg` | 0.042 | 0.040 | 0.000003 |
| `fine-logreg` | 0.461 | 0.419 | 0.000003 |
| `fine-change-logreg` | 0.935 | 0.852 | 0.000004 |

## Files

- Verdicts: `results/simset_dev/verdicts_frozen__<row>_<condition>.jsonl`.
- Fitted heads: `results/frozen/heads_<row>.npz`. Choices and folds: `results/frozen/fit_log.json`.
- Every dev prediction: `results/frozen/dev_predictions.jsonl`.
- Leak check: `results/frozen/leakcheck.md`. Smoke test: `results/frozen/smoke.log`.

