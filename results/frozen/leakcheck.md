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

