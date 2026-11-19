# First look: zero-shot vision judge

Model: `LiquidAI/LFM3-VL-2B-MLX-8bit` (snapshot `6990b981e8235d1a41a3aa3c50b3402631659abb`), mlx-vlm 0.8.0, prompt v1, thinking on, temperature 0.0, run date 2026-11-19.

Truth is the simulator's frozen pass rule. Unparsed replies count as wrong answers. Ranges are exact 95% (Clopper-Pearson).

## Results

| condition | n | true pass judged PASS | true pass judged FAIL | true fail judged FAIL | true fail judged PASS | unparsed | pass recall | fail recall | balanced accuracy | false-pass rate | median s per take |
|---|---|---|---|---|---|---|---|---|---|---|---|
| after | 59 | 4 | 6 | 46 | 3 | 0 | 4 of 10 (12-74%) | 46 of 49 (83-99%) | 0.67 | 3 of 49 (1-17%) | 12.1 |
| always PASS (reference) | 59 | 10 | 0 | 0 | 49 | 0 | 10 of 10 (69-100%) | 0 of 49 (0-7%) | 0.50 | 49 of 49 (93-100%) | - |
| always FAIL (reference) | 59 | 0 | 10 | 49 | 0 | 0 | 0 of 10 (0-31%) | 49 of 49 (93-100%) | 0.50 | 0 of 49 (0-7%) | - |

## Per task, condition `after`

| task | n | true pass | judged PASS on true pass | true fail | judged FAIL on true fail | unparsed |
|---|---|---|---|---|---|---|
| bottle_to_shelf | 3 | 0 | 0 | 3 | 3 | 0 |
| plate_to_hotel_bottom | 3 | 0 | 0 | 3 | 3 | 0 |
| plate_to_hotel_middle | 3 | 0 | 0 | 3 | 3 | 0 |
| plate_to_hotel_top | 3 | 0 | 0 | 3 | 3 | 0 |
| plate_to_tray | 6 | 2 | 1 | 4 | 4 | 0 |
| rack_carry | 3 | 2 | 0 | 1 | 1 | 0 |
| rack_into_box | 3 | 0 | 0 | 3 | 2 | 0 |
| small_tubes_to_rack | 3 | 0 | 0 | 3 | 3 | 0 |
| sort_by_cap | 3 | 1 | 1 | 2 | 1 | 0 |
| stack_plates | 3 | 0 | 0 | 3 | 3 | 0 |
| stack_tip_boxes | 3 | 0 | 0 | 3 | 3 | 0 |
| tube_handover | 3 | 1 | 0 | 2 | 2 | 0 |
| tubes_to_rack | 14 | 2 | 1 | 12 | 11 | 0 |
| tubes_to_waste | 3 | 2 | 1 | 1 | 1 | 0 |
| unstack_plate | 3 | 0 | 0 | 3 | 3 | 0 |

## Control (not a result): `leak_control`

The judge saw the uncropped last frame, with the banner that prints the stage count. This only tests whether the model reads on-screen text. On 3 takes from a different recorder the banner shows only the task and the clock.

| condition | n | true pass judged PASS | true pass judged FAIL | true fail judged FAIL | true fail judged PASS | unparsed | pass recall | fail recall | balanced accuracy | false-pass rate | median s per take |
|---|---|---|---|---|---|---|---|---|---|---|---|
| leak_control | 59 | 5 | 4 | 44 | 4 | 2 | 5 of 10 (19-81%) | 44 of 49 (78-97%) | 0.70 | 4 of 49 (2-20%) | 14.4 |

## Notes

- n is small: 10 true passes. One take more or less moves pass recall by 10 points.
- These takes were already seen during development of the simulation. This is a first look and a pipeline check, not a sealed test.
- Simulation only. Nothing here says how the judge does on real camera frames.
