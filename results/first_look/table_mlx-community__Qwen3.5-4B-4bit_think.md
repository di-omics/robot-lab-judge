# First look: zero-shot vision judge

Model: `mlx-community/Qwen3.5-4B-4bit` (snapshot `0e7ffd5c629ef7719d4cbc04069232580bfa9d9c`), mlx-vlm 0.8.0, prompt v1, thinking on, temperature 0.0, run date 2026-11-19.

Truth is the simulator's frozen pass rule. Unparsed replies count as wrong answers. Ranges are exact 95% (Clopper-Pearson).

## Results

| condition | n | true pass judged PASS | true pass judged FAIL | true fail judged FAIL | true fail judged PASS | unparsed | pass recall | fail recall | balanced accuracy | false-pass rate | median s per take |
|---|---|---|---|---|---|---|---|---|---|---|---|
| after | 59 | 2 | 4 | 27 | 1 | 25 | 2 of 10 (3-56%) | 27 of 49 (40-69%) | 0.38 | 1 of 49 (0-11%) | 85.3 |
| always PASS (reference) | 59 | 10 | 0 | 0 | 49 | 0 | 10 of 10 (69-100%) | 0 of 49 (0-7%) | 0.50 | 49 of 49 (93-100%) | - |
| always FAIL (reference) | 59 | 0 | 10 | 49 | 0 | 0 | 0 of 10 (0-31%) | 49 of 49 (93-100%) | 0.50 | 0 of 49 (0-7%) | - |

## Per task, condition `after`

| task | n | true pass | judged PASS on true pass | true fail | judged FAIL on true fail | unparsed |
|---|---|---|---|---|---|---|
| bottle_to_shelf | 3 | 0 | 0 | 3 | 1 | 2 |
| plate_to_hotel_bottom | 3 | 0 | 0 | 3 | 1 | 2 |
| plate_to_hotel_middle | 3 | 0 | 0 | 3 | 1 | 2 |
| plate_to_hotel_top | 3 | 0 | 0 | 3 | 2 | 1 |
| plate_to_tray | 6 | 2 | 0 | 4 | 2 | 2 |
| rack_carry | 3 | 2 | 0 | 1 | 1 | 0 |
| rack_into_box | 3 | 0 | 0 | 3 | 1 | 2 |
| small_tubes_to_rack | 3 | 0 | 0 | 3 | 2 | 1 |
| sort_by_cap | 3 | 1 | 1 | 2 | 0 | 1 |
| stack_plates | 3 | 0 | 0 | 3 | 2 | 1 |
| stack_tip_boxes | 3 | 0 | 0 | 3 | 1 | 2 |
| tube_handover | 3 | 1 | 0 | 2 | 2 | 1 |
| tubes_to_rack | 14 | 2 | 0 | 12 | 9 | 5 |
| tubes_to_waste | 3 | 2 | 1 | 1 | 0 | 2 |
| unstack_plate | 3 | 0 | 0 | 3 | 2 | 1 |

## Notes

- n is small: 10 true passes. One take more or less moves pass recall by 10 points.
- These takes were already seen during development of the simulation. This is a first look and a pipeline check, not a sealed test.
- Simulation only. Nothing here says how the judge does on real camera frames.
