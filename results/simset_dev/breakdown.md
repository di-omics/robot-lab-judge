# Scene set: where each judge says PASS

Each cell is the number of scenes the run judged PASS, of the scenes it saw. A good judge says PASS on every true pass and on no true fail. An unparsed reply is not a PASS.

## By kind of end state

| kind | truth | n | LFM2.5-VL-450M-MLX-bf16, after | LFM2.5-VL-450M-MLX-bf16, before_after | LFM3-VL-2B-MLX-8bit, thinking, after | LFM3-VL-2B-MLX-8bit, thinking, before_after | Qwen3.5-4B-4bit, after | Qwen3.5-4B-4bit, before_after |
|---|---|---|---|---|---|---|---|---|
| clean | pass | 80 | 0 of 80 | 0 of 80 | 24 of 80 | 50 of 80 | 63 of 80 | 80 of 80 |
| offset | pass | 134 | 0 of 134 | 0 of 134 | 69 of 134 | 97 of 134 | 118 of 134 | 134 of 134 |
| tilt | pass | 69 | 0 of 69 | 0 of 69 | 28 of 69 | 40 of 69 | 53 of 69 | 69 of 69 |
| offset | fail | 102 | 0 of 102 | 0 of 102 | 18 of 102 | 51 of 102 | 86 of 102 | 102 of 102 |
| tilt | fail | 11 | 0 of 11 | 0 of 11 | 0 of 11 | 2 of 11 | 9 of 11 | 11 of 11 |
| partial | fail | 12 | 0 of 12 | 0 of 12 | 0 of 12 | 0 of 12 | 12 of 12 | 12 of 12 |
| untouched | fail | 28 | 0 of 28 | 0 of 28 | 0 of 28 | 0 of 28 | 27 of 28 | 28 of 28 |
| trap | fail | 39 | 0 of 39 | 0 of 39 | 7 of 39 | 14 of 39 | 36 of 39 | 37 of 39 |

## Offset scenes with no turn, by how far the moved prop ended from its place

Distance in 3 dimensions from where the task's oracle left the prop, after settling. Offset scenes that also turned the prop are left out here, so distance is the only change. Chores differ in how far a prop can move and still pass, and a judge can call a whole chore PASS or FAIL, so read this table next to the by-task table.

| distance | truth | n | LFM2.5-VL-450M-MLX-bf16, after | LFM2.5-VL-450M-MLX-bf16, before_after | LFM3-VL-2B-MLX-8bit, thinking, after | LFM3-VL-2B-MLX-8bit, thinking, before_after | Qwen3.5-4B-4bit, after | Qwen3.5-4B-4bit, before_after |
|---|---|---|---|---|---|---|---|---|
| 0 to 2 mm | pass | 11 | 0 of 11 | 0 of 11 | 3 of 11 | 11 of 11 | 11 of 11 | 11 of 11 |
| 2 to 5 mm | pass | 21 | 0 of 21 | 0 of 21 | 1 of 21 | 6 of 21 | 15 of 21 | 21 of 21 |
| 5 to 10 mm | pass | 5 | 0 of 5 | 0 of 5 | 0 of 5 | 1 of 5 | 4 of 5 | 5 of 5 |
| 10 to 20 mm | pass | 11 | 0 of 11 | 0 of 11 | 0 of 11 | 1 of 11 | 6 of 11 | 11 of 11 |
| 20 to 40 mm | pass | 19 | 0 of 19 | 0 of 19 | 15 of 19 | 17 of 19 | 19 of 19 | 19 of 19 |
| 40 mm and over | pass | 38 | 0 of 38 | 0 of 38 | 32 of 38 | 38 of 38 | 38 of 38 | 38 of 38 |
| 5 to 10 mm | fail | 9 | 0 of 9 | 0 of 9 | 0 of 9 | 0 of 9 | 7 of 9 | 9 of 9 |
| 10 to 20 mm | fail | 21 | 0 of 21 | 0 of 21 | 1 of 21 | 5 of 21 | 14 of 21 | 21 of 21 |
| 20 to 40 mm | fail | 18 | 0 of 18 | 0 of 18 | 3 of 18 | 12 of 18 | 17 of 18 | 18 of 18 |
| 40 mm and over | fail | 21 | 0 of 21 | 0 of 21 | 10 of 21 | 13 of 21 | 21 of 21 | 21 of 21 |

## Balanced accuracy by task

| task | n | LFM2.5-VL-450M-MLX-bf16, after | LFM2.5-VL-450M-MLX-bf16, before_after | LFM3-VL-2B-MLX-8bit, thinking, after | LFM3-VL-2B-MLX-8bit, thinking, before_after | Qwen3.5-4B-4bit, after | Qwen3.5-4B-4bit, before_after |
|---|---|---|---|---|---|---|---|
| bottle_to_shelf | 48 | 0.50 | 0.50 | 0.81 | 0.85 | 0.50 | 0.50 |
| plate_to_hotel_bottom | 44 | 0.50 | 0.50 | 0.50 | 0.44 | 0.31 | 0.50 |
| plate_to_hotel_top | 48 | 0.50 | 0.50 | 0.50 | 0.52 | 0.50 | 0.50 |
| plate_to_tray | 48 | 0.50 | 0.50 | 0.46 | 0.36 | 0.29 | 0.50 |
| sort_by_cap | 48 | 0.50 | 0.50 | 0.91 | 0.91 | 0.50 | 0.59 |
| stack_plates | 48 | 0.50 | 0.50 | 0.44 | 0.28 | 0.26 | 0.50 |
| stack_tip_boxes | 48 | 0.50 | 0.50 | 0.52 | 0.61 | 0.50 | 0.50 |
| tubes_to_rack | 47 | 0.50 | 0.50 | 0.76 | 0.78 | 0.50 | 0.50 |
| tubes_to_waste | 48 | 0.50 | 0.50 | 0.69 | 0.94 | 0.50 | 0.50 |
| unstack_plate | 48 | 0.50 | 0.50 | 0.48 | 0.44 | 0.50 | 0.50 |
