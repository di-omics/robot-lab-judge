# The first look: recorded takes

Before the scene set existed, the same judges were tried on recorded robot attempts. This page keeps that record.
It is an early pipeline check on takes that were already seen while the simulation was being built. The
[scene set](scene-set.md) replaced it as the place where numbers come from.

## What was judged

59 recorded takes from the runs of [robot-lab-handling](https://github.com/di-omics/robot-lab-handling): 10
passes and 49 fails by the simulator's frozen rule. 16 more takes were stopped by an error and left out. 56 of
the 59 are zero-shot takes of a robot model. 3 are camera-guided takes from another recorder (2 of the 10 passes).
1 of the 59 is an early practice take that is not in that repo's public records. The take videos are not public,
so this page cannot be reproduced from the 2 repos alone. The verdict files are in `results/first_look/`.

## The label leak in the take videos

Every frame of a take video is 640 x 566: a text banner in rows 0 to 25, the top camera in rows 26 to 385, and
the 2 wrist cameras below. The banner prints the stage count and the word DONE. A judge that can read gets the
answer for free. So `extract` cuts the banner away before any judge sees a frame, and the tests fail if a judged
picture still holds banner rows.

The uncropped frame was kept as a control, never as a result. With the banner left in, 7 of the 59 replies of the
reasoning model quoted the banner's step count, and all 7 were right. The leak is real, so the crop stays.

## Conditions

- `after`: the cropped last frame (top view over both wrist views).
- `before_after`: the cropped first frame, then the cropped last frame.
- `leak_control`: the uncropped last frame, banner included. A control only.

## What came out

The full table is [`results/first_look/board.md`](../results/first_look/board.md). The main lines, last frame:

| Judge | Says PASS on true passes | False passes | Balanced accuracy |
|---|---|---|---|
| `LiquidAI/LFM3-VL-2B-MLX-8bit`, thinking on | 4 of 10 | 3 of 49 | 0.67 |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16` | 3 of 10 | 2 of 49 | 0.63 |
| `mlx-community/Qwen3.5-4B-4bit` | 10 of 10 | 47 of 49 | 0.52 |

With only 10 passes these 3 rows are within noise of each other. Two things from this look held up later:

- The small Liquid model's 3 caught passes were all `rack_carry` or `tube_handover`. On the scene set, which
  leaves those 2 chores out, it answers FAIL on every scene.
- LFM3-VL-2B with thinking off is not usable for this: 30 of 59 replies were not PASS or FAIL.

## What this look does not mean

- n is small: 59 takes, 10 of them passes. The ranges are wide.
- These takes were already seen during development. This is a floor for later work, not a test.
- The judge sees still frames from a simulator.
