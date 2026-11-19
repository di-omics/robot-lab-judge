# Board: vision judges on the first-look takes

Every saved run, best balanced accuracy first. Truth is the simulator's frozen pass rule. Unparsed replies count as wrong answers. Ranges are exact 95% (Clopper-Pearson).

## Results

| model | thinking | condition | n | pass recall | fail recall | false-pass rate | balanced accuracy | unparsed | median s per take |
|---|---|---|---|---|---|---|---|---|---|
| `LiquidAI/LFM3-VL-2B-MLX-8bit` | on | after | 59 | 4 of 10 (12-74%) | 46 of 49 (83-99%) | 3 of 49 (1-17%) | 0.67 | 0 | 12.1 |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16` | off | after | 59 | 3 of 10 (7-65%) | 47 of 49 (86-100%) | 2 of 49 (0-14%) | 0.63 | 0 | 0.2 |
| `mlx-community/Qwen3.5-4B-4bit` | off | after | 59 | 10 of 10 (69-100%) | 2 of 49 (0-14%) | 47 of 49 (86-100%) | 0.52 | 0 | 1.8 |
| `mlx-community/Qwen3.5-4B-4bit` | off | before_after | 59 | 10 of 10 (69-100%) | 2 of 49 (0-14%) | 47 of 49 (86-100%) | 0.52 | 0 | 3.3 |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16` | off | before_after | 59 | 0 of 10 (0-31%) | 49 of 49 (93-100%) | 0 of 49 (0-7%) | 0.50 | 0 | 0.4 |
| `mlx-community/Qwen3.5-4B-4bit` | on | after | 59 | 2 of 10 (3-56%) | 27 of 49 (40-69%) | 1 of 49 (0-11%) | 0.38 | 25 | 85.3 |
| `LiquidAI/LFM3-VL-2B-MLX-8bit` | off | after | 59 | 0 of 10 (0-31%) | 26 of 49 (38-67%) | 0 of 49 (0-7%) | 0.27 | 30 | 0.7 |
| `LiquidAI/LFM3-VL-2B-MLX-8bit` | off | before_after | 59 | 1 of 10 (0-45%) | 3 of 49 (1-17%) | 7 of 49 (6-27%) | 0.08 | 48 | 1.0 |
| always PASS (reference) | - | - | 59 | 10 of 10 (69-100%) | 0 of 49 (0-7%) | 49 of 49 (93-100%) | 0.50 | 0 | - |
| always FAIL (reference) | - | - | 59 | 0 of 10 (0-31%) | 49 of 49 (93-100%) | 0 of 49 (0-7%) | 0.50 | 0 | - |

## Control (not a result): the banner left in

The judge saw the uncropped last frame, with the banner that prints the stage count. This only tests whether a model reads on-screen text.

| model | thinking | condition | n | pass recall | fail recall | false-pass rate | balanced accuracy | unparsed | median s per take |
|---|---|---|---|---|---|---|---|---|---|
| `LiquidAI/LFM3-VL-2B-MLX-8bit` | on | leak_control | 59 | 5 of 10 (19-81%) | 44 of 49 (78-97%) | 4 of 49 (2-20%) | 0.70 | 2 | 14.4 |
| `mlx-community/Qwen3.5-4B-4bit` | off | leak_control | 59 | 10 of 10 (69-100%) | 4 of 49 (2-20%) | 45 of 49 (80-98%) | 0.54 | 0 | 1.9 |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16` | off | leak_control | 59 | 1 of 10 (0-45%) | 48 of 49 (89-100%) | 1 of 49 (0-11%) | 0.54 | 0 | 0.2 |
| `LiquidAI/LFM3-VL-2B-MLX-8bit` | off | leak_control | 59 | 1 of 10 (0-45%) | 42 of 49 (73-94%) | 4 of 49 (2-20%) | 0.48 | 6 | 0.7 |

## Notes

- n is small: 10 true passes. One take more or less moves pass recall by 10 points.
- These takes were already seen during development of the simulation. This is a first look and a pipeline check, not a sealed test.
- Simulation only. Nothing here says how the judge does on real camera frames.
