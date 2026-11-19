# Board: vision judges on the scene set, dev split

Every saved run, best balanced accuracy first. Truth is the simulator's frozen pass rule. Unparsed replies count as wrong answers. Ranges are exact 95% (Clopper-Pearson).

## Results

| model | thinking | condition | n | pass recall | fail recall | false-pass rate | balanced accuracy | unparsed | median s per take |
|---|---|---|---|---|---|---|---|---|---|
| `LiquidAI/LFM3-VL-2B-MLX-8bit` | on | after | 475 | 121 of 283 (37-49%) | 167 of 192 (81-91%) | 25 of 192 (9-19%) | 0.65 | 0 | 9.9 |
| `LiquidAI/LFM3-VL-2B-MLX-8bit` | on | before_after | 475 | 187 of 283 (60-72%) | 120 of 192 (55-69%) | 67 of 192 (28-42%) | 0.64 | 5 | 13.2 |
| `mlx-community/Qwen3.5-4B-4bit` | off | before_after | 475 | 283 of 283 (99-100%) | 2 of 192 (0-4%) | 190 of 192 (96-100%) | 0.51 | 0 | 1.9 |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16` | off | after | 475 | 0 of 283 (0-1%) | 192 of 192 (98-100%) | 0 of 192 (0-2%) | 0.50 | 0 | 0.1 |
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16` | off | before_after | 475 | 0 of 283 (0-1%) | 192 of 192 (98-100%) | 0 of 192 (0-2%) | 0.50 | 0 | 0.2 |
| `mlx-community/Qwen3.5-4B-4bit` | off | after | 475 | 234 of 283 (78-87%) | 22 of 192 (7-17%) | 170 of 192 (83-93%) | 0.47 | 0 | 1.1 |
| always PASS (reference) | - | - | 475 | 283 of 283 (99-100%) | 0 of 192 (0-2%) | 192 of 192 (98-100%) | 0.50 | 0 | - |
| always FAIL (reference) | - | - | 475 | 0 of 283 (0-1%) | 192 of 192 (98-100%) | 0 of 192 (0-2%) | 0.50 | 0 | - |

## Notes

- These are simulator scenes, not robot attempts. Both arms rest at home in every scene. A real attempt ends with the arms anywhere.
- Truth is the frozen pass rule on the settled end state. Some rules are loose (a bottle anywhere on the platform passes), so a true pass can look untidy.
- This is the dev split. It is for choosing prompts and thresholds, so it is not a final number. Final numbers come from the sealed test split only.
- Simulation only. Nothing here says how the judge does on real camera frames.
