# What the best judge says on clean plate scenes

Run: `LiquidAI/LFM3-VL-2B-MLX-8bit`, thinking on, last frame, dev split. A clean scene is one where
every prop is where the chore's oracle left it, so it passes. The patterns are matched on the
lower-cased thinking text of each reply. Made by `tools/update_readme.py`.

| Chore | Clean scenes | Judged FAIL | Its reasoning speaks of | In |
|---|---|---|---|---|
| `stack_plates` | 8 | 8 of 8 | a single plate | 8 of 8 |
| `plate_to_tray` | 8 | 8 of 8 | no tray, or the plate is not on a tray | 7 of 8 |
| `unstack_plate` | 8 | 8 of 8 | - | - |
| `plate_to_hotel_top` | 8 | 8 of 8 | - | - |
| `plate_to_hotel_bottom` | 8 | 8 of 8 | - | - |
| `stack_tip_boxes` | 8 | 5 of 8 | - | - |

Patterns:

- `stack_plates`: `only one|single plate|one plate|a single|just one`
- `plate_to_tray`: `no (black )?tray|not on (any|a|the) (black )?tray|(don['\u2019]t|do not|cannot|can['\u2019]t) see a (black )?tray`
