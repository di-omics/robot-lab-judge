# Scene set calibration: where the label flips

Date 2026-11-19. Bench code `92c256eb309b`, mujoco 4.0.1.
Layout seeds 10000, 10001, 10002 (train range). Per layout, 3 random directions, each with one random placed prop and no yaw turn. The offset is bisected on a log scale from 0.5 to 60 mm (5 steps after the 2 ends). A discarded scene counts as not a pass.
The flip distance is the planar offset where the frozen rule's label on the settled state turns from pass to fail.
Discard reasons: `blocked` means the moved prop still overlapped something after being raised 80 mm (for example a plate pushed into a side wall of the hotel); `not at rest` means a prop still moved after 3 rest windows. Both count as not a pass here, so flips in directions that block early are short. In the generated splits a discarded scene is drawn again (up to 12 tries for an offset, 4 for others) and counted in info.json. When the sweep saw no fail at all, `generate` aims at the bisection median instead.

Then, per layout, 16 random offsets drawn the way `generate` draws them (log-uniform 0.5 to 60 mm, 30% with a yaw turn, a discarded scene drawn again), and one threshold fitted to them: the fit flip. `generate` aims its offsets around the fit flip, because the bisection counts blocked scenes as fails.

| task | median flip mm | flips mm (each direction) | fit flip mm (sweep scenes it gets wrong) | tilt scenes that pass | discarded scenes | notes |
|---|---|---|---|---|---|---|
| tubes_to_rack | 3.77 | 3.24, 3.77, 3.77, 4.38, 3.77, 3.24, 3.77, 3.24, 3.77 | 3.82 (0 of 47) | 8 of 13 | 40 of 146 (a prop left the table 13, blocked 1, not at rest 26) | - |
| bottle_to_shelf | 55.68 | 60, 60, 55.68, 47.94, 60, 47.94, 47.94, 47.94, 60 | 49.49 (0 of 48) | 11 of 11 | 7 of 106 (a prop left the table 2, not at rest 5) | s10000 bottle 159 deg: passes at 60 mm; s10000 bottle -165 deg: passes at 60 mm; s10001 bottle -34 deg: passes at 60 mm; s10002 bottle 147 deg: passes at 60 mm |
| plate_to_tray | 5.9 | 5.9, 5.08, 5.9, 5.9, 5.08, 5.08, 5.9, 5.9, 5.08 | 5.57 (3 of 48) | 8 of 15 | 0 of 126 | - |
| sort_by_cap | 60.0 | 60, 47.94, 47.94, 55.68, 60, 60, 60, 60, 60 | 39.29 (0 of 48) | 10 of 12 | 5 of 98 (a prop left the table 3, not at rest 2) | s10000 tube_red 14 deg: passes at 60 mm; s10001 tube_red 1 deg: passes at 60 mm; s10001 tube_red -132 deg: passes at 60 mm; s10002 tube_blue -146 deg: passes at 60 mm; s10002 tube_red 2 deg: passes at 60 mm; s10002 tube_red 133 deg: passes at 60 mm |
| tubes_to_waste | 41.28 | 35.54, 60, 60, 55.68, 60, 35.54, 35.54, 41.28, 26.35 | 60.0 (0 of 48) | 14 of 14 | 19 of 115 (blocked 18, not at rest 1) | s10000 tube_a -115 deg: passes at 60 mm; s10000 tube_b -42 deg: passes at 60 mm; s10001 tube_b -50 deg: passes at 60 mm |
| stack_tip_boxes | 26.35 | 26.35, 26.35, 26.35, 26.35, 26.35, 26.35, 26.35, 26.35, 26.35 | 29.07 (8 of 48) | 15 of 15 | 0 of 126 | - |
| plate_to_hotel_top | 5.08 | 5.08, 5.08, 16.82, 5.9, 5.08, 5.9, 5.08, 5.08, 5.08 | 6.64 (6 of 48) | 14 of 15 | 0 of 126 | - |
| plate_to_hotel_middle | 5.9 | 16.82, 5.9, 6.86, 5.08, 5.9, 12.47, 5.08, 16.82, 5.08 | 60.0 (0 of 46) | 13 of 13 | 83 of 180 (blocked 83) | - |
| plate_to_hotel_bottom | 5.9 | 5.08, 19.54, 5.08, 16.82, 5.9, 5.08, 7.96, 16.82, 5.08 | 18.25 (0 of 46) | 11 of 11 | 62 of 159 (blocked 62) | - |
| stack_plates | 12.47 | 12.47, 14.48, 12.47, 12.47, 12.47, 12.47, 12.47, 12.47, 12.47 | 11.88 (5 of 48) | 15 of 15 | 0 of 126 | - |
| unstack_plate | 5.08 | 5.08, 5.08, 5.9, 4.38, 5.9, 4.38, 5.9, 5.9, 5.08 | 5.69 (5 of 48) | 10 of 15 | 0 of 126 | - |
| small_tubes_to_rack | 6.86 | 5.9, 5.9, 6.86, 22.69, 30.6, 6.86, 5.9, 22.69, 5.9 | 6.16 (3 of 48) | 9 of 11 | 36 of 137 (not at rest 36) | - |
| rack_into_box | 3.77 | 3.24, 3.77, 3.77, 4.38, 3.77, 3.24, 4.38, 3.24, 30.6 | 3.77 (11 of 48) | 5 of 14 | 22 of 135 (a prop left the table 1, not at rest 21) | - |

## Tasks left out

A still frame cannot show the way there, so tasks whose pass also needs an earlier stage are left out.

- `rack_carry`: success also needs earlier stages: rack_lifted_level.
- `tube_handover`: success also needs earlier stages: right_hand_has_it.
