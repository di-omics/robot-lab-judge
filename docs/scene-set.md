# The scene set

A set of pictures made straight from the simulator, with end states close to the line between pass and fail.
It exists because recorded robot attempts are few and mostly fail far from the line: a judge can look good on
them without seeing anything fine.

## What a scene is

A scene is 1 chore, 1 layout seed and 1 end state of the labware, with both arms at home and the physics settled
and at rest. Each scene has 3 pictures: the start of the layout (`first_views`), the end (`last_views`) and the
end from the top camera alone (`last_top`). They use the same camera layout as a recorded take of
[robot-lab-handling](https://github.com/di-omics/robot-lab-handling), with no text banner.

Truth is that repo's frozen pass rule, read on the settled final state. A pass that only flickers while things
settle does not count.

## Kinds of end state

All start from where the chore's scripted oracle leaves the labware.

| Kind | What was done | Label |
|---|---|---|
| `clean` | Every prop where the oracle left it, moved by 0.5 mm or less | must pass |
| `offset` | 1 prop moved in the table plane, sometimes turned by up to 20 degrees, then dropped and settled | from the rule |
| `tilt` | 1 prop tipped by 4 to 35 degrees, then dropped and settled | from the rule |
| `partial` | 1 prop left where it started (chores with several moved props) | from the rule |
| `untouched` | Nothing moved | from the rule |
| `trap` | One of the chore's own look-alike failures that needs no hand to hold it | must fail |

The label always comes from the rule, never from the kind. A scene that is not at rest after settling, where a
prop left the table, or where the physics blew up is thrown away and counted in `info.json`.

Only chores whose pass rule reads the final state alone are used. `rack_carry` and `tube_handover` also need an
earlier moment (a level lift, a hand-over), which a still picture cannot show, so they are left out.

## How far a prop can move before the chore fails

`calibrate` measures, per chore, the offset at which the label turns from pass to fail. `generate` aims its
offsets around that distance, so the scenes sit near the line. The full table is in
[`results/simset/calibration.md`](../results/simset/calibration.md). In short:

| Chore | Flip distance |
|---|---|
| `tubes_to_rack`, `rack_into_box` | about 4 mm |
| `plate_to_tray`, `unstack_plate`, plate hotel | about 5 to 6 mm |
| `small_tubes_to_rack` | about 6 to 7 mm |
| `stack_plates` | about 12 mm |
| `stack_tip_boxes` | about 26 to 29 mm |
| `bottle_to_shelf`, `sort_by_cap`, `tubes_to_waste` | 40 to 60 mm, or never within 60 mm |

The direction matters as well as the distance, so passes and fails overlap in distance. 1 pixel of the top view is
about 2.5 mm on the table (an estimate from the size of a plate in the picture), so a plain 6 mm shift of a plate
is 2 to 3 pixels.

## Splits

Splits are by layout seed and never share one.

| Split | Seeds | Scenes | Passes |
|---|---|---|---|
| train | 10000 to 10029 | 1,768 | 1,046 |
| dev | 20000 to 20007 | 475 | 283 |
| test | 30000 to 30019 | 1,537 | sealed |

`plate_to_hotel_middle`, `small_tubes_to_rack` and `rack_into_box` appear only in test (`seen_task` is false
there), to measure chores that a judge fitted on the train split has not seen.

The train and dev manifests are in the repo (`data/simset/<split>/manifest.jsonl`). Each row names the scene, its
kind, the commanded move, the measured final error, the rule text and the truth. The pictures are not in the repo.

## What sealed means here

- `seal` writes `data/simset/test/SEALED.md` with the hashes of the test manifest and pictures.
- Prompts, thresholds and training use train and dev only.
- When the folder given to `judge --data` holds `SEALED.md`, `judge` adds 1 line per model and condition to
  `results/sealed_log.jsonl` before it starts. So every look at the test split is on record.
- `check` compares sealed scenes with a fresh run of the simulator, and never prints their labels.
- The test manifest is not in the repo. Anyone can make the test scenes again from the seeds, so the seal is not
  a secret. It is a rule this repo keeps for its own numbers.

One thing to know: while the set was being built, before sealing, the build logs printed the pass share of each
test chore and the labels of 5 test scenes. No model has been scored on the test split. This is the first line of
`results/sealed_log.jsonl`.

## Making the scenes again

This needs the simulator. Check out robot-lab-handling at commit `3a734a0` (its code fingerprint is
`92c256eb309b`; each split's `info.json` records it) and use the Python that has its dependencies
(`dreamlabs==1.0.42`, `mujoco==4.0.1`). Here that Python is called `SIMPY`:

```
export LAB_HANDLING_BENCH=/path/to/robot-lab-handling
$SIMPY -m labjudge.simset calibrate
$SIMPY -m labjudge.simset generate --split train
$SIMPY -m labjudge.simset generate --split dev
$SIMPY -m labjudge.simset generate --split test
$SIMPY -m labjudge.simset check
$SIMPY -m labjudge.simset sheet --split dev
```

`generate` is the same on every run for a given seed. `check` rebuilds 20 scenes in fresh simulator instances and
compares the labels; on the saved set all 20 matched. The 3 splits take about 30 minutes on a laptop and 1.5 GB of
disk.

## Limits of the scene set

- Both arms are at home in every scene, so the wrist views show almost no labware. Real attempts end with the
  arms anywhere.
- Scenes are clean renders saved as PNG. Recorded takes go through video compression.
- The pass share is above 65% on `bottle_to_shelf`, `sort_by_cap` and `tubes_to_waste`, whose rules are loose, and
  on the 3 chores that appear only in test. Balanced accuracy and the false-pass rate do not depend on that share.
- A moved prop is raised until it overlaps nothing and then dropped. So a tube moved more than about 2 mm in its
  well lands on the rim and falls over: an offset fail for tubes is a tube on its side, not a small shift.
- 2 look-alike failures are left out because a hand holds their end state (a tube held over a well, a bottle held
  over the shelf).
