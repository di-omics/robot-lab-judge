# Results

| Folder or file | What it holds |
|---|---|
| `simset_dev/` | Every judge run on the dev split of the scene set: `board.md` (all runs in 1 table), `breakdown.md` (by kind of end state, by distance, by chore), `reasoning_keywords.md`, and 1 `verdicts_*.jsonl` file per run |
| `simset/` | `calibration.md` and `calibration.json`: how far a prop can move before each chore fails |
| `first_look/` | The earlier look at 59 recorded takes, see [docs/first-look.md](../docs/first-look.md) |
| `sealed_log.jsonl` | 1 line for every look at the sealed test split |

## A verdicts file

The name is `verdicts_<model>[_think]_<condition>.jsonl`. `_think` marks a run with thinking on. The conditions
are `after` (last frame), `before_after` (first and last frame) and, in the first look only, `leak_control`.

Line 1 is a header: model id, snapshot hash, library versions, prompt version, thinking, temperature, token limit,
date, data folder and condition.

Every other line is 1 scene: `take_id`, `task`, `seed`, `truth` (pass or fail, from the simulator), the pictures
shown, the model's full `reply`, the parsed `verdict` (PASS, FAIL or unparsed) and `seconds`.

An unparsed reply counts as a wrong answer everywhere.

## Making the tables again

```
python -m labjudge board --results results/simset_dev
python -m labjudge breakdown --data data/simset/dev --results results/simset_dev
python tools/update_readme.py
```
