# The linear probe on changed pictures

The saved heads, the dev pictures changed a little, no refit. `none` checks this script against the
saved verdicts. Made by `tools/probe_perturbed.py`, which lists the changes.

| Row | Change | Scenes | Same verdict as saved | Says PASS on true passes | False passes | Balanced accuracy |
|---|---|---|---|---|---|---|
| `top-logreg` | none | 475 | 475 of 475 | 274 of 283 (94-99%) | 14 of 192 (4-12%) | 0.95 |
| `top-logreg` | jpeg | 475 | 378 of 475 | 214 of 283 (70-81%) | 27 of 192 (9-20%) | 0.81 |
| `top-logreg` | shift | 475 | 370 of 475 | 185 of 283 (60-71%) | 12 of 192 (3-11%) | 0.80 |
| `top-logreg` | all | 475 | 354 of 475 | 176 of 283 (56-68%) | 19 of 192 (6-15%) | 0.76 |
| `fine-logreg` | none | 40 | 40 of 40 | 28 of 28 (88-100%) | 0 of 12 (0-26%) | 1.00 |
| `fine-logreg` | all | 475 | 338 of 475 | 210 of 283 (69-79%) | 65 of 192 (27-41%) | 0.70 |
