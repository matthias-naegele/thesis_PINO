# Resistivities used per dataset

Resistivity (η) per simulation, base dataset vs. extended `*_zusatz` dataset
(38 train / 11 valid sims instead of 23 / 6).

η is in natural (code) units and quoted here in units of 10⁻⁴: a table entry
of 4.69 corresponds to `eta0 = 4.69e-04` in that run's `amrvac.par` (see e.g.
`orszangTang_simulation/amrvac.par`, which is datagen run 11 = train value 4.69).

**Verified 2026-07-03** against the `eta` channel of every produced h5 under
`datagen/finished*`: all tables below match the files exactly, in order. The
run numbering encodes the split so that the dataloader's sorted-glob split
works out: `output-0001…0023` base train, `output-0901…0919` added train
(zusatz), `output-1024…1029` base valid, `output-2904…2920` added valid
(run number of the base runs = datagen dir number, e.g. output-0011 = dir 11).

`_zusatz` is a strict superset: the 23 base train and 6 base valid runs are
reused unchanged; only new values are added.

Base resistivities were chosen at random, with even η-range coverage verified
after sampling. The `_zusatz` additions skew toward lower η (mostly in
[1.00, 3.76] vs. a base range up to 9.87).

## Base dataset (23 train / 6 valid)

### Train (23)

| η | η | η | η | η |
|---|---|---|---|---|
| 7.13 | 7.19 | 9.09 | 6.57 | 3.38 |
| 8.13 | 2.37 | 1.92 | 6.74 | 2.36 |
| 4.69 | 1.49 | 4.14 | 6.81 | 3.32 |
| 2.72 | 1.10 | 3.92 | 9.87 | 7.29 |
| 7.57 | 2.33 | 4.61 | | |

### Valid (6)

| η | η | η | η | η | η |
|---|---|---|---|---|---|
| 1.25 | 1.08 | 1.41 | 5.59 | 1.27 | 5.84 |

## Extended dataset (`*_zusatz`, 38 train / 11 valid)

### Train (38 = 23 base + 15 added)

Base: same 23 values as above. Added (15):

| η | η | η | η | η |
|---|---|---|---|---|
| 1.00 | 1.01 | 1.02 | 1.04 | 1.05 |
| 1.07 | 1.15 | 1.26 | 1.52 | 1.32 |
| 2.00 | 2.65 | 3.23 | 3.31 | 3.76 |

### Valid (11 = 6 base + 5 added)

Base: same 6 values as above. Added (5):

| η | η | η | η | η |
|---|---|---|---|---|
| 1.03 | 1.09 | 1.54 | 2.96 | 5.00 |
