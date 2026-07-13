# Handy commands — BHAC → FNO converter

Convert one finished datagen run to a uniform-grid h5. `RUN_ROOT` must be the
directory `run_tmp.sh` wrote to (its `OUTROOT`), `BHAC_EXE` the converter `bhac`
binary, `LEVEL_IO` the output level (1 = base 256², 2 = 512², …):

```bash
sbatch --export=ALL,\
RUN_ROOT=$HOME/data/BHAC_output/orszagTang/converge/datagen/<N>,\
LEVEL_IO=1,\
BHAC_EXE=<your converter bhac binary> \
bhac_to_fno.sh
```

Note: don't use a `*` glob in `RUN_ROOT`.

Quick-look the produced h5 (renders per-timestep PNGs of one field):

```bash
python plot_fno_h5.py --input <RUN_ROOT>/output-0000/fno/fno_uniform_level1.h5 \
    --var eta --outdir png_eta --robust
```
