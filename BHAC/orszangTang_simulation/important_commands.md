# Handy commands — Orszag-Tang datagen run

Submit one dataset ("datagen") simulation:

```bash
sbatch --export=ALL,BHAC_DIR=<your BHAC framework checkout> run_tmp.sh
```

`run_tmp.sh` compiles `./bhac` locally (`make`) and runs it; this folder is a
BHAC *setup* dir, not the framework source. `$BHAC_DIR` must point at a BHAC
framework tree (the one `setup.pl` and the physics modules come from) — clone and
install it per the upstream BHAC guide:
<https://bhac.science/> → Documentation
(`git clone https://gitlab.itp.uni-frankfurt.de/BHAC-release/bhac.git`).

Each run writes to `$HOME/data/BHAC_output/orszagTang/converge/datagen/<N>`
(`output-0000/` + `inputs/amrvac.par`) — this is the `RUN_ROOT` the converter in
`../converter_flatten_to_h5/` then reads.

Set the run's resistivity via `eta0` in `amrvac.par` (&usrlist) before submitting;
see `../resistivities_used.md` for the value each datagen run used.
