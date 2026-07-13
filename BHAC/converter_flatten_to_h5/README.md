
##################################################################################################

BHAC -> FNO converter: flatten AMR snapshots to uniform-grid HDF5

##################################################################################################

Purpose
-------
Turns a finished BHAC run (AMR data####.dat snapshots under RUN_ROOT/output-????/)
into one uniform-grid HDF5 file per run. RUN_ROOT must be the directory the
simulation launcher archived to, i.e. run_tmp.sh's OUTROOT
(../orszangTang_simulation/), which holds output-????/ plus inputs/amrvac.par:

    output-????/fno/fno_uniform_level<L>.h5
    datasets: fields[T,C,Ny,Nx], t, x, y, varnames

This is exactly the input format of dataloaders/datasets.py (BHACUniformDataset).

Each bhac_to_fno*.sh does two stages per output directory:
1. BHAC convert-restarts: every data####.dat is re-read with level_io=<L> and
   convert_type='oneblock', producing uniform ASCII block files. The conversion
   parfile is copied from the run's own inputs/amrvac.par, so the run's eta0 (and
   hence the 'eta' output channel) is preserved. The script symlinks the conversion
   parfile onto amrvac.par because initglobaldata_usr reads that fixed file name.
   writew drops the entropy Ds; the resulting channels are:
   rho u1 u2 u3 p b1 b2 b3 e1 e2 e3 lfac xi jz divB eta.
2. A python packer grids the oneblock files and stacks them in time into the h5.

Which variant to use
--------------------
- bhac_to_fno.sh            — canonical for LEVEL_IO <= 2 (this produced the FNO
                              training data). In-memory packer (float64); fine at
                              level 1-2, OOMs at high levels.
- bhac_to_fno_lowRAM_dat.sh — highly likely (per user's recollection) used for
                              the LEVEL_IO=5 conversions. Streams one snapshot at
                              a time into a chunked+gzipped float32 h5 instead of
                              holding the whole run in RAM, avoiding the OOM
                              bhac_to_fno.sh hits at that level.
- bhac_to_fno_lowRAM_dat_ideal.sh — identical to lowRAM_dat except the
                              NaN-after-gridding guard is commented out; needed for
                              the ideal (eta = 0) runs whose converted output
                              tripped that guard.

(bhac_to_fno_lowRAM.sh, pick_up_at_blk.sh, temporaer.sh, and the leftover
run_BHAC_vanilla.sh / run_simple.sh / restart.sh / amrvac_restart.par copies
from the simulation directory were removed 2026-07-03 as stale/unused — see
git history.)

Provenance (VERIFIED against the produced dataset files, 2026-07-03):
every converter variant leaves a fingerprint in the h5 ('fields' chunk layout:
the lowRAM variants set explicit chunks (1,1,ny,nx) / (1,C,ny,nx); bhac_to_fno.sh
lets h5py auto-chunk). Scanning all 185 files
under datagen/finished* showed:
- ALL final datasets — level 1 (256^2, finished/ + finished_256_zusatz/), level 2
  (512^2, finished_512/ + finished_512_zusatz/) and level 3 (1024^2,
  finished_1024/) — have h5py auto-chunks and 16 channels: written by
  bhac_to_fno.sh in this directory (.dat era).
  Level 1 is the resolution used by the base config (config/mhd_bhac.yaml);
  level 2 is used by config/stride8_t9_25_t10_512p{,_zusatz}.yaml and
  config/coarse8_t9_25_t10_512p{,_zusatz}.yaml (fno_uniform_level2.h5).
- No final dataset file was written by a lowRAM variant; finished_4096/ is empty,
  i.e. the level-5 conversions (highly likely, per user's recollection:
  bhac_to_fno_lowRAM_dat.sh, needed because bhac_to_fno.sh OOMs at that level;
  both carry the SLURM job name 75_5 = eta 7e-5, level 5) never became a
  collected dataset.
- All files: 401 snapshots, dt = 0.025, eta channel constant per file and matching
  ../resistivities_used.md exactly (run output-00NN = datagen dir NN).

Build notes
-----------
The BHAC user files here differ from the simulation directory on purpose:
- definitions.h has #define HDF5. This is a leftover from an early, abandoned
  attempt to let BHAC write HDF5 snapshots and convert those directly: that does
  not work — BHAC must output .dat snapshots (which this converter re-reads). The
  flag is harmless (the .dat path ignores it); the simulation dir keeps
  #undefine HDF5.
- mod_indices.t enlarges ngridshi to 1.8e6 (and nlevelshi to 13) so the convert
  restart can hold the fully refined grid of the big runs.

Quick-look
----------
plot_fno_h5.py and plot_rho_lfac.py render per-timestep PNGs from a produced
fno_uniform_level<L>.h5 for a fast sanity check of the conversion (pick a field
via --var; see important_commands.md for an example).
