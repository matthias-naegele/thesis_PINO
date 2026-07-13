
##################################################################################################

Orszag-Tang vortex — BHAC setup used to generate the FNO training data

Orszag, SA, Tang, C-M: Small-scale structure of two-dimensional magnetohydrodynamic turbulence.
J. Fluid Mech. 90, 129-143 (1979). doi:10.1017/S002211207900210X

O. Porth, H. Olivares, Y. Mizuno, Z. Younsi, L. Rezzolla, M. Moscibrodzka, H. Falcke, and
M. Kramer. The black hole accretion code. Computational Astrophysics and Cosmology,
4:1, May 2017. doi: 10.1186/s40668-017-0020-2.

##################################################################################################

Purpose
-------
This directory is one example of the resistive SRMHD Orszag-Tang simulations that
produced the FNO training/validation dataset. Each dataset simulation is a copy of
this directory with a different resistivity eta0 in amrvac.par (&usrlist); this copy
was "datagen" run 11 (eta0 = 4.69e-04, a train-set value — see ../resistivities_used.md).
Run parameters: 2pi x 2pi periodic box, 256^2 base grid, mxnest=5, tmax=10,
dtsave(2)=0.025 -> 401 .dat snapshots per run.

The finished runs are turned into uniform-grid HDF5 files for the FNO by the scripts
in ../converter_flatten_to_h5/. The OUTROOT this launcher archives to
($HOME/data/BHAC_output/orszagTang/converge/datagen/<N>, holding output-0000/ and
inputs/amrvac.par) is exactly the RUN_ROOT the converter reads back.

Resistivity wiring (verified, incl. numerically on the produced data)
----------------------------------------------------------------------
Numerical verification (2026-07-03, on the converted datagen h5 files): runs
start from bitwise-identical initial data and differ only in eta0; at t=5 two
runs with eta = 1.08e-4 vs 1.10e-4 differ by 0.27% in rho (rel. L2) while
eta = 1.10e-4 vs 9.87e-4 differ by 28% — the dynamics respond strongly and
monotonically to eta. In addition, the stored jz channel satisfies the
resistive Ohm's law e3 - (v x B)_3 = eta * jz / lfac with the file's eta
exactly (getcurrent computes J from the Ohm inversion of the rrmhd module),
confirming the resistive module with eqpar(eta_) = eta0 produced the data.

- The makefile and all launchers build PHYSICS=rrmhd, BHAC's *resistive* relativistic
  MHD module: the state vector evolves the electric fields (e1,e2,e3), and amrvac.par
  selects the IMEX integrator (typeadvance='ImEx12') plus typeinversionresis, which
  only exist for the stiff resistive Ohm's-law source term. The runs are genuinely
  resistive, not silently ideal.
- eta0 is read from &usrlist in ./amrvac.par and assigned to eqpar(eta_), the uniform
  resistivity of the rrmhd module (initglobaldata_usr in amrvacusr.t). eta is in
  natural (code) units.
- eta is additionally written into every snapshot as an output channel ('eta',
  specialvar_output). The FNO dataloader reads its eta conditioning from that channel,
  so the value the network sees is always the one the simulation actually ran with.

CAUTION: initglobaldata_usr opens the *hard-coded* file name './amrvac.par', ignoring
the parfile passed via `-i`. Any run, restart, or convert invocation must have the
correct amrvac.par present in the working directory (the launchers and the converter
take care of this by copying/symlinking their parfile onto amrvac.par).

Which launcher is which
-----------------------
Current:
- run_tmp.sh    — the dataset ("datagen") launcher used for the training runs
                  (job name 11_datag; archives output-0000/ plus inputs/amrvac.par
                  to ~/data/BHAC_output/orszagTang/converge/datagen/<N>, the layout
                  the converter expects). Runs on node-local /tmp.

(run_BHAC_vanilla.sh, restart.sh, run_simple.sh and an old bhac_to_fno.sh copy
were removed 2026-07-03 as stale, unused leftovers — see git history for their
content and the bugs/gotchas they had.)

