#!/bin/bash
#SBATCH -J 75_5_bhac_to_fno
#SBATCH -p small_cpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=2-00:00:00
#SBATCH --mem=256G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

############################################
# USER SETTINGS	 			sbatch --export=ALL,RUN_ROOT=$HOME/data/BHAC_output/orszagTang/2194918,LEVEL_IO=1,BHAC_EXE=$HOME/thesis/physicsnemo/examples/cfd/mhd_pino/bhac_runs/orszagTang/bhac bhac_to_fno.sh
############################################
RUN_ROOT="${RUN_ROOT:-$HOME/data/BHAC_output/orszagTang/2194918}"
BHAC_EXE="${BHAC_EXE:-$HOME/thesis/physicsnemo/examples/cfd/mhd_pino/bhac_runs/orszagTang/bhac}"
LEVEL_IO="${LEVEL_IO:-1}"          # 1 -> base grid (e.g. 128x128), 2 -> finest everywhere (e.g. 256x256 if mxnest=2)
KEEP_INTERMEDIATE="${KEEP_INTERMEDIATE:-0}"  # 1 keep converted text files, 0 delete after packing
############################################

echo "RUN_ROOT = ${RUN_ROOT}"
echo "BHAC_EXE = ${BHAC_EXE}"
echo "LEVEL_IO = ${LEVEL_IO}"

if [[ ! -x "${BHAC_EXE}" ]]; then
  echo "ERROR: BHAC_EXE not found or not executable: ${BHAC_EXE}" >&2
  exit 1
fi

# Find a template parfile (used only to keep physics/varname settings consistent)
TEMPLATE_PAR=""
if [[ -f "${RUN_ROOT}/inputs/amrvac.par" ]]; then
  TEMPLATE_PAR="${RUN_ROOT}/inputs/amrvac.par"
else
  shopt -s nullglob
  candidates=( "${RUN_ROOT}/inputs/"amrvac_*.par )
  shopt -u nullglob
  if [[ ${#candidates[@]} -gt 0 ]]; then
    TEMPLATE_PAR="${candidates[0]}"
  fi
fi

if [[ -z "${TEMPLATE_PAR}" || ! -f "${TEMPLATE_PAR}" ]]; then
  echo "ERROR: Could not find template parfile under ${RUN_ROOT}/inputs" >&2
  exit 1
fi
echo "TEMPLATE_PAR = ${TEMPLATE_PAR}"

# Patch a key=value in a namelist-style parfile
patch_parfile() {
  local par_path="$1"
  local key="$2"
  local value="$3"
  python3 - <<PY
from pathlib import Path
import re

p = Path("${par_path}")
lines = p.read_text().splitlines()

key = "${key}"
val = "${value}"

pat = re.compile(rf"^(\\s*{re.escape(key)}\\s*=\\s*).*$")
done = False
for i, line in enumerate(lines):
    m = pat.match(line)
    if m:
        lines[i] = f"{m.group(1)}{val}"
        done = True
        break

# If key not present, insert it into &filelist (best-effort)
if not done:
    out = []
    inserted = False
    for line in lines:
        out.append(line)
        if (not inserted) and line.strip().lower().startswith("&filelist"):
            out.append(f"        {key} = {val}")
            inserted = True
    lines = out

p.write_text("\\n".join(lines) + "\\n")
PY
}

convert_dir() {
  local outdir="$1"
  echo "------------------------------------------------------------"
  echo "Converting directory: ${outdir}"

  # Must run inside outdir to keep paths short
  pushd "${outdir}" >/dev/null

  # Confirm snapshots exist
  shopt -s nullglob
  local snaps=( data[0-9][0-9][0-9][0-9].dat )
  shopt -u nullglob
  if [[ ${#snaps[@]} -eq 0 ]]; then
    echo "No data????.dat found; skipping ${outdir}"
    popd >/dev/null
    return
  fi

  local blk_dir="fno_blk_level${LEVEL_IO}"
  local fno_dir="fno"
  mkdir -p "${blk_dir}" "${fno_dir}"

  # Build conversion parfile with SHORT RELATIVE filenameini/filenameout
  local conv_par="convert_fno_level${LEVEL_IO}.par"
  cp -f "${TEMPLATE_PAR}" "${conv_par}"

  patch_parfile "${conv_par}" "filenameini"  "'data'"
  patch_parfile "${conv_par}" "filenameout"  "'${blk_dir}/data'"
  patch_parfile "${conv_par}" "autoconvert"  "F"
  patch_parfile "${conv_par}" "write_xdmf"   "F"

  # Restart from dat snapshots
  patch_parfile "${conv_par}" "hdf5_ini"     "F"
  # Uniformize at convert stage
  patch_parfile "${conv_par}" "level_io"     "${LEVEL_IO}"

  # Use oneblock ASCII output (easy to regrid/pack for ML)
  patch_parfile "${conv_par}" "convert"      ".true."
  # 'oneblock
  patch_parfile "${conv_par}" "convert_type" "'oneblock'"
  patch_parfile "${conv_par}" "saveprim"     ".true."
  patch_parfile "${conv_par}" "writew" "11*.true., .false., 2*.true."
  ln -sf "${conv_par}" amrvac.par
  # Convert each snapshot index
  for f in "${snaps[@]}"; do
    local idx="${f##*data}"
    idx="${idx%.dat}"
    idx=$((10#${idx}))
    echo "  -> converting restart ${idx} (${f})"
    mpirun -np 1 "${BHAC_EXE}" -i "${conv_par}" -restart "${idx}" -convert
  done

  # Pack "oneblock" outputs into FNO HDF5:
  # Writes: fno/fno_uniform_level${LEVEL_IO}.h5 with datasets:
  # fields[T,C,Ny,Nx], x[Nx], y[Ny], t[T], varnames[C]
  python3 - <<'PY'
import os, re, glob
import numpy as np
import h5py

level_io = int(os.environ["LEVEL_IO"])
blk_dir = f"fno_blk_level{level_io}"
fno_dir = "fno"
os.makedirs(fno_dir, exist_ok=True)

# oneblock usually produces one ASCII file per snapshot. Extension varies across forks/builds.
# Grab everything matching data####.* except the input dat.
candidates = sorted(glob.glob(os.path.join(blk_dir, "data[0-9][0-9][0-9][0-9].*")))
candidates = [p for p in candidates if not p.endswith(".dat")]

if not candidates:
    # fallback: any file with 4 digits
    candidates = sorted(glob.glob(os.path.join(blk_dir, "*[0-9][0-9][0-9][0-9]*")))

if not candidates:
    raise SystemExit(f"No converted files found under {blk_dir}. Check BHAC convert_type output naming.")

def snap_index(path: str) -> int:
    m = re.search(r"(\d{4})", os.path.basename(path))
    return int(m.group(1)) if m else -1

candidates = sorted(candidates, key=snap_index)

def read_oneblock(path: str):
    # oneblock is ASCII; format is typically:
    # line1: x y <var1> <var2> ...
    # line2: Ntot Nx Ny (or similar)
    # line3: time
    # rest: columns per cell
    with open(path, "r") as f:
        def next_nonempty():
            while True:
                line = f.readline()
                if not line:
                    return None
                s = line.strip()
                if s and not s.startswith("#"):
                    return s

        header = next_nonempty()
        if header is None:
            raise ValueError(f"Empty file: {path}")
        header_cols = header.split()
        if len(header_cols) < 4:
            raise ValueError(f"Unexpected header in {path}: {header_cols}")

        # assume 2D: first two columns are coords
        varnames = header_cols[2:]

        dims = next_nonempty()
        if dims is None:
            raise ValueError(f"Missing dims line in {path}")
        dims_cols = dims.split()
        if len(dims_cols) < 3:
            raise ValueError(f"Unexpected dims line in {path}: {dims_cols}")

        # robust parse: allow floats in the dims line
        ntot = int(float(dims_cols[0]))
        nx = int(float(dims_cols[1]))
        ny = int(float(dims_cols[2]))

        tline = next_nonempty()
        if tline is None:
            raise ValueError(f"Missing time line in {path}")
        t = float(tline)

        data = np.loadtxt(f, dtype=np.float32)
        if data.ndim == 1:
            data = data[None, :]

        # Expect N rows = ntot ~ nx*ny
        if data.shape[0] not in (ntot, nx * ny):
            raise ValueError(f"Row count mismatch in {path}: got {data.shape[0]}, expected {ntot} or {nx*ny}")

        x = data[:, 0]
        y = data[:, 1]
        vals = data[:, 2:]  # (N, C)

        C = vals.shape[1]
        if C != len(varnames):
            raise ValueError(f"Channel mismatch in {path}: vals has {C}, header has {len(varnames)}")

        ux = np.unique(x)
        uy = np.unique(y)

        # Sometimes nx/ny in file are swapped depending on ordering; trust uniques
        nx = ux.size
        ny = uy.size

        # Map each row into (j,i)
        dx = (ux[-1] - ux[0]) / (ux.size - 1)
        dy = (uy[-1] - uy[0]) / (uy.size - 1)
        i = np.round((x - ux[0]) / dx).astype(int)
        j = np.round((y - uy[0]) / dy).astype(int)

        grid = np.empty((C, ny, nx), dtype=np.float32)
        grid[:] = np.nan
        for k in range(C):
            grid[k, j, i] = vals[:, k]

        # if np.isnan(grid).any():
            # raise ValueError(f"NaNs after gridding {path}; oneblock output may not be strictly structured.")

        return t, ux.astype(np.float32), uy.astype(np.float32), varnames, grid

T = len(candidates)

t0, x_ref, y_ref, varnames_ref, grid0 = read_oneblock(candidates[0])
C, ny, nx = grid0.shape

out_h5 = os.path.join(fno_dir, f"fno_uniform_level{level_io}.h5")
with h5py.File(out_h5, "w") as h5:
    fields = h5.create_dataset(
        "fields",
        shape=(T, C, ny, nx),
        dtype=np.float32,
        chunks=(1, 1, ny, nx),
        compression="gzip",
        compression_opts=4,
    )
    times = h5.create_dataset("t", shape=(T,), dtype=np.float32)
    h5.create_dataset("x", data=x_ref)
    h5.create_dataset("y", data=y_ref)
    dt = h5py.string_dtype(encoding="utf-8")
    h5.create_dataset("varnames", data=np.array(varnames_ref, dtype=object), dtype=dt)
    h5.attrs["layout"] = "fields[T,C,Ny,Nx]"

    fields[0] = grid0
    times[0] = np.float32(t0)

    for n, path in enumerate(candidates[1:], start=1):
        t, ux, uy, varnames, grid = read_oneblock(path)

        if varnames != varnames_ref:
            raise ValueError(f"Variable list changed in {path}")
        if grid.shape != (C, ny, nx):
            raise ValueError(f"Grid shape changed in {path}: got {grid.shape}, expected {(C, ny, nx)}")
        if not (np.allclose(x_ref, ux) and np.allclose(y_ref, uy)):
            raise ValueError(f"Grid coordinates differ across snapshots (bad for FNO).")

        fields[n] = grid
        times[n] = np.float32(t)

print(f"Wrote {out_h5} with fields shape {(T, C, ny, nx)}")
PY
  LEVEL_IO="${LEVEL_IO}"

  if [[ "${KEEP_INTERMEDIATE}" != "1" ]]; then
    rm -f "${blk_dir}"/data[0-9][0-9][0-9][0-9].* 2>/dev/null || true
  fi

  echo "Done: ${outdir}/fno/fno_uniform_level${LEVEL_IO}.h5"
  popd >/dev/null
}

# Process all output-???? directories
shopt -s nullglob
outdirs=( "${RUN_ROOT}"/output-[0-9][0-9][0-9][0-9] )
shopt -u nullglob

if [[ ${#outdirs[@]} -eq 0 ]]; then
  echo "ERROR: No output-???? directories found under ${RUN_ROOT}" >&2
  exit 1
fi

for d in "${outdirs[@]}"; do
  convert_dir "${d}"
done

echo "All conversions complete."
