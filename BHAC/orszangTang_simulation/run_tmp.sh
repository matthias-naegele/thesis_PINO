#!/bin/bash
#SBATCH -J 11_datag
#SBATCH -p small_cpu
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=2-00:00:00
#SBATCH --mem=256G
#SBATCH --tmp=500G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=matthias.naegele@mail.de

OUTROOT="$HOME/data/BHAC_output/orszagTang/converge/datagen/11"
TMPROOT="/tmp/mnaegele/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"

rm -r ./output*
mkdir output-0000

# tmp
mkdir -p "$TMPROOT"
cp -r ./ "$TMPROOT/"
cd "$TMPROOT"

ls
ls $BHAC_DIR

$BHAC_DIR/setup.pl -d=23 -phi=2 -z=3 -g=12,12 -p=rrmhd -eos=default -nf=0 -ndust=0 -u=nul -coord=cart -arch=gfortran10_hdf5

export OMPI_MCA_btl="^smcuda"

make clean && make

mpirun --mca btl ^smcuda --np "${SLURM_NTASKS}" ./bhac -i amrvac.par

mkdir -p "$OUTROOT/inputs"
cp -r ./output-0000 "$OUTROOT/"
cp amrvac.par "$OUTROOT/inputs/"
