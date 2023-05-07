#!/usr/bin/bash

#SBATCH --time=1:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=1G   # memory per CPU core
#SBATCH -J "lammps"   # job name

source /etc/profile.d/modules.sh
cd $SLURM_SUBMIT_DIR

job=lammps

setenv LD_LIBRARY_PATH /central/groups/wag/programs/lammps/lammps-16Mar18/src:$LD_LIBRARY_PATH
setenv PATH /central/groups/wag/programs/lammps/lammps-16Mar18/src:$PATH
module unload intel/18.1
module load mpich/3.2.1 fftw/3.3.8 gcc/7.3.0
lmp=/central/groups/wag/programs/lammps/lammps-16Mar18/src/lmp_mpi

mpirun -np 1 $lmp -in in.lammps > out.${job}
