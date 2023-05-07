#!/bin/bash -l 
 
#SBATCH -t 1:00:00 
#SBATCH -N 1 
#SBATCH -J J:alde.in 
#SBATCH -n 2 
 
export SCHRODINGER_TMPDIR=/central/scratch/musgrave/
export TMPDIR=/central/scratch/musgrave/
/central/groups/wag/programs/Schrodinger_2022_3/jaguar run -TMPDIR /central/scratch/musgrave -WAIT -PARALLEL 2  alde.in 
