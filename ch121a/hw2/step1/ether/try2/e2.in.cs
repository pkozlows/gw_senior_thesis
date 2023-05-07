#!/bin/bash -l 
 
#SBATCH -t 5:00:00 
#SBATCH -N 1 
#SBATCH -J J:e2.in 
#SBATCH -n 1 
 
export SCHRODINGER_TMPDIR=/central/scratch/musgrave/
export TMPDIR=/central/scratch/musgrave/
/central/groups/wag/programs/Schrodinger_2022_3/jaguar run -TMPDIR /central/scratch/musgrave -WAIT -PARALLEL 1  e2.in 
