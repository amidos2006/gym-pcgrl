#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --mem=40GB
#SBATCH --job-name=pcgrl
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=sam.earle@nyu.edu
#SBATCH --output=pcgrl_%j.out

cd /scratch/se2161/pcgrl
#conda init bash
source activate

while getopts c: flag
do
    case "$flag}" in
        e) conditionals=${OPTARG};;
    esac
done

singularity exec python train.py --conditionals $experiment

#python train.py --conditionals "regions"
#python train.py --conditionals "path-length"
#python train.py --conditionals "regions" "path-length"

