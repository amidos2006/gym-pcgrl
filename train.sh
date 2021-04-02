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
conda activate pcgrl

### BINARY

## TURTLE

#python train.py --conditionals "regions"
#python train.py --conditionals "path-length"
#python train.py --conditionals "regions" "path-length"

### ZELDA

## TURTLE

#python train.py --problem "zeldagoal" --conditionals "enemies" --representation "turtle"
#python train.py --problem "zeldagoal" --conditionals "path-length" --representation "turtle"
#python train.py --problem "zeldagoal" --conditionals "enemies" length" --representation "turtle"

## NARROW

python train.py --problem "zeldagoal" --conditionals "enemies" --representation "narrow"
#python train.py --problem "zeldagoal" --conditionals "path-length" --representation "narrow"
#python train.py --problem "zeldagoal" --conditionals "path-length" --representation "narrow"
