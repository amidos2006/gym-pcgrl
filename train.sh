#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
##SBATCH --gres=gpu:1
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



### BINARY ###


## NARROW


## TURTLE

#python train.py --problem "binarygoal" --conditionals "regions" --representation "turtle"
#python train.py --problem "binarygoal" --conditionals "path-length" --representation "turtle"
#python train.py --problem "binarygoal" --conditionals "regions" "path-length" --representation "turtle"


## WIDE


## WIDE - CA

#python train.py --problem "binarygoal" --conditionals "regions" "path-length" --representation "wide" --ca_action



### ZELDA ###


## NARROW

#python train.py --problem "zeldagoal" --conditionals "enemies" --representation "narrow"
#python train.py --problem "zeldagoal" --conditionals "path-length" --representation "narrow"
#python train.py --problem "zeldagoal" --conditionals "enemies" "path-length" --representation "narrow"
#python train.py --problem "zeldagoal" --conditionals "nearest-enemy" --representation "narrow"
#python train.py --problem "zeldagoal" --conditionals "player" "key" "door" "enemies" "regions" "nearest-enemy" "path-length" --representation "narrow"

## TURTLE

#python train.py --problem "zeldagoal" --conditionals "enemies" --representation "turtle"
#python train.py --problem "zeldagoal" --conditionals "path-length" --representation "turtle"
#python train.py --problem "zeldagoal" --conditionals "enemies" "path-length" --representation "turtle"
#python train.py --problem "zeldagoal" --conditionals "nearest-enemy" --representation "turtle"
#python train.py --problem "zeldagoal" --conditionals "player" "key" "door" "enemies" "regions" "nearest-enemy" "path-length" --representation "turtle"

## WIDE

#python train.py --problem "zeldagoal" --conditionals "player" "key" "door" "enemies" "regions" "nearest-enemy" "path-length" --representation "wide"

## WIDE - CA

python train.py --problem "zeldagoal" --conditionals "player" "key" "door" "enemies" "regions" "nearest-enemy" "path-length" --representation "wide" --ca_action

### SOKOBAN

## NARROW

#python train.py --problem "sokobangoal" --conditionals "player" "crate" "sol-length" --representation "narrow"

## TURTLE

#python train.py --problem "sokobangoal" --conditionals "crate" --representation "turtle"
#python train.py --problem "sokobangoal" --conditionals "sol-length" --representation "turtle"
#python train.py --problem "sokobangoal" --conditionals "crate" "sol-length" --representation "turtle"
#python train.py --problem "sokobangoal" --conditionals "player" "crate" "sol-length" --representation "turtle"

## WIDE

#python train.py --problem "sokobangoal" --conditionals "crate" --representation "wide"
#python train.py --problem "sokobangoal" --conditionals "sol-length" --representation "wide"
#python train.py --problem "sokobangoal" --conditionals "crate" "sol-length" --representation "wide"
#python train.py --problem "sokobangoal" --conditionals "player" "crate" "sol-length" --representation "wide"
 
## WIDE - CA

#python train.py --problem "sokobangoal" --conditionals "crate" --representation "wide" --ca_action
#python train.py --problem "sokobangoal" --conditionals "sol-length" --representation "wide" --ca_action
#python train.py --problem "sokobangoal" --conditionals "crate" "sol-length" --representation "wide" --ca_action
#python train.py --problem "sokobangoal" --conditionals "player" "crate" "sol-length" --representation "wide" --ca_action
