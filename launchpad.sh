#!/bin/bash

#SBATCH -A danielk_gpu
#SBATCH --partition a 100
#SBATCH --qos=qos_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --job-name="HW6 CS 601.471/671 homework"


module load anaconda

# init virtual environment if needed
conda create -n toy_classification_env python=3.7

conda activate toy_classification_env # open the Python environment

pip install -r requirements.txt --no-cache-dir # install Python dependencies
conda install -c pytorch faiss-gpu cudatoolkit=10.2 
pip install torch --no-cache-dir 
# runs your code
srun python hw7-work.py --experiment "overfit" --small_subset --device cuda --model "facebook/rag-token-nq" --batch_size "4" --lr 1e-4 --num_epochs 20