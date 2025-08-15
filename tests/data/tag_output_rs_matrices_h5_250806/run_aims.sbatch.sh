#!/bin/bash -l

### Job Name
#SBATCH --job-name test_bg_tags

### Standard output and error
#SBATCH --output=./%x.%j.out
#SBATCH --error=./%x.%j.err
###SBATCH --array=???
###SBATCH --output ./logs/%x.%A_%a.out
###SBATCH --error ./logs/%x.%A_%a.err

### Job resources, for mem use either

#SBATCH --partition p.ada
#SBATCH --nvmps
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=72
#SBATCH --gres=gpu:a100:4
#SBATCH --time=00:30:00

### Initial working dir, abs or rel to where sbatch is called from
#SBATCH --chdir ./

source ~/scripts/slurm_utils.sh
print_slurm_info
setup_parallel_env

source /u/zklou/projects/aims/250806/.env.aims.sh
export OMP_NUM_THREADS=1
export OMP_PLACES=cores  # for FHI-aims gpu version
ulimit -s unlimited

TASK_DPATH=/u/zklou/projects/2508_tbg_epc/tests/tag_output_rs_matrices_h5_250806

################ run FHI-aims calculation ################

cd ${TASK_DPATH}

echo "FHI-aims realpath: $(realpath ${AIMS_CUDA_HDF5})"
echo "[$(date)] Running AIMS at $(pwd)"
srun --export=ALL --ntasks=64 ${AIMS_CUDA_HDF5} > aims.out
if [[ $(grep -c "Have a nice day" aims.out) -eq 0 ]]; then
    echo "[$(date)] ERROR!!!!!! ${cal_idx}: not finished"
fi
echo "[$(date)] Finished AIMS at $(pwd)"

# 313141
