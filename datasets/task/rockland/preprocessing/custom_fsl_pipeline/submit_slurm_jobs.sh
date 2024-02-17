#!/bin/bash
#SBATCH --job-name=NKI_TEST    # TODO: why is this called TEST?
#SBATCH --time=10:00:00
#SBATCH --output=/mnt/Data/neuro-dynamic-covariance/datasets/task/rockland/slurm_logs/custom_fsl_pipeline_preprocessing/CHECKERBOARD645/%A_%a.out
#SBATCH --error=/mnt/Data/neuro-dynamic-covariance/datasets/task/rockland/slurm_logs/custom_fsl_pipeline_preprocessing/CHECKERBOARD645/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1-296%20

CohortRawDataDir=/mnt/Data/neuro-dynamic-covariance/datasets/task/rockland/CHECKERBOARD645/raw

# this is zero-indexed, but the folder itself (i.e. "raw") takes up the first (zero) index
dirs=( $(find $CohortRawDataDir -maxdepth 1 -type d -exec basename {} \; | sort -u) )
for dir in "${dirs[@]}"; do
    echo "$((i++)) $dir"
done

SubID=${dirs[$SLURM_ARRAY_TASK_ID]}  # SLURM_ARRAY_TASK_ID is one-indexed (not zero-indexed)
SesID=ses-BAS1  # baseline 1 visit
TASK=CHECKERBOARD
TR=645  # in ms
echo ""
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "SubID:               $SubID"

echo "Submitting: $SLURM_ARRAY_TASK_ID -- $SubID $SesID $TASK"
srun --cpu_bind=threads --distribution=block:block bash datasets/task/rockland/preprocessing/custom_fsl_pipeline/nki_bids_mpp.sh "$SubID" $SesID $TR $TASK
