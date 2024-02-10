#!/bin/bash
set -e

# preprocess_single_subject.sh <SubID> <SesID> <TR> <TASK>

# INPUTS:
# All paths assume BIDS structure
#   SubID         : BIDS format SubjectID (including sub-, e.g. sub-A00001234)
#   SesID         : BIDS format Session ID (including ses-, e.g. ses-BAS1)
#   TR            : TR in ms
#   TASK          : TaskID

# OUTPUTS:
#   FSL format output directories

# This runs the standard FSL preprocessing pipeline as used in the FSL GUI
# This is a script to run the FLS GUI on a single subject, instead of running them one-by-one in the GUI
# That is, in Step 2 we replace the GUI workflow with the commands written here
# All relevant parameters such as smoothing parameter will be stored in the design file
# This assumes your data is already in `.nii.gz` format (i.e. not in `.nii` format!)
# It also assumes your data is in BIDS format
#
# --------------------------------------------

pwd
printf 'Using Python %s\n\n' "$(which python)"

SubID=$1
SesID=$2
TR=$3

FWHMl=5  # smoothing parameter, in mm

#
# set paths
#

ICAAROMADir="/mnt/Data/Toolboxes/ICA-AROMA"
export FSLDir="/usr/local/fsl"
. ${FSLDir}/etc/fslconf/fsl.sh

Path2Act=/mnt/Data/neuro-dynamic-covariance/datasets/task/rockland
GitBaseDir=/home/opk20/git_repos/neuro-dynamic-covariance

RawImageDir=${Path2Act}/CHECKERBOARD${TR}/raw/${SubID}/${SesID}
AnatImgFileName=${SubID}_${SesID}_T1w
AnatImgFilePath=${RawImageDir}/anat/${AnatImgFileName}
FuncImgFileName=${SubID}_${SesID}_task-CHECKERBOARD_acq-${TR}_bold
FuncImgFilePath=${RawImageDir}/func/${FuncImgFileName}

StandardImgFilePath=${FSLDir}/data/standard/MNI152_T1_2mm_brain

ResultsBaseDir=${Path2Act}/CHECKERBOARD${TR}/default_fsl_pipeline/${SubID}/${SesID}/${FuncImgFileName}_mpp

ica_dir=${ResultsBaseDir}/ica-aroma_fwhm${FWHMl}

echo "AnatImgFilePath:     ${AnatImgFilePath}"
echo "FuncImgFilePath:     ${FuncImgFilePath}"
echo "StandardImgFilePath: ${StandardImgFilePath}"
echo "ResultsDir:          ${ResultsBaseDir}"

mkdir -p "${ResultsBaseDir}"
cd "${ResultsBaseDir}"  # This is necessary for FOV and STEP 4.
cp "${AnatImgFilePath}".nii.gz "${ResultsBaseDir}"/  # Copy the anatomical image to the results directory.

# Step 1a: reorient anatomical (T1) image
fslreorient2std "${AnatImgFileName}.nii.gz" "${AnatImgFileName}_reorient.nii.gz"

# Step 1b: remove the neck area
robustfov -i "${AnatImgFileName}_reorient.nii.gz" -r "${AnatImgFileName}_reorient_FOV.nii.gz"

# Step 1c: brain extraction on the anatomical image
bet "${AnatImgFileName}_reorient_FOV.nii.gz" "${AnatImgFileName}_reorient_FOV_brain.nii.gz" -f 0.4 -R -m

# Step 2: create new design file, replacing subject/session number
# We need to change 3 things: the output directory, the rs-fMRI image path, and the T1 image path.
more ${GitBaseDir}/datasets/task/rockland/preprocessing/default_fsl_pipeline/template_design.fsf | sed s/sub-A01234567/"${SubID}"/g > design.fsf
# Remove first 3 lines that are somehow added?
sed -i -e 1,3d design.fsf

# Step 2: run preprocessing for the new setup/subject
cat design.fsf
feat design.fsf

# Step 3: de-noising using ICA-AROMA
#rm -rf "${ica_dir}"
#python ${ICAAROMADir}/ICA_AROMA.py \
#-in "${FinalResultFilePath}".nii.gz \
#-affmat "${ResultsBaseDir}"/reg/example_func2standard.mat \
#-mc "${ResultsBaseDir}"/prefiltered_func_data_mcf.par \
#-out "${ica_dir}" \
#-den both
