#!/bin/bash
set -e

# nki_bids_mpp.sh <SubID> <SesID> <TR> <TASK>

# INPUTS:
# All paths assume BIDS structure
#   SubID         : BIDS format SubjectID (including sub-, e.g. sub-A00001234)
#   SesID         : BIDS format Session ID (including ses-, e.g. ses-BAS1)
#   TR            : TR in ms
#   TASK          : TaskID

# OUTPUTS:
#   FSL format output directories

# Workflow:
#   Step 1) Brain extraction (using FSL BET)
#   Step 2) Fixing field of view & orientation
#   Step 2) FEAT registration/alignment (including linear transformation EPI (the functional images) <> Anat using BBR (brain boundary registration))
#   Step 3) Non-linear registration to MNI 2mm using FNIRT
#   Step 4) Motion correction (MC) using MCFLIRT, aligned to first scan
#   Step 5) Smoothing using a Gaussian kernel (5mm) + fixing for boundary issues due to smoothing.
#   Step 6) High-pass filtering (de-trending) using a moving Gaussian kernel of 100s
#   Step 7) ICA-AROMA (fancy de-noising) -- assumes Python3.8+ is used (in a conda environment)
#     This extracts IC components and determines automatically which ones can be regressed out of the data.
#
# SA, JZ, 2021
# --------------------------------------------

pwd
printf 'Using Python %s\n\n' "$(which python)"

SubID=$1
SesID=$2
TR=$3
AQLAB=$4  # sub-A00008326_ses-BAS1_task-CHECKERBOARD_acq-645_bold.nii.gz -- sub-A00008326_ses-DS2_645_bold

flag_feat1=0;  # TODO: why are we not running this step?

FWHMl=5  # smoothing parameter, in mm
HPFs=100 # High pass filter in seconds

coblar=0 # for testing

#
# set paths
#

ICAAROMADir="/mnt/Data/Toolboxes/ICA-AROMA"
export FSLDir="/usr/local/fsl"
. ${FSLDir}/etc/fslconf/fsl.sh

Path2Act=/mnt/Data/neuro-dynamic-covariance/datasets/task/rockland

RawImageDir=${Path2Act}/CHECKERBOARD${TR}/raw/${SubID}/${SesID}
AnatImgFileName=${SubID}_${SesID}_T1w
AnatImgFilePath=${RawImageDir}/anat/${AnatImgFileName}
FuncImgFileName=${SubID}_${SesID}_task-${AQLAB}_acq-${TR}_bold
FuncImgFilePath=${RawImageDir}/func/${FuncImgFileName}

StandardImgFilePath=${FSLDir}/data/standard/MNI152_T1_2mm_brain

fMRIPREP=${Path2Act}/CHECKERBOARD${TR}/custom_fsl_pipeline/mpp/${SubID}/${SesID}/${FuncImgFileName}_mpp

ica_dir=${fMRIPREP}/ica-aroma_fwhm${FWHMl}

echo "AnatImgFilePath:     ${AnatImgFilePath}"
echo "FuncImgFilePath:     ${FuncImgFilePath}"
echo "StandardImgFilePath: ${StandardImgFilePath}"
echo "ResultsDir:          ${fMRIPREP}"

if [ "$flag_feat1" == 0 ]; then
	[ "$coblar" == 1 ] && rm -rf "${fMRIPREP}"
fi

mkdir -p "${fMRIPREP}"
cd "${fMRIPREP}"  # This is necessary for FOV and STEP 4.
cp "${AnatImgFilePath}".nii.gz "${fMRIPREP}"/  # Copy the anatomical image to the mpp directory.

#
# start preprocessing
#

printf "\n%s > STEP 1: brain extraction (BET) (from structural scan)...\n" "$(date)"
# TODO: this gives two 'Operation not permitted' errors at the moment (though it seems harmless?)
${FSLDir}/bin/bet "${AnatImgFilePath}".nii.gz "${fMRIPREP}"/"${AnatImgFileName}"_brain.nii.gz -f 0.3 -o -m -n -R -S -B
printf "%s > STEP 1 (BET) completed.\n" "$(date)"

# Clean up the field-of-view (FOV).
${FSLDir}/bin/fslmaths "${FuncImgFilePath}" prefiltered_func_data -odt float
TotalVol=$(${FSLDir}/bin/fslnvols prefiltered_func_data)
TargetVol=$(bc -l <<< "$TotalVol/2")
printf '\nTargetVol: %s\n' "${TargetVol}"
${FSLDir}/bin/fslroi prefiltered_func_data example_func "$TargetVol" 1

if [ "$flag_feat1" == 1 ]; then
  printf "\n> STEP 2: FEAT registration\n"
	# Do a boundary-based registration (BBR).
	echo "> Running the first bit of FEAT..."
	${FSLDir}/bin/mainfeatreg \
	-F 6.00 \
	-d ${fMRIPREP} \
	-l ${fMRIPREP}/feat2_pre \
	-R ${fMRIPREP}/report_unwarp.html \
	-r ${fMRIPREP}/report_reg.html  \
	-i ${fMRIPREP}/example_func.nii.gz  \
	-n 10 \
	-h ${AnatImgFileName}_brain \
	-w BBR -x 90 \
	-s ${StandardImgFilePath} -y 12 -z 90
else
	echo "> Not running FEAT registration."
fi

printf "%s > STEP 3: non-linear registration to MNI space.\n" "$(date)"
# ${FSLDir}/bin/imcp
bash /home/"$(whoami)"/git_repos/neuro-dynamic-covariance/datasets/task/rockland/preprocessing/custom_fsl_pipeline/reg_fun2mni.sh "${fMRIPREP}"/"${AnatImgFileName}"
printf "%s > STEP 3 (non-linear registration) completed.\n" "$(date)"

printf "%s > STEP 4: running the second bit of FEAT: motion correction (MC) (i.e. inter-slice correction)...\n" "$(date)"
${FSLDir}/bin/mcflirt -in prefiltered_func_data -out prefiltered_func_data_mcf -mats -plots -reffile example_func -rmsrel -rmsabs -spline_final
${FSLDir}/bin/fsl_tsplot -i prefiltered_func_data_mcf.par -t 'MCFLIRT estimated rotations (radians)' -u 1 --start=1 --finish=3 -a x,y,z -w 640 -h 144 -o rot.png
${FSLDir}/bin/fsl_tsplot -i prefiltered_func_data_mcf.par -t 'MCFLIRT estimated translations (mm)' -u 1 --start=4 --finish=6 -a x,y,z -w 640 -h 144 -o trans.png
${FSLDir}/bin/fsl_tsplot -i prefiltered_func_data_mcf_abs.rms,prefiltered_func_data_mcf_rel.rms -t 'MCFLIRT estimated mean displacement (mm)' -u 1 -w 640 -h 144 -a absolute,relative -o disp.png
${FSLDir}/bin/fslmaths prefiltered_func_data_mcf -Tmean mean_func
${FSLDir}/bin/bet2 mean_func mask -f 0.3 -n -m
${FSLDir}/bin/immv mask_mask mask
${FSLDir}/bin/fslmaths prefiltered_func_data_mcf -mas mask prefiltered_func_data_bet
rm prefiltered_func_data_mcf.nii.gz prefiltered_func_data.nii.gz
printf "%s > STEP 4 (motion correction) completed.\n" "$(date)"

printf "%s > STEP 5: smoothing...\n" "$(date)"
OrigImgFilePath=${fMRIPREP}/prefiltered_func_data_bet
FinalResultFilePath=${OrigImgFilePath}_fwhm${FWHMl}
MaskImgFilePath=${fMRIPREP}/mask
smoothparSig=$(bc -l <<< "${FWHMl}/2.3548")

echo "-IN:   ${OrigImgFilePath}"
echo "-OUT:  ${FinalResultFilePath}"
echo "-MASK: ${MaskImgFilePath}"
echo "-KERNEL(mm): ${FWHMl} , sigma: ${smoothparSig}"

# Take care of the smoothing effect on the edges.
${FSLDir}/bin/fslmaths "$OrigImgFilePath" -s "$smoothparSig" -mas "$MaskImgFilePath" tmp_result1_tmp
${FSLDir}/bin/fslmaths "$MaskImgFilePath" -s "$smoothparSig" -mas "$MaskImgFilePath" "$FinalResultFilePath"
${FSLDir}/bin/fslmaths tmp_result1_tmp -div "$FinalResultFilePath" "$FinalResultFilePath"
${FSLDir}/bin/imrm tmp_result1_tmp
printf "%s > STEP 5 (smoothing) completed.\n" "$(date)"

printf "%s > STEP 6: running high-pass filter...\n" "$(date)"
hpfsigma=$(bc -l <<< "1/(2*${TR}*1/${HPFs})")  # sigma in volumes
${FSLDir}/bin/fslmaths "${FinalResultFilePath}" -Tmean tempMean
${FSLDir}/bin/fslmaths "${FinalResultFilePath}" -bptf "${hpfsigma}" -1 -add tempMean "${FinalResultFilePath}"
${FSLDir}/bin/imrm tempMean
printf "%s > STEP 6 (high-pass filtering) completed.\n" "$(date)"

printf "%s > STEP 7: de-noising using ICA-AROMA...\n" "$(date)"
rm -rf "${ica_dir}"
python ${ICAAROMADir}/ICA_AROMA.py \
-in "${FinalResultFilePath}".nii.gz \
-affmat "${fMRIPREP}"/reg/example_func2standard.mat \
-mc "${fMRIPREP}"/prefiltered_func_data_mcf.par \
-out "${ica_dir}" \
-den both
printf "%s > STEP 7 (ICA-AROMA de-noising) completed.\n" "$(date)"
