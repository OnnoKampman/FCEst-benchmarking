#bin/bash
set -e

# It is almost copy of what is happened in FSL FEAT
# SA, Ox, 2020

# We should already have example_func
# The output of this function will be example_func2standard.mat, but what is important is the warp files
# The warp files can be used later to take the whole 4D image into the standard space.
# OR they can be used to take the stat images into MNI

HEADIMG=$1

ImgDir=$(dirname ${HEADIMG})
REGDIR="${ImgDir}/reg"

STNDHEAD=${FSLDIR}/data/standard/MNI152_T1_2mm

echo ""
echo "HighRes : ${HEADIMG}"
echo "Reg dir : ${REGDIR}"
echo ""

mkdir -p "${REGDIR}"

TotalVol=$(${FSLDIR}/bin/fslnvols prefiltered_func_data)
TargetVol=$(bc -l <<< "$TotalVol/2")
echo "TargetVol: $TargetVol"
${FSLDIR}/bin/fslroi prefiltered_func_data ${REGDIR}/example_func $TargetVol 1

cd ${REGDIR}

${FSLDIR}/bin/fslmaths ${HEADIMG}_brain highres
${FSLDIR}/bin/fslmaths ${HEADIMG}  highres_head
${FSLDIR}/bin/fslmaths ${STNDHEAD}_brain standard
${FSLDIR}/bin/fslmaths ${STNDHEAD} standard_head
${FSLDIR}/bin/fslmaths ${STNDHEAD}_brain_mask_dil standard_mask

###################################################
# BBR: epi >> highres_head
###################################################
echo "> Running BBR..."
${FSLDIR}/bin/epi_reg --epi=example_func --t1=highres_head --t1brain=highres --out=example_func2highres
${FSLDIR}/bin/convert_xfm -inverse -omat highres2example_func.mat example_func2highres.mat

# Take a snapshot
${FSLDIR}/bin/slicer example_func2highres highres -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png
${FSLDIR}/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png example_func2highres1.png
${FSLDIR}/bin/slicer highres example_func2highres -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png
${FSLDIR}/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png example_func2highres2.png
${FSLDIR}/bin/pngappend example_func2highres1.png - example_func2highres2.png example_func2highres.png
/bin/rm -f sl?.png example_func2highres2.png

/bin/rm example_func2highres1.png

###################################################
# FLIRT: highres >> standard
###################################################
echo "FLIRT: highres >> standard"
${FSLDIR}/bin/flirt -in highres -ref standard -out highres2standard -omat highres2standard.mat \
-cost corratio -dof 12 -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -interp trilinear

###################################################
# FNIRT: highres >> standard
###################################################
echo "FNIRT: highres >> standard"
${FSLDIR}/bin/fnirt --iout=highres2standard_head --in=highres_head \
--aff=highres2standard.mat --cout=highres2standard_warp --iout=highres2standard --jout=highres2highres_jac \
--config=T1_2_MNI152_2mm --ref=standard_head --refmask=standard_mask --warpres=10,10,10

###################################################
# APPLYWARP: highres >> standard
###################################################
echo "APPLYWARP: highres >> standard"
${FSLDIR}/bin/applywarp -i highres -r standard -o highres2standard -w highres2standard_warp

${FSLDIR}/bin/convert_xfm -inverse -omat standard2highres.mat highres2standard.mat

# Take a snapshot
${FSLDIR}/bin/slicer highres2standard standard -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
${FSLDIR}/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png highres2standard1.png ; 
${FSLDIR}/bin/slicer standard highres2standard -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
${FSLDIR}/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png highres2standard2.png ; 
${FSLDIR}/bin/pngappend highres2standard1.png - highres2standard2.png highres2standard.png; 
/bin/rm -f sl?.png highres2standard2.png

/bin/rm highres2standard1.png

###################################################
# APPLYWARP: example_func >> standard
###################################################
echo "APPLYWARP: example_func >> standard"
${FSLDIR}/bin/convert_xfm -omat example_func2standard.mat -concat highres2standard.mat example_func2highres.mat
${FSLDIR}/bin/convertwarp --ref=standard --premat=example_func2highres.mat --warp1=highres2standard_warp --out=example_func2standard_warp

${FSLDIR}/bin/applywarp --ref=standard --in=example_func --out=example_func2standard --warp=example_func2standard_warp

${FSLDIR}/bin/convert_xfm -inverse -omat standard2example_func.mat example_func2standard.mat

${FSLDIR}/bin/slicer example_func2standard standard -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
${FSLDIR}/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png example_func2standard1.png ; 
${FSLDIR}/bin/slicer standard example_func2standard -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
${FSLDIR}/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png example_func2standard2.png ; 
${FSLDIR}/bin/pngappend example_func2standard1.png - example_func2standard2.png example_func2standard.png;
/bin/rm -f sl?.png example_func2standard2.png
