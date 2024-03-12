#
# Save TVFC estimates
#

python benchmarks/fmri/rs/HCP_PTN1200_recon2/train_models/save_TVFC_estimates.py d15 all correlation SVWP_joint
python benchmarks/fmri/rs/HCP_PTN1200_recon2/train_models/save_TVFC_estimates.py d15 all correlation sFC

python benchmarks/fmri/rs/HCP_PTN1200_recon2/train_models/save_TVFC_estimates.py d15 LEOO correlation SVWP_joint
python benchmarks/fmri/rs/HCP_PTN1200_recon2/train_models/save_TVFC_estimates.py d15 LEOO correlation sFC

#
# Save TVFC estimates summary measures
#

python benchmarks/fmri/rs/HCP_PTN1200_recon2/TVFC_summary_measures/save_TVFC_estimates_summary_measures.py d15 correlation SVWP_joint
python benchmarks/fmri/rs/HCP_PTN1200_recon2/TVFC_summary_measures/save_TVFC_estimates_summary_measures.py d15 correlation DCC_joint
python benchmarks/fmri/rs/HCP_PTN1200_recon2/TVFC_summary_measures/save_TVFC_estimates_summary_measures.py d15 correlation SW_cross_validated
python benchmarks/fmri/rs/HCP_PTN1200_recon2/TVFC_summary_measures/save_TVFC_estimates_summary_measures.py d15 correlation sFC

#
# Imputation benchmark
#

python benchmarks/fmri/rs/HCP_PTN1200_recon2/imputation_study/compute_LEOO_likelihoods.py d15 SVWP_joint multivariate
python benchmarks/fmri/rs/HCP_PTN1200_recon2/imputation_study/compute_LEOO_likelihoods.py d15 DCC_joint multivariate
python benchmarks/fmri/rs/HCP_PTN1200_recon2/imputation_study/compute_LEOO_likelihoods.py d15 SW_cross_validated multivariate
python benchmarks/fmri/rs/HCP_PTN1200_recon2/imputation_study/compute_LEOO_likelihoods.py d15 sFC multivariate

#
# Test-retest
#

python benchmarks/fmri/rs/HCP_PTN1200_recon2/test_retest/compute_I2C2_scores.py d15 SVWP_joint
python benchmarks/fmri/rs/HCP_PTN1200_recon2/test_retest/compute_I2C2_scores.py d15 DCC_joint
python benchmarks/fmri/rs/HCP_PTN1200_recon2/test_retest/compute_I2C2_scores.py d15 SW_cross_validated
python benchmarks/fmri/rs/HCP_PTN1200_recon2/test_retest/compute_I2C2_scores.py d15 sFC

python benchmarks/fmri/rs/HCP_PTN1200_recon2/test_retest/compute_ICC_edgewise_matrices.py d15 SVWP_joint
python benchmarks/fmri/rs/HCP_PTN1200_recon2/test_retest/compute_ICC_edgewise_matrices.py d15 DCC_joint
python benchmarks/fmri/rs/HCP_PTN1200_recon2/test_retest/compute_ICC_edgewise_matrices.py d15 SW_cross_validated
python benchmarks/fmri/rs/HCP_PTN1200_recon2/test_retest/compute_ICC_edgewise_matrices.py d15 sFC

python benchmarks/fmri/rs/HCP_PTN1200_recon2/test_retest/run_statistical_test_mean_ICC_score.py d15
