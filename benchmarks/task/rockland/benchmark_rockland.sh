#
# Save TVFC estimates
#

python benchmarks/task/rockland/train_models/save_TVFC_estimates.py all correlation SVWP_joint 645
python benchmarks/task/rockland/train_models/save_TVFC_estimates.py all correlation sFC 645

python benchmarks/task/rockland/train_models/save_TVFC_estimates.py LEOO covariance SVWP_joint 645
python benchmarks/task/rockland/train_models/save_TVFC_estimates.py LEOO covariance sFC 645

#
# Imputation study
#

python benchmarks/task/rockland/imputation_study/compute_LEOO_likelihoods.py SVWP_joint
python benchmarks/task/rockland/imputation_study/compute_LEOO_likelihoods.py DCC_joint
python benchmarks/task/rockland/imputation_study/compute_LEOO_likelihoods.py DCC_bivariate_loop
python benchmarks/task/rockland/imputation_study/compute_LEOO_likelihoods.py SW_cross_validated
python benchmarks/task/rockland/imputation_study/compute_LEOO_likelihoods.py sFC

python benchmarks/task/rockland/imputation_study/run_statistical_test_between_estimation_methods.py
