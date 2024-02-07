#
# Save TVFC estimates
#

# d2

python benchmarks/simulations/train_models/save_TVFC_estimates.py d2 all N0400_T0200 correlation SVWP
python benchmarks/simulations/train_models/save_TVFC_estimates.py d2 all N0400_T0200 correlation sFC

python benchmarks/simulations/train_models/save_TVFC_estimates.py d2 LEOO N0400_T0200 correlation SVWP
python benchmarks/simulations/train_models/save_TVFC_estimates.py d2 LEOO N0400_T0200 correlation sFC

# d4s

python benchmarks/simulations/train_models/save_TVFC_estimates.py d4s all N0400_T0010 correlation SVWP_joint
python benchmarks/simulations/train_models/save_TVFC_estimates.py d4s all N0400_T0010 correlation sFC

python benchmarks/simulations/train_models/save_TVFC_estimates.py d4s LEOO N0400_T0010 covariance SVWP_joint
python benchmarks/simulations/train_models/save_TVFC_estimates.py d4s LEOO N0400_T0010 covariance sFC

#
# Evaluation
#

# d2

python benchmarks/simulations/evaluation/compute_average_quantitative_results.py d2 all N0400_T0200

# d4s

python benchmarks/simulations/evaluation/compute_average_quantitative_results.py d4s all N0400_T0010

#
# Imputation benchmark
#

# d2

python benchmarks/simulations/imputation_benchmark/compute_LEOO_likelihoods.py d2 N0400_T0200 SVWP
python benchmarks/simulations/imputation_benchmark/compute_LEOO_likelihoods.py d2 N0400_T0200 DCC_joint
python benchmarks/simulations/imputation_benchmark/compute_LEOO_likelihoods.py d2 N0400_T0200 SW_cross_validated
python benchmarks/simulations/imputation_benchmark/compute_LEOO_likelihoods.py d2 N0400_T0200 sFC

python benchmarks/simulations/imputation_benchmark/run_statistical_test_between_estimation_methods.py d2 N0400_T0200

# d4s

python benchmarks/simulations/imputation_benchmark/compute_LEOO_likelihoods.py d4s N0400_T0010 SVWP_joint
python benchmarks/simulations/imputation_benchmark/compute_LEOO_likelihoods.py d4s N0400_T0010 DCC_joint
python benchmarks/simulations/imputation_benchmark/compute_LEOO_likelihoods.py d4s N0400_T0010 SW_cross_validated
python benchmarks/simulations/imputation_benchmark/compute_LEOO_likelihoods.py d4s N0400_T0010 sFC

python benchmarks/simulations/imputation_benchmark/run_statistical_test_between_estimation_methods.py d4s N0400_T0010
