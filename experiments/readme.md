This directory contains the experiments and their results.
Final results:
- *baseline_model_with_std.ipynb* and *baseline_model.ipynb* show results for baseline models
- *full_model_results_expected.ipynb*
and *full_model_results_sampled.ipynb* show results for both methods of discretizing size. 
- *accuracy_vs_coverage.ipynb* creates accuracy vs coverage plots for full models described in files above

Return status model:
- *HierarchicalReturnStatus_demo.ipynb* - SGD method for optimizing parameter $w$
- *hierarchical_status_w_sampling* - results for sampled $w$  
  
Dataset:
- *dataset_exploration.ipynb* describes the Renttherunway dataset

Other files:
- *config_preparation.ipynb* was used to select and prepare the configs of best runs to be slightly modified and run again
- *no_training_tests.ipynb* was used to generate results of models before training to compare them to trained models
- *HierarchicalSize_demo.ipynb* and *new_hierarchical_size_test.ipynb* were used at an early stages of experiments to examine size model
- *full_model_results.ipynb* served a base for *full_model_results_sampled.ipynb* and *full_model_results_expected.ipynb*