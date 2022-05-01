# Contents
- `tenfold_classifier.py`: Python script for 10-fold C.V. on categorical labels (required for running greedy_classifier.py)
- `tenfold_regressor.py`: Python script for 10-fold C.V. on numerical labels (required for running greedy_regressor.py)
- `greedy_classifier.py`: Python script for bottom-up feature selection on categorical labels.
- `greedy_regressor.py`: Python script for bottom-up feature selection on numerical labels.

# Be aware of...
- Depending on algorithm/feature, some of features give warning message during very early stage of greedy feature selection.

# Available algorithms and scoring metrics:
- algorithms : [GB,XGBOOST,RF,M5P,GAUSSIAN,ADABOOST,KNN,SVC,NEURAL]
- scoring metrics (classification task): ['roc_auc','matthew','bacc','f1']

# Installation
1. You need to install a conda environment as follows:
```bash
conda env create -f requirements.yaml -n greedyFS
```

2. Activate your new conda envrionement :
```bash
source activate greedyFS
```

3. Run 10-fold C.V or Greey Feature selection as below :
```bash
python tenfold_classifier.py ExtraTrees ../test_files/training_classifier.csv age
python tenfold_regressor.py ExtraTrees ../test_files/training_regressor.csv dG
python greedy_classifier.py XGBOOST ../test_files/training_classifier.csv age
python greedy_regressor.py XGBOOST ../test_files/training_regressor.csv dG
```
