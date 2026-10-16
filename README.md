# Holiday Package Purchase Prediction

A notebook studying which customers are likely to purchase a travel package, using customer attributes and sales-contact data. The prediction target is `ProdTaken`.

## Contents

- `Holiday_Package.ipynb`: cleans inconsistent category labels, fills missing values, and creates a `TotalVisiting` feature.
- Compares Random Forest, Decision Tree, Logistic Regression, Gradient Boosting, AdaBoost, and XGBoost classifiers.
- Uses one-hot encoding and standard scaling, an 80/20 train/test split, and randomized hyperparameter searches.
- `Travel.csv`: the input dataset included in this repository.

## Run locally

Use Python 3 and run Jupyter from the repository directory.

```bash
git clone https://github.com/Sanskar1314/Holiday_Package_Prediction_using_RandomForest.git
cd Holiday_Package_Prediction_using_RandomForest
python -m venv .venv
source .venv/bin/activate
python -m pip install jupyterlab numpy pandas matplotlib seaborn plotly scikit-learn xgboost
jupyter lab
```

On Windows, activate the environment with `.venv\Scripts\activate`. Open the notebook listed above and run its cells in order. Results and visualizations appear in the notebook.

## Reproduction notes

Run with `Travel.csv` in the working directory. The notebook identifies its source as the [Kaggle holiday package purchase dataset](https://www.kaggle.com/datasets/susant4learning/holiday-package-purchase-prediction).

The tuning section includes a standalone `adaboost_params` reference whose definition is commented out; skip it or define the variable. Several search candidates use legacy options such as `max_features='auto'`, `deviance`, or `mse`; update unsupported candidates for your installed scikit-learn version. Imputation is performed before the train/test split in this experiment, so use a training-only preprocessing pipeline for a stricter evaluation.
