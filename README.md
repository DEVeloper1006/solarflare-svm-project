# Solar Flare Prediction using SVM

## Overview

This project implements a Support Vector Machine (SVM) model for solar flare prediction using different feature sets and datasets. The model is trained and evaluated with cross-validation and incorporates experiments for feature selection, data normalization, and no-shuffle validation.

## Project Structure

```
data/
  ├── data-2010-15/
  ├── data-2020-24/
.gitignore
model.py
README.md
requirements.txt
```

### Files and Directories:
- `data/`: Directory containing the datasets used for training and testing.
  - `data-2010-15/` and `data-2020-24/`: These folders contain the dataset files for two different time periods.
- `.gitignore`: A Git ignore file to exclude unnecessary files from version control.
- `model.py`: Python file implementing the SVM model, data processing, and experiments.
- `README.md`: This README file.
- `requirements.txt` : File to install all dependencies

## Dependencies

This project requires the following Python libraries:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`

You can install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

Alternatively, you can manually install each package using `pip`:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Usage

### 1. Load and Preprocess Data

Data is loaded from the `data/` directory. The dataset contains feature matrices and corresponding labels for solar flare prediction.

The feature matrices are created from various sources of solar flare data:
- `pos_features_main_timechange.npy`, `neg_features_main_timechange.npy`
- `pos_features_historical.npy`, `neg_features_historical.npy`
- `pos_features_maxmin.npy`, `neg_features_maxmin.npy`

### 2. Feature Selection and Model Training

The `SolarFlareSVM` class is used to train the SVM model. It includes functionality for:
- Feature creation and selection (`feature_creation`)
- Preprocessing with standard normalization (`preprocess`)
- Cross-validation and performance evaluation (`cross_validation`)
- Calculation of the TSS score for each fold of the cross-validation (`tss`)

### 3. Experiments

The project includes three main experiments:

#### Feature Experiment
The feature experiment tests different feature combinations and identifies the best feature set based on the TSS (True Skill Statistic) score.

```python
best_combination = feature_experiment()
```

#### Data Experiment
The data experiment tests the model's performance across different datasets and plots the TSS scores for each dataset.

```python
data_experiment(best_combination=best_combination)
```

#### No Shuffle Experiment
This experiment disables data shuffling in the k-fold cross-validation and tests the model's performance based on the fixed order of data in `data_order.npy`.

```python
no_shuffle_experiment(best_combination=best_combination)
```

### 4. Output

For each experiment, the model will output:
- The confusion matrix for each fold of the cross-validation.
- The TSS scores and average accuracy for each combination or dataset.
- Plots for the TSS scores across different feature combinations or datasets.

### Example:

```bash
python model.py
```

This will run the feature experiment, data experiment, and no-shuffle experiment sequentially and plot the results.

## Author

- Dev Mody
- McMaster University
- October 2024