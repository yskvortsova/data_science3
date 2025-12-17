# Predictive machine learning models for determining sequence based protein-protein interaction 
#### *Yana Skvortsova, 2025*

Developing and analysing tree-based predictive machine learning models to determine protein-protein interactions based on amino acid sequences.

Aims: 
- To extract numerical features from a string of paired amino acid sequences,
- Train and tune machine learning models to predict if two proteins will interact, given their amino acid sequenes,
- Optimise hyperparameters of the models,
- Evaluate feature importances,
- Identify the best model and analyse performance.

## Dataset

The labelled protein-protein interaction dataset can be obtained from Kaggle (Spandan Sureja).

Available at: https://www.kaggle.com/datasets/spandansureja/ppi-dataset 

This url allows to download two files:
- positive_protein_sequences, with 36,652 interacting protein pair sequences.
- negative_protein_sequnces, with 36,480 non-interacting protein pair sequences.

The sequences are made up of letters representing amino acids, which make up proteins.

## Setup

Dwnload the data files into the data folder in your project folder.

### Requirements

The following libraries have to be installed:
- pandas
- numpy
- seaborn
- matplotlib
- sklearn
- biopython
- xgboost

# Workflow 

## 1. Dataset preprocessing

- Load in negative and positive sequences.
  
```python
negative = pd.read_csv(r"data\negative_protein_sequences.csv")
positive = pd.read_csv(r"data\positive_protein_sequences.csv")
```

- Briefly visualise both datasets

- Make a classification target to label interacting sequences and non-interacting sequences.

```python
positive["target"] = 1
negative["target"] = 0
```

- Combine the two datasets into one DataFrame

## 2. Data cleaning

- Remove sequence pairs that do not contain standard amino acids
- Important for biolgical accuracy, and also for downstream BioPython analysis.

## 3. Feature extraction from amino acid sequences

Extract the following features from each sequence (1 and 2):

- Sequence length
- Sequence length similarity
- Amino acid composition (proportion of each amino acid in the sequence)
- Physiochemical properties 
    - Molecular weight
    - Aromaticity
    - Instability index
    - Isoelectric point
    - Mean Flexibility
    - Secondary structure proportions for:
        - Helix
        - Turn
        - Sheet

Concatenate the dataframes of the extracted features into one final DataFrame (proteins_full).
The final DataFrame will have the two sequence column, target (0 or 1 for non interactors and interactors), and the feature columns, with features for each sequence (seq 1 and 2), as well as the pairwise similarity features. 


## 4. Data splitting

4.1 Define the X and y variables.

```python
X = proteins_full.drop(columns=["protein_sequences_1", "protein_sequences_2", "target"]) 
y = proteins_full["target"]
```

4.2 Optional - explore the features distribution and difference for each feature between interacting and non ineracting sequences. The output is quite large.

4.3 Split into testing and training subgroups. 

```python
X_train_unscaled, X_test_unscaled, y_train, y_test = train_test_split(X , y, test_size=0.25, random_state=42, stratify=y)
```
## 5. Data scaling

Scale the testing and training data.

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_unscaled)
X_test = scaler.transform(X_test_unscaled)
```

## 6. Machine Learning Model 1 - Random Forest Model

### 6.1 Train and evaluate the base Random Forest model.

- Accuracy score
- Classification report
- Confusion matrix

### 6.2 Perform hyperparameter tuning using HalvingGridSearchCV with the following grid.

```python
Parameter_grid =  {
    "max_depth": [None, 10, 20, 30],
    "max_features": ["sqrt", "log2", 0.3],
    "min_samples_leaf": [1, 5, 15],
    "n_estimators": [100, 250, 500]
}
```

### 6.3 Optimised hyperparameters - Random Forest 

*To save time on running the paramter tuning on loading again, you can use the determined hyperparameters that were established from the run above to run and fit the tuned Random Forest model:*

```python
Best_RandomForest_model = RandomForestClassifier(n_estimators=500, max_depth=30, max_features="sqrt", min_samples_leaf=1, n_jobs=-1, verbose=False, random_state=42)
Best_RandomForest_model.fit(X_train,y_train)
Best_RandomForest_model
```
### 6.4 Evaluate the tuned Random Forest Model

- Accuracy score
- Classification report
- Confusion matrix

## 6.5 Feature importance analysis - Random Forest Model

Analyse all feature importances.

Determine the top 20 feature importances for the tuned Random Forest model.

```python
importances = Best_RandomForest_model.feature_importances_
importances_df = pd.Series(importances, index=X.columns)
top_20importances = importances_df.sort_values(ascending=False).head(20)
```

## 6.6 Permutation importance analysis - Random Forest Model

Find out the permutation importance for all features. 

```python
result = permutation_importance(Best_RandomForest_model, X_test, y_test, n_repeats=5, random_state=1, n_jobs=-1,scoring="accuracy") 
importancesPermutation = result.importances_mean 
importancesPermutation_df = pd.Series(importancesPermutation, index=X.columns)
importancesPermutation_sort = importancesPermutation_df.sort_values(ascending=False)
```

## 6.7 ROC Curve - Random Forest Model

Generate an ROC curve and AUC value.

```python
y_prob_rf = Best_RandomForest_model.predict_proba(X_test)[:, 1] 
false_positive_rf, true_positive_rf, _ = roc_curve(y_test, y_prob_rf) 
roc_auc_rf = auc(false_positive_rf, true_positive_rf) 

RocCurveDisplay(fpr=false_positive_rf, 
                tpr=true_positive_rf, 
                estimator_name=f"Random Forest (AUC = {roc_auc_rf:.4f})"
).plot()
```

## 7. Machine Learning Model 2 - XGBoost Model

### 7.1 Train and evaluate the base XGBoost model.

- Accuracy score
- Classification report
- Confusion matrix

### 7.2 Perform hyperparameter tuning using HalvingGridSearchCV with the following grid.

```python
Parameter_grid =  {
    "n_estimators":[200, 600, 800],
    "learning_rate": [0.1, 0.05, 0.02],
    "max_depth": [3, 6, 10],
    "subsample": [0.4, 0.8],
    "colsample_bytree": [0.5, 0.8]
}
```

### 7.3 Optimised hyperparameters - XGBoost

*To save time on running the paramter tuning on loading again, you can use the determined hyperparameters that were established from the run above to run and fit the tuned Random Forest model:*

```python
Best_XGBoostModel = XGBClassifier(n_estimators = 600, learning_rate = 0.1, max_depth = 6, subsample = 0.8, colsample_bytree = 0.5, n_jobs = -1, random_state = 42, verbose =False)
Best_XGBoostModel.fit(X_train,y_train)
Best_XGBoostModel
```
### 7.4 Evaluate the tuned XGBoost model

- Accuracy score
- Classification report
- Confusion matrix

## 7.5 Feature importance analysis - XGBoost model

Analyse all feature importances.

Determine the top 20 feature importances for the tuned Random Forest model.

```python
importancesXGB = Best_XGBoostModel.feature_importances_
importancesXGB_df = pd.Series(importancesXGB, index=X.columns)
sorted_importancesXGB = importancesXGB_df.sort_values(ascending=False).head(20)
```

## 7.6 Permutation importance analysis - XGBoost model

Find out the permutation importance for all features. 

```python
resultXGBoost = permutation_importance(Best_XGBoostModel, X_test, y_test, n_repeats=5, random_state=1, n_jobs=-1, scoring="accuracy") 
importancesPermutationXGBoost = resultXGBoost.importances_mean 
importancesPermutationXGBoost_df = pd.Series(importancesPermutationXGBoost, index=X.columns) 
importancesPermutationXGBoost_sort = importancesPermutationXGBoost_df.sort_values(ascending=False)
```

## 7.7 ROC Curve - XGBoost model

Generate an ROC curve and AUC value.

```python
y_prob_xgb = Best_XGBoostModel.predict_proba(X_test)[:, 1] 
false_positive_xgb, true_positive_xgb, _ = roc_curve(y_test, y_prob_xgb) 
roc_auc_xgb = auc(false_positive_xgb, true_positive_xgb) 
RocCurveDisplay(fpr=false_positive_xgb, 
                tpr=true_positive_xgb, 
                estimator_name=f"XGBoost (AUC = {roc_auc_xgb:.4f})"
).plot()
```
