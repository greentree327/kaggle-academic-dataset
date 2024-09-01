# kaggle-academic-dataset
Classification with an Academic Success Dataset - Playground Series, Season 4, Episode 6

## Achievement

**Description:** My notebook ranked 24th out of 2684 in the competition

<img src="https://github.com/user-attachments/assets/93cd552f-e8da-46fe-b1e4-24bdba65a795" alt="I achieved Top 1% in this Kaggle competition, ranked 24th out of 2684" width="45%" />

**Link:** [Kaggle Competition](https://www.kaggle.com/competitions/playground-series-s4e6)

### Summary of Findings

1. **Model Performance**:
   - XGBoost consistently outperforms other models in this classification task.

2. **Prediction and Accuracy**:
   - Using `model.predict_proba` followed by `normalize_predictions` and then calculating `accuracy_score` provides higher accuracy compared to using `model.predict` directly.

---

### Dataset Overview

The dataset includes features and a target variable, used to train and evaluate various machine learning models.

### Data Inspection and Preprocessing

1. **Data Inspection**:
   - Checked for missing values in the training dataset (`df`).
   - Generated a correlation matrix using Seaborn's heatmap for the test dataset (`df_test`), excluding the 'id' column.
   - Analyzed the distribution of the target variable by checking the unique categories and the number of samples per category.
   - Applied label encoding to transform the categorical labels in the 'Target' column to numerical values using `LabelEncoder`.

2. **Feature and Target Separation**:
   - Separated features and target variables for both training (`x_train`, `y_train`) and testing datasets (`x_test`).

### Model Training and Evaluation

1. **Cross-Validation**:
   - Implemented K-Fold and Stratified K-Fold cross-validation to ensure proper training and testing splits.
   - Developed a `get_score` function for model evaluation.
   - Demonstrated the difference between regular K-Fold and Stratified K-Fold, and the effect of shuffling data in Stratified K-Fold.

2. **Model Evaluation Metrics**:
   - Explained log-loss calculation and its importance.
   - Used `log_loss` from scikit-learn to compute log-loss for a sample dataset.
   - Demonstrated the concept of `argmax` for obtaining predicted class labels from probability predictions.

3. **Model Training and Hyperparameter Tuning**:
   - Cross-validated various models using a custom `cross_validate` function that computes out-of-fold (OOF) and test predictions.
   - Trained and validated several models:
     - Logistic Regression
     - Random Forest (untuned and tuned using Grid Search and Random Search)
     - Extra Trees (using Random Search)
     - XGBoost (untuned and tuned using Optuna for Bayesian optimization)
   - Saved the best hyperparameters for each tuned model using the `pickle` library for later use.

4. **Using Pickle for Saving Checkpoints**:
   - Saved and loaded the best hyperparameters obtained from Grid Search, Random Search, and Bayesian optimization using `pickle`.

### Results

The results section compares the performance of different models, identifying XGBoost as the best-performing model. It also demonstrates that using `model.predict_proba` followed by `normalize_predictions` and calculating `accuracy_score` yields higher accuracy compared to using `model.predict` directly.
