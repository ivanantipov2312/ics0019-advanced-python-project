Network Intrusion Detection System (NIDS) - AI Classification Report

#  **1. Methods and Tools Used**

## 1.1 Libraries used:

- Pandas for dataset loading and preprocessing.

- Scikit-learn for label encoding, evaluation metrics, and cross-validation.

- Imbalanced-learn (SMOTE) to oversample minority classes and reduce class imbalance.

- XGBoost as the primary classifier due to its strong performance on tabular datasets.

- Joblib for saving the trained model and predictions.


The NSL-KDD dataset was downloaded directly from GitHub. Categorical features such as protocol type, service, and connection flags were encoded using LabelEncoder. The original attack labels were grouped into five broader categories to simplify classification.

## 1.2 Dataset info:

***Training samples: 125973**

***Test samples: 22544**

***Number of features: 40**


## 1.3 Class distribution (train – original):

Table : Class distribution before SMOTE

| Class | Count |
| - | - |
| Normal | 67343 |
| DoS | 45927 |
| Probe | 11656 |
| R2L | 995 |
| U2R | 52 |


# 2. Approaches Tried and Observations

The primary challenge in the dataset was severe class imbalance. The R2L and U2R categories contained significantly fewer samples than the majority classes. To address this issue, SMOTE (Synthetic Minority Oversampling Technique) was applied to generate synthetic samples for minority categories. The chosen model was XGBoost because: It handles structured/tabular data efficiently. It performs well with nonlinear feature interactions. It supports multiclass classification directly. Cross-validation was performed using 5-fold validation and macro F1-score as the evaluation metric. Macro F1-score was selected because it treats all classes equally, making it more suitable for imbalanced datasets.

# 3. What Worked and What Did Not

The model achieved strong performance on the majority classes such as DoS, Probe, and Normal traffic. The use of XGBoost combined with oversampling improved recall for minority classes compared to training without balancing. However, several limitations remained: R2L and U2R attacks were still difficult to classify correctly. Applying SMOTE before cross-validation introduced data leakage and inflated the validation score. The dataset contains very few U2R examples, limiting the model's ability to generalize.

# **4. Final Classification Results**

Table : Resulting scores

| Class | Precision | Recall | F1-Score |
| - | - | - | - |
| DoS | 0.96 | 0.83 | 0.89 |
| Normal | 0.70 | 0.97 | 0.81 |
| Probe | 0.84 | 0.79 | 0.81 |
| R2L | 0.99 | 0.14 | 0.25 |
| U2R | 0.62 | 0.24 | 0.34 |


**Final Macro F1-score**: 0.62

# 5. Cross-Validation vs Test Performance

The model achieved a cross-validation macro F1-score of approximately 0.9998, while the final test macro F1-score was only 0.62. This large discrepancy indicates that the cross-validation procedure was overly optimistic. The most likely reason is that SMOTE was applied before cross-validation. Synthetic samples generated during oversampling may appear in both training and validation folds, causing data leakage. As a result, the validation performance does not accurately represent real-world generalization.

***Macro F1 (CV mean): 0.9998**

***Macro F1 (CV std): 0.0001**

# 6. Confusion Matrix Visualization

The following confusion matrix is an illustrative visualization based on the classification behavior described in the experiment results.
![confusion_matrix.png](./confusion_matrix.png)

# 7. Reflection and Future Improvements

If more time were available, several improvements could be explored: Using a proper machine learning pipeline where SMOTE is applied inside each cross-validation fold. Trying alternative balancing techniques such as ADASYN or class weighting. Performing feature selection and hyperparameter optimization. Testing deep learning approaches such as autoencoders or LSTM-based intrusion detection systems. Using more modern cybersecurity datasets to improve real-world applicability. Overall, the project demonstrates that gradient boosting models can effectively classify common network attacks, but minority attack categories remain challenging due to limited training examples.
