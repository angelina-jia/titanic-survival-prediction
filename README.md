# Titanic Survival Prediction

A machine learning project that predicts passenger survival on the RMS Titanic using Python and scikit-learn.

## Overview

The sinking of the RMS Titanic on April 15, 1912, was one of the most infamous maritime disasters in history. This project analyzes passenger data to predict survival outcomes based on characteristics like age, sex, passenger class, and family relationships.

Using the classic Kaggle Titanic dataset, I developed and compared multiple classification models to identify patterns that influenced survival during this historic tragedy.

## Results

| Model | Cross-Validation Accuracy | Validation Accuracy | Kaggle Score |
|-------|--------------------------|-------------------|--------------|
| **Random Forest** | **80.7%** | **83%** | **0.751** |
| Logistic Regression | 79.9% | 81% | - |

### Key Performance Metrics (Random Forest):
- **Precision (Death)**: 88%
- **Precision (Survival)**: 75% 
- **Recall (Death)**: 84%
- **Recall (Survival)**: 81%
- **Total Misclassifications**: 31 out of 179 validation samples

## Dataset

The dataset contains information about 891 passengers from the Titanic, including:
- **Survived**: Target variable (0 = No, 1 = Yes)
- **Pclass**: Passenger class (1st, 2nd, 3rd)
- **Sex**: Gender
- **Age**: Age in years
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Fare**: Passenger fare
- **Embarked**: Port of embarkation

## Methodology

### Data Preprocessing
- **Missing Value Handling**: Median imputation for Age column
- **Feature Selection**: Removed non-predictive columns (Name, Ticket, PassengerId)
- **Categorical Encoding**: Dummy variables for Sex, Embarked, and Pclass
- **Data Alignment**: Ensured train/test feature consistency

### Model Development
- **Cross-Validation**: 5-fold cross-validation for robust evaluation
- **Algorithm Comparison**: Random Forest vs Logistic Regression
- **Performance Metrics**: Confusion matrices, precision, recall, F1-score
- **Model Selection**: Random Forest chosen for superior performance

### Evaluation
- Confusion matrix analysis for detailed error assessment
- Cross-validation scores for generalization performance
- Kaggle competition submission for real-world validation

## Key Findings

- **Random Forest outperformed Logistic Regression** across all metrics
- **Both models struggled more with predicting survival than death**, reflecting the tragic reality
- **Passenger characteristics like class, age, and sex** contained meaningful predictive signals
- **Model performance aligns with historical accounts** of evacuation protocols and socioeconomic factors

## Technologies Used

- **Python 3.x**
- **pandas** - Data manipulation and analysis
- **scikit-learn** - Machine learning algorithms and evaluation
- **numpy** - Numerical computations

- **Jupyter Notebook** - Development environment

## Files

- `titanic_analysis.ipynb` - Complete analysis notebook
- `titanic_submission.csv` - Kaggle competition submission
- `README.md` - Project documentation

## Installation & Usage

```bash
# Clone the repository
git clone https://github.com/yourusername/titanic-survival-prediction.git

# Navigate to project directory
cd titanic-survival-prediction

# Install required packages
pip install pandas scikit-learn numpy jupyter

# Run the notebook
jupyter notebook titanic_analysis.ipynb
```

## Future Improvements

- **Feature Engineering**: Extract titles from names, create family size variables
- **Hyperparameter Tuning**: Optimize model parameters using GridSearchCV
- **Advanced Models**: Explore Gradient Boosting, XGBoost, or ensemble methods
- **Missing Data**: Implement more sophisticated imputation techniques

## Historical Context

This analysis provides insights into the social and economic factors that influenced survival during the Titanic disaster. The predictive power of features like passenger class and gender validates historical accounts of "women and children first" evacuation protocols and the correlation between socioeconomic status and survival rates.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Kaggle for providing the Titanic dataset
- The scikit-learn community for excellent machine learning tools
- Historical records that make this analysis possible
