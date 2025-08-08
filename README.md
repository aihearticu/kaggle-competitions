# Kaggle Competition Solutions

A collection of machine learning solutions for various Kaggle competitions and datasets.

## 📊 Projects

### 1. Titanic - Machine Learning from Disaster
- **Score**: 77.99% accuracy
- **Model**: Random Forest with feature engineering
- **Features**: Title extraction, family size analysis

### 2. ConnectX
- **Score**: 722.2 
- **Rank**: #33
- **Approach**: Minimax with alpha-beta pruning, 8-10 ply depth

### 3. House Prices - Advanced Regression
- **Target**: Top 15%
- **Model**: 8-model ensemble
- **Features**: 200+ engineered features

### 4. Healthcare ML Models
Collection of predictive models for healthcare datasets:
- Heart Disease Prediction (93.48% ROC-AUC)
- Stroke Prediction (78.85% ROC-AUC)
- Diabetes Prediction (82.15% ROC-AUC)
- Sepsis Early Warning (100% ROC-AUC on synthetic data)
- Chronic Kidney Disease (100% ROC-AUC)

## 🛠️ Tech Stack
- Python 3.8+
- scikit-learn
- pandas, numpy
- imbalanced-learn

## 📁 Repository Structure
```
kaggle-competitions/
├── titanic/
├── connectx/
├── house-prices/
├── heart-failure-prediction/
├── stroke-prediction/
├── diabetes-prediction/
├── sepsis-prediction/
└── chronic-kidney-disease/
```

## 🚀 Getting Started

```bash
# Clone repository
git clone https://github.com/aihearticu/kaggle-competitions.git

# Install dependencies
pip install -r requirements.txt

# Run any model
cd titanic
python3 titanic_improved.py
```

## 📈 Results

| Competition | Score | Metric |
|------------|-------|--------|
| Titanic | 77.99% | Accuracy |
| ConnectX | 722.2 | ELO Rating |
| Heart Disease | 93.48% | ROC-AUC |
| Stroke | 78.85% | ROC-AUC |
| Diabetes | 82.15% | ROC-AUC |
| CKD | 100% | ROC-AUC |

## 📝 License
MIT