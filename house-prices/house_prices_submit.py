"""
House Prices: Advanced Regression Techniques - Competition Submission
Kaggle Competition: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
Author: AIHeartICU
Target: Top 15% of leaderboard

Optimized for Kaggle submission without heavy dependencies
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm, skew
from scipy.special import boxcox1p

from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso, Ridge, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline

import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("HOUSE PRICES KAGGLE COMPETITION - OPTIMIZED SOLUTION")
print("="*80)

# ========== 1. LOAD DATA ==========
print("\n1. LOADING DATA")
print("-"*60)

# Download competition data from GitHub mirror
import urllib.request
import os

if not os.path.exists('train.csv'):
    print("Downloading House Prices competition data...")
    urls = {
        'train': 'https://raw.githubusercontent.com/agconti/kaggle-house-prices/master/data/train.csv',
        'test': 'https://raw.githubusercontent.com/agconti/kaggle-house-prices/master/data/test.csv'
    }
    
    for name, url in urls.items():
        try:
            urllib.request.urlretrieve(url, f'{name}.csv')
            print(f"✓ Downloaded {name}.csv")
        except:
            print(f"✗ Could not download {name}.csv")

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Save IDs
train_ID = train['Id']
test_ID = test['Id']

# ========== 2. REMOVE OUTLIERS ==========
print("\n2. OUTLIER REMOVAL")
print("-"*60)

print(f"Original: {len(train)} samples")

# Remove GrLivArea outliers
train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)

# Remove LotFrontage outliers
train = train.drop(train[train['LotFrontage'] > 300].index)

print(f"After removal: {len(train)} samples")

# ========== 3. TARGET TRANSFORMATION ==========
print("\n3. TARGET VARIABLE ANALYSIS")
print("-"*60)

# Log transform the target
y_train = np.log1p(train['SalePrice'].values)

print(f"Original price range: ${train['SalePrice'].min():,.0f} - ${train['SalePrice'].max():,.0f}")
print(f"Log transformed range: {y_train.min():.2f} - {y_train.max():.2f}")

# ========== 4. FEATURE ENGINEERING ==========
print("\n4. FEATURE ENGINEERING")
print("-"*60)

# Combine datasets
ntrain = train.shape[0]
ntest = test.shape[0]
all_data = pd.concat((train.drop(['SalePrice'], axis=1), test)).reset_index(drop=True)

print(f"Combined size: {all_data.shape}")

# Missing value analysis
missing = all_data.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(f"\nFeatures with missing values: {len(missing)}")

# Handle missing values by feature type
# Features where NA means "None"
none_features = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
                 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                 'MasVnrType', 'MSSubClass']

for col in none_features:
    if col in all_data.columns:
        all_data[col] = all_data[col].fillna('None')

# Features where NA means 0
zero_features = ['GarageYrBlt', 'GarageArea', 'GarageCars',
                 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']

for col in zero_features:
    if col in all_data.columns:
        all_data[col] = all_data[col].fillna(0)

# Mode imputation for categorical
mode_features = ['MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 
                 'Exterior2nd', 'SaleType', 'Functional']

for col in mode_features:
    if col in all_data.columns:
        all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

# LotFrontage: Group by Neighborhood
if 'LotFrontage' in all_data.columns:
    all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median()))

# Utilities: Drop (almost all same value)
if 'Utilities' in all_data.columns:
    all_data = all_data.drop(['Utilities'], axis=1)

# Create new features
print("\nCreating new features...")

# Total SF
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

# Total Bathrooms
all_data['TotalBathrooms'] = (all_data['FullBath'] + (0.5 * all_data['HalfBath']) +
                               all_data['BsmtFullBath'] + (0.5 * all_data['BsmtHalfBath']))

# Total porch SF
porch_cols = ['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
all_data['TotalPorchSF'] = all_data[porch_cols].sum(axis=1)

# Has features
all_data['HasPool'] = all_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['Has2ndFloor'] = all_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['HasGarage'] = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['HasBsmt'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['HasFireplace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

# Age features
all_data['YearsSinceBuilt'] = all_data['YrSold'] - all_data['YearBuilt']
all_data['YearsSinceRemod'] = all_data['YrSold'] - all_data['YearRemodAdd']

# Quality interactions
all_data['OverallQual_sq'] = all_data['OverallQual'] ** 2
all_data['OverallQual_x_Cond'] = all_data['OverallQual'] * all_data['OverallCond']
all_data['Qual_x_GrLiv'] = all_data['OverallQual'] * all_data['GrLivArea']

print(f"Features after engineering: {all_data.shape[1]}")

# ========== 5. ENCODING AND TRANSFORMATION ==========
print("\n5. ENCODING CATEGORICAL FEATURES")
print("-"*60)

# Label Encoding for ordinal features
ordinal_features = {
    'ExterQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'ExterCond': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'BsmtQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'BsmtCond': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'HeatingQC': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'KitchenQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'FireplaceQu': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'GarageQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'GarageCond': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'PoolQC': {'None': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
    'BsmtExposure': {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},
    'BsmtFinType1': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
    'BsmtFinType2': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
    'Functional': {'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8},
    'Fence': {'None': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4},
}

for col, mapping in ordinal_features.items():
    if col in all_data.columns:
        all_data[col] = all_data[col].map(mapping).fillna(0)

# Get remaining categorical columns
categorical_cols = all_data.select_dtypes(include=['object']).columns

print(f"Categorical features to encode: {len(categorical_cols)}")

# One-hot encoding
all_data = pd.get_dummies(all_data, columns=categorical_cols)

print(f"Features after encoding: {all_data.shape[1]}")

# ========== 6. SKEWNESS CORRECTION ==========
print("\n6. FIXING SKEWED FEATURES")
print("-"*60)

# Find skewed features
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[abs(skewed_feats) > 0.75]

print(f"Skewed features: {len(skewed_feats)}")

# Box-Cox transformation
from scipy.special import boxcox1p
lam = 0.15
for feat in skewed_feats.index:
    all_data[feat] = boxcox1p(all_data[feat], lam)

# ========== 7. PREPARE FINAL DATA ==========
print("\n7. PREPARING FINAL DATASETS")
print("-"*60)

# Split back
X_train = all_data[:ntrain]
X_test = all_data[ntrain:]

print(f"Final train: {X_train.shape}")
print(f"Final test: {X_test.shape}")

# Scale features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========== 8. MODEL TRAINING ==========
print("\n8. TRAINING MODELS")
print("-"*60)

# Cross-validation setup
n_folds = 5
kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

def rmsle_cv(model):
    rmse = np.sqrt(-cross_val_score(model, X_train_scaled, y_train, 
                                    scoring="neg_mean_squared_error", cv=kfold))
    return rmse.mean()

print("Training models with 5-fold CV...")

# 1. LASSO
lasso = Lasso(alpha=0.0005, random_state=42, max_iter=10000)
print(f"\n1. LASSO: CV Score = {rmsle_cv(lasso):.4f}")
lasso.fit(X_train_scaled, y_train)

# 2. Elastic Net
enet = ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=42, max_iter=10000)
print(f"2. Elastic Net: CV Score = {rmsle_cv(enet):.4f}")
enet.fit(X_train_scaled, y_train)

# 3. Ridge
ridge = Ridge(alpha=10, random_state=42)
print(f"3. Ridge: CV Score = {rmsle_cv(ridge):.4f}")
ridge.fit(X_train_scaled, y_train)

# 4. Gradient Boosting
gbr = GradientBoostingRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=4,
    max_features='sqrt',
    min_samples_leaf=15,
    min_samples_split=10,
    loss='huber',
    random_state=42
)
print(f"4. Gradient Boosting: CV Score = {rmsle_cv(gbr):.4f}")
gbr.fit(X_train_scaled, y_train)

# 5. Random Forest
rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
print(f"5. Random Forest: CV Score = {rmsle_cv(rf):.4f}")
rf.fit(X_train_scaled, y_train)

# 6. Extra Trees
et = ExtraTreesRegressor(
    n_estimators=500,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
print(f"6. Extra Trees: CV Score = {rmsle_cv(et):.4f}")
et.fit(X_train_scaled, y_train)

# 7. SVR
svr = SVR(C=10, epsilon=0.01)
print(f"7. SVR: CV Score = {rmsle_cv(svr):.4f}")
svr.fit(X_train_scaled, y_train)

# 8. Kernel Ridge
kr = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
print(f"8. Kernel Ridge: CV Score = {rmsle_cv(kr):.4f}")
kr.fit(X_train_scaled, y_train)

# ========== 9. ENSEMBLE PREDICTIONS ==========
print("\n9. CREATING ENSEMBLE")
print("-"*60)

# Get predictions
lasso_pred = np.expm1(lasso.predict(X_test_scaled))
enet_pred = np.expm1(enet.predict(X_test_scaled))
ridge_pred = np.expm1(ridge.predict(X_test_scaled))
gbr_pred = np.expm1(gbr.predict(X_test_scaled))
rf_pred = np.expm1(rf.predict(X_test_scaled))
et_pred = np.expm1(et.predict(X_test_scaled))
svr_pred = np.expm1(svr.predict(X_test_scaled))
kr_pred = np.expm1(kr.predict(X_test_scaled))

# Weighted ensemble
ensemble = (
    lasso_pred * 0.10 +
    enet_pred * 0.10 +
    ridge_pred * 0.05 +
    gbr_pred * 0.30 +
    rf_pred * 0.20 +
    et_pred * 0.15 +
    svr_pred * 0.05 +
    kr_pred * 0.05
)

print("\nEnsemble weights:")
print("• Gradient Boosting: 30%")
print("• Random Forest: 20%")
print("• Extra Trees: 15%")
print("• LASSO: 10%")
print("• Elastic Net: 10%")
print("• Ridge: 5%")
print("• SVR: 5%")
print("• Kernel Ridge: 5%")

# ========== 10. CREATE SUBMISSION ==========
print("\n10. CREATING SUBMISSION FILE")
print("-"*60)

# Create submission
submission = pd.DataFrame()
submission['Id'] = test_ID
submission['SalePrice'] = ensemble

# Sanity checks
submission['SalePrice'] = np.clip(submission['SalePrice'], 40000, 800000)

# Save
submission.to_csv('submission.csv', index=False)

print(f"\n✓ Submission saved: submission.csv")
print(f"Predictions: ${submission['SalePrice'].min():,.0f} - ${submission['SalePrice'].max():,.0f}")
print(f"Mean: ${submission['SalePrice'].mean():,.0f}")
print(f"Median: ${submission['SalePrice'].median():,.0f}")

# ========== 11. SUBMISSION INSTRUCTIONS ==========
print("\n" + "="*80)
print("KAGGLE SUBMISSION INSTRUCTIONS")
print("="*80)

print("""
🎯 TO SUBMIT AND SEE YOUR RANKING:

1. Go to: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/submit

2. Upload: submission.csv

3. Description: "8-model ensemble with 100+ features"

4. Click "Make Submission"

📊 EXPECTED PERFORMANCE:
• Target: < 0.12000 (Top 15%)
• Features: 200+ after encoding
• Models: 8 different algorithms
• Ensemble: Weighted average

🏆 YOUR RANKING WILL APPEAR:
• Public Score: Immediately visible
• Your rank out of ~5000 teams
• Leaderboard position updates instantly

💡 TIPS FOR IMPROVEMENT:
• Try different ensemble weights
• Add more feature interactions
• Tune hyperparameters further
• Stack models instead of averaging

Good luck! Check the leaderboard to see your rank!
""")

print("\n" + "="*80)
print("READY FOR KAGGLE SUBMISSION!")
print("="*80)