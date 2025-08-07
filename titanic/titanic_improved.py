"""
Kaggle Titanic - Improved Solution for 0.80+ accuracy
Focused on proven strategies from top solutions
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
test_ids = test_df['PassengerId']

# Combine for feature engineering
df = pd.concat([train_df, test_df], sort=False)

# ========== CRITICAL FEATURE ENGINEERING ==========

# 1. Title extraction - MOST IMPORTANT FEATURE
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 
                                   'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

# 2. Family size - CRITICAL
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# 3. Simple but effective age imputation
median_ages = df.groupby(['Title', 'Pclass'])['Age'].transform('median')
df['Age'].fillna(median_ages, inplace=True)

# 4. Fare imputation
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# 5. Embarked imputation
df['Embarked'].fillna('S', inplace=True)

# 6. Cabin - just has/hasn't
df['HasCabin'] = df['Cabin'].notna().astype(int)

# 7. Create bins for continuous variables
df['AgeBin'] = pd.cut(df['Age'], 5)
df['FareBin'] = pd.qcut(df['Fare'], 5)

# ========== ENCODING ==========

# Map categorical features to numbers
df['Sex'] = df['Sex'].map({'female': 1, 'male': 0})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
df['Title'] = df['Title'].map(title_mapping)

# Encode bins
df['AgeBin'] = pd.cut(df['Age'], 5, labels=[0, 1, 2, 3, 4])
df['FareBin'] = pd.qcut(df['Fare'], 5, labels=[0, 1, 2, 3, 4])

# ========== FEATURE SELECTION ==========
# Only use proven important features
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Survived']
df = df.drop(drop_elements, axis=1, errors='ignore')

# Fill any remaining NaN
df = df.fillna(0)

# Split back
train_len = len(train_df)
X_train = df[:train_len]
X_test = df[train_len:]
y_train = train_df['Survived']

# ========== OPTIMIZED RANDOM FOREST ==========
# Based on research, properly tuned RF can achieve 0.82+ alone

rf_optimized = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

# Cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(rf_optimized, X_train, y_train, cv=cv, scoring='accuracy')
print(f"CV Score: {scores.mean():.4f}")

# Train and predict
rf_optimized.fit(X_train, y_train)
predictions = rf_optimized.predict(X_test)

# Create submission
submission = pd.DataFrame({
    'PassengerId': test_ids,
    'Survived': predictions
})
submission.to_csv('submission_improved.csv', index=False)
print("Submission saved as submission_improved.csv")