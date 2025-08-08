"""
Diabetes Prediction with ICU Clinical Insights
Author: ICU Nurse at Kaiser - Healthcare AI/ML Portfolio
Dataset: Pima Indians Diabetes Dataset (Classic ML benchmark)

Clinical Context:
As an ICU nurse, I've seen the devastating effects of uncontrolled diabetes:
- DKA (Diabetic Ketoacidosis) requiring ICU admission
- Hyperosmolar hyperglycemic state (HHS)  
- Multi-organ failure from chronic complications
- Critical care management of diabetic emergencies

This model focuses on early detection and risk stratification to prevent
these ICU admissions through proactive care.

Key Clinical Insights Applied:
1. BMI + Glucose interaction (metabolic syndrome)
2. Age-adjusted risk thresholds
3. Family history impact (genetics)
4. Insulin resistance patterns
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, auc)
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DIABETES PREDICTION - ICU PREVENTION PERSPECTIVE")
print("="*80)

# ========== 1. DATA LOADING ==========
print("\n1. LOADING DIABETES DATA")
print("-"*60)

# Download Pima Indians Diabetes dataset
import urllib.request
import os

if not os.path.exists('diabetes.csv'):
    print("Downloading Pima Indians Diabetes dataset...")
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
    urllib.request.urlretrieve(url, 'diabetes.csv')
    print("Dataset downloaded successfully")

# Load with proper column names
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
          'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

df = pd.read_csv('diabetes.csv', names=columns)
print(f"Total patients: {df.shape[0]:,}")
print(f"Clinical features: {df.shape[1]-1}")

# ========== 2. CLINICAL DATA QUALITY ASSESSMENT ==========
print("\n2. CLINICAL DATA QUALITY ASSESSMENT")
print("-"*60)

print("\nDataset Overview:")
print(df.info())

print("\nDiabetes Prevalence:")
diabetes_counts = df['Outcome'].value_counts()
print(f"• No Diabetes: {diabetes_counts[0]:,} ({diabetes_counts[0]/len(df)*100:.1f}%)")
print(f"• Diabetes: {diabetes_counts[1]:,} ({diabetes_counts[1]/len(df)*100:.1f}%)")

print("\nClinical Statistics:")
print(df.describe().round(2))

# ========== 3. HANDLING MISSING VALUES (ZEROS ARE MISSING) ==========
print("\n" + "="*80)
print("3. CLINICAL DATA CLEANING")
print("="*80)

# In medical data, zeros in these fields are actually missing values
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

print("\nMissing Values (recorded as zeros):")
for col in zero_cols:
    zero_count = (df[col] == 0).sum()
    zero_pct = zero_count / len(df) * 100
    print(f"• {col}: {zero_count} ({zero_pct:.1f}%)")

# Replace zeros with NaN for proper imputation
df_processed = df.copy()
for col in zero_cols:
    df_processed[col] = df_processed[col].replace(0, np.nan)

# Clinical imputation using KNN (similar patients)
print("\nImputing missing values using KNN (similar patient profiles)...")
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(
    imputer.fit_transform(df_processed),
    columns=df_processed.columns
)

# ========== 4. CLINICAL FEATURE ENGINEERING ==========
print("\n" + "="*80)
print("4. CLINICAL FEATURE ENGINEERING")
print("="*80)

print("\nCreating clinically relevant features...")

# 1. Glucose categories (ADA guidelines)
df_imputed['Glucose_Category'] = pd.cut(df_imputed['Glucose'],
                                         bins=[0, 100, 125, 200, 500],
                                         labels=['Normal', 'Prediabetic', 'Diabetic', 'Severe'])

# 2. BMI categories (WHO classification)
df_imputed['BMI_Category'] = pd.cut(df_imputed['BMI'],
                                     bins=[0, 18.5, 25, 30, 35, 100],
                                     labels=['Underweight', 'Normal', 'Overweight', 'Obese', 'Severely_Obese'])

# 3. Blood Pressure categories (AHA guidelines)
df_imputed['BP_Category'] = pd.cut(df_imputed['BloodPressure'],
                                   bins=[0, 80, 90, 120, 140, 200],
                                   labels=['Low', 'Normal', 'Elevated', 'High', 'Crisis'])

# 4. Age risk groups (diabetes risk increases with age)
df_imputed['Age_Group'] = pd.cut(df_imputed['Age'],
                                 bins=[0, 30, 40, 50, 60, 100],
                                 labels=['Young', 'Middle', 'PreSenior', 'Senior', 'Elder'])

# 5. Metabolic syndrome indicators
df_imputed['MetabolicSyndrome_Score'] = 0
df_imputed.loc[df_imputed['Glucose'] > 100, 'MetabolicSyndrome_Score'] += 1
df_imputed.loc[df_imputed['BloodPressure'] > 130, 'MetabolicSyndrome_Score'] += 1
df_imputed.loc[df_imputed['BMI'] > 30, 'MetabolicSyndrome_Score'] += 1
# Note: Would add HDL and Triglycerides if available

# 6. Insulin resistance proxy (HOMA-IR approximation)
df_imputed['Insulin_Resistance'] = (df_imputed['Glucose'] * df_imputed['Insulin']) / 405

# 7. Pregnancy risk (gestational diabetes history)
df_imputed['High_Pregnancy_Risk'] = (df_imputed['Pregnancies'] >= 3).astype(int)

# 8. Combined risk score
df_imputed['Risk_Score'] = (
    (df_imputed['Glucose'] > 125).astype(int) * 3 +
    (df_imputed['BMI'] > 30).astype(int) * 2 +
    (df_imputed['Age'] > 45).astype(int) * 2 +
    (df_imputed['DiabetesPedigreeFunction'] > 0.5).astype(int) * 2 +
    (df_imputed['BloodPressure'] > 140).astype(int)
)

print("Created features:")
print("• Glucose categories (ADA guidelines)")
print("• BMI categories (WHO)")
print("• Blood pressure categories (AHA)")
print("• Age risk groups")
print("• Metabolic syndrome score")
print("• Insulin resistance index")
print("• Pregnancy risk indicator")
print("• Combined risk score")

# ========== 5. PREPARE FOR MODELING ==========
print("\n" + "="*80)
print("5. PREPARING DATA FOR ML")
print("="*80)

# Encode categorical features
from sklearn.preprocessing import LabelEncoder

categorical_cols = ['Glucose_Category', 'BMI_Category', 'BP_Category', 'Age_Group']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df_imputed[col] = le.fit_transform(df_imputed[col])
    label_encoders[col] = le

# Prepare features and target
X = df_imputed.drop('Outcome', axis=1)
y = df_imputed['Outcome'].astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} patients")
print(f"Test set: {X_test.shape[0]} patients")
print(f"Features: {X_train.shape[1]}")

# ========== 6. HANDLE CLASS IMBALANCE ==========
print("\n" + "="*80)
print("6. ADDRESSING CLASS IMBALANCE")
print("="*80)

# Apply SMOTE for balanced training
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"Original training distribution: {pd.Series(y_train).value_counts().to_dict()}")
print(f"Balanced training distribution: {pd.Series(y_train_balanced).value_counts().to_dict()}")

# ========== 7. MODEL DEVELOPMENT ==========
print("\n" + "="*80)
print("7. CLINICAL ML MODEL DEVELOPMENT")
print("="*80)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nTraining multiple models for ensemble...")

# 1. Logistic Regression
print("\n1. Logistic Regression (Interpretable):")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train_balanced)
lr_scores = cross_val_score(lr, X_train_scaled, y_train_balanced, cv=cv, scoring='roc_auc')
print(f"   CV ROC-AUC: {lr_scores.mean():.4f} (+/- {lr_scores.std():.4f})")

# 2. Random Forest
print("\n2. Random Forest (Feature Interactions):")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    random_state=42
)
rf.fit(X_train_balanced, y_train_balanced)
rf_scores = cross_val_score(rf, X_train_balanced, y_train_balanced, cv=cv, scoring='roc_auc')
print(f"   CV ROC-AUC: {rf_scores.mean():.4f} (+/- {rf_scores.std():.4f})")

# 3. Gradient Boosting
print("\n3. Gradient Boosting (High Accuracy):")
gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
gb.fit(X_train_balanced, y_train_balanced)
gb_scores = cross_val_score(gb, X_train_balanced, y_train_balanced, cv=cv, scoring='roc_auc')
print(f"   CV ROC-AUC: {gb_scores.mean():.4f} (+/- {gb_scores.std():.4f})")

# 4. SVM
print("\n4. Support Vector Machine (Non-linear):")
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train_scaled, y_train_balanced)
svm_scores = cross_val_score(svm, X_train_scaled, y_train_balanced, cv=cv, scoring='roc_auc')
print(f"   CV ROC-AUC: {svm_scores.mean():.4f} (+/- {svm_scores.std():.4f})")

# 5. Ensemble Voting Classifier
print("\n5. Ensemble Model (Combined Strengths):")
ensemble = VotingClassifier(
    estimators=[
        ('lr', lr),
        ('rf', rf),
        ('gb', gb),
        ('svm', svm)
    ],
    voting='soft'
)
ensemble.fit(X_train_scaled, y_train_balanced)

# ========== 8. MODEL EVALUATION ==========
print("\n" + "="*80)
print("8. CLINICAL MODEL EVALUATION")
print("="*80)

# Evaluate ensemble model
y_pred = ensemble.predict(X_test_scaled)
y_proba = ensemble.predict_proba(X_test_scaled)[:, 1]

print("\nEnsemble Model Performance:")
print(classification_report(y_test, y_pred, 
                           target_names=['No Diabetes', 'Diabetes']))

# Confusion Matrix Analysis
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix Analysis:")
print(f"True Negatives: {tn} (Correctly identified healthy)")
print(f"False Positives: {fp} (Further screening needed)")
print(f"False Negatives: {fn} (⚠️ CRITICAL - Missed diabetes)")
print(f"True Positives: {tp} (Correctly identified diabetes)")

# Clinical metrics
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0
roc_auc = roc_auc_score(y_test, y_proba)

print("\nClinical Performance Metrics:")
print(f"Sensitivity (Recall): {sensitivity:.1%}")
print(f"Specificity: {specificity:.1%}")
print(f"PPV (Precision): {ppv:.1%}")
print(f"NPV: {npv:.1%}")
print(f"ROC-AUC: {roc_auc:.4f}")

# Number Needed to Screen
if ppv > 0:
    nns = int(1 / ppv)
    print(f"\nNumber Needed to Screen: {nns}")
    print(f"(Screen {nns} high-risk patients to identify 1 diabetic)")

# ========== 9. RISK STRATIFICATION ==========
print("\n" + "="*80)
print("9. DIABETES RISK STRATIFICATION")
print("="*80)

# Create risk categories
risk_thresholds = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
risk_labels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']

risk_df = pd.DataFrame({
    'Probability': y_proba,
    'Actual': y_test,
    'Risk_Category': pd.cut(y_proba, bins=risk_thresholds, labels=risk_labels)
})

risk_summary = risk_df.groupby('Risk_Category')['Actual'].agg(['count', 'sum', 'mean'])
risk_summary.columns = ['Patients', 'Diabetics', 'Diabetes_Rate']
print("\nRisk Stratification:")
print(risk_summary)

# ========== 10. FEATURE IMPORTANCE ==========
print("\n" + "="*80)
print("10. CLINICAL FEATURE IMPORTANCE")
print("="*80)

# Get feature importance from Random Forest
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Diabetes Risk Factors:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"{row['feature']:30s}: {row['importance']:.4f}")

# ========== 11. CLINICAL RECOMMENDATIONS ==========
print("\n" + "="*80)
print("11. ICU PREVENTION RECOMMENDATIONS")
print("="*80)

print("""
DIABETES PREVENTION INSIGHTS FROM ICU PERSPECTIVE:

1. **Glucose is the Strongest Predictor**
   • Implement continuous glucose monitoring for high-risk
   • ICU Protocol: Tight glucose control prevents complications
   • Target: Fasting <100, Post-meal <140

2. **BMI + Metabolic Syndrome Critical**
   • Combined metabolic factors compound risk
   • ICU Reality: Obese diabetics have worse outcomes
   • Action: Aggressive lifestyle intervention programs

3. **Age-Adjusted Screening Essential**
   • Risk increases dramatically after 45
   • ICU Observation: Older diabetics = more complications
   • Protocol: Annual screening after 45, biannual after 60

4. **Insulin Resistance Early Marker**
   • Detected before glucose elevation
   • ICU Impact: Insulin resistance = poor ICU outcomes
   • Intervention: Metformin consideration for prediabetics

5. **Family History (Pedigree Function)**
   • Strong genetic component confirmed
   • ICU Pattern: Familial clustering of complications
   • Response: Earlier and more frequent screening

ICU PREVENTION STRATEGIES:

IMMEDIATE ACTIONS (High Risk: >60% probability):
• Start metformin if prediabetic
• Continuous glucose monitoring
• Quarterly HbA1c monitoring
• Dietitian referral
• Exercise prescription (150 min/week)

MODERATE ACTIONS (Moderate Risk: 40-60%):
• Biannual glucose testing
• Weight loss program (5-7% goal)
• Blood pressure management
• Lipid optimization
• Lifestyle coaching

SURVEILLANCE (Low Risk: <40%):
• Annual screening
• Education on risk factors
• Promote healthy lifestyle
• Monitor BMI trends

ICU ADMISSION PREVENTION:
• DKA Prevention: Education on sick day management
• HHS Prevention: Regular glucose monitoring in elderly
• Infection Prevention: Vaccination protocols
• Foot Care: Prevent amputations
• Cardiovascular: Aggressive risk factor modification

QUALITY METRICS TO TRACK:
• Screening compliance rate
• Time to diagnosis from first abnormal glucose
• DKA admission rate (Target: <2%)
• HbA1c <7% achievement rate
• Complication development rate
""")

# ========== 12. SAVE RESULTS ==========
print("\n" + "="*80)
print("12. SAVING CLINICAL MODELS AND RESULTS")
print("="*80)

# Save predictions
predictions_df = pd.DataFrame({
    'Patient_ID': range(len(y_test)),
    'Actual': y_test.values,
    'Predicted': y_pred,
    'Probability': y_proba,
    'Risk_Category': pd.cut(y_proba, bins=risk_thresholds, labels=risk_labels)
})
predictions_df.to_csv('diabetes_predictions.csv', index=False)

# Save model performance
with open('diabetes_model_performance.txt', 'w') as f:
    f.write("DIABETES PREDICTION MODEL PERFORMANCE\n")
    f.write("="*50 + "\n\n")
    f.write(f"Model: Ensemble (LR + RF + GB + SVM)\n")
    f.write(f"ROC-AUC Score: {roc_auc:.4f}\n")
    f.write(f"Sensitivity: {sensitivity:.1%}\n")
    f.write(f"Specificity: {specificity:.1%}\n")
    f.write(f"PPV: {ppv:.1%}\n")
    f.write(f"NPV: {npv:.1%}\n")
    f.write(f"Number Needed to Screen: {nns if ppv > 0 else 'N/A'}\n\n")
    f.write("Top Risk Factors:\n")
    for idx, row in feature_importance.head(10).iterrows():
        f.write(f"• {row['feature']}: {row['importance']:.4f}\n")

# Save risk stratification
risk_summary.to_csv('diabetes_risk_stratification.csv')

print("\nFiles saved:")
print("• diabetes_predictions.csv - Individual risk predictions")
print("• diabetes_model_performance.txt - Model metrics")
print("• diabetes_risk_stratification.csv - Risk category analysis")

print("\n" + "="*80)
print("DIABETES PREDICTION MODEL READY FOR CLINICAL VALIDATION")
print("Prevention today prevents ICU admission tomorrow")
print("="*80)