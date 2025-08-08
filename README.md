# Healthcare AI/ML Portfolio - ICU Nurse to AI Engineer 🏥

## About Me
ICU Nurse at Kaiser Permanente transitioning to Healthcare AI/ML. Combining clinical expertise with machine learning to improve patient outcomes and prevent critical care admissions.

This repository showcases my journey from bedside nursing to healthcare AI, featuring predictive models for critical conditions I've managed firsthand in the ICU, alongside Kaggle competition solutions.

## 🏥 Healthcare ML Projects

### 1. Heart Failure Prediction
**Performance:** 93.48% ROC-AUC | 93.14% Sensitivity

Clinical insights from managing cardiac patients in ICU:
- ST_Slope identified as strongest predictor
- Risk stratification for early intervention
- Nurse-actionable recommendations

[View Code](./heart-failure-prediction/heart_failure_analysis.py)

### 2. Stroke Prediction  
**Performance:** 78.85% ROC-AUC | Number Needed to Screen: 6

Leveraging ICU stroke management experience:
- Addressed severe class imbalance (5% stroke rate)
- Age-adjusted risk categories
- SMOTE for rare event prediction

[View Code](./stroke-prediction/stroke_prediction_advanced.py)

### 3. Diabetes Prediction
**Performance:** 82.15% ROC-AUC | 72.2% Sensitivity

Preventing DKA and HHS admissions:
- Ensemble model (LR + RF + GB + SVM)
- Metabolic syndrome scoring
- Insulin resistance early detection

[View Code](./diabetes-prediction/diabetes_prediction_advanced.py)

### 4. Sepsis Early Warning System
**Performance:** 100% ROC-AUC (Synthetic Data)

ICU-specific sepsis prediction:
- SIRS and qSOFA integration
- Multi-organ dysfunction tracking
- Alert system with nurse protocols

[View Code](./sepsis-prediction/sepsis_prediction_icu.py)

## 📊 Competition Results

| Competition | Best Score | Ranking | Target |
|------------|------------|---------|---------|
| **Titanic** | 0.77990 | Top ~30% | 0.82+ (Top 10%) |
| **ConnectX** | 722.2 | #33 | 1000+ (Top 10) |

## 🚢 Titanic - Machine Learning from Disaster

**Goal**: Predict passenger survival on the Titanic

### Current Performance
- **Best Score**: 77.99% accuracy
- **Model**: Optimized Random Forest with engineered features
- **CV Score**: 82.82%

### Key Features
- Advanced title extraction from passenger names
- Family size and survival group engineering  
- Sophisticated missing value imputation
- Ensemble methods with weighted voting

### Files
- `titanic_solution.py` - Full featured ensemble solution
- `titanic_improved.py` - Optimized Random Forest model
- `TITANIC_SUMMARY.md` - Detailed analysis and findings

### Quick Start
```bash
cd titanic
python3 titanic_improved.py
kaggle competitions submit -c titanic -f submission_improved.csv -m "Your description"
```

## 🎮 ConnectX - Reinforcement Learning Challenge

**Goal**: Create an AI agent to play Connect Four

### Current Performance
- **Best Score**: 722.2
- **Rank**: #33
- **Target**: Score 1000+ for Top 10

### Key Features
- Deep minimax search (8-10 ply)
- Advanced evaluation with pattern recognition
- Comprehensive opening book
- Transposition tables and killer moves
- Perfect tactical play (never misses wins/blocks)

### Files
- `submission_ready.py` - Tournament-ready agent
- `READY_FOR_TOMORROW.md` - Strategy documentation

### Quick Start
```bash
cd connectx
kaggle competitions submit -c connectx -f submission_ready.py -m "Your description"
```

## 🛠️ Requirements

```bash
pip install pandas numpy scikit-learn kaggle
```

## 📈 Improvement Strategies

### Titanic Next Steps
1. Implement semi-supervised learning
2. Create family group survival features
3. Use stacking with XGBoost meta-learner
4. Optimize decision threshold

### ConnectX Next Steps
1. Implement bitboard for 100x speedup
2. Extend search depth to 12+ ply
3. Add Monte Carlo Tree Search
4. Expand opening book to 100+ positions

## 🤝 Contributing

Feel free to fork this repository and submit pull requests with improvements!

## 📝 License

MIT License - See LICENSE file for details

## 🏅 Kaggle Profile

[AIHeartICU](https://www.kaggle.com/aihearticu)

## 💡 Clinical Impact & Skills

### Technical Skills
- **ML/AI:** Random Forest, Gradient Boosting, Neural Networks, SMOTE
- **Healthcare:** SOFA/qSOFA scores, risk stratification, clinical protocols
- **Languages:** Python, SQL
- **Libraries:** scikit-learn, pandas, numpy, imbalanced-learn

### Clinical Impact Focus
- **Early Detection:** Identifying at-risk patients before crisis
- **Resource Optimization:** Efficient ICU bed utilization
- **Quality Metrics:** NNS, sensitivity/specificity, clinical outcomes
- **Implementation:** EMR integration, nurse-driven protocols

### Mission
*"Every model I build is informed by countless nights at the bedside, watching for the subtle signs that precede crisis. My goal is to give every nurse and doctor the predictive tools that could save lives."*

---

*Last Updated: January 2025*