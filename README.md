# Tennis Match Outcome Prediction

## Project Overview

This repository presents a machine learning solution for predicting the outcome of professional men's tennis matches. Leveraging a dataset of over 36,000 matches, the project applies multiple supervised learning models, including Lasso, XGBoost, and CatBoost, to assess the probability that a given player will win a match.

By incorporating player attributes, match conditions, and bookmaker odds, this solution aims to enable accurate and interpretable match forecasting. The final model is deployed through a Streamlit interface, enabling practical use by analysts, coaches, and researchers in sports analytics.

---

## Data Description

The dataset includes 36,005 professional ATP-level men's matches and contains metadata, player statistics, and bookmaker-provided betting odds.

### Core Fields:

- **ATP**: Unique tournament identifier  
- **Location**: Venue where the match took place  
- **Tournament**: Name of the tournament  
- **Date**: Match date  
- **Series**: ATP series (e.g., Grand Slam, Masters 1000)  
- **Court**: Indoor or outdoor court type  
- **Surface**: Surface type (Hard, Clay, Grass)  
- **Round**: Tournament round (e.g., Quarterfinal, Final)  
- **Best of**: Maximum number of sets played  
- **Winner / Loser**: Player names  
- **WRank / LRank**: ATP rankings of the winner and loser  
- **WPts / LPts**: ATP ranking points of the winner and loser  
- **W1–W5 / L1–L5**: Games won in sets 1–5 by each player  
- **Wsets / Lsets**: Total sets won by each player  
- **Comment**: Match completion status (e.g., Completed, Walkover, Retirement)  
- **Odds**: Betting odds from various sources including Bet365, Expekt, and Unibet  
- **pl1_*** and **pl2_***: Player features such as height, weight, handedness, nationality, and pro start year

---

## Exploratory Data Analysis (EDA)

Comprehensive exploratory analysis was conducted to guide model selection and feature design. The following aspects were examined:

- **Univariate Distributions**: Histograms and boxplots of numerical features such as rank, odds, and physical attributes revealed skewness, outliers, and missing data.
- **Categorical Distributions**: Frequency plots of surface types, rounds, and court categories identified dominant classes and potential imbalance.
- **Match Duration**: Distributions of sets played and games per set across different surfaces helped assess surface impact.
- **Bookmaker Odds**: Distributions of `AvgW`, `AvgL`, and `log_Avg_ratio` showed patterns between early and late rounds.
- **Temporal Trends**: Match dates were analyzed to ensure temporal integrity of train/test splits and avoid leakage.
- **Correlations**: Pearson correlation matrices were used to identify and eliminate multicollinearity (e.g., between `WRank` and `WPts`).

These findings informed both the feature engineering strategy and the model evaluation process.

---

## Feature Engineering

To enhance predictive performance and capture non-linear patterns, a variety of transformations were applied:

- **Ranking Features**:
  - `rank_diff`: Difference between ATP ranks of Player 1 and Player 2.
  - `custom_log_rank_diff`: Log-transformed signed difference to smooth large disparities.
- **Odds-Based Features**:
  - `bet_diff_Avg`: Difference between average odds of the two players.
  - `log_Avg_ratio`: Logarithmic ratio of the odds to indicate relative favoritism.
- **Polynomial Features**:
  - Squared and interaction terms for `rank_diff`, `height_diff`, and `weight_diff` to capture non-linearity.
- **Interaction Terms**:
  - `rank_surface_Hard_interaction`: Multiplies rank difference by an indicator for hard court.
- **Categorical Encoding**:
  - Categorical fields (e.g., `Surface`, `Series`, `Round`, `Court`) encoded via `LabelEncoder` or passed natively to CatBoost.
- **Standardization**:
  - Numerical features were scaled to have zero mean and unit variance using `StandardScaler`.

The final dataset included approximately 50 features, with careful attention paid to avoiding data leakage and preserving interpretability.

---

## Modeling Pipeline

### Models Implemented:

1. **Lasso Logistic Regression**
   - L1-penalized linear model for baseline interpretability
   - AUC: 0.69 | Accuracy: 0.65 | F1-score: 0.66

2. **XGBoost**
   - Gradient boosting with decision trees
   - AUC: 0.72 | Accuracy: 0.66 | F1-score: 0.65

3. **CatBoost**
   - Gradient boosting with categorical support
   - AUC: 0.72 | Accuracy: 0.66 | F1-score: 0.66

4. **CatBoost Extended**
   - Includes betting odds as predictive features
   - AUC: 0.85 | Accuracy: 0.75 | F1-score: 0.75

All models were trained using stratified 5-fold cross-validation and were calibrated using isotonic or Platt scaling for improved probability estimates.

---

## Evaluation

Models were evaluated based on:

- Accuracy  
- ROC AUC  
- F1-score  
- Confusion matrices  
- Feature importance (via model-native tools)

| Model             | Accuracy | AUC   | F1    | Training Time |
|------------------|----------|-------|-------|----------------|
| Lasso            | 0.65     | 0.69  | 0.66  | ~2.5 minutes   |
| XGBoost          | 0.66     | 0.72  | 0.65  | ~0.5 minutes   |
| CatBoost         | 0.66     | 0.72  | 0.66  | ~15 minutes    |
| CatBoost Extended| 0.75     | 0.85  | 0.75  | ~9 minutes     |

---

## Statistical Hypothesis Testing

Two hypotheses were tested using the Kolmogorov–Smirnov test:

1. **Surface Effect on Match Duration**:  
   - Result: No significant difference between Hard and Clay surfaces  

2. **Bookmaker Odds by Match Round**:  
   - Result: Statistically significant difference between early and final rounds  
   - Implication: Odds provide stronger signals in later rounds

---

## Streamlit Application

An interactive application built using **Streamlit** allows users to:

- Input match data manually (e.g., player ranks, odds, surface)  
- Receive predicted probabilities of Player 1 winning  
- Visualize confidence and key prediction features

The deployed model is the calibrated **CatBoost Extended**.

---

## Conclusion

This project demonstrates that accurate and practical prediction of tennis match outcomes is achievable through the use of gradient boosting models, domain-specific feature engineering, and bookmaker odds. The inclusion of odds-based features significantly enhances model performance, as evidenced by the CatBoost Extended model outperforming all other approaches.

The work confirms that even in structured sports like tennis, external contextual data (e.g., odds) provides predictive value beyond traditional player and match statistics. Furthermore, the integration of this pipeline into a user-facing Streamlit app highlights the potential for real-time application in sports analytics, coaching, and media coverage.

This study contributes a robust, interpretable, and reproducible pipeline for outcome prediction and sets the foundation for future advancements incorporating dynamic, video-based, or multi-modal data.

---

## Limitations

- Model does not account for dynamic in-season factors (e.g., fatigue, injury)  
- Dataset excludes lower-tier events and WTA matches  
- Training time may be non-trivial on constrained hardware  
- Betting odds are not always available for historical matches

---

## Future Work

1. Include real-time and dynamic features (recent form, injury news)  
2. Use computer vision to track player behavior and shot patterns  
3. Deploy real-time prediction tools for live betting or commentary  
4. Expand dataset to include WTA, ITF, and Challenger matches  
5. Evaluate fairness and demographic bias in prediction performance

---

## Repository Structure and File Descriptions

| File | Description |
|------|-------------|
| `.catboost.info` | Internal CatBoost training log file; used for model diagnostics |
| `.gitignore` | Specifies files and directories ignored by Git |
| `catboost_extended_model.cbm` | Trained CatBoost Extended model file including bookmaker odds |
| `catboost_model.cbm` | Trained CatBoost model file without betting data |
| `Documentation.pdf` | Full research paper with detailed analysis, methodology, and results |
| `hypotheses.ipynb` | Jupyter notebook performing statistical hypothesis testing (e.g., KS Test) |
| `lasso_model.pkl` | Pickled Lasso Logistic Regression model |
| `README.md` | Project documentation file (this document) |
| `streamlit_tennis.py` | Streamlit app script for deploying prediction interface |
| `tennis_data_cleaned.csv` | Preprocessed and cleaned version of the raw dataset |
| `tennis_data_corrected.csv` | Version of dataset with corrections applied (e.g., missing values) |
| `tennis_data_processing.ipynb` | Main notebook for feature engineering and model training |
| `tennis_data.csv` | Raw dataset containing all original match records |
| `xgboost_model.pkl` | Pickled XGBoost model object |

---

## Citation

If you use this project in your work, please cite the original report `Documentation.pdf`.

---

## Contact

For questions or collaboration inquiries, please contact the project author via the HSE University Faculty of Computer Science.
