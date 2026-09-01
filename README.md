# Australian Vehicle Price Prediction

An end-to-end data science project that cleans, explores, and models the **Australian Vehicle Prices** dataset to understand what drives used-car prices in Australia and to predict them using machine learning.

## Overview

This notebook walks through the full data science pipeline:

1. **Data Cleaning & Preprocessing** — handling messy string fields, missing values, and type conversions
2. **Exploratory Data Analysis (EDA)** — visualizing distributions, relationships, and correlations
3. **Regression Assumption Checks** — linearity, autocorrelation, heteroscedasticity, multicollinearity, influential points, and normality
4. **Model Building** — Ridge Regression, Decision Tree, and Random Forest regressors
5. **Model Evaluation & Feature Importance**
6. **Conclusions & Insights**

## Dataset

The analysis uses `Australian Vehicle Prices.csv`, a dataset of car listings in the Australian market (2023), containing features such as:

- `Brand`, `Model`, `Year`, `UsedOrNew`
- `Transmission`, `DriveType`, `FuelType`, `FuelConsumption`
- `Engine`, `CylindersinEngine`, `Doors`, `Seats`
- `BodyType`, `Location`, `Kilometres`, `Price`


## Workflow

### 1. Data Cleaning & Preprocessing
- Dropped high-cardinality/unusable columns (`Model`, `Car/Suv`, `Title`, `ColourExtInt`)
- Parsed compound string fields (`Engine`, `FuelConsumption`, `Location`, `CylindersinEngine`, `Doors`, `Seats`) into usable numeric values
- Replaced placeholder values (`-`, `POA`) with `NaN`
- Dropped rows missing critical fields; imputed missing values in numeric fields using the mode
- Cast columns to appropriate data types

### 2. Exploratory Data Analysis
- Count plots for `Brand`, `UsedOrNew`, `Transmission`, `DriveType`, `FuelType`, `Location`, `BodyType`
- Scatter plots of `Price` vs. `Kilometres`, `Year`, and `FuelConsumption`
- Outlier identification and removal (e.g., a vintage Ferrari priced far outside the normal range)
- Box plots of `Price` by `Location` and `Brand`
- Correlation heatmap and KDE plots to inspect skewness

### 3. Data Preparation
- Label-encoded `Brand`
- One-hot encoded categorical columns (`UsedOrNew`, `Transmission`, `DriveType`, `FuelType`, `BodyType`)

### 4. Regression Assumption Checks
- **Linearity** — residuals vs. fitted values plot
- **Autocorrelation** — Durbin–Watson statistic
- **Heteroscedasticity** — Breusch–Pagan test
- **Multicollinearity** — Variance Inflation Factor (VIF)
- **Influential points** — Cook's distance
- **Normality** — Q-Q plot and Shapiro–Wilk test, followed by a Box-Cox transformation of the target variable to address non-normality and heteroscedasticity

### 5. Model Building
Three regression models were trained and hyperparameter-tuned (via `GridSearchCV` where applicable):

| Model | Rationale |
|---|---|
| **Ridge Regression** | Handles multicollinearity while preserving interpretability |
| **Decision Tree Regressor** | Captures non-linear relationships without requiring transformations |
| **Random Forest Regressor** | Handles complex feature interactions on a larger dataset with reduced overfitting risk |

### 6. Model Evaluation
Models were compared using R², Adjusted R², RMSE, MAE, and MAPE.

**Best model: Random Forest Regressor**
- R² Score: **0.86**
- RMSE: **5085.20**
- MAE: **3492.49**
- MAPE: **0.134**

### 7. Feature Importance
The most influential features on predicted price were: **Year, Kilometres, DriveType_Front, Engine, Brand,** and **FuelType_Unleaded** — with vehicle age (`Year`) as the single dominant driver of price, reflecting depreciation.

## Key Insights

- **Toyota** had the highest number of sales in the Australian market in 2023, followed by Hyundai, Mazda, and Holden; **Smart** had the fewest.
- The market is dominated by **used, automatic** vehicles.
- **New South Wales** leads in sales volume, followed by Victoria.
- **SUVs** are the most in-demand body type.
- **Lamborghini** has the widest price range (highest-priced segment), followed by McLaren, Aston Martin, and Ferrari, while Nissan, Honda, and Hyundai represent more affordable brands.
- Vehicle age (`Year`) is by far the strongest driver of price — more than double the combined effect of mileage, drivetrain, and engine size.

## Requirements

```
python 3.13.9
numpy 2.3.5
pandas 2.3.3
matplotlib 3.11.0
seaborn 0.13.2
scikit-learn 1.9.0
statsmodels 0.14.5
scipy 1.16.3
```

Install with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels scipy
```

## Usage

1. Clone this repository
2. Place `Australian Vehicle Prices.csv` in the project directory
3. Open and run the notebook cell by cell:

```bash
jupyter notebook Code.ipynb
```

## Project Structure

```
.
├── Code.ipynb                          # Main analysis notebook
├── Australian Vehicle Prices.csv       # Dataset (not included, see above)
└── README.md
```
