# 🚗 Australian Vehicle Price Prediction

An end-to-end machine learning pipeline that predicts the resale price (AUD) of used and new vehicles listed in Australia. The project covers data cleaning, leak-free imputation, exploratory analysis, feature engineering, model comparison, and a formal set of regression diagnostics that explain *why* linear models fall short on this problem.


---
## 🛠️ Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `statsmodels` · `SciPy` · `Matplotlib` · `Seaborn`

---

## 📌 Highlights

- **16,734 vehicle listings**, 19 raw columns (brand, model, year, kilometres, engine, fuel, transmission, body type, location, etc.).
- **No data leakage**: the train/test split happens *before* any imputation, grouping, or binning. Every statistic is learned from the training set only.
- **Champion model: Random Forest**, with a test R² of **0.896** on log-price and a test MAE of about **AUD 6,191**, roughly AUD 1,500 better than the OLS baseline.
- **Statistical rigor**: Durbin-Watson, Breusch-Pagan, Shapiro-Wilk, Jarque-Bera, VIF, and Cook's Distance are used to justify the choice of model.

## 🗂️ Data

The dataset is the **Australian Vehicle Prices** listing data (Kaggle). Each row is one vehicle listing.

| Type | Columns |
| :--- | :--- |
| Target | `Price` (AUD) |
| Numeric | `Year`, `Kilometres`, `Engine`, `CylindersinEngine`, `FuelConsumption`, `Doors`, `Seats` |
| Categorical | `Brand`, `Model`, `UsedOrNew`, `Transmission`, `DriveType`, `FuelType`, `BodyType`, `ColourExtInt`, `Location` |


---

## 🔬 Methodology

### 1. Data Cleaning
- Dropped `Car/Suv` (a noisier duplicate of `BodyType`), `Title`, and the high-cardinality sub-model/feature text.
- Dropped `ColourInt` because about 46% of its values were missing.
- Split `ColourExtInt` into exterior and interior colour, and extracted `State` from `Location`.
- Converted placeholder values (`-`, `POA`, `0`) to `NaN`.
- Parsed numeric values out of text fields: engine capacity (`EngineCC`), cylinders, fuel consumption, doors, and seats.
- Replaced junk colour labels such as `"3 years"` (warranty text) with `Unknown`.

### 2. Leak-Free Train/Test Split and Imputation
The data is split **80/20** (`random_state=42`) before any imputation. All fill values come from the training set only.

| Features | Imputation strategy |
| :--- | :--- |
| `Doors`, `Seats`, `FuelType` | Hierarchical mode: (Brand, BodyType) → BodyType / Brand → global |
| `EngineCC`, `CylindersinEngine`, `FuelConsumption` | **Hot-deck random sampling** from similar cars: (Brand, BodyType, FuelType) → (Brand, BodyType) → Brand → global. This avoids the artificial spike a single median creates. |
| `Kilometres` | Grouped median: (UsedOrNew, Year) → UsedOrNew → global |
| `Transmission` | Hierarchical mode: (Brand, Year) → Brand → global |
| `BodyType`, `ColourExt`, `State` | Filled with `Unknown` |

**Missingness indicators** (`specs_missing`, `km_missing`) are added so the models can tell imputed rows apart. The notebook found that the five spec columns tend to be missing *together* (not at random), which is why a flag is useful.

### 3. Exploratory Data Analysis
- Price is strongly right-skewed, so the target is modelled as `log1p(Price)`.
- Depreciation curves for `Year` and `Kilometres` against log-price.
- Price comparisons across `UsedOrNew`, `Transmission`, and `BodyType`.
- The 15 most common brands against their median price.
- A correlation heatmap of the numeric features, which shows `EngineCC` ↔ `CylindersinEngine` (r ≈ 0.75) and `Year` ↔ `Kilometres` (r ≈ −0.72).

### 4. Feature Engineering
- **Outlier filtering (train only):** `1,500 ≤ Price ≤ 500,000`, `Year ≥ 1995`, `Kilometres ≤ 500,000`. Test rows are never dropped, so evaluation stays honest. Test mileage is capped at the training 99.5th percentile.
- **`Vehicle_Age`** = 2024 − `Year`.
- **`Market_Segment`**: a 5-tier feature (`Budget`, `Economy`, `Mid_Range`, `Premium`, `Luxury`) built from the median price of each (Brand, BodyType) pair in the training set. This replaces 600+ sparse one-hot `Model` columns.
- One-hot encoding (`drop_first=True`) with train/test column alignment (131 final features), and `StandardScaler` on the continuous columns.

### 5. Modelling
Four models are trained on `log1p(Price)` and evaluated on the held-out test set:

- OLS Linear Regression
- Ridge Regression (`alpha=10`)
- Decision Tree (`max_depth=10`, `min_samples_leaf=10`)
- Random Forest (`n_estimators=200`, `max_depth=15`, `min_samples_leaf=3`)

---

## 📊 Results

| Model | Train R² (log) | Test R² (log) | Test R² (AUD) | Test MAE (AUD) | Test RMSE (AUD) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| OLS Linear Regression | 0.8517 | 0.8489 | 0.7067 | $7,681 | $20,344 |
| Ridge Regression (α = 10) | 0.8463 | 0.8414 | 0.6344 | $8,051 | $22,713 |
| Decision Tree | 0.8776 | 0.8184 | 0.6720 | $8,250 | $21,512 |
| **Random Forest** | **0.9580** | **0.8957** | **0.7774** | **$6,191** | **$17,721** |

### Why not a linear model? Regression diagnostics

| Diagnostic | Result | Takeaway |
| :--- | :--- | :--- |
| Durbin-Watson | 2.0364 | Errors are independent (passed) |
| Breusch-Pagan |  p < 0.001 | Heteroscedasticity: luxury and older cars have wider price spread |
| Shapiro-Wilk |  p < 0.001 | Heavy-tailed residuals from flagship and performance trims |
| Cook's Distance (> 4/n) | 671 rows (5.05% of train) | A noticeable share of influential points |
| Correlation / VIF | r = 0.75 and −0.72 | Multicollinearity, which motivates regularisation |

These tests explain why tree-based ensembles outperform OLS here: they handle non-linearity, interactions, and non-normal errors without distributional assumptions.

---

