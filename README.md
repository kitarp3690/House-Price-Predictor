# House Price Prediction

A machine learning project that predicts residential house prices using the Ames Housing Dataset. Built with Linear Regression covering the full data science pipeline — from raw data to an evaluated, exportable model.

---

## Results

| Metric | Score |
|---|---|
| R² Score | 0.8978 |
| RMSE | 0.1375 (log scale) |
| Avg Actual Price | $189,651 |
| Avg Predicted Price | $185,860 |
| Training Set | 2,344 houses |
| Test Set | 586 houses |

The model explains **89.78%** of the variance in house prices.

---

## Tech Stack

- **Python** — core language
- **Pandas** — data loading, cleaning, manipulation
- **NumPy** — numerical operations, log transform
- **Matplotlib** — EDA visualisations, correlation charts
- **Scikit-learn** — model training, scaling, evaluation
- **Jupyter Notebook** — development environment

---

## Dataset

**Ames Housing Dataset** — 2,930 residential house sales in Ames, Iowa with 82 features covering size, quality, age, and condition.

Source: [Kaggle — Ames Housing Dataset](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset)

---

## Project Structure

```
house-price-prediction/
│
├── data/
│   └── AmesHousing.csv        # raw dataset
│
├── house_price.ipynb          # main notebook
├── house_model_bundle.pkl     # saved model + scaler
└── README.md
```

---

## Pipeline

### 1. Data Cleaning
- Dropped columns with 90%+ missing values (`Pool QC`, `Misc Feature`, `Alley`)
- Filled categorical nulls with `"None"` (no garage, no fireplace, etc.)
- Filled numeric nulls with median or `0`
- Applied log transform to `SalePrice` to fix right-skewed distribution

### 2. Feature Engineering
- `Total SF` — basement + above ground living area combined
- `House Age` — years between build year and sale year
- `Remod Age` — years since last remodel
- `Total Baths` — full + half baths across all floors
- `Has Garage`, `Has Fireplace`, `Has Basement` — binary presence features

### 3. Top Predictive Features

| Feature | Correlation with Price |
|---|---|
| Overall Qual | 0.83 |
| Total SF | 0.78 |
| Gr Liv Area | 0.70 |
| Total Baths | 0.67 |
| Garage Area | 0.65 |
| House Age | -0.62 |

### 4. Model Training
- 80/20 train-test split with `random_state=42`
- Features scaled with `StandardScaler`
- Trained `LinearRegression` from Scikit-learn

### 5. Evaluation
- RMSE of 0.1375 on log scale
- R² of 0.8978 — model explains ~90% of price variance

---

## How to Run

1. Clone the repository and navigate to the project folder
2. Install dependencies
   ```bash
   pip install pandas numpy matplotlib scikit-learn jupyter
   ```
3. Download `AmesHousing.csv` from Kaggle and place it in the `data/` folder
4. Open the notebook
   ```bash
   jupyter notebook house_price.ipynb
   ```
5. Run all cells from top to bottom

---

## Loading the Saved Model

```python
import pickle
import numpy as np

with open('house_model_bundle.pkl', 'rb') as f:
    bundle = pickle.load(f)

model  = bundle['model']
scaler = bundle['scaler']

# Example prediction (use same 18 features in same order)
sample = scaler.transform([your_feature_array])
log_price = model.predict(sample)
price = np.exp(log_price)
print(f"Predicted price: ${price[0]:,.0f}")
```
