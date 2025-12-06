# Tel Aviv Rent Price Forecasting 🏠

## Overview

This project focuses on predicting residential rental prices in **Tel Aviv, Israel**, using machine learning techniques. The goal was to build a robust regression model that estimates monthly rent based on property characteristics, location data, and socio-economic factors.

The project handles the full data science lifecycle: data cleaning, feature engineering (including external API integration), model selection, hyperparameter tuning, and evaluation.

## 🛠️ Tech Stack

  * **Language:** Python 3.x
  * **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Requests, Missingno.
  * **Techniques:** NLP (Regex), API Integration (Google Maps), Pipelines, Regularization (ElasticNet), Ensemble Learning (Random Forest).

## 📊 Data Processing & Feature Engineering

The raw data required significant cleaning and enrichment to improve model performance. Key steps included:

### 1\. Data Cleaning

  * **Outlier Removal:** Filtered out extreme prices (kept range 2,500–40,000 NIS) and commercial properties (stores, parking spots, sublets).
  * **Imputation via NLP:** Used **Regular Expressions (Regex)** to extract missing `Room Number` and `Area (sq meters)` directly from the free-text `Description` column.
  * **Logic-based Imputation:** Filled missing floor numbers based on median values and keywords (e.g., "Ground floor" = 0, "Basement" = -1).

### 2\. Feature Engineering

  * **📍 Distance from City Center:** Utilized the **Google Maps Distance Matrix API** to calculate the precise distance from each apartment to the city center (Dizengoff Center), replacing rough estimates.
  * **💰 Socio-Economic Scoring:** Created a custom mapping dictionary to assign a socio-economic rank (1-10) to specific neighborhoods based on municipal data.
  * **📝 Text Features:** Generated features based on the length and word count of the property description.
  * **Standardization:** Normalized `Property Type` into broader categories (e.g., Apartment, Garden Apartment, Penthouse/Roof, Unit).

## 🤖 Models & Methodology

I implemented Scikit-Learn **Pipelines** to prevent data leakage and ensure reproducible preprocessing (OneHotEncoding for categoricals, Scaling for numerics).

Two primary models were evaluated and optimized using `RandomizedSearchCV`:

1.  **Elastic Net Regression:** A linear model combining L1 (Lasso) and L2 (Ridge) regularization to handle multicollinearity and perform feature selection.
2.  **Random Forest Regressor:** A tree-based ensemble method to capture non-linear relationships and feature interactions.

## 📈 Results

The models were evaluated using RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), and $R^2$.

| Model | $R^2$ Score | RMSE | MAE |
| :--- | :--- | :--- | :--- |
| **Random Forest** | **0.63** | **2,843** | **1,825** |
| Elastic Net | 0.61 | 2,945 | 1,886 |

**Key Findings:**

  * **Random Forest** slightly outperformed Elastic Net, suggesting non-linear relationships in the data.
  * **Feature Importance:** The most critical predictor of price was **Apartment Area ($m^2$)**, followed by **Distance from Center** and specific **Neighborhoods**.

## 🚀 Getting Started

### Prerequisites

1.  Clone the repo:
    ```bash
    git clone https://github.com/KD-g/Rent-Price-Forecasting-Project.git
    ```
2.  Install required packages:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn requests missingno
    ```

### Usage

1.  Ensure `train.csv` (and optionally `test.csv`) is in the root directory.
2.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook "Rent Price Forecasting Project (1).ipynb"
    ```
3.  **Note on API:** The code uses the Google Maps API. If running the distance calculation cells afresh, ensure you provide a valid API key in the `get_distance` function.

## 📂 Repository Structure

  * `Rent Price Forecasting Project (1).ipynb`: The main notebook containing analysis, cleaning, and modeling code.
  * `train.csv`: The training dataset (not included in repo, assumes local availability).

## 👤 Author

**Kfir Diamond**

  * [LinkedIn](https://www.linkedin.com/in/kfir-diamond-631571266/)

-----

*This project was developed as part of a Data Mining course (2025).*
