Dynamic Pricing Optimization

A Machine Learning project that analyzes retail pricing, customer demand, competitor pricing, and revenue trends to build an intelligent dynamic pricing system.

Project Overview

This project focuses on:

Exploratory Data Analysis (EDA)
Revenue trend analysis
Demand forecasting
Competitor price analysis
Machine Learning price prediction
Model comparison using regression metrics

The system helps businesses make data-driven pricing decisions.

Technologies Used
Programming Language
Python
Libraries
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
XGBoost
Statsmodels
Tools
VS Code
Git
GitHub

Project Structure

Dynamic_Pricing/
│
├── data/
│   └── retail_price_data.csv
│
├── screenshots/
│   ├── correlation_heatmap.png
│   ├── unit_price_distribution.png
│   ├── total_price_distribution.png
│   ├── qty_distribution.png
│   ├── monthly_revenue_trend.png
│   └── model_comparison.png
│
├── src/
│   └── dynamic_pricing.py
│
├── requirements.txt
├── README.md
├── .gitignore
│
└── venv/
 
 Dataset Information

Dataset contains:

50,000 rows
30 columns

Important features:

| Feature     | Description             |
| ----------- | ----------------------- |
| unit_price  | Product selling price   |
| qty         | Quantity sold           |
| total_price | Total transaction value |
| comp_1      | Competitor 1 price      |
| comp_2      | Competitor 2 price      |
| comp_3      | Competitor 3 price      |
| revenue     | Revenue generated       |
| customers   | Customer count          |
| holiday     | Holiday indicator       |

Workflow
1. Data Loading

Dataset is loaded using Pandas.

df = pd.read_csv("data/retail_price_data.csv")
2. Data Cleaning

Performed operations:

Duplicate removal
Date conversion
Missing value handling
Feature engineering

Output:

Shape : (50000, 30)
Duplicate rows: 0
Exploratory Data Analysis (EDA)

EDA helps understand pricing behavior, demand patterns, and relationships between variables.

Distribution of Unit Price
Most products are in low-to-medium price ranges
Few expensive products create a long-tail distribution

Distribution of Total Price
Most transactions are low-value purchases
Few transactions contribute very high revenue

Distribution of Quantity Sold
Small quantities are sold more frequently
High quantity purchases are less common

Monthly Revenue Trend
Revenue changes month-to-month
Seasonal fluctuations are visible

Correlation Heatmap

Shows relationships between numerical features.

Key insights:

Revenue strongly correlates with quantity
Competitor prices affect product pricing
Lag prices influence future pricing

Machine Learning Models Used

The project compares multiple regression algorithms.

| Model             | Purpose                     |
| ----------------- | --------------------------- |
| Linear Regression | Baseline prediction model   |
| Ridge Regression  | Reduces overfitting         |
| Random Forest     | Ensemble learning           |
| XGBoost           | Advanced boosting algorithm |
|_________________________________________________|

Measures average prediction error.

Formula:

MAE = Average(|Actual - Predicted|)

Lower MAE = Better model

RMSE (Root Mean Squared Error)

Penalizes large prediction errors more heavily.

Formula:

RMSE = sqrt(mean((Actual - Predicted)^2))

Lower RMSE = Better model

R² Score

Measures how well the model explains the data.

Range:

1 → Perfect prediction
0 → Poor prediction

Higher R² = Better model

Model Comparison Output
Model comparison (Price prediction):

Model                  MAE         RMSE        R2
Linear Regression      0.0000      0.0000      1.0000
Ridge Regression       0.0043      0.0065      1.0000
Random Forest          0.2714      0.7287      0.9998
XGBoost                0.7258      1.0825      0.9996

Best model selected:

Best price model: Linear Regression
Key Insights
Revenue depends strongly on:
Product pricing
Quantity sold
Competitor prices
Competitor pricing directly affects market behavior
Linear Regression performed best on this dataset
Seasonal trends impact revenue and demand
Future Improvements

Possible future enhancements:

Real-time pricing engine
Flask or Streamlit deployment
Dashboard integration
Deep Learning models
Live competitor pricing APIs
How to Run the Project
Clone Repository
git clone https://github.com/YOUR_USERNAME/Dynamic_Pricing_Optimization.git
Create Virtual Environment
python -m venv venv

Activate environment:

Windows
venv\Scripts\activate
Install Requirements
pip install -r requirements.txt
Run Project
python src/dynamic_pricing.py
Requirements
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
statsmodels
Applications
E-commerce pricing
Retail analytics
Revenue optimization
Demand forecasting
Smart pricing systems
Author

Shrujana

GitHub:
https://github.com/DudalaShrujana