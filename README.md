Dynamic Pricing Optimization

An end-to-end Machine Learning + Data Analytics project that analyzes retail sales data, customer demand, competitor pricing, and revenue trends to build an intelligent pricing prediction system.

Project Overview

This project focuses on dynamic pricing optimization using retail transaction data.

The system:

Performs Exploratory Data Analysis (EDA)
Analyzes:
Revenue trends
Demand trends
Competitor price gaps
Product performance
Builds Machine Learning models to:
Predict product prices
Analyze demand behavior
Compares multiple ML algorithms using evaluation metrics:
MAE
RMSE
R² Score

The project demonstrates how businesses can use data-driven pricing strategies to maximize revenue and improve decision-making.

Problem Statement

Traditional pricing strategies are static and do not react to:

Market demand
Competitor prices
Seasonal trends
Customer behavior

This project aims to build a system that can:

Analyze retail market conditions
Predict optimal pricing behavior
Understand revenue and demand patterns
Compare different ML models for pricing accuracy
Technologies Used
Programming Language
Python
Libraries & Frameworks
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
├── src/
│   └── dynamic_pricing.py
│
├── README.md
├── requirements.txt
├── .gitignore
│
└── venv/
Dataset Information

The dataset contains around:

50,000 rows
30 columns

Important features include:

Feature	Description
unit_price	Product selling price
qty	Quantity sold
total_price	Total transaction amount
comp_1, comp_2, comp_3	Competitor prices
customers	Number of customers
holiday	Holiday indicator
weekend	Weekend indicator
revenue	Generated revenue
lag_price	Previous pricing values
Workflow
1. Data Loading

The CSV dataset is loaded using Pandas.

df = pd.read_csv("data/retail_price_data.csv")
2. Data Cleaning

Performed operations:

Checked dataset shape
Removed duplicate rows
Converted date columns
Handled missing values

Output:

Shape : (50000, 30)
Duplicate rows: 0
Exploratory Data Analysis (EDA)

EDA helps understand data patterns before building ML models.

Distribution of Unit Price

Shows how product prices are distributed.

Most products lie in the low-to-medium price range
Few expensive products create a long-tail distribution

Distribution of Total Price

Analyzes transaction value distribution.

Observations:

Most purchases are lower-value transactions
Few high-value purchases generate large revenue

Distribution of Quantity Sold

Shows demand behavior.

Observations:

Smaller quantities are sold more frequently
High quantities occur less often

Monthly Revenue Trend

Tracks how revenue changes over time.

Insights:

Revenue fluctuates monthly
Some seasonal spikes are visible

Correlation Heatmap

Displays relationships between numerical variables.

Insights:

Strong positive relationships between:
quantity
revenue
lag prices
Competitor prices influence product pricing

Machine Learning Models Used

The project compares multiple regression algorithms.

Model	Purpose
Linear Regression	Baseline prediction model
Ridge Regression	Handles multicollinearity
Random Forest	Ensemble learning model
XGBoost	Advanced boosting algorithm
Evaluation Metrics
MAE (Mean Absolute Error)

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

Model Comparison Results
Model	MAE	RMSE	R²
Linear Regression	Very Low	Very Low	1.000
Ridge Regression	Low	Low	1.000
Random Forest	Moderate	Moderate	0.999
XGBoost	Moderate	Moderate	0.999

Best model selected:

Best price model: Linear Regression
Key Insights
Revenue strongly depends on:
Quantity sold
Product pricing
Competitor pricing
Competitor prices significantly influence market pricing
Linear Regression performed exceptionally well on this dataset
Demand and revenue show seasonal variations
Future Improvements

Possible enhancements:

Deploy using Flask or Streamlit
Add real-time pricing recommendations
Use Deep Learning models
Integrate live competitor pricing APIs
Build interactive dashboards
How to Run the Project
Clone Repository
git clone https://github.com/YOUR_USERNAME/Dynamic_Pricing_Optimization.git
Create Virtual Environment
python -m venv venv

Activate:

Windows
venv\Scripts\activate
Install Dependencies
pip install -r requirements.txt
Run Project
python src/dynamic_pricing.py
Sample Output
Shape : (50000, 30)
Duplicate rows: 0

--- EDA ---
Share of rows in top 1% revenue: 0.010

--- ML: Price Prediction (target = unit_price) ---

Best price model: Linear Regression
Applications

This project can be used in:

E-commerce platforms
Retail businesses
Inventory optimization
Smart pricing systems
Revenue forecasting
Author

Shrujana

Machine Learning & Data Analytics Project

GitHub: https://github.com/DudalaShrujana