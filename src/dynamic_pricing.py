"""
Dataset: retail_price_data.csv
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans

from xgboost import XGBRegressor

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

from sklearn.inspection import permutation_importance
from statsmodels.tsa.statespace.sarimax import SARIMAX


# ------------------------------
# Helper functions
# ------------------------------

def standardize_columns(df):
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )
    return df


def cap_iqr(s):
    s = pd.to_numeric(s, errors="coerce")
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return s.clip(low, high)


def safe_to_datetime(series, fmt="%d-%m-%Y"):
    return pd.to_datetime(series, format=fmt, errors="coerce")


# ------------------------------
# Load  data                     df = pd.read_csv("data/retail_price_data.csv")
# ------------------------------

path = r"C:\Users\Dell\OneDrive\Desktop\Dynamic_Pricing\data\\"
csv_file = "retail_price_data.csv"
df = pd.read_csv(path  + csv_file)

print("Shape :", df.shape)

df = standardize_columns(df)

# ------------------------------
# Basic Cleaning
# ------------------------------

dup = df.duplicated().sum()
print("Duplicate rows:", dup)
df = df.drop_duplicates().reset_index(drop=True)

cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
num_cols = [c for c in df.columns if c not in cat_cols]

for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

for c in num_cols:
    if df[c].isna().sum() > 0:
        df[c] = df[c].fillna(df[c].median())

for c in cat_cols:
    if df[c].isna().sum() > 0:
        df[c] = df[c].fillna(df[c].mode().iloc[0])


if "month_year" in df.columns:
    dt = safe_to_datetime(df["month_year"])
    if dt.isna().sum() > 0 and "month" in df.columns and "year" in df.columns:
        dt2 = pd.to_datetime(
            df["year"].astype(int).astype(str) + "-" +
            df["month"].astype(int).astype(str) + "-01",
            errors="coerce"
        )
        dt = dt.fillna(dt2)

    df["month_year_dt"] = dt
    df["month"] = df["month_year_dt"].dt.month.astype(int)
    df["year"] = df["month_year_dt"].dt.year.astype(int)


for col in ["unit_price", "total_price", "qty", "freight_price", "customers", "volume"]:
    if col in df.columns:
        df[col] = cap_iqr(df[col])


# ------------------------------
# Feature Engineering
# ------------------------------

for i in [1, 2, 3]:
    ccol = f"comp_{i}"
    if ccol in df.columns:
        df[f"comp{i}_diff"] = df["unit_price"] - df[ccol]

    fcol = f"fp{i}"
    if fcol in df.columns:
        df[f"fp{i}_diff"] = df["freight_price"] - df[fcol]


df["revenue"] = df["unit_price"] * df["qty"]
df["avg_spend_per_customer"] = df["total_price"] / (df["customers"] + 1e-9)

df = df.sort_values(["product_id", "month_year_dt"]).reset_index(drop=True)

df["qty_lag1"] = df.groupby("product_id")["qty"].shift(1)
df["price_lag1"] = df.groupby("product_id")["unit_price"].shift(1)

df["qty_lag1"] = df["qty_lag1"].fillna(df["qty"].median())
df["price_lag1"] = df["price_lag1"].fillna(df["unit_price"].median())

df["qty_roll3"] = (
    df.groupby("product_id")["qty"]
      .rolling(window=3, min_periods=1)
      .mean()
      .reset_index(level=0, drop=True)
)

df["price_roll3"] = (
    df.groupby("product_id")["unit_price"]
      .rolling(window=3, min_periods=1)
      .mean()
      .reset_index(level=0, drop=True)
)

df["is_q1"] = df["month"].isin([1, 2, 3]).astype(int)
df["is_q4"] = df["month"].isin([10, 11, 12]).astype(int)
df["is_festive_proxy"] = ((df.get("holiday", 0) == 1) |
                           (df.get("weekend", 0) > 0)).astype(int)

df["elasticity_cat"] = np.nan
if "product_category_name" in df.columns:
    for cat, g in df.groupby("product_category_name"):
        if g.shape[0] >= 20:
            x = np.log(g["unit_price"].clip(lower=1e-6))
            y = np.log(g["qty"].clip(lower=1e-6))
            b = np.cov(x, y)[0, 1] / (np.var(x) + 1e-9)
            df.loc[g.index, "elasticity_cat"] = b

df["elasticity_cat"] = df["elasticity_cat"].fillna(df["elasticity_cat"].median())

if "product_category_name" in df.columns:
    df = pd.get_dummies(df, columns=["product_category_name"], drop_first=True)


# ------------------------------
# EDA
# ------------------------------

print("\n--- EDA ---")

for col in ["unit_price", "total_price", "qty"]:
    if col in df.columns:
        plt.figure()
        plt.hist(df[col], bins=30)
        plt.title(f"Distribution: {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

monthly = df.groupby("month_year_dt").agg({
    "revenue": "sum",
    "qty": "sum",
    "customers": "sum",
    "unit_price": "mean"
}).reset_index().sort_values("month_year_dt")

plt.figure()
plt.plot(monthly["month_year_dt"], monthly["revenue"])
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(monthly["month_year_dt"], monthly["qty"])
plt.title("Monthly Demand (Qty) Trend")
plt.xlabel("Month")
plt.ylabel("Quantity")
plt.tight_layout()
plt.show()

prod_rev = df.groupby("product_id")["revenue"].sum().sort_values(ascending=False).head(15)
plt.figure(figsize=(10, 4))
prod_rev.plot(kind="bar")
plt.title("Top 15 Products by Revenue")
plt.xlabel("product_id")
plt.ylabel("Revenue")
plt.tight_layout()
plt.show()

# Competitor gap analysis
for i in [1, 2, 3]:
    c = f"comp{i}_diff"
    if c in df.columns:
        plt.figure()
        plt.hist(df[c].dropna(), bins=30)
        plt.title(f"Competitor {i} Price Gap (unit_price - comp_{i})")
        plt.xlabel(c)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

# Correlation heatmap (numeric only)
num_cols2 = df.select_dtypes(include=[np.number]).columns
corr = df[num_cols2].corr()

plt.figure(figsize=(11, 7))
plt.imshow(corr, aspect="auto")
plt.colorbar()
plt.title("Correlation Heatmap (Numeric Features)")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=6)
plt.yticks(range(len(corr.index)), corr.index, fontsize=6)
plt.tight_layout()
plt.show()

# Outlier spikes: show top 1% revenue rows
thr = df["revenue"].quantile(0.99)
spikes = (df["revenue"] >= thr).mean()
print(f"Share of rows in top 1% revenue: {spikes:.3f}")

# ------------------------------
# ML: Price Prediction (Regression)
# ------------------------------
print("\n--- ML: Price Prediction (target = unit_price) ---")

# Drop non-numeric and IDs for modeling
drop_cols = ["month_year", "month_year_dt"]
X = df.drop(columns=[c for c in drop_cols if c in df.columns] +
            ["unit_price"], errors="ignore")
y = df["unit_price"].copy()

# Ensure X is numeric
X = X.select_dtypes(include=[np.number]).copy()
X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale for linear models
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

models = []
models.append(("Linear Regression", LinearRegression(), X_train_sc, X_test_sc))
models.append(("Ridge Regression", Ridge(alpha=1.0), X_train_sc, X_test_sc))
models.append(("Random Forest",
               RandomForestRegressor(n_estimators=300, random_state=42),
               X_train, X_test))

models.append(("XGBoost",
               XGBRegressor(
                   n_estimators=500,
                   max_depth=5,
                   learning_rate=0.05,
                   subsample=0.9,
                   colsample_bytree=0.9,
                   random_state=42
               ),
               X_train, X_test))

results = []
trained_models = {}

for name, model, Xtr, Xte in models:
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
    trained_models[name] = model

results_df = (
    pd.DataFrame(results)
      .sort_values("RMSE")
      .reset_index(drop=True)
)

print("\nModel comparison (Price prediction):")
print(results_df)

best_price_model_name = results_df.loc[0, "Model"]
best_price_model = trained_models[best_price_model_name]
print("\nBest price model:", best_price_model_name)

# ------------------------------
# ML: Demand Model (qty) for Optimization
# ------------------------------
print("\n--- ML: Demand Model (target = qty) ---")

# Demand features should include price and competitor signals
demand_drop = ["month_year", "month_year_dt"]
Xd = df.drop(columns=[c for c in demand_drop if c in df.columns] +
             ["qty"], errors="ignore")
yd = df["qty"].copy()

Xd = Xd.select_dtypes(include=[np.number]).copy()
Xd = Xd.replace([np.inf, -np.inf], np.nan).fillna(Xd.median())

Xd_train, Xd_test, yd_train, yd_test = train_test_split(
    Xd, yd, test_size=0.2, random_state=42
)


demand_model = RandomForestRegressor(n_estimators=400, random_state=42)
demand_model.fit(Xd_train, yd_train)
yd_pred = demand_model.predict(Xd_test)

print("Demand RF -> MAE:", mean_absolute_error(yd_test, yd_pred),
      "RMSE:", np.sqrt(mean_squared_error(yd_test, yd_pred)),
      "R2:", r2_score(yd_test, yd_pred))

# ------------------------------
# Optimization: suggest optimal price for a sample row
# ------------------------------
print("\n--- Price Optimization (revenue maximization) ---")

def suggest_optimal_price(row_numeric, price_col="unit_price",
                          pct_range=0.25, steps=25):
    """
    Vary price within +/- pct_range and pick the price that maximizes
    predicted revenue.
    Uses the trained demand_model to predict qty.
    """
    base_price = float(row_numeric.get(price_col, 0.0))
    if base_price <= 0:
        base_price = float(df["unit_price"].median())

    lo = base_price * (1 - pct_range)
    hi = base_price * (1 + pct_range)
    grid = np.linspace(lo, hi, steps)

    best = {"price": base_price, "pred_qty": None, "pred_revenue": -1}

    for p in grid:
        rr = row_numeric.copy()
        rr[price_col] = p

        # demand model expects same columns as Xd_train
        xvec = pd.DataFrame([rr])[Xd_train.columns].copy()
        xvec = xvec.replace([np.inf, -np.inf], np.nan).fillna(Xd_train.median())

        qhat = float(demand_model.predict(xvec)[0])
        rev = p * max(qhat, 0)

        if rev > best["pred_revenue"]:
            best = {"price": float(p),
                    "pred_qty": float(qhat),
                    "pred_revenue": float(rev)}

    return best

# Demonstration with one random row
sample_row = df.sample(1, random_state=7).iloc[0]
row_dict = None  # placeholder

row_numeric = sample_row.to_dict()
# ensure numeric row_dict for demand model columns
row_numeric = {k: row_numeric.get(k, 0) for k in Xd_train.columns}

opt = suggest_optimal_price(row_numeric)
print("Sample current price:", float(sample_row["unit_price"]))
print("Suggested price:", opt["price"], "Pred qty:", opt["pred_qty"],
      "Pred revenue:", opt["pred_revenue"])

# ------------------------------
# 8) Time-series forecasting (monthly revenue + qty)
# ------------------------------
print("\n--- Time-Series Forecasting (Monthly) ---")

ts = monthly.set_index("month_year_dt")[["revenue", "qty"]].asfreq("MS").fillna(method="ffill")

def forecast_series(series, steps=6):
    """Forecast using SARIMAX if available; else moving average fallback."""
    if series.dropna().shape[0] >= 24:
        model = SARIMAX(series,
                        order=(1, 1, 1),
                        seasonal_order=(1, 1, 1, 12),
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        res = model.fit(disp=False)
        fc = res.forecast(steps=steps)
        return fc
    else:
        # fallback: rolling mean forecast
        last = series.iloc[-6:].mean()
        idx = pd.date_range(series.index.max() + pd.offsets.MonthBegin(1),
                            periods=steps, freq="MS")
        return pd.Series([last] * steps, index=idx)

rev_fc = forecast_series(ts["revenue"], steps=6)
qty_fc = forecast_series(ts["qty"], steps=6)

plt.figure()
plt.plot(ts.index, ts["revenue"], label="history")
plt.plot(rev_fc.index, rev_fc.values, label="forecast")
plt.title("Revenue Forecast (Next 6 months)")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(ts.index, ts["qty"], label="history")
plt.plot(qty_fc.index, qty_fc.values, label="forecast")
plt.title("Demand Forecast (Next 6 months)")
plt.xlabel("Month")
plt.ylabel("Quantity")
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------
# Clustering: segment products by pricing sensitivity
# ------------------------------
print("\n--- Clustering (KMeans) ---")

# Build product-level table
prod = df.groupby("product_id").agg({
    "unit_price": "mean",
      "qty": "mean",
    "revenue": "sum",
    "customers": "mean",
    "elasticity_cat": "mean" if "elasticity_cat" in df.columns else "mean"
}).reset_index()

# Fill missing
for c in prod.columns:
    if c != "product_id":
        prod[c] = pd.to_numeric(prod[c], errors="coerce").fillna(prod[c].median())

cluster_cols = [c for c in ["unit_price", "qty", "customers", "elasticity_cat"]
                if c in prod.columns]
Xc = prod[cluster_cols].copy()
sc = StandardScaler()
Xc_sc = sc.fit_transform(Xc)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
prod["cluster"] = kmeans.fit_predict(Xc_sc)

print("Cluster counts:\n", prod["cluster"].value_counts())

plt.figure()
plt.scatter(prod["unit_price"], prod["qty"], c=prod["cluster"], s=12)
plt.title("Product Segments: Mean Price vs Mean Qty")
plt.xlabel("Mean unit_price")
plt.ylabel("Mean qty")
plt.tight_layout()
plt.show()

# ------------------------------
# Explainable AI (SHAP / fallback importance)
# ------------------------------
print("\n--- Explainability ---")

# Use best price model if it is tree-based; otherwise use RF for explanation
explain_model = best_price_model
tree_like = ("Forest" in best_price_model_name) or ("XGBoost" in best_price_model_name)

if tree_like:
    try:
        # shap wants raw feature matrix (not scaled)
        if best_price_model_name in ["Random Forest", "XGBoost"]:
            X_explain = X_test.copy()
        else:
            X_explain = X_test.copy()

        explainer = shap.TreeExplainer(explain_model)
        shap_values = explainer.shap_values(X_explain)

        # Summary plot
        shap.summary_plot(shap_values, X_explain, show=False)
        plt.title("SHAP Summary (Price Model)")
        plt.tight_layout()
        plt.show()

        print("SHAP summary plotted.")
    except Exception as e:
        print("SHAP failed:", e)
        tree_like = False

if not tree_like:
    # Use RF for permutation importance (works for any fitted model)
    base_model = trained_models.get("Random Forest", None)
    if base_model is not None:
        imp = permutation_importance(base_model, X_test, y_test,
                                     n_repeats=5, random_state=42)
        imp_df = pd.DataFrame({
            "feature": X_test.columns,
            "importance": imp.importances_mean
        }).sort_values("importance", ascending=False).head(15)

        plt.figure(figsize=(10, 4))
        plt.bar(imp_df["feature"], imp_df["importance"])
        plt.title("Top 15 Feature Importances (Permutation)")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

        print("Permutation importances plotted.")
    else:
        print("Neither SHAP nor permutation importance available in this environment.")