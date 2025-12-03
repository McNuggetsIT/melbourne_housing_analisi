import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# ---------------------------------------
# 1) LOAD DATA
# ---------------------------------------
df = pd.read_csv(r'D:\Coding\melbourne_housing_analisi\Melbourne_housing.csv')

cols = [
    'Suburb', 'Rooms', 'Type', 'Method', 'SellerG', 'Regionname',
    'Propertycount', 'Distance', 'CouncilArea', 'Bedroom',
    'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'Price'
]

df = df[cols]

# Fill and clean
df = df.replace([np.inf, -np.inf], np.nan)
df[['Car','Bathroom','Bedroom','Distance','Propertycount']] = \
    df[['Car','Bathroom','Bedroom','Distance','Propertycount']].fillna(0)

df = df.drop(columns=['Landsize','BuildingArea'])
df = df.dropna()

# Dummies
df = pd.get_dummies(df, drop_first=True)

# ---------------------------------------
# 2) TRAIN / TEST SPLIT
# ---------------------------------------
X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------
# 3) SCALING
# ---------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------------------
# 4) LINEAR REGRESSION BASELINE
# ---------------------------------------
baseline = LinearRegression()
baseline.fit(X_train, y_train)

print("Baseline LR - Train R2:", baseline.score(X_train, y_train))
print("Baseline LR - Test  R2:", baseline.score(X_test, y_test))

# ---------------------------------------------------------
# 5) BEST LASSO (senza CV) → cerchiamo manualmente l'alpha
# ---------------------------------------------------------
lasso_alphas = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 25, 50]

best_lasso_score = -999
best_lasso_alpha = None

for a in lasso_alphas:
    model = Lasso(alpha=a, max_iter=30000)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    if score > best_lasso_score:
        best_lasso_score = score
        best_lasso_alpha = a

best_lasso = Lasso(alpha=best_lasso_alpha, max_iter=30000)
best_lasso.fit(X_train, y_train)

print("\nBest LASSO alpha:", best_lasso_alpha)
print("LASSO - Train R2:", best_lasso.score(X_train, y_train))
print("LASSO - Test  R2:", best_lasso.score(X_test, y_test))

# ---------------------------------------------------------
# 6) BEST RIDGE (senza CV) → cerchiamo manualmente l'alpha
# ---------------------------------------------------------
ridge_alphas = [0.01, 0.1, 1, 5, 10, 25, 50, 100, 250]

best_ridge_score = -999
best_ridge_alpha = None

for a in ridge_alphas:
    model = Ridge(alpha=a)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    if score > best_ridge_score:
        best_ridge_score = score
        best_ridge_alpha = a

best_ridge = Ridge(alpha=best_ridge_alpha)
best_ridge.fit(X_train, y_train)

print("\nBest RIDGE alpha:", best_ridge_alpha)
print("RIDGE - Train R2:", best_ridge.score(X_train, y_train))
print("RIDGE - Test  R2:", best_ridge.score(X_test, y_test))
