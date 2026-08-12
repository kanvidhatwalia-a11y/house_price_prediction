import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("train (1).csv")

# Features used by the Streamlit app
features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
]

# Remove rows with missing values
data = df[features + ["SalePrice"]].dropna()

X = data[features]
y = data["SalePrice"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save the new model
joblib.dump(model, "house_price_model.pkl")

print("Model trained successfully!")
print("Features:", features)
