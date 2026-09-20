import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. Load the dataset
# --------------------------------------------------

data = pd.read_csv("car data.csv")

print("First 5 rows:")
print(data.head())

print("\nDataset shape:", data.shape)
print("\nMissing values:")
print(data.isnull().sum())


# --------------------------------------------------
# 2. Clean column names
# --------------------------------------------------

data.columns = data.columns.str.strip()

# Rename columns to make them easier to work with
data = data.rename(columns={
    "Car_Name": "car_name",
    "Year": "year",
    "Selling_Price": "selling_price",
    "Present_Price": "present_price",
    "Driven_kms": "driven_kms",
    "Fuel_Type": "fuel_type",
    "Selling_type": "selling_type",
    "Transmission": "transmission",
    "Owner": "owner"
})


# --------------------------------------------------
# 3. Basic data cleaning
# --------------------------------------------------

# Remove duplicate rows
data = data.drop_duplicates()

# Remove rows containing missing values
data = data.dropna()

print("\nDataset after cleaning:", data.shape)


# --------------------------------------------------
# 4. Feature Engineering
# --------------------------------------------------

# Current year can be changed when the model is used
current_year = 2026

# Calculate how old the car is
data["car_age"] = current_year - data["year"]

# Avoid negative ages in case the dataset contains
# a future year accidentally
data = data[data["car_age"] >= 0]


# --------------------------------------------------
# 5. Select features and target
# --------------------------------------------------

X = data[
    [
        "car_name",
        "year",
        "present_price",
        "driven_kms",
        "fuel_type",
        "selling_type",
        "transmission",
        "owner",
        "car_age"
    ]
]

y = data["selling_price"]


# --------------------------------------------------
# 6. Split numerical and categorical features
# --------------------------------------------------

categorical_features = [
    "car_name",
    "fuel_type",
    "selling_type",
    "transmission"
]

numerical_features = [
    "year",
    "present_price",
    "driven_kms",
    "owner",
    "car_age"
]


# --------------------------------------------------
# 7. Preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# --------------------------------------------------
# 8. Create the regression model
# --------------------------------------------------

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)


# --------------------------------------------------
# 9. Create complete ML pipeline
# --------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", model)
    ]
)


# --------------------------------------------------
# 10. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# --------------------------------------------------
# 11. Train the model
# --------------------------------------------------

pipeline.fit(X_train, y_train)

print("\nModel training completed!")


# --------------------------------------------------
# 12. Make predictions
# --------------------------------------------------

y_pred = pipeline.predict(X_test)


# --------------------------------------------------
# 13. Evaluate the model
# --------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("-------------------------")
print("Mean Absolute Error:", round(mae, 2))
print("Root Mean Squared Error:", round(rmse, 2))
print("R2 Score:", round(r2, 2))


# --------------------------------------------------
# 14. Compare actual and predicted prices
# --------------------------------------------------

comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("\nActual vs Predicted Prices:")
print(comparison.head(10))


# --------------------------------------------------
# 15. Visualize actual vs predicted prices
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.7)

plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Car Prices")

# Perfect prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.tight_layout()
plt.show()


# --------------------------------------------------
# 16. Predict the price of a new car
# --------------------------------------------------

new_car = pd.DataFrame({
    "car_name": ["Honda City"],
    "year": [2020],
    "present_price": [8.5],
    "driven_kms": [25000],
    "fuel_type": ["Petrol"],
    "selling_type": ["Dealer"],
    "transmission": ["Manual"],
    "owner": [0],
    "car_age": [current_year - 2020]
})

predicted_price = pipeline.predict(new_car)

print("\nPredicted Selling Price:")
print(round(predicted_price[0], 2))