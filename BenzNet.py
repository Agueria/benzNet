import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# --- Load Data ---
DATA_PATH = "data.xlsx"
df = pd.read_excel(DATA_PATH)

# --- Basic Cleaning: Remove Obvious Outliers & Duplicates ---
if 'year' in df.columns:
    df = df[df['year'] > 1980]
df = df.drop_duplicates()
df = df.dropna(subset=['price'])

# Remove extreme prices (top/bottom 1%)
q_low = df['price'].quantile(0.01)
q_high = df['price'].quantile(0.99)
df = df[(df['price'] >= q_low) & (df['price'] <= q_high)]

# --- Handle Categorical Features ---
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
if 'transmission' in cat_cols:
    cat_cols.remove('transmission')
if 'transmission' in df.columns:
    df = df.drop('transmission', axis=1)  # Drop problematic column

# One-Hot Encode all categorical columns (if any)
if cat_cols:
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(df[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols), index=df.index)
    df = pd.concat([df.drop(cat_cols, axis=1), encoded_df], axis=1)

# --- Drop Any Remaining Missing Data ---
df = df.dropna(axis=0)

# --- Prepare Features and Target ---
X = df.drop('price', axis=1)
y = df['price']

# --- Split and Scale Data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Define and Train Neural Network Model ---
model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(64, activation="relu"),
    Dropout(0.1),
    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=120,
    batch_size=32,
    verbose=1
)

# --- Plot Loss Curves ---
plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training/Validation Loss")
plt.legend()
plt.tight_layout()
plt.show()

# --- Model Evaluation ---
y_pred = model.predict(X_test_scaled).flatten()
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R^2 Score: {r2:.3f}")

# --- Scatter Plot: True vs Predicted ---
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("True Price")
plt.ylabel("Predicted Price")
plt.title("True vs Predicted Car Prices")
plt.tight_layout()
plt.show()

# --- Feature Importance (Permutation Importance) ---
try:
    import eli5
    from eli5.sklearn import PermutationImportance
    import warnings
    warnings.filterwarnings('ignore')
    from sklearn.linear_model import Ridge
    # Train simple linear model for feature importances
    ridge = Ridge().fit(X_train_scaled, y_train)
    perm = PermutationImportance(ridge, random_state=1).fit(X_test_scaled, y_test)
    feature_names = X.columns
    sorted_idx = perm.feature_importances_.argsort()[::-1]
    print("Top 10 Feature Importances:")
    for idx in sorted_idx[:10]:
        print(f"{feature_names[idx]}: {perm.feature_importances_[idx]:.4f}")
except ImportError:
    print("eli5 not installed: skipping feature importance.")

# --- Save Model (optional) ---
# model.save("car_price_model.h5")
