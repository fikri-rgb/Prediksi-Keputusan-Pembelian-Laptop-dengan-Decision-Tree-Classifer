# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. Load dataset
data = pd.read_csv("data_pembelian_laptop.csv")

print("Nama kolom:", data.columns)

# 2. Encoding data kategorikal
le = LabelEncoder()

data["Age"] = le.fit_transform(data["Age"])
data["Income"] = le.fit_transform(data["Income"])
data["Student"] = le.fit_transform(data["Student"])
data["CreditRating"] = le.fit_transform(data["CreditRating"])
data["BuysComputer"] = le.fit_transform(data["BuysComputer"])

print("\nDataset setelah encoding:")
print(data.head())

# 3. Pisahkan fitur dan target
X = data.drop(["SNo", "BuysComputer"], axis=1)
y = data["BuysComputer"]

print("\nFitur:", X.columns.tolist())


# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train model Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 6. Simpan model
joblib.dump(model, "model.pkl")

print("\n✅ Model Decision Tree berhasil dilatih & disimpan sebagai model.pkl")
