import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("data.csv")

# Input features
X = data[
    [
        "pregnancies",
        "glucose",
        "blood_pressure",
        "skin_thickness",
        "insulin",
        "bmi",
        "diabetes_pedigree",
        "age"
    ]
]

# Target
y = data["outcome"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale the data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("=" * 40)
print("DIABETES PREDICTION SYSTEM")
print("=" * 40)

print("Model: Logistic Regression")
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(f"Dataset Records: {len(data)}")

# Take patient details
print("\nPredict New Patient")
print("=" * 40)

pregnancies = float(input("Enter number of pregnancies: "))
glucose = float(input("Enter glucose level: "))
blood_pressure = float(input("Enter blood pressure: "))
skin_thickness = float(input("Enter skin thickness: "))
insulin = float(input("Enter insulin level: "))
bmi = float(input("Enter BMI: "))
diabetes_pedigree = float(input("Enter diabetes pedigree: "))
age = float(input("Enter age: "))

# Create patient data
patient = [[
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree,
    age
]]

# Scale patient data
patient_scaled = scaler.transform(patient)

# Prediction
prediction = model.predict(patient_scaled)[0]

# Probability
probability = model.predict_proba(patient_scaled)[0][1]

print("\n" + "=" * 40)

if prediction == 1:
    print("🔴 Prediction: HIGH RISK OF DIABETES")
    print(f"Estimated diabetes probability: {probability * 100:.2f}%")
else:
    print("🟢 Prediction: LOW RISK OF DIABETES")
    print(f"Estimated diabetes probability: {probability * 100:.2f}%")

print("=" * 40)
print("Project completed successfully!")