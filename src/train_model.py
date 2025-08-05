import pandas as pd

# Load the dataset
df = pd.read_csv("data/training_data.csv")

# Preview
print(df.head())
print(df['clicked'].value_counts(normalize=True))
from sklearn.model_selection import train_test_split

# Define input features and target
X = df.drop(columns=['user_id', 'clicked'])
y = df['clicked']

# One-hot encode categorical features
X_encoded = pd.get_dummies(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

print("Encoded features:")
print(X_encoded.columns)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on test set
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

# Evaluate
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print(f"AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
import joblib

joblib.dump(clf, 'models/behavior_model.pkl')
print("✅ Model saved to models/behavior_model.pkl")
